from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.adapters.whatsapp.webhook import parse_incoming_message, validate_twilio_signature
from app.admin import admin_router
from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.errors import (
    KisanAwaazError,
    kisanaawaaz_error_handler,
    unhandled_error_handler,
)
from app.models.conversation import Conversation
from app.models.database import ConversationStore

logger = get_logger(__name__)


def _build_router_and_whatsapp():
    """Wire Phase 1-5 adapters and pipelines into the MessageRouter.

    Heavy imports stay inside this function so importing app.main stays light.
    Each pipeline is only built when its credentials/artifacts are available;
    the router degrades gracefully to Urdu placeholders otherwise.
    """
    from app.adapters.llm.qwen_adapter import QwenLLMAdapter
    from app.adapters.speech.whisper_adapter import WhisperAdapter
    from app.adapters.vision.qwen_adapter import QwenVisionAdapter
    from app.adapters.whatsapp.meta_adapter import MetaWhatsAppAdapter
    from app.adapters.whatsapp.twilio_adapter import TwilioAdapter
    from app.pipelines.image_pipeline import ImagePipeline
    from app.pipelines.response_validator import ResponseValidator
    from app.pipelines.text_pipeline import TextPipeline
    from app.pipelines.voice_pipeline import VoicePipeline
    from app.router import MessageRouter

    # Prefer Meta's WhatsApp Cloud API when configured (free, no region
    # restrictions); fall back to Twilio otherwise.
    if settings.meta_access_token and settings.meta_phone_number_id:
        whatsapp = MetaWhatsAppAdapter(
            access_token=settings.meta_access_token,
            phone_number_id=settings.meta_phone_number_id,
            api_version=settings.meta_api_version,
        )
        logger.info("Router wiring: using Meta WhatsApp Cloud API adapter")
    else:
        whatsapp = TwilioAdapter(
            account_sid=settings.twilio_account_sid,
            auth_token=settings.twilio_auth_token,
            whatsapp_number=settings.twilio_whatsapp_number,
        )
        logger.info("Router wiring: using Twilio WhatsApp adapter")

    rag_retrieve = None
    if Path(settings.faiss_index_path).exists() and Path(settings.metadata_path).exists():
        try:
            from app.rag.embeddings import SentenceTransformerEmbeddings
            from app.rag.service import RAGService

            rag_service = RAGService(
                raw_dir=settings.rag_raw_dir,
                processed_dir=settings.rag_processed_dir,
                embedding_provider=SentenceTransformerEmbeddings(settings.embedding_model),
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            )
            rag_service.load_index()
            rag_retrieve = rag_service.retrieve
            logger.info("Router wiring: RAG retrieval enabled")
        except Exception as e:
            logger.warning("RAG index unavailable, continuing without RAG: %s", e)
    else:
        logger.info("No FAISS index at %s; pipelines run without RAG", settings.faiss_index_path)

    validator = ResponseValidator()

    llm_adapter = None
    if settings.dashscope_api_key:
        llm_adapter = QwenLLMAdapter(
            api_key=settings.dashscope_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
        )

    text_pipeline = None
    if llm_adapter is not None:
        text_pipeline = TextPipeline(
            llm_adapter=llm_adapter,
            rag_retrieve=rag_retrieve,
            validator=validator,
            rag_top_k=settings.rag_top_k,
        )
        logger.info("Router wiring: text pipeline enabled")

    image_pipeline = None
    if settings.dashscope_api_key:
        image_pipeline = ImagePipeline(
            vision_adapter=QwenVisionAdapter(
                api_key=settings.dashscope_api_key,
                model=settings.vision_model,
                timeout=settings.vision_timeout,
            ),
            rag_retrieve=rag_retrieve,
            llm_adapter=llm_adapter,
            validator=validator,
            max_image_bytes=settings.vision_max_image_bytes,
            rag_top_k=settings.rag_top_k,
        )
        logger.info("Router wiring: image pipeline enabled")

    voice_pipeline = None
    if settings.openai_api_key and text_pipeline is not None:
        voice_pipeline = VoicePipeline(
            speech_adapter=WhisperAdapter(
                api_key=settings.openai_api_key,
                model=settings.whisper_model,
                timeout=settings.speech_timeout,
                max_audio_bytes=settings.speech_max_audio_bytes,
            ),
            text_pipeline=text_pipeline,
            max_audio_bytes=settings.speech_max_audio_bytes,
        )
        logger.info("Router wiring: voice pipeline enabled")

    router = MessageRouter(
        image_pipeline=image_pipeline,
        whatsapp_adapter=whatsapp,
        text_pipeline=text_pipeline,
        voice_pipeline=voice_pipeline,
    )
    return router, whatsapp


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Kisan Awaaz starting up | env=%s | log_level=%s", settings.app_env, settings.log_level)
    router, whatsapp = _build_router_and_whatsapp()
    conversation_store = ConversationStore(settings.database_url)
    conversation_store.init_schema()
    app.state.router = router
    app.state.whatsapp = whatsapp
    app.state.conversation_store = conversation_store
    yield
    logger.info("Kisan Awaaz shutting down")


app = FastAPI(
    title="Kisan Awaaz",
    description="WhatsApp-based AI agricultural advisory for Pakistani farmers",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_exception_handler(KisanAwaazError, kisanaawaaz_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_error_handler)  # type: ignore[arg-type]
app.include_router(admin_router)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "kisan-awaaz",
        "version": app.version,
        "env": settings.app_env,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist_conversation(
    request: Request,
    message,
    reply: str,
    created_at: str,
    responded_at: str,
) -> None:
    store = getattr(request.app.state, "conversation_store", None)
    if store is None:
        logger.warning("Webhook: conversation store not wired; skipping persistence")
        return
    conversation = Conversation(
        farmer_phone=message.from_number,
        message_in=message.body or None,
        media_type={"image": "image", "audio": "voice"}.get(message.media_type, "text"),
        media_url=message.media_url,
        response_out=reply,
        created_at=created_at,
        responded_at=responded_at,
    )
    try:
        store.save(conversation)
    except Exception as e:
        logger.warning("Webhook: failed to persist conversation: %s", e)


async def _process_and_reply(request: Request, message) -> JSONResponse | dict:
    """Shared pipeline: route an already-parsed WhatsAppMessage, reply, persist."""
    router = getattr(request.app.state, "router", None)
    whatsapp = getattr(request.app.state, "whatsapp", None)
    if router is None or whatsapp is None:
        logger.error("Webhook: router not wired (lifespan did not run)")
        return JSONResponse(status_code=503, content={"error": "Service not ready"})

    created_at = _utc_now_iso()
    from app.flow import handle_message

    reply, voice_lang = await handle_message(router, whatsapp, message)
    responded_at = _utc_now_iso()
    await whatsapp.send_text(message.from_number, reply)
    if voice_lang and hasattr(whatsapp, "send_audio_bytes"):
        try:
            from app.tts import synthesize

            await whatsapp.send_audio_bytes(message.from_number, await synthesize(reply, voice_lang))
        except Exception as e:  # voice is a bonus; text already sent
            logger.warning("Webhook: voice reply failed: %s", e)
    _persist_conversation(request, message, reply, created_at, responded_at)
    return {"status": "ok"}


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """Twilio incoming WhatsApp webhook: validate signature -> parse -> route -> reply."""
    # Twilio posts application/x-www-form-urlencoded; parsed with the stdlib
    # because starlette's request.form() requires python-multipart.
    raw_body = await request.body()
    form = dict(parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))

    auth_token = settings.twilio_auth_token
    signature = request.headers.get("X-Twilio-Signature", "")
    if auth_token:
        if not validate_twilio_signature(str(request.url), form, signature, auth_token):
            logger.warning("Webhook: rejected request with invalid Twilio signature")
            return JSONResponse(status_code=403, content={"error": "Invalid Twilio signature"})
    else:
        logger.warning("Webhook: TWILIO_AUTH_TOKEN not set; skipping signature validation")

    message = parse_incoming_message(form)
    return await _process_and_reply(request, message)


@app.get("/webhook/whatsapp/meta")
async def meta_webhook_verify(request: Request):
    """Meta's one-time webhook verification handshake.

    When you enter your webhook URL + verify token in the Meta app dashboard,
    Meta sends this GET request. You must echo back hub.challenge exactly
    (as plain text) if hub.verify_token matches your configured value.
    """
    from fastapi.responses import PlainTextResponse

    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == settings.meta_verify_token and challenge:
        logger.info("Meta webhook: verification handshake succeeded")
        return PlainTextResponse(challenge)

    logger.warning("Meta webhook: verification failed (mode=%s, token_match=%s)", mode, token == settings.meta_verify_token)
    return JSONResponse(status_code=403, content={"error": "Verification failed"})


@app.post("/webhook/whatsapp/meta")
async def meta_webhook(request: Request):
    """Meta WhatsApp Cloud API incoming webhook: validate signature -> parse -> route -> reply."""
    from app.adapters.whatsapp.meta_webhook import parse_meta_incoming_message, validate_meta_signature

    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256", "")

    if settings.meta_app_secret:
        if not validate_meta_signature(raw_body, signature, settings.meta_app_secret):
            logger.warning("Meta webhook: rejected request with invalid signature")
            return JSONResponse(status_code=403, content={"error": "Invalid signature"})
    else:
        logger.warning("Meta webhook: META_APP_SECRET not set; skipping signature validation")

    payload = await request.json()
    message = parse_meta_incoming_message(payload)
    if message is None:
        # Status callback (sent/delivered/read), not an actual incoming message.
        return {"status": "ignored"}

    return await _process_and_reply(request, message)
