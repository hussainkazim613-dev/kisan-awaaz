"""Text query pipeline: RAG retrieval -> LLM generation -> response validation."""

from __future__ import annotations

from collections.abc import Callable

from app.adapters.llm.base import LLMAdapter
from app.core.errors import PipelineError, ValidationError
from app.core.logging import get_logger
from app.pipelines.response_validator import ResponseValidator
from app.rag.models import RetrievalResult

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Kisan Awaaz, an agricultural advisor helping Pakistani farmers.\n"
    "Answer ONLY using the retrieved knowledge passages provided below.\n"
    "If the knowledge base does not contain relevant information, say so clearly in Urdu.\n"
    "Respond in simple Urdu that a farmer can understand.\n"
    "Never invent chemical names, dosages, or treatment schedules.\n"
    "If uncertain, recommend visiting the local agricultural extension office."
)

UNGROUNDED_FALLBACK = (
    "معذرت، ہم مخصوص علاج کی تصدیق نہیں کر سکے۔ "
    "براہ کرم اپنے قریبی زرعی توسیع کے دفتر سے رجوع کریں۔"
)


class TextPipeline:
    """Processes text queries through RAG retrieval and LLM generation.

    Flow: validate text -> RAG retrieve -> build context -> LLM generate -> validate -> return
    """

    def __init__(
        self,
        llm_adapter: LLMAdapter,
        rag_retrieve: Callable[..., list[RetrievalResult]] | None = None,
        validator: ResponseValidator | None = None,
        rag_top_k: int = 5,
    ):
        self._llm = llm_adapter
        self._rag_retrieve = rag_retrieve
        self._validator = validator or ResponseValidator()
        self._rag_top_k = rag_top_k

    async def process(self, text: str, vision_context: dict | None = None) -> str:
        if not text or not text.strip():
            raise ValidationError("Query text must not be empty")

        from app.flow import OFF_TOPIC, is_farming_query

        if vision_context is None and not is_farming_query(text):
            logger.info("TextPipeline: off-topic query blocked (no LLM call)")
            return OFF_TOPIC

        logger.info("TextPipeline: processing query (%d chars)", len(text))

        query = self._build_query(text, vision_context)

        rag_results: list[RetrievalResult] = []
        if self._rag_retrieve is not None:
            try:
                rag_results = self._rag_retrieve(query, top_k=self._rag_top_k)
                logger.info("TextPipeline: retrieved %d RAG results", len(rag_results))
            except Exception as e:
                logger.warning("TextPipeline: RAG retrieval failed: %s", e)

        rag_texts = [r.text for r in rag_results]
        rag_sources = [r.document_id for r in rag_results]
        context = "\n\n".join(rag_texts)

        try:
            response_text = await self._llm.generate(text, context=context)
        except Exception as e:
            raise PipelineError("TextPipeline", f"LLM generation failed: {e}") from e

        report = self._validator.validate(
            response_text,
            rag_sources=rag_sources or None,
            rag_texts=rag_texts or None,
        )

        if report.has_ungrounded_treatment:
            logger.warning("TextPipeline: ungrounded treatment detected, using fallback")
            response_text = UNGROUNDED_FALLBACK
        elif report.disclaimer:
            response_text = f"{response_text}\n\n{report.disclaimer}"

        logger.info("TextPipeline: response generated (%d chars)", len(response_text))
        return response_text

    @staticmethod
    def _build_query(text: str, vision_context: dict | None) -> str:
        if not vision_context:
            return text
        crop = vision_context.get("crop", "")
        symptoms = vision_context.get("symptoms", [])
        if not crop:
            return text
        parts = [text, crop]
        if symptoms:
            parts.extend(symptoms)
        return " ".join(parts)
