"""Conversation flow: language choice, farmer gate, off-topic guard.

Everything here is rule-based (zero LLM tokens). The LLM is only called once a
farmer has chosen a language AND named a real crop.
"""
from __future__ import annotations

import contextvars
import re
import time
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

# Per-request context, read by the LLM adapter and voice pipeline.
current_language: contextvars.ContextVar[str] = contextvars.ContextVar("current_language", default="ur")
current_crop: contextvars.ContextVar[str] = contextvars.ContextVar("current_crop", default="")

# code -> (name for the LLM, whisper code, aliases)
LANGUAGES: dict[str, dict] = {
    "ur": {"name": "Urdu", "whisper": "ur", "aliases": ["1", "urdu", "اردو"]},
    "pa": {"name": "Punjabi (Shahmukhi, Arabic script, as spoken in Pakistan)", "whisper": "ur",
           "aliases": ["2", "punjabi", "panjabi", "پنجابی"]},
    "sd": {"name": "Sindhi (Arabic script)", "whisper": "sd",
           "aliases": ["3", "sindhi", "sindi", "سندھی", "سنڌي"]},
    "ps": {"name": "Pashto", "whisper": "ps",
           "aliases": ["4", "pashto", "pushto", "pukhto", "پشتو", "پښتو"]},
    "skr": {"name": "Saraiki (Arabic script)", "whisper": "ur",
            "aliases": ["5", "saraiki", "seraiki", "siraiki", "سرائیکی", "سرایکی"]},
}

# canonical -> (urdu display, aliases)
CROPS: dict[str, tuple[str, list[str]]] = {
    "wheat": ("گندم", ["gandum", "gandam", "wheat", "گندم"]),
    "cotton": ("کپاس", ["kapas", "cotton", "کپاس"]),
    "rice": ("چاول", ["chawal", "dhan", "dhaan", "rice", "چاول", "دھان", "دهان"]),
    "sugarcane": ("گنا", ["ganna", "sugarcane", "گنا", "گنے"]),
    "maize": ("مکئی", ["makai", "maize", "corn", "مکئی", "مکی"]),
    "potato": ("آلو", ["aloo", "alu", "potato", "آلو"]),
    "tomato": ("ٹماٹر", ["tamatar", "tomato", "ٹماٹر"]),
    "onion": ("پیاز", ["pyaz", "piyaz", "onion", "پیاز"]),
    "chilli": ("مرچ", ["mirch", "chilli", "chili", "مرچ"]),
    "mustard": ("سرسوں", ["sarson", "mustard", "سرسوں"]),
    "gram": ("چنا", ["chana", "gram", "چنا", "چنے"]),
    "mango": ("آم", ["aam", "mango", "آم"]),
    "citrus": ("کینو", ["kinnow", "malta", "citrus", "orange", "کینو", "مالٹا", "مالٹے"]),
    "banana": ("کیلا", ["kela", "کیلا"]),
    "guava": ("امرود", ["amrood", "guava", "امرود"]),
    "sunflower": ("سورج مکھی", ["sunflower", "سورج"]),
    "vegetables": ("سبزیاں", ["sabzi", "sabziyan", "vegetable", "vegetables", "سبزی", "سبزیاں"]),
    "fodder": ("چارہ", ["chara", "fodder", "berseem", "چارہ", "برسیم"]),
}

# Words that mark a message as farming-related (besides crop names).
AGRI_WORDS = set("""
fasal fasl faslon kheti khet kheton bimari bemar bimar keera keray kiray patta patte pattay khad khaad
pani paani sinchai spray dawa dawai dawaii beej bij zameen zamin mitti paudha poda podha phal sabzi
ugana ugaon ugana boni kaat katai nursery tukhm sundi sundiyan makhi sool
crop crops plant leaf leaves disease pest pests fertilizer fertiliser soil seed seeds irrigation farm
farming insect insects fungus spray yellow wilt rust blight weed harvest yield urea dap
فصل فصلوں کھیت کھیتی بیماری بیمار کیڑا کیڑے پتے پتا پتوں کھاد پانی سپرے دوا دوائی بیج زمین مٹی پودا پودے
پودوں پھل سبزی کاشت سنڈی سنڈیاں پھپھوندی زنگ مکھی سوکھ پیلا پیلے یوریا ڈی اے پی
""".split())

TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def match_language(text: str) -> str | None:
    toks = _tokens(text)
    for code, info in LANGUAGES.items():
        for alias in info["aliases"]:
            if alias in toks:
                return code
    return None


def match_crop(text: str) -> str | None:
    toks = set(_tokens(text))
    for canonical, (_, aliases) in CROPS.items():
        if any(a in toks for a in aliases):
            return canonical
    return None


TOPICS = {
    "disease and pests": ["1", "bimari", "bemar", "bimar", "keera", "keray", "kiray", "disease", "pest",
                          "بیماری", "بیمار", "کیڑا", "کیڑے", "سنڈی"],
    "fertilizer": ["2", "khad", "khaad", "urea", "dap", "fertilizer", "fertiliser", "کھاد", "یوریا"],
    "irrigation and water": ["3", "pani", "paani", "sinchai", "water", "irrigation", "پانی"],
    "identifying a problem from a photo": ["4", "photo", "tasveer", "tasweer", "image", "picture", "تصویر", "فوٹو"],
}
GREETINGS = {"hi", "hello", "hey", "salam", "salaam", "assalam", "assalamualaikum", "aoa", "slam",
             "سلام", "السلام", "علیکم", "ہیلو"}


def match_topic(text: str) -> str | None:
    toks = set(_tokens(text))
    for name, aliases in TOPICS.items():
        if any(a in toks for a in aliases):
            return name
    return None


def is_greeting(text: str) -> bool:
    toks = _tokens(text)
    return bool(toks) and all(t in GREETINGS for t in toks)


def is_farming_query(text: str) -> bool:
    toks = set(_tokens(text))
    if toks & AGRI_WORDS:
        return True
    return any(a in toks for _, aliases in CROPS.values() for a in aliases)


WELCOME = "السلام علیکم! میں کسان آواز ہوں، آپ کا فصل ڈاکٹر۔ 🌾\n\n"
LANG_MENU = (
    "اپنی زبان چنیں (نمبر بھیجیں یا نام بولیں):\n"
    "1️⃣ اردو\n2️⃣ پنجابی\n3️⃣ سندھی\n4️⃣ پشتو\n5️⃣ سرائیکی"
)
CROP_Q = "شکریہ! اب بتائیں، آپ کون سی فصل اگاتے ہیں؟ (مثلاً گندم، کپاس، چاول، گنا، آلو)"
CROP_RETRY = "معذرت، مجھے فصل کا نام سمجھ نہیں آیا۔ براہ کرم اپنی فصل کا نام بتائیں، جیسے گندم، کپاس یا ٹماٹر۔"
NOT_FARMER = (
    "کسان آواز صرف فصلوں اور کاشتکاری کے سوالات کے لیے ہے۔ "
    "اگر آپ کاشتکار ہیں تو اپنی فصل کا نام لکھیں یا بولیں۔"
)
TOPIC_Q = (
    "آپ کو کس بارے میں مدد چاہیے؟ نمبر بھیجیں یا بولیں:\n"
    "1️⃣ بیماری یا کیڑے\n2️⃣ کھاد\n3️⃣ پانی / آبپاشی\n4️⃣ بیمار پتے کی تصویر سے پہچان"
)
TOPIC_RETRY = "براہ کرم 1، 2، 3 یا 4 بھیجیں، یا بتائیں: بیماری، کھاد، پانی یا تصویر۔"
LIMIT_MSG = "آج کے لیے آپ کے سوالات کی حد پوری ہو گئی ہے۔ کل دوبارہ پوچھیں۔ شکریہ!"
READY_MSG = (
    "ٹھیک ہے! آپ کی فصل: {crop}۔\n"
    "اب فصل کی بیماری، کیڑوں، کھاد یا پانی کے بارے میں سوال پوچھیں، یا بیمار پتے کی تصویر بھیجیں۔\n"
    "زبان بدلنے کے لیے 'زبان' لکھیں۔"
)
OFF_TOPIC = (
    "معذرت، میں صرف فصلوں اور کاشتکاری کے سوالات کا جواب دے سکتا ہوں۔ "
    "براہ کرم اپنی فصل، بیماری، کیڑے، کھاد یا پانی کے بارے میں پوچھیں۔"
)
RESET_WORDS = {"زبان", "zaban", "language", "reset", "menu"}

SESSION_TTL = 24 * 3600
DAILY_LIMIT = 20


@dataclass
class Session:
    state: str = "NEW"  # NEW -> ASK_LANG -> ASK_CROP -> ASK_TOPIC -> READY
    lang: str = "ur"
    crop: str = ""
    fails: int = 0
    topic: str = ""
    strikes: int = 0  # off-topic strikes
    day: str = ""
    used: int = 0  # questions answered today
    last_seen: float = field(default_factory=time.time)


SESSIONS: dict[str, Session] = {}


def _get_session(number: str) -> Session:
    s = SESSIONS.get(number)
    if s is None or time.time() - s.last_seen > SESSION_TTL:
        s = Session()
        SESSIONS[number] = s
    s.last_seen = time.time()
    return s


async def _transcribe(router, whatsapp, message) -> str:
    """Transcribe a voice note during onboarding (Whisper, no LLM)."""
    vp = getattr(router, "_voice_pipeline", None)
    if vp is None or not message.media_url:
        return ""
    try:
        audio = await whatsapp.download_media(message.media_url)
        return await vp._speech.transcribe(audio, language="ur")
    except Exception as e:  # noqa: BLE001
        logger.warning("Flow: onboarding transcription failed: %s", e)
        return ""


async def handle_message(router, whatsapp, message) -> tuple[str, str | None]:
    """Return (reply_text, voice_language). voice_language is set only when the
    farmer sent a voice note, so the reply should also be spoken."""
    s = _get_session(message.from_number)
    is_audio = message.media_type == "audio"
    text = (message.body or "").strip()

    if s.state == "READY":
        if not is_audio and text.lower() in RESET_WORDS:
            s.state, s.fails = "ASK_LANG", 0
            return LANG_MENU, None
        if not is_audio and is_greeting(text):
            return READY_MSG.format(crop=CROPS[s.crop][0]), None
        today = time.strftime("%Y-%m-%d")
        if s.day != today:
            s.day, s.used = today, 0
        if s.used >= DAILY_LIMIT:
            return LIMIT_MSG, None
        current_language.set(s.lang)
        current_crop.set(f"{s.crop}; main concern: {s.topic}")
        reply = await router.route(message)
        if reply == OFF_TOPIC:
            s.strikes += 1
            if s.strikes >= 3:  # not interested in farming: re-qualify, no more LLM calls
                s.state, s.fails, s.strikes = "ASK_CROP", 0, 0
                return NOT_FARMER, None
        else:
            s.used += 1
            s.strikes = 0
        return reply, (s.lang if is_audio else None)

    voice = "ur" if is_audio else None

    if s.state == "NEW":
        s.state = "ASK_LANG"
        return WELCOME + LANG_MENU, voice

    if is_audio:
        text = await _transcribe(router, whatsapp, message)

    if s.state == "ASK_LANG":
        code = match_language(text)
        if code is None:
            s.fails += 1
            if s.fails < 3:
                return LANG_MENU, voice
            code = "ur"  # language is not a gate; default to Urdu
        s.lang, s.state, s.fails = code, "ASK_CROP", 0
        return CROP_Q, s.lang if is_audio else None

    # ASK_CROP: the farmer gate. No LLM call happens until a crop is named.
    if s.state == "ASK_CROP":
        crop = match_crop(text)
        if crop is None:
            s.fails += 1
            return (CROP_RETRY if s.fails < 3 else NOT_FARMER), (s.lang if is_audio else None)
        s.crop, s.state, s.fails = crop, "ASK_TOPIC", 0
        return TOPIC_Q, (s.lang if is_audio else None)

    # ASK_TOPIC
    topic = match_topic(text)
    if topic is None:
        s.fails += 1
        return (TOPIC_RETRY if s.fails < 3 else NOT_FARMER), (s.lang if is_audio else None)
    s.topic, s.state, s.fails = topic, "READY", 0
    return READY_MSG.format(crop=CROPS[s.crop][0]), (s.lang if is_audio else None)
