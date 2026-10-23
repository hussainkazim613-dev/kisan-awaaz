import asyncio
from types import SimpleNamespace

from app import flow


class FakeRouter:
    _voice_pipeline = None

    async def route(self, m):
        return "LLM-CALLED"


def msg(body, n="923001", media=None):
    return SimpleNamespace(from_number=n, body=body, media_type=media, media_url=None)


def run(coro):
    return asyncio.run(coro)


def test_gate_and_flow():
    flow.SESSIONS.clear()
    r = FakeRouter()
    reply, _ = run(flow.handle_message(r, None, msg("hi")))
    assert "اپنی زبان" in reply
    reply, _ = run(flow.handle_message(r, None, msg("2")))
    assert flow.SESSIONS["923001"].lang == "pa"
    reply, _ = run(flow.handle_message(r, None, msg("burger banana hai")))
    assert reply == flow.CROP_RETRY and flow.SESSIONS["923001"].state == "ASK_CROP"
    reply, _ = run(flow.handle_message(r, None, msg("meri gandum ki fasal")))
    assert flow.SESSIONS["923001"].state == "ASK_TOPIC" and flow.SESSIONS["923001"].crop == "wheat"
    reply, _ = run(flow.handle_message(r, None, msg("burger")))
    assert reply == flow.TOPIC_RETRY
    reply, _ = run(flow.handle_message(r, None, msg("1")))
    assert flow.SESSIONS["923001"].state == "READY"
    reply, _ = run(flow.handle_message(r, None, msg("hi")))
    assert "فصل" in reply and reply != "LLM-CALLED"
    reply, _ = run(flow.handle_message(r, None, msg("gandum mein keera hai")))
    assert reply == "LLM-CALLED"
    reply, _ = run(flow.handle_message(r, None, msg("زبان")))
    assert "اپنی زبان" in reply


def test_off_topic_guard():
    assert not flow.is_farming_query("burger banane ka tareeqa batao")
    assert not flow.is_farming_query("who won the cricket match")
    assert flow.is_farming_query("mera tamatar bemar hai")
    assert flow.is_farming_query("کپاس میں سنڈی لگ گئی ہے")
    assert flow.is_farming_query("wheat leaf yellow")


def test_strikes_and_limit():
    flow.SESSIONS.clear()

    class Off(FakeRouter):
        async def route(self, m):
            return flow.OFF_TOPIC

    s = flow._get_session("9230")
    s.state, s.crop, s.topic = "READY", "wheat", "fertilizer"
    for _ in range(3):
        reply, _ = run(flow.handle_message(Off(), None, msg("burger", n="9230")))
    assert reply == flow.NOT_FARMER and s.state == "ASK_CROP"
    s.state, s.used, s.day = "READY", flow.DAILY_LIMIT, __import__("time").strftime("%Y-%m-%d")
    reply, _ = run(flow.handle_message(FakeRouter(), None, msg("gandum khad", n="9230")))
    assert reply == flow.LIMIT_MSG
