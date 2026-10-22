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
    assert flow.SESSIONS["923001"].state == "READY" and flow.SESSIONS["923001"].crop == "wheat"
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
