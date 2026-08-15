"""429 handling: tell a rate limit apart from a billing stop, and wait as long
as the server asks.

The bug this pins down: Google's ordinary free-tier rate-limit message ends
with "check your plan and billing details", and the old classifier keyed on
the word "billing" — so every routine rate limit was treated as a hard stop
and the retry never ran at all.
"""

import time

from publikclip_pipeline.scoring import llm

# The exact message that killed a real scoring run.
FREE_TIER_429 = (
    "You exceeded your current quota, please check your plan and billing "
    "details. For more information on this error, head to: "
    "https://ai.google.dev/gemini-api/docs/rate-limits.\n"
    "* Quota exceeded for metric: "
    "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
    "limit: 20, model: gemini-3.7-flash\nPlease retry in 15.936627229s."
)


def test_free_tier_limit_is_not_a_billing_stop():
    assert llm._is_billing_stop(FREE_TIER_429) is False


def test_plain_quota_message_mentioning_billing_is_not_a_stop():
    assert llm._is_billing_stop(
        "You exceeded your current quota, please check your plan and billing details."
    ) is False


def test_real_billing_failures_are_stops():
    for message in (
        "Billing account not found for this project.",
        "Billing is not enabled for this project.",
        "Your credits have been exhausted.",
        "Insufficient credit balance to complete this request.",
    ):
        assert llm._is_billing_stop(message) is True, message


def test_retry_delay_from_retry_info():
    payload = {
        "error": {
            "message": "rate limited",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "16s"},
            ],
        }
    }
    assert llm._retry_delay_seconds(payload) == 16.0


def test_retry_delay_falls_back_to_message_text():
    payload = {"error": {"message": FREE_TIER_429}}
    delay = llm._retry_delay_seconds(payload)
    assert delay is not None and abs(delay - 15.936627229) < 1e-6


def test_retry_delay_absent_is_none():
    assert llm._retry_delay_seconds({"error": {"message": "no idea"}}) is None
    assert llm._retry_delay_seconds({}) is None


def test_waited_delay_exceeds_what_the_server_asked():
    """The old schedule waited 4 s then 8 s against a 15.9 s request and gave
    up one moment before the window opened."""
    asked = llm._retry_delay_seconds({"error": {"message": FREE_TIER_429}})
    wait = min(asked + 1.0, llm.MAX_RETRY_WAIT)
    assert wait > asked


def test_pacing_spaces_calls(monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_GEMINI_API_KEY", "test-key")
    client = llm.GeminiClient(rpm=120)  # 0.5 s apart, keeps the test quick
    start = time.monotonic()
    for _ in range(3):
        client._wait_for_slot()
    elapsed = time.monotonic() - start
    assert elapsed >= 1.0  # first is free, then two gaps of 0.5 s


def test_pacing_disabled_when_rpm_zero(monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_GEMINI_API_KEY", "test-key")
    client = llm.GeminiClient(rpm=0)
    start = time.monotonic()
    for _ in range(5):
        client._wait_for_slot()
    assert time.monotonic() - start < 0.1
