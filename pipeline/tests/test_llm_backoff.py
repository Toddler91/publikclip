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


def test_free_tier_limit_is_not_terminal():
    assert llm._is_terminal_429(FREE_TIER_429) is False


def test_plain_quota_message_mentioning_billing_is_not_terminal():
    assert llm._is_terminal_429(
        "You exceeded your current quota, please check your plan and billing details."
    ) is False


def test_depleted_prepay_credits_are_terminal():
    """The message that burned five retries and reported nothing useful."""
    assert llm._is_terminal_429(
        "Your prepayment credits are depleted. Please go to AI Studio at "
        "https://ai.studio/projects to manage your project and billing. Learn "
        "more at https://ai.google.dev/gemini-api/docs/billing#prepay."
    ) is True


def test_real_billing_failures_are_terminal():
    for message in (
        "Billing account not found for this project.",
        "Billing is not enabled for this project.",
        "Your credits have been exhausted.",
        "Insufficient credit balance to complete this request.",
        "Your credit balance is too low.",
        "You have run out of credits.",
        "Please enable billing to continue.",
    ):
        assert llm._is_terminal_429(message) is True, message


def test_daily_cap_detected_from_quota_id():
    """The real one. A per-DAY free-tier cap's prose is identical to a
    per-minute limit — same wording, same "Please retry in 47s" — so only the
    structured quotaId distinguishes them. Missing it retried a used-up daily
    allowance five times over."""
    payload = {
        "error": {
            "message": (
                "You exceeded your current quota, please check your plan and billing "
                "details. * Quota exceeded for metric: "
                "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                "limit: 20, model: gemini-3.7-flash\nPlease retry in 47.4s."
            ),
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                        "quotaValue": "20",
                    }],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "47s"},
            ],
        }
    }
    message = payload["error"]["message"]
    assert llm._is_terminal_429(message) is False        # prose alone cannot tell
    assert llm._is_terminal_429(message, payload) is True  # the quotaId can


def test_per_minute_quota_id_stays_retryable():
    payload = {
        "error": {
            "message": "free_tier requests, limit: 20",
            "details": [{
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [{
                    "quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
                }],
            }],
        }
    }
    assert llm._is_terminal_429(payload["error"]["message"], payload) is False


def test_quota_ids_tolerates_junk():
    assert llm._quota_ids({}) == []
    assert llm._quota_ids({"error": {"details": None}}) == []
    assert llm._quota_ids({"error": {"details": ["nonsense", 3]}}) == []


def test_daily_cap_is_terminal_even_on_free_tier():
    """A per-minute limit clears in seconds; a per-day cap does not, so
    retrying it inside one run is pure waiting."""
    assert llm._is_terminal_429(
        "Quota exceeded for metric: generate_content_free_tier_requests_per_day, "
        "limit: 50"
    ) is True


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


def test_countdown_ticks_down_once_a_second():
    """A silent wait reads as a hang; the countdown proves the run is alive."""
    seen = []
    start = time.monotonic()
    llm._sleep_with_countdown(2.5, seen.append)
    elapsed = time.monotonic() - start

    assert 2.4 <= elapsed < 4.0
    assert len(seen) >= 2
    numbers = [int(m.split("retrying in ")[1].split("s")[0]) for m in seen]
    assert numbers == sorted(numbers, reverse=True)  # counts down, never up
    assert numbers[0] <= 3 and numbers[-1] == 1


def test_countdown_without_progress_still_sleeps():
    start = time.monotonic()
    llm._sleep_with_countdown(0.3, None)
    assert time.monotonic() - start >= 0.29


def test_pacing_disabled_when_rpm_zero(monkeypatch):
    monkeypatch.setenv("PUBLIKCLIP_GEMINI_API_KEY", "test-key")
    client = llm.GeminiClient(rpm=0)
    start = time.monotonic()
    for _ in range(5):
        client._wait_for_slot()
    assert time.monotonic() - start < 0.1
