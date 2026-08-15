"""LLM backends: Gemini (BYO key, default) and Ollama (local fallback).

One interface: generate_json(prompt, schema, images) → dict, with disk
caching keyed on (backend, model, prompt, schema) so re-runs never re-spend
— the M2 gate requires cache hits on identical inputs.

Key resolution: PUBLIKCLIP_GEMINI_API_KEY env var, then
PUBLIKCLIP_HOME/secrets.json {"gemini_api_key": "..."} (written by the
app's onboarding). Ollama needs no key — just a running daemon.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import httpx

from .. import config

# The rolling alias, deliberately: Google retires pinned models for NEW api
# keys while still advertising them in ListModels (learned live — 404 "no
# longer available to new users" on gemini-2.5-flash with a fresh key).
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OLLAMA_URL = "http://localhost:11434"
LLM_TIMEOUT = 120.0


class LlmError(Exception):
    """User-actionable LLM failure (bad key, daemon down, model missing)."""


# Free-tier Gemini allows a small number of requests per minute, and scoring
# makes one call per candidate — 35+ on an hour-long stream. Unpaced they go
# out as fast as the network allows and the stage dies on the first 429.
# Pacing is cheaper than retrying and keeps a run inside the quota instead of
# repeatedly bouncing off it.
GEMINI_FREE_TIER_RPM = 20
MIN_CALL_INTERVAL = 60.0 / GEMINI_FREE_TIER_RPM
# Retries wait as long as the API asks: it reports its own backoff, routinely
# longer than a guess. The old fixed 4 s/8 s schedule gave up while the server
# was still asking for ~16 s.
MAX_RETRY_WAIT = 90.0
RETRY_ATTEMPTS = 5

_BILLING_PHRASES = (
    "billing account",
    "billing is not enabled",
    "billing has not been enabled",
    "credits have been exhausted",
    "credit balance",
    "insufficient credit",
)


def _retry_delay_seconds(payload: dict) -> float | None:
    """The server's requested backoff, from RetryInfo or the message text."""
    error = payload.get("error") or {}
    for detail in error.get("details") or []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("RetryInfo"):
            match = re.match(r"([\d.]+)s?$", str(detail.get("retryDelay", "")).strip())
            if match:
                return float(match.group(1))
    match = re.search(r"retry in ([\d.]+)\s*s", str(error.get("message", "")), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _is_billing_stop(message: str) -> bool:
    """True only for 'the account has actually run out of money'.

    Deliberately not keyed on the bare word "billing": Google's ordinary
    free-tier rate-limit message ends with "check your plan and billing
    details", so matching that word classified every routine rate limit as a
    hard stop and skipped the retry entirely.
    """
    low = message.lower()
    if "free_tier" in low or "free tier" in low:
        return False  # a free-tier quota metric is a rate limit, not a bill
    return any(phrase in low for phrase in _BILLING_PHRASES)


def gemini_api_key() -> str | None:
    key = os.environ.get("PUBLIKCLIP_GEMINI_API_KEY")
    if key:
        return key
    secrets_path = config.home_dir() / "secrets.json"
    if secrets_path.exists():
        try:
            return json.loads(secrets_path.read_text()).get("gemini_api_key")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_dir() -> Path:
    path = config.home_dir() / "llm-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_key(backend: str, model: str, prompt: str, schema: dict, images: list[bytes]) -> str:
    h = hashlib.sha256()
    h.update(backend.encode())
    h.update(model.encode())
    h.update(prompt.encode())
    h.update(json.dumps(schema, sort_keys=True).encode())
    for img in images:
        h.update(hashlib.sha256(img).digest())
    return h.hexdigest()[:32]


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


class GeminiClient:
    backend = "gemini"

    def __init__(self, model: str = GEMINI_MODEL, rpm: int = GEMINI_FREE_TIER_RPM):
        self.model = model
        key = gemini_api_key()
        if not key:
            raise LlmError(
                "No Gemini API key found. Add one in Settings (or set "
                "PUBLIKCLIP_GEMINI_API_KEY), or switch to Ollama mode."
            )
        self._key = key
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last_call = 0.0

    def _wait_for_slot(self) -> None:
        """Space calls out so a scoring pass stays inside the free-tier rate."""
        if self._min_interval <= 0:
            return
        gap = time.monotonic() - self._last_call
        if gap < self._min_interval:
            time.sleep(self._min_interval - gap)
        self._last_call = time.monotonic()

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        images: list[bytes] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        images = images or []
        cache_file = _cache_dir() / f"{_cache_key(self.backend, self.model, prompt, schema, images)}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        parts: list[dict[str, Any]] = [{"text": prompt}]
        for img in images:
            import base64

            parts.append(
                {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img).decode()}}
            )
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
                "temperature": 0.2,
            },
        }
        last_err: Exception | None = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                self._wait_for_slot()
                res = httpx.post(
                    GEMINI_URL.format(model=self.model),
                    params={"key": self._key},
                    json=body,
                    timeout=LLM_TIMEOUT,
                )
                if res.status_code in (401, 403):
                    raise LlmError("Gemini rejected the API key. Check it in Settings.")
                if res.status_code == 429:
                    # Surface the API's own words — a quota backoff and a
                    # "credits depleted" billing stop look identical as bare
                    # 429s but need opposite user actions.
                    try:
                        payload = res.json()
                        detail = payload["error"]["message"]
                    except Exception:  # noqa: BLE001
                        payload, detail = {}, "rate limited"
                    last_err = LlmError(f"Gemini 429: {detail}")
                    if _is_billing_stop(detail):
                        raise last_err
                    if attempt == RETRY_ATTEMPTS - 1:
                        break
                    # Honour the server's own backoff; fall back to widening
                    # waits, and add a margin so we return *after* the window
                    # rather than one moment before it opens.
                    asked = _retry_delay_seconds(payload)
                    wait = min(asked + 1.0 if asked else 5.0 * (attempt + 1), MAX_RETRY_WAIT)
                    if progress:
                        progress(f"Gemini rate limit — waiting {wait:.0f}s…")
                    time.sleep(wait)
                    continue
                res.raise_for_status()
                payload = res.json()
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(_strip_fences(text))
                cache_file.write_text(json.dumps(data))
                return data
            except LlmError:
                raise
            except (httpx.HTTPError, KeyError, json.JSONDecodeError, IndexError) as err:
                last_err = err
        raise LlmError(f"Gemini call failed after retries: {last_err}")


class OllamaClient:
    backend = "ollama"

    def __init__(self, model: str | None = None):
        try:
            res = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
            res.raise_for_status()
        except httpx.HTTPError as err:
            raise LlmError(
                "Ollama isn't running. Start it (`ollama serve`) or switch to Gemini mode."
            ) from err
        models = [m["name"] for m in res.json().get("models", [])]
        if not models:
            raise LlmError("Ollama has no models. Pull one, e.g. `ollama pull llama3.1:8b`.")
        self.model = model if model in models else _pick_ollama_model(models)

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        images: list[bytes] | None = None,
        progress: Callable[[str], None] | None = None,  # local: never rate limited
    ) -> dict:
        if images:
            # Text-only fallback: the caller records visual as signals_missing.
            images = []
        cache_file = _cache_dir() / f"{_cache_key(self.backend, self.model, prompt, schema, [])}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            res = httpx.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=600.0)
            res.raise_for_status()
            data = json.loads(_strip_fences(res.json()["message"]["content"]))
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as err:
            raise LlmError(f"Ollama call failed: {err}") from err
        cache_file.write_text(json.dumps(data))
        return data


def _pick_ollama_model(models: list[str]) -> str:
    """Prefer capable general models, and among them the LARGEST — list
    order once handed us qwen2.5:3b while 7b sat right there."""
    import re

    def size_of(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
        return float(m.group(1)) if m else 0.0

    candidates = [
        name
        for prefix in ("llama3.1", "llama3", "qwen2.5", "qwen3", "mistral", "gemma2", "gemma3")
        for name in models
        if name.startswith(prefix)
    ]
    if candidates:
        return max(candidates, key=size_of)
    return models[0]


def make_client(llm_mode: str):
    if llm_mode == "ollama":
        return OllamaClient()
    return GeminiClient()
