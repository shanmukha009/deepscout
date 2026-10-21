"""Provider-agnostic LLM client with token, cost and latency accounting.

Resilient: provider calls use the SDK's own timeout + retries, and are wrapped in
an additional exponential-backoff loop. A call that still fails after all retries
returns empty text instead of raising, so one flaky request never aborts a whole
research run or a 150-question benchmark. When DS_CACHE=1, identical requests are
served from a local disk cache (zero cost, near-zero latency).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict

from .cache import cache, make_key
from .config import settings

# USD per 1M tokens (input, output). Source: Anthropic API pricing, October 2026.
PRICING: dict[str, tuple[float, float]] = {
    "mock-model": (0.0, 0.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}

_MAX_RETRIES = 4           # our backoff loop, on top of the SDK's own retries
_REQUEST_TIMEOUT = 60.0    # seconds per HTTP request
_SDK_RETRIES = 4           # retries inside the OpenAI/Anthropic client


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False


def _price(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return round((in_tok / 1e6) * pin + (out_tok / 1e6) * pout, 6)


class LLM:
    def __init__(self, model: str | None = None):
        self.model = model or settings.model

    def _call_provider(self, system: str, user: str, max_tokens: int) -> tuple[str, int, int]:
        if settings.provider == "mock":
            text = "[mock completion] " + user[:120]
            return text, len(user) // 4, len(text) // 4
        if settings.provider == "anthropic":
            import anthropic as _anthropic
            client = _anthropic.Anthropic(
                api_key=settings.anthropic_api_key or None,
                default_headers={"Accept-Encoding": "identity"},
                timeout=_REQUEST_TIMEOUT,
                max_retries=_SDK_RETRIES,
            )
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens
        if settings.provider == "openai":
            from openai import OpenAI as _OpenAI
            client = _OpenAI(
                api_key=settings.openai_api_key or None,
                default_headers={"Accept-Encoding": "identity"},
                timeout=_REQUEST_TIMEOUT,
                max_retries=_SDK_RETRIES,
            )
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = response.choices[0].message.content or ""
            return text, response.usage.prompt_tokens, response.usage.completion_tokens
        raise ValueError(f"Unknown provider {settings.provider}")

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> tuple[str, Usage]:
        key = make_key("complete", settings.provider, self.model, system, user, max_tokens)
        hit = cache.get(key)
        if hit is not None:
            u = hit["usage"]
            return hit["text"], Usage(u["input_tokens"], u["output_tokens"], 0.0, 0.0, cached=True)

        t0 = time.perf_counter()
        delay = 1.0
        for attempt in range(_MAX_RETRIES):
            try:
                text, in_tok, out_tok = self._call_provider(system, user, max_tokens)
                usage = Usage(
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=_price(self.model, in_tok, out_tok),
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                cache.set(key, {"text": text, "usage": asdict(usage)})
                return text, usage
            except ValueError:
                raise  # genuine config error (unknown provider) — don't mask it
            except Exception:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2  # 1s, 2s, 4s
                    continue
                # Exhausted retries: degrade gracefully instead of crashing the run.
                return "", Usage(latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        return "", Usage()

    def complete_json(
        self, system: str, user: str, max_tokens: int = 1024
    ) -> tuple[dict | list, Usage]:
        json_system = system + "\n\nRespond with valid JSON only. No prose, no markdown fences."
        text, usage = self.complete(json_system, user, max_tokens)
        try:
            return json.loads(text), usage
        except json.JSONDecodeError:
            text2, usage2 = self.complete(json_system, user, max_tokens)
            merged = Usage(
                input_tokens=usage.input_tokens + usage2.input_tokens,
                output_tokens=usage.output_tokens + usage2.output_tokens,
                cost_usd=usage.cost_usd + usage2.cost_usd,
                latency_ms=usage.latency_ms + usage2.latency_ms,
            )
            return json.loads(text2), merged
