"""Provider-agnostic LLM client with token, cost and latency accounting.

`complete()` returns (text, usage). In provider="mock" it runs with no API key so
the scaffold is runnable before any agent is built.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .config import settings

# USD per 1M tokens (input, output). Source: Anthropic API pricing, October 2026.
PRICING: dict[str, tuple[float, float]] = {
    "mock-model": (0.0, 0.0),
    # Anthropic — current models
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet": (3.0, 15.0),      # short alias used in .env.example
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    # OpenAI — current models
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


def _price(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return round((in_tok / 1e6) * pin + (out_tok / 1e6) * pout, 6)


class LLM:
    def __init__(self, model: str | None = None):
        self.model = model or settings.model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> tuple[str, Usage]:
        t0 = time.perf_counter()
        if settings.provider == "mock":
            text = "[mock completion] " + user[:120]
            in_tok, out_tok = len(user) // 4, len(text) // 4
        elif settings.provider == "anthropic":
            import anthropic as _anthropic  # lazy import — only needed when wired
            client = _anthropic.Anthropic(
                api_key=settings.anthropic_api_key or None,
                default_headers={"Accept-Encoding": "identity"},
            )
            response = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = response.content[0].text
            in_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
        elif settings.provider == "openai":
            from openai import OpenAI as _OpenAI  # lazy import
            client = _OpenAI(
                api_key=settings.openai_api_key or None,
                default_headers={"Accept-Encoding": "identity"},
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
            in_tok = response.usage.prompt_tokens
            out_tok = response.usage.completion_tokens
        else:
            raise ValueError(f"Unknown provider {settings.provider}")

        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=_price(self.model, in_tok, out_tok),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return text, usage

    def complete_json(
        self, system: str, user: str, max_tokens: int = 1024
    ) -> tuple[dict | list, Usage]:
        """Ask for strict JSON, parse it; retry once on parse failure.

        Accumulates Usage across both attempts so callers see the full cost.
        Raises json.JSONDecodeError if both attempts fail.
        """
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
