"""Central config, read from the environment. See .env.example."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    provider: str = os.getenv("DS_PROVIDER", "mock")  # mock | anthropic | openai
    model: str = os.getenv("DS_MODEL", "mock-model")
    search_provider: str = os.getenv("DS_SEARCH", "mock")  # mock | tavily | serper
    max_sub_questions: int = int(os.getenv("DS_MAX_SUBQ", "5"))
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    search_api_key: str = os.getenv("SEARCH_API_KEY", "")


settings = Settings()
