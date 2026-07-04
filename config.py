import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Model:
    """Represent a single NVIDIA NIM model endpoint."""
    name: str
    api_key: str
    extra_body: dict = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return "https://integrate.api.nvidia.com/v1"

    @property
    def model_id(self) -> str:
        return self.name


@dataclass
class Config:
    api_keys: list[str] = field(default_factory=lambda: _load_api_keys())
    models: list[str] = field(default_factory=lambda: _load_models())
    timeout: float = float(os.getenv("REQUEST_TIMEOUT", "30"))
    sticky_ttl: int = int(os.getenv("STICKY_TTL", "300"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "0"))
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "10"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    thinking_mode: str = os.getenv("THINKING_MODE", "normalize")
    auth_api_key: str = os.getenv("AUTH_API_KEY", "")


def _load_api_keys() -> list[str]:
    raw = os.getenv("NVIDIA_API_KEYS", "")
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


def _load_models() -> list[str]:
    raw = os.getenv("NVIDIA_MODELS", "")
    if not raw:
        return DEFAULT_MODELS
    return [m.strip() for m in raw.split(",") if m.strip()]


DEFAULT_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek-ai/deepseek-v4-flash",
    "minimaxai/minimax-m2.7",
    "z-ai/glm-5.1",
]

MODEL_EXTRA = {
    "deepseek-ai/deepseek-v4-flash": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
    "deepseek-ai/deepseek-v4-pro": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
    "moonshotai/kimi-k2.6": {
        "chat_template_kwargs": {"thinking": True}
    },
    "z-ai/glm-5.2": {
        "chat_template_kwargs": {"thinking": True}
    },
}

config = Config()