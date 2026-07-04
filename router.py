import asyncio
import time
import threading
from collections import OrderedDict
from typing import AsyncIterator, Optional

import httpx
from config import config, Model, MODEL_EXTRA
from normalizer import ThinkingMode, normalize_response, StreamNormalizer

STICKY_CACHE: dict[str, tuple[str, float]] = {}
STICKY_LOCK = threading.Lock()
KEY_ROUND_ROBIN = 0
KEY_RR_LOCK = threading.Lock()


def _clean_expired():
    """Remove expired sticky entries."""
    now = time.time()
    with STICKY_LOCK:
        expired = [k for k, (_, ts) in STICKY_CACHE.items() if now - ts > config.sticky_ttl]
        for k in expired:
            del STICKY_CACHE[k]


def get_sticky(session_id: str) -> Optional[str]:
    """Return the sticky model name for a session, or None."""
    _clean_expired()
    with STICKY_LOCK:
        entry = STICKY_CACHE.get(session_id)
        if entry:
            model_name, ts = entry
            if time.time() - ts <= config.sticky_ttl:
                return model_name
            del STICKY_CACHE[session_id]
    return None


def set_sticky(session_id: str, model_name: str):
    with STICKY_LOCK:
        STICKY_CACHE[session_id] = (model_name, time.time())


def _build_ordered_models(sticky: Optional[str]) -> list[Model]:
    """Return models in the order they should be tried.

    If a sticky model exists, it goes first; the rest follow in default order.
    API keys are round-robined: each request starts from the next key in rotation.
    """
    api_keys = config.api_keys
    if not api_keys:
        raise RuntimeError("NVIDIA_API_KEYS is not configured")

    global KEY_ROUND_ROBIN
    with KEY_RR_LOCK:
        start_idx = KEY_ROUND_ROBIN % len(api_keys)
        KEY_ROUND_ROBIN += 1

    # Rotate keys so the round-robin key comes first
    rotated_keys = api_keys[start_idx:] + api_keys[:start_idx]

    models = []
    for model_name in config.models:
        extra = MODEL_EXTRA.get(model_name, {})
        for key in rotated_keys:
            models.append(Model(name=model_name, api_key=key, extra_body=extra))

    if sticky:
        sticky_models = [m for m in models if m.name == sticky]
        other_models = [m for m in models if m.name != sticky]
        models = sticky_models + other_models

    return models


def _is_retryable(status: int) -> bool:
    return status in (429, 500, 502, 503, 504)


def _is_fatal(status: int) -> bool:
    """Status codes that mean this model should be skipped permanently this request."""
    return status in (429, 503)


async def _try_single(
    client: httpx.AsyncClient,
    model: Model,
    body: dict,
) -> tuple[Optional[dict], Optional[str]]:
    """Try a single model. Returns (response_json, error_reason)."""
    url = f"{model.base_url}/chat/completions"

    request_body = {**body, "model": model.model_id}
    if "stream" in request_body:
        request_body.pop("stream")

    if model.extra_body:
        request_body = {**request_body, **model.extra_body}

    try:
        resp = await client.post(
            url,
            json=request_body,
            headers={
                "Authorization": f"Bearer {model.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.timeout, connect=10.0),
        )

        if resp.status_code == 200:
            return resp.json(), None
        elif _is_fatal(resp.status_code):
            return None, f"{model.name}:{resp.status_code}"
        else:
            return None, f"{model.name}:{resp.status_code}({resp.text[:200]})"

    except httpx.TimeoutException:
        return None, f"{model.name}:timeout"
    except Exception as e:
        return None, f"{model.name}:{type(e).__name__}"


async def route_chat(
    body: dict,
    session_id: Optional[str] = None,
    thinking_mode: ThinkingMode = ThinkingMode.normalize,
) -> dict:
    """Route a chat completion request through the model list.

    Strategy:
    1. If session_id is provided, try the sticky model first.
    2. Fall through the remaining models in order.
    3. On first success, update sticky cache and return.
    4. Normalize thinking/reasoning content for consistency.
    """
    sticky = get_sticky(session_id) if session_id else None
    models = _build_ordered_models(sticky)

    fatal_models: set[str] = set()
    errors: list[str] = []

    async with httpx.AsyncClient() as client:
        for model in models:
            if model.name in fatal_models:
                continue

            resp, err = await _try_single(client, model, body)

            if resp is not None:
                if session_id:
                    set_sticky(session_id, model.name)
                return normalize_response(resp, thinking_mode)

            if err:
                errors.append(err)
                if "429" in err or "503" in err:
                    fatal_models.add(model.name)

    raise RouteError(f"All models failed: {'; '.join(errors)}")


async def route_chat_stream(
    body: dict,
    session_id: Optional[str] = None,
    thinking_mode: ThinkingMode = ThinkingMode.normalize,
) -> AsyncIterator[str]:
    """Streaming version: yields normalized SSE chunks from the first successful model.

    Uses StreamNormalizer to unify thinking content across different models.
    """
    sticky = get_sticky(session_id) if session_id else None
    models = _build_ordered_models(sticky)

    fatal_models: set[str] = set()
    errors: list[str] = []

    async with httpx.AsyncClient() as client:
        for model in models:
            if model.name in fatal_models:
                continue

            url = f"{model.base_url}/chat/completions"
            request_body = {**body, "model": model.model_id, "stream": True}
            if model.extra_body:
                request_body = {**request_body, **model.extra_body}

            try:
                async with client.stream(
                    "POST",
                    url,
                    json=request_body,
                    headers={
                        "Authorization": f"Bearer {model.api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=httpx.Timeout(config.timeout, connect=10.0),
                ) as resp:
                    if resp.status_code == 200:
                        if session_id:
                            set_sticky(session_id, model.name)
                        normalizer = StreamNormalizer(thinking_mode)
                        async for line in resp.aiter_lines():
                            for result in normalizer.feed(line):
                                yield result
                        for result in normalizer.feed("data: [DONE]"):
                            yield result
                        return
                    elif _is_fatal(resp.status_code):
                        fatal_models.add(model.name)
                        errors.append(f"{model.name}:{resp.status_code}")
                    else:
                        errors.append(f"{model.name}:{resp.status_code}")

            except (httpx.TimeoutException, Exception) as e:
                errors.append(f"{model.name}:{type(e).__name__}")

    raise RouteError(f"All models failed: {'; '.join(errors)}")


class RouteError(Exception):
    pass