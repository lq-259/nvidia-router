import asyncio
import copy
import json
import logging
import time
import threading
from typing import AsyncIterator, Optional

import httpx
from config import config, Model, MODEL_EXTRA
from normalizer import ThinkingMode, normalize_response, StreamNormalizer

logger = logging.getLogger("nvidia-router")

STICKY_CACHE: dict[str, tuple[str, float]] = {}
STICKY_LOCK = threading.Lock()
KEY_ROUND_ROBIN = 0
KEY_RR_LOCK = threading.Lock()


def _clean_expired():
    now = time.time()
    with STICKY_LOCK:
        expired = [k for k, (_, ts) in STICKY_CACHE.items() if now - ts > config.sticky_ttl]
        for k in expired:
            del STICKY_CACHE[k]


def get_sticky(session_id: str) -> Optional[str]:
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


def _build_probe_models(sticky: Optional[str]) -> list[Model]:
    """Assign one round-robin key to each model for probing."""
    api_keys = config.api_keys
    if not api_keys:
        raise RuntimeError("NVIDIA_API_KEYS is not configured")

    global KEY_ROUND_ROBIN
    with KEY_RR_LOCK:
        start_idx = KEY_ROUND_ROBIN % len(api_keys)
        KEY_ROUND_ROBIN += 1

    models = []
    for i, model_name in enumerate(config.models):
        key = api_keys[(start_idx + i) % len(api_keys)]
        extra = MODEL_EXTRA.get(model_name, {})
        models.append(Model(name=model_name, api_key=key, extra_body=extra))

    if sticky:
        sticky_models = [m for m in models if m.name == sticky]
        other_models = [m for m in models if m.name != sticky]
        models = sticky_models + other_models

    return models


def _is_fatal(status: int) -> bool:
    return status in (429, 503)


async def _probe_one(
    client: httpx.AsyncClient,
    model: Model,
    sample_msg: str,
) -> tuple[Model, Optional[float], Optional[str]]:
    """Send a lightweight probe. Returns (model, latency_seconds, error)."""
    url = f"{model.base_url}/chat/completions"
    request_body = {
        "model": model.model_id,
        "messages": [{"role": "user", "content": sample_msg}],
        "max_tokens": 1,
    }
    # Turn off thinking for probe — faster, save tokens
    if model.extra_body:
        probe_extra = copy.deepcopy(model.extra_body)
        if "chat_template_kwargs" in probe_extra:
            probe_extra["chat_template_kwargs"] = {
                **probe_extra["chat_template_kwargs"],
                "thinking": False,
            }
        request_body = {**request_body, **probe_extra}

    start = time.monotonic()
    try:
        resp = await client.post(
            url,
            json=request_body,
            headers={
                "Authorization": f"Bearer {model.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.timeout, connect=5.0),
        )
        latency = time.monotonic() - start
        if resp.status_code == 200:
            return model, latency, None
        else:
            return model, None, f"{resp.status_code}"
    except httpx.TimeoutException:
        return model, None, "timeout"
    except Exception as e:
        return model, None, type(e).__name__


async def _try_single(
    client: httpx.AsyncClient,
    model: Model,
    body: dict,
) -> tuple[Optional[dict], Optional[str]]:
    url = f"{model.base_url}/chat/completions"
    request_body = {**body, "model": model.model_id}
    if "stream" in request_body:
        request_body.pop("stream")
    if model.extra_body:
        request_body = {**request_body, **model.extra_body}

    # Log message types and tool_call_id presence for debugging
    msgs = request_body.get("messages", [])
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    missing_tcid = any("tool_call_id" not in m for m in tool_msgs)
    if missing_tcid and config.debug:
        logger.warning(f"Request to {model.name}: {len(msgs)} msgs, {len(tool_msgs)} tool msgs, SOME MISSING tool_call_id!")
        for i, m in enumerate(tool_msgs):
            if "tool_call_id" not in m:
                logger.warning(f"  tool msg[{i}]: keys={list(m.keys())}")

    try:
        resp = await client.post(
            url,
            json=request_body,
            headers={
                "Authorization": f"Bearer {model.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.full_timeout, connect=10.0),
        )
        if resp.status_code == 200:
            return resp.json(), None
        elif _is_fatal(resp.status_code):
            return None, f"{model.name}:{resp.status_code}"
        else:
            return None, f"{model.name}:{resp.status_code}({resp.text[:200]})"
    except (httpx.TimeoutException, asyncio.CancelledError, Exception) as e:
        return None, f"{model.name}:{type(e).__name__}"


async def route_chat(
    body: dict,
    session_id: Optional[str] = None,
    thinking_mode: ThinkingMode = ThinkingMode.normalize,
) -> dict:
    """Route via concurrent probe + full request."""
    logger.info(f"route_chat: session={session_id}, msgs={len(body.get('messages',[]))}, has_tools={'tools' in body}")
    if len(body.get('messages', [])) > 50:
        logger.warning(f"route_chat: large context ({len(body['messages'])} msgs), may need longer timeout")
    sticky = get_sticky(session_id) if session_id else None
    if sticky and config.debug:
        logger.info(f"route_chat: sticky hit -> {sticky}")
    probe_models = _build_probe_models(sticky)
    sample_msg = _extract_sample(body)

    async with httpx.AsyncClient() as client:
        if sticky:
            resp, err = await _try_single(client, probe_models[0], body)
            if resp is not None:
                if config.debug:
                    logger.info(f"route_chat: sticky success -> {sticky}")
                return normalize_response(resp, thinking_mode)

        # Phase 1: Race probes, first to respond gets full request
        probe_tasks = {
            asyncio.create_task(_probe_one(client, m, sample_msg)): m
            for m in probe_models
        }
        ranked: list[tuple[Model, float]] = []
        pending = set(probe_tasks.keys())

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                try:
                    result = task.result()
                except (asyncio.CancelledError, Exception):
                    continue
                if isinstance(result, tuple) and result[1] is not None:
                    winner, probe_lat = result[0], result[1]
                    ranked.append((winner, probe_lat))

            if not ranked and not pending:
                raise RouteError("All models failed to respond to probe")

            if not ranked:
                continue

            # Try full request to the fastest probe, racing against remaining probes
            winner, probe_lat = ranked[0]
            full_task = asyncio.create_task(_try_single(client, winner, body))

            full_done, pending = await asyncio.wait(
                pending | {full_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if full_task in full_done:
                try:
                    resp, err = full_task.result()
                except (asyncio.CancelledError, Exception):
                    resp, err = None, "exception"
                if resp is not None:
                    for t in pending:
                        t.cancel()
                    if session_id:
                        set_sticky(session_id, winner.name)
                    if config.debug:
                        logger.info(f"route_chat: full request success -> {winner.name}")
                    return normalize_response(resp, thinking_mode)
                # Full request failed, remove from ranked
                ranked.pop(0)
                pending.discard(full_task)
            else:
                # Another probe finished before full request completed
                new_probes = False
                for task in full_done:
                    try:
                        result = task.result()
                    except (asyncio.CancelledError, Exception):
                        continue
                    if isinstance(result, tuple) and result[1] is not None:
                        ranked.append((result[0], result[1]))
                        new_probes = True
                if new_probes:
                    full_task.cancel()
                    try:
                        ranked.sort(key=lambda x: float(x[1]))
                    except (ValueError, TypeError):
                        pass
                else:
                    # Only failed probes completed, wait for full request
                    await full_task
                    try:
                        resp, err = full_task.result()
                    except (asyncio.CancelledError, Exception):
                        resp, err = None, "exception"
                    if resp is not None:
                        if session_id:
                            set_sticky(session_id, winner.name)
                        return normalize_response(resp, thinking_mode)
                    ranked.pop(0)

            if not pending and not ranked:
                raise RouteError("All models failed to respond to probe")

    # Phase 2: Fallback — try remaining probes in speed order
    errors: list[str] = []
    async with httpx.AsyncClient() as client:
        for model, probe_lat in ranked:
            if model is None:
                continue
            resp, err = await _try_single(client, model, body)
            if resp is not None:
                if session_id:
                    set_sticky(session_id, model.name)
                if config.debug:
                    logger.info(f"route_chat: fallback success -> {model.name}")
                return normalize_response(resp, thinking_mode)
            if err:
                errors.append(f"{model.name}({probe_lat:.1f}s):{err}")

    logger.error(f"route_chat: all failed: {errors}")
    raise RouteError(f"All models failed: {'; '.join(errors)}")


def _extract_sample(body: dict) -> str:
    msgs = body.get("messages", [])
    for m in reversed(msgs):
        if m.get("role") == "user":
            content = m.get("content", "hi")
            if isinstance(content, str):
                return content[:100]
    return "hi"


async def route_chat_stream(
    body: dict,
    session_id: Optional[str] = None,
    thinking_mode: ThinkingMode = ThinkingMode.normalize,
) -> AsyncIterator[str]:
    """Streaming version: probe first, then stream from fastest model."""
    sticky = get_sticky(session_id) if session_id else None
    if sticky and config.debug:
        logger.info(f"route_chat_stream: sticky hit -> {sticky}")
    if len(body.get('messages', [])) > 50:
        logger.warning(f"route_chat_stream: large context ({len(body['messages'])} msgs)")
    probe_models = _build_probe_models(sticky)
    sample_msg = _extract_sample(body)

    async with httpx.AsyncClient() as client:
        if sticky:
            async for chunk in _stream_from_model(client, probe_models[0], body, session_id, thinking_mode):
                yield chunk
            return

        probe_tasks = {
            asyncio.create_task(_probe_one(client, m, sample_msg)): m
            for m in probe_models
        }
        ranked: list[tuple[Model, float]] = []
        pending = set(probe_tasks.keys())

        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                try:
                    result = task.result()
                except (asyncio.CancelledError, Exception):
                    continue
                if isinstance(result, tuple) and result[1] is not None:
                    ranked.append((result[0], result[1]))

            if not ranked:
                continue

            if not pending and not ranked:
                yield f"data: {json.dumps({'error': 'All models failed to respond to probe'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            winner, probe_lat = ranked[0]
            if config.debug:
                logger.info(f"route_chat_stream: probe winner -> {winner.name} ({probe_lat:.1f}s)")
            async for chunk in _stream_from_model(client, winner, body, session_id, thinking_mode):
                yield chunk
            return

        if not ranked:
            yield f"data: {json.dumps({'error': 'All models failed to respond to probe'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        errors: list[str] = []
        for model, probe_lat in ranked:
            had_content = False
            async for chunk in _stream_from_model(client, model, body, session_id, thinking_mode):
                if chunk.startswith('data: {"choices"'):
                    had_content = True
                yield chunk
            if had_content:
                return
            errors.append(f"{model.name}({probe_lat:.1f}s)")

        errors_str = "; ".join(errors)
        yield f"data: {json.dumps({'error': f'All models failed: {errors_str}'})}\n\n"
        yield "data: [DONE]\n\n"


async def _stream_from_model(
    client: httpx.AsyncClient,
    model: Model,
    body: dict,
    session_id: Optional[str],
    thinking_mode: ThinkingMode,
) -> AsyncIterator[str]:
    url = f"{model.base_url}/chat/completions"
    request_body = {**body, "model": model.model_id, "stream": True}
    if model.extra_body:
        request_body = {**request_body, **model.extra_body}

    logger.info(f"Streaming from {model.name} key={model.api_key[:12]}...")

    try:
        async with client.stream(
            "POST",
            url,
            json=request_body,
            headers={
                "Authorization": f"Bearer {model.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.full_timeout, connect=10.0),
        ) as resp:
            logger.info(f"Stream response: {model.name} status={resp.status_code}")
            if resp.status_code == 200:
                if session_id:
                    set_sticky(session_id, model.name)
                normalizer = StreamNormalizer(thinking_mode)
                async for line in resp.aiter_lines():
                    for result in normalizer.feed(line):
                        yield result
            else:
                logger.warning(f"Stream failed: {model.name} status={resp.status_code}")
                yield "data: [DONE]\n\n"
    except (httpx.TimeoutException, asyncio.CancelledError, Exception) as e:
        logger.error(f"Stream exception from {model.name}: {type(e).__name__}: {e}")
        yield "data: [DONE]\n\n"


class RouteError(Exception):
    pass