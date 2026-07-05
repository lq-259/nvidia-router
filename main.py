import json
import logging
import uuid
from typing import Optional, Any

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from config import config
from router import route_chat, route_chat_stream, RouteError
from normalizer import ThinkingMode

logger = logging.getLogger("nvidia-router")

app = FastAPI(title="NVIDIA Model Router", version="1.0.0")
security = HTTPBearer(auto_error=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)


def verify_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Verify API key for this service. Skipped if AUTH_API_KEY is not set."""
    if not config.auth_api_key:
        return
    if credentials is None or credentials.credentials != config.auth_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ChatMessage(BaseModel):
    role: str
    content: Any = None


class ChatRequest(BaseModel):
    model_config = {"extra": "allow"}
    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stream: bool = False
    session_id: Optional[str] = Field(
        default=None,
        description="Sticky session ID. Same ID gets same model on success.",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "models": config.models}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, _=Depends(verify_key)):
    # Build body from all request fields, minus internal ones
    body = {
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
    }
    for key, val in req.model_dump().items():
        if key in ("model", "messages", "temperature", "max_tokens", "top_p", "stream", "session_id"):
            continue
        if val is not None:
            body[key] = val

    session_id = req.session_id
    thinking_mode = ThinkingMode(config.thinking_mode)

    logger.info(f"chat_completions: stream={req.stream} session={session_id} msgs={len(req.messages)} tools={'tools' in body}")

    try:
        if req.stream:
            return StreamingResponse(
                _stream_response(body, session_id, thinking_mode),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            result = await route_chat(body, session_id, thinking_mode)
            return result
    except RouteError as e:
        raise HTTPException(status_code=502, detail=str(e))


async def _stream_response(
    body: dict,
    session_id: Optional[str],
    thinking_mode: ThinkingMode,
):
    try:
        chunk_count = 0
        async for chunk in route_chat_stream(body, session_id, thinking_mode):
            if chunk_count == 0:
                logger.info(f"Stream: first chunk [{len(chunk)} chars]")
            chunk_count += 1
            yield chunk
        logger.info(f"Stream: done, {chunk_count} chunks yielded")
    except RouteError as e:
        logger.error(f"Stream error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model"} for m in config.models
        ],
    }


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get the sticky model for a session (for debugging)."""
    from router import get_sticky
    sticky = get_sticky(session_id)
    return {"session_id": session_id, "sticky_model": sticky}


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a sticky session."""
    from router import STICKY_CACHE, STICKY_LOCK
    with STICKY_LOCK:
        STICKY_CACHE.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)