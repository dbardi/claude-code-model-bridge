"""The HTTP seam: the OpenAI-compatible surface clients talk to."""

import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from claude_code_model_bridge.claude_cli import ClaudeCli
from claude_code_model_bridge.translation import (
    build_invocation,
    completion_from_events,
    stream_chunks,
)


def create_app(claude_cli: ClaudeCli) -> Starlette:
    """Builds the application, taking the Claude seam as a dependency."""

    async def create_chat_completion(request: Request) -> JSONResponse:
        body = await request.json()
        invocation = build_invocation(body)
        if body.get("stream"):
            return StreamingResponse(
                _server_sent_events(
                    claude_cli.run(invocation),
                    model=body["model"],
                    include_usage=bool(
                        (body.get("stream_options") or {}).get("include_usage")
                    ),
                ),
                media_type="text/event-stream",
            )
        events = [event async for event in claude_cli.run(invocation)]
        return JSONResponse(completion_from_events(events, model=body["model"]))

    async def _server_sent_events(events, model: str, include_usage: bool):
        async for chunk in stream_chunks(events, model=model, include_usage=include_usage):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return Starlette(
        routes=[Route("/v1/chat/completions", create_chat_completion, methods=["POST"])]
    )
