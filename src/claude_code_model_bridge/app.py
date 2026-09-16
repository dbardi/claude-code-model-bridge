"""The HTTP seam: the OpenAI-compatible surface Hermes talks to."""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from claude_code_model_bridge.claude_cli import ClaudeCli
from claude_code_model_bridge.translation import build_invocation, completion_from_events


def create_app(claude_cli: ClaudeCli) -> Starlette:
    """Builds the application, taking the Claude seam as a dependency."""

    async def create_chat_completion(request: Request) -> JSONResponse:
        body = await request.json()
        invocation = build_invocation(body)
        events = [event async for event in claude_cli.run(invocation)]
        return JSONResponse(completion_from_events(events, model=body["model"]))

    return Starlette(
        routes=[Route("/v1/chat/completions", create_chat_completion, methods=["POST"])]
    )
