"""The HTTP seam: the OpenAI-compatible surface clients talk to."""

import asyncio
import json

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from claude_code_model_bridge.catalog import ModelCatalog, UnknownModel
from claude_code_model_bridge.claude_cli import ClaudeCli
from claude_code_model_bridge.failures import failure_from, failure_in
from claude_code_model_bridge.request_log import RequestRecord
from claude_code_model_bridge.translation import (
    build_invocation,
    completion_from_events,
    stream_chunks,
    usage_from_events,
)


def create_app(
    claude_cli: ClaudeCli, catalog: ModelCatalog, max_concurrent: int = 4
) -> Starlette:
    """Builds the application, taking the Claude seam as a dependency.

    `max_concurrent` caps how many calls run at once. Callers make requests
    of their own accord (summaries, titles, retries), and each one costs
    from the same usage window, so the bridge holds the rest waiting rather
    than starting everything at once.
    """
    running = asyncio.Semaphore(max_concurrent)

    async def create_chat_completion(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            resolution = catalog.resolve(body["model"])
        except UnknownModel:
            return _unknown_model(body["model"])
        invocation = build_invocation(body, resolution)
        streaming = bool(body.get("stream"))
        record = RequestRecord(
            model=body["model"],
            tools=bool(body.get("tools")),
            streaming=streaming,
        )
        if streaming:
            events = _limited(claude_cli.run(invocation))
            try:
                opening = await _until_output(events)
            except Exception as error:
                return _reported(failure_from(error), record)
            failure = failure_in(opening)
            if failure is not None:
                return _reported(failure, record)
            return StreamingResponse(
                _server_sent_events(
                    _replayed(opening, events),
                    model=body["model"],
                    include_usage=bool(
                        (body.get("stream_options") or {}).get("include_usage")
                    ),
                    record=record,
                ),
                media_type="text/event-stream",
            )
        try:
            events = [event async for event in _limited(claude_cli.run(invocation))]
        except Exception as error:
            return _reported(failure_from(error), record)
        record.note_usage(usage_from_events(events))
        failure = failure_in(events)
        if failure is not None:
            return _reported(failure, record)
        completion = completion_from_events(events, model=body["model"])
        record.finished()
        return JSONResponse(completion)

    async def _until_output(events) -> list:
        """Reads ahead until the answer starts, so failures still have a status.

        Once a stream has begun there is no status left to send: the headers
        are gone. A failed run produces no output at all, so reading up to
        the first content event costs nothing and keeps the failure
        reportable.
        """
        opening = []
        async for event in events:
            opening.append(event)
            if event.get("type") in ("stream_event", "assistant"):
                break
            if event.get("type") == "result":
                break
        return opening

    async def _replayed(opening: list, rest):
        """Re-emits what was read ahead, then continues with the live run."""
        for event in opening:
            yield event
        async for event in rest:
            yield event

    async def _limited(events):
        """Runs one call, waiting its turn if too many are already running."""
        async with running:
            async for event in events:
                yield event

    async def _server_sent_events(events, model: str, include_usage: bool, record):
        outcome = "success"
        try:
            async for chunk in stream_chunks(events, model=model, include_usage=True):
                usage = chunk.get("usage")
                if usage:
                    record.note_usage(usage)
                    if not include_usage:
                        continue
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            outcome = "failed"
            raise
        finally:
            record.finished(outcome)

    async def list_models(request: Request) -> JSONResponse:
        return JSONResponse(catalog.listing())

    def _reported(failure, record) -> JSONResponse:
        """Records the failure, then hands the caller what it needs to react."""
        record.finished(failure.code)
        return JSONResponse(
            failure.body(), status_code=failure.status, headers=failure.headers()
        )

    def _unknown_model(model: str) -> JSONResponse:
        """Reports an unusable model id the way callers expect to read it."""
        return JSONResponse(
            {
                "error": {
                    "message": f"The model `{model}` does not exist.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
            status_code=404,
        )

    return Starlette(
        routes=[
            Route("/v1/chat/completions", create_chat_completion, methods=["POST"]),
            Route("/v1/models", list_models, methods=["GET"]),
        ]
    )
