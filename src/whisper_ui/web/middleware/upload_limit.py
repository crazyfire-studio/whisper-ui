"""Bound the size of an upload request before its body reaches the disk.

The upload route takes ``File()`` parameters, so Starlette parses the whole
multipart body into spooled temporary files before the handler runs; the
per-file ``MAX_UPLOAD_SIZE`` check in ``routes/upload.py`` only sees a file
after every byte of the request has been written. Without a bound on the
request itself, one POST can fill the frontend container's disk.

This pure ASGI middleware refuses a declared ``Content-Length`` over the
limit without reading the body, and counts the body as it streams so a
request without one (chunked) is cut off once it passes the limit. The
mid-stream case raises ``HTTPException``, which FastAPI re-raises from body
parsing rather than turning it into a 400, so both paths answer 413.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from starlette.exceptions import HTTPException
from starlette.responses import PlainTextResponse

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)


class UploadSizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int, paths: frozenset[str], message: str) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.paths = paths
        self.message = message

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.max_bytes:
            logger.warning("upload request refused: content_length=%d max=%d", declared, self.max_bytes)
            await PlainTextResponse(self.message, status_code=413)(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    logger.warning("upload request cut off: received=%d max=%d", received, self.max_bytes)
                    raise HTTPException(status_code=413, detail=self.message)
            return message

        await self.app(scope, limited_receive, send)


def _content_length(scope: Scope) -> int | None:
    for name, value in scope["headers"]:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
