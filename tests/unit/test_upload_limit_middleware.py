"""Tests for UploadSizeLimitMiddleware: the per-request cap on upload bodies."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.testclient import TestClient

from whisper_ui.web.middleware.upload_limit import UploadSizeLimitMiddleware, _content_length

_LIMIT = 1000
_MESSAGE = "batch too large"
_BOUNDARY = "testboundary"


def _make_app() -> tuple[FastAPI, list[int]]:
    """Build an app whose upload route records how many file bytes it saw."""
    seen: list[int] = []
    app = FastAPI()
    app.add_middleware(UploadSizeLimitMiddleware, max_bytes=_LIMIT, paths=frozenset({"/upload"}), message=_MESSAGE)

    @app.post("/upload")
    async def upload(files: Annotated[list[UploadFile], File()]):
        for f in files:
            seen.append(len(await f.read()))
        return {"ok": True}

    @app.post("/other")
    async def other(request: Request):
        return {"bytes": len(await request.body())}

    @app.get("/upload")
    async def page():
        return {"page": True}

    return app, seen


def _multipart(payload: bytes) -> bytes:
    return (
        (
            f"--{_BOUNDARY}\r\n"
            'Content-Disposition: form-data; name="files"; filename="a.mp3"\r\n'
            "Content-Type: audio/mpeg\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{_BOUNDARY}--\r\n".encode()
    )


def _chunks(body: bytes, size: int = 300):
    for i in range(0, len(body), size):
        yield body[i : i + size]


_MULTIPART_HEADERS = {"content-type": f"multipart/form-data; boundary={_BOUNDARY}"}


def test_request_under_limit_reaches_the_route():
    app, seen = _make_app()
    resp = TestClient(app).post("/upload", files=[("files", ("a.mp3", b"x" * 500, "audio/mpeg"))])
    assert resp.status_code == 200
    assert seen == [500]


def test_declared_length_over_limit_is_refused_without_reaching_the_route():
    app, seen = _make_app()
    resp = TestClient(app).post("/upload", files=[("files", ("a.mp3", b"x" * 2000, "audio/mpeg"))])
    assert resp.status_code == 413
    assert resp.text == _MESSAGE
    assert seen == []


def test_streamed_body_without_length_is_cut_off_at_the_limit():
    app, seen = _make_app()
    resp = TestClient(app).post("/upload", content=_chunks(_multipart(b"x" * 2000)), headers=_MULTIPART_HEADERS)
    assert resp.status_code == 413
    assert resp.json() == {"detail": _MESSAGE}
    assert seen == []


def test_streamed_body_under_limit_reaches_the_route():
    app, seen = _make_app()
    resp = TestClient(app).post("/upload", content=_chunks(_multipart(b"x" * 500)), headers=_MULTIPART_HEADERS)
    assert resp.status_code == 200
    assert seen == [500]


def test_unparsable_content_length_is_treated_as_absent():
    # The body is then bounded by counting alone.
    assert _content_length({"headers": [(b"content-length", b"12ab")]}) is None
    assert _content_length({"headers": []}) is None
    assert _content_length({"headers": [(b"content-length", b"42")]}) == 42


def test_other_paths_and_methods_are_not_limited():
    app, _ = _make_app()
    client = TestClient(app)
    assert client.post("/other", content=b"x" * 5000).json() == {"bytes": 5000}
    assert client.get("/upload").status_code == 200
