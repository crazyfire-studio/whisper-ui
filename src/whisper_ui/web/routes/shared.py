from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from whisper_ui.core.models import Job, JobStatus
from whisper_ui.storage import users_repo
from whisper_ui.web.deps import DbDep, FileStoreDep, make_content_disposition, templates

from .viewer import _MEDIA_MIME_TYPES, _VIDEO_EXTENSIONS, _build_speaker_colors

router = APIRouter(prefix="/shared")


def _shared_job(db, share_token: str) -> Job | None:
    """Return the job behind a share link, or None when the link is not live.

    A link stops working while its owner is deactivated, so deactivating an
    account also cuts off what it shared. Jobs from before accounts existed
    have no owner; only admins can see and share them.
    """
    job = db.get_job_by_share_token(share_token)
    if job is None or job.owner_id is None:
        return job
    owner = users_repo.get_user_by_id(db.conn, job.owner_id)
    return job if owner is not None and owner.is_active else None


@router.get("/{share_token}", response_class=HTMLResponse)
async def shared_viewer(request: Request, share_token: str, db: DbDep, filestore: FileStoreDep):
    job = _shared_job(db, share_token)
    if job is None or job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=404)

    result = await asyncio.to_thread(filestore.load_result, job.id)
    if result is None:
        raise HTTPException(status_code=404)

    speaker_colors = _build_speaker_colors(result.segments)
    media_path = filestore.get_any_media_path(job.id, job.filepath)
    media_available = media_path is not None
    media_is_video = media_available and media_path.suffix.lower() in _VIDEO_EXTENSIONS

    return templates.TemplateResponse(
        request=request,
        name="shared_viewer.html",
        context={
            "job": job,
            "result": result,
            "speaker_colors": speaker_colors,
            "media_available": media_available,
            "media_is_video": media_is_video,
            "share_token": share_token,
        },
    )


@router.get("/{share_token}/media")
async def shared_media(share_token: str, db: DbDep, filestore: FileStoreDep):
    job = _shared_job(db, share_token)
    if job is None:
        raise HTTPException(status_code=404)

    media_path = filestore.get_any_media_path(job.id, job.filepath)
    if media_path is None or not media_path.is_file():
        raise HTTPException(status_code=404)

    ext = media_path.suffix.lower()
    mime = _MEDIA_MIME_TYPES.get(ext, "application/octet-stream")
    filename = f"{Path(job.filename).stem}{ext}"

    return FileResponse(
        path=media_path,
        media_type=mime,
        headers={"Content-Disposition": make_content_disposition(filename, "inline")},
    )
