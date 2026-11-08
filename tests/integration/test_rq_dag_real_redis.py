"""The pipeline DAG scenarios from ``test_pipeline_dag_parallelism`` against a real Redis.

Those tests drive ``enqueue_pipeline`` through in-process ``rq.SimpleWorker``
burst runs on fakeredis. fakeredis does not run RQ's server-side behaviour
the way Redis does, so a green unit run cannot show that an RQ upgrade keeps
the DAG working. These wrappers run the same scenarios unchanged on a real
server: dependency release, the success and failure callbacks, cancelling
the rest of a failed pipeline, retry isolation and the failed-job TTL.

Set ``REDIS_TEST_URL`` to a Redis database the tests may empty, for example
``redis://localhost:6379/15``. Without it these tests skip. With it, an
unreachable server is an error, not a skip, so CI cannot pass them silently.
The database must be empty when a test starts; the tests refuse to run on one
that holds keys, so a mistyped URL cannot wipe real data.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import pytest
import redis

import tests.unit.test_pipeline_dag_parallelism as dag
from tests.unit.test_pipeline_dag_parallelism import fake_settings  # noqa: F401 - fixture reused as-is

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration


@pytest.fixture
def real_redis() -> Iterator[redis.Redis]:
    url = os.environ.get("REDIS_TEST_URL")
    if not url:
        pytest.skip("REDIS_TEST_URL is not set")
    conn = redis.Redis.from_url(url)
    if conn.dbsize():
        pytest.fail(f"refusing to use a Redis database that already holds keys: {url}")
    try:
        yield conn
    finally:
        conn.flushdb()
        conn.close()


def test_full_dag_completes(monkeypatch, real_redis, fake_settings, tmp_path):  # noqa: F811
    dag.test_full_dag_runs_through_simple_worker_and_marks_job_completed(
        monkeypatch, real_redis, fake_settings, tmp_path
    )


def test_stage_failure_cancels_the_rest(monkeypatch, real_redis, fake_settings, tmp_path):  # noqa: F811
    dag.test_stage_failure_cancels_downstream_and_marks_job_failed(monkeypatch, real_redis, fake_settings, tmp_path)


def test_retry_ignores_stale_sibling_writes(monkeypatch, real_redis, fake_settings, tmp_path):  # noqa: F811
    dag.test_retry_isolates_new_attempt_from_stale_sibling_writes(monkeypatch, real_redis, fake_settings, tmp_path)


def test_failed_sub_jobs_expire_within_a_week(real_redis, fake_settings, tmp_path):  # noqa: F811
    dag.test_every_sub_job_expires_from_the_failed_registry_within_a_week(real_redis, fake_settings, tmp_path)


def test_stage_tasks_see_their_own_rq_job(real_redis):
    """RQ 3's get_current_job() reads a contextvar instead of a thread-local.

    The stage tasks read their generation and timeout through it; if it
    returned None they would fall back silently and the scenarios above would
    still pass. Run both readers as RQ jobs on a SimpleWorker, as the GPU
    workers do, and check that each sees the job it runs in.
    """
    from rq import Queue, SimpleWorker

    from whisper_ui.worker.stage_tasks import _current_generation, _current_job_timeout

    queue = Queue("stage-context", connection=real_redis)
    generation_job = queue.enqueue(_current_generation, meta={"generation": 7})
    timeout_job = queue.enqueue(_current_job_timeout, job_timeout=1234)

    SimpleWorker([queue], connection=real_redis).work(burst=True, with_scheduler=False)

    assert generation_job.latest_result().return_value == 7
    assert timeout_job.latest_result().return_value == 1234


def _drain_like_production(conn) -> None:
    """Run every queue the way compose does: the GPU queue on ``SimpleWorker``,
    the io, cpu and llm queues on RQ's default forking ``Worker``."""
    from rq import Queue, SimpleWorker, Worker

    from whisper_ui.core.constants import WORKER_QUEUE_CPU, WORKER_QUEUE_GPU, WORKER_QUEUE_IO, WORKER_QUEUE_LLM

    queues = {
        name: Queue(name, connection=conn)
        for name in (WORKER_QUEUE_IO, WORKER_QUEUE_GPU, WORKER_QUEUE_CPU, WORKER_QUEUE_LLM)
    }
    for _ in range(50):
        progressed = False
        for name, queue in queues.items():
            if queue.count == 0:
                continue
            worker_class = SimpleWorker if name == WORKER_QUEUE_GPU else Worker
            worker_class([queue], connection=conn).work(burst=True, with_scheduler=False)
            progressed = True
        if not progressed:
            break


def _record_callbacks(monkeypatch, conn) -> None:
    """Leave a Redis key when a pipeline callback runs.

    A forking worker runs the job, and its callbacks, in a child process, so the
    in-memory job and database stand-ins the unit scenarios assert on never
    change in the test process. Redis is shared, so the callbacks mark it.
    """
    from whisper_ui.worker import pipeline_dispatcher

    for name in ("finalize_success", "finalize_failure"):
        original = getattr(pipeline_dispatcher, name)

        def wrapper(rq_job, *args, _original=original, _name=name):
            conn.set(f"test:{_name}", rq_job.meta["parent_job_id"])
            return _original(rq_job, *args)

        monkeypatch.setattr(pipeline_dispatcher, name, wrapper)


def _stage_statuses(conn, sub_ids) -> dict:
    from rq.job import Job as RQJob

    return {
        RQJob.fetch(i, connection=conn).meta["stage"]: RQJob.fetch(i, connection=conn).get_status(refresh=True)
        for i in sub_ids
    }


def _enqueue_upload_job(conn, fake_settings, tmp_path, job_id):  # noqa: F811
    from unittest.mock import MagicMock

    from whisper_ui.core.models import Job, JobStatus
    from whisper_ui.worker import pipeline_dispatcher
    from whisper_ui.worker.pipeline_dispatcher import enqueue_pipeline

    job = Job(
        id=job_id,
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        duration=60.0,
        enable_diarization=True,
    )
    filestore = MagicMock()
    filestore.prepare_upload_path.return_value = tmp_path / "subdir" / "m.mp3"
    filestore.save_result.return_value = tmp_path / "result.json"
    enqueue_pipeline(job, redis=conn, settings=fake_settings, filestore=filestore)
    gen = pipeline_dispatcher._current_generation(conn, job.id)
    return job, filestore, pipeline_dispatcher._load_subjob_ids(conn, job.id, gen)


def test_dag_completes_on_production_worker_classes(monkeypatch, real_redis, fake_settings, tmp_path):  # noqa: F811
    """io (preprocess) -> gpu (transcribe, diarize) -> cpu (assign, postprocess),
    with the io and cpu stages in forked work horses."""
    from unittest.mock import MagicMock

    from rq.job import JobStatus as RQJobStatus

    from tests.helpers.dag_harness import fake_runtime_factory, install_runtime, stub_stages

    _record_callbacks(monkeypatch, real_redis)
    with contextlib.ExitStack() as stack:
        stack.enter_context(stub_stages(monkeypatch, []))
        db = MagicMock()
        job, filestore, sub_ids = _enqueue_upload_job(real_redis, fake_settings, tmp_path, "job-prod-workers")
        db.get_job.return_value = job
        _, builder = stack.enter_context(fake_runtime_factory(real_redis, fake_settings, db, filestore))
        install_runtime(monkeypatch, builder)
        _drain_like_production(real_redis)

    statuses = _stage_statuses(real_redis, sub_ids)
    assert set(statuses.values()) == {RQJobStatus.FINISHED}, statuses
    assert real_redis.get("test:finalize_success") == b"job-prod-workers"
    assert real_redis.get("test:finalize_failure") is None


def test_gpu_failure_cancels_forked_stages(monkeypatch, real_redis, fake_settings, tmp_path):  # noqa: F811
    """A failure in the SimpleWorker's transcribe stage cancels the cpu stages
    that the forking workers would have run."""
    from unittest.mock import MagicMock

    from rq.job import JobStatus as RQJobStatus

    from tests.helpers.dag_harness import fake_runtime_factory, install_runtime, stub_stages

    class _ExplodingTranscribe:
        name = "transcribe"

        def execute(self, context, on_progress=None):
            raise RuntimeError("gpu kaboom")

        def cleanup(self):
            pass

    _record_callbacks(monkeypatch, real_redis)
    with contextlib.ExitStack() as stack:
        stack.enter_context(stub_stages(monkeypatch, []))
        monkeypatch.setattr("whisper_ui.worker.stage_tasks.TranscribeStage", lambda *a, **kw: _ExplodingTranscribe())
        db = MagicMock()
        job, filestore, sub_ids = _enqueue_upload_job(real_redis, fake_settings, tmp_path, "job-prod-fail")
        db.get_job.return_value = job
        _, builder = stack.enter_context(fake_runtime_factory(real_redis, fake_settings, db, filestore))
        install_runtime(monkeypatch, builder)
        _drain_like_production(real_redis)

    statuses = _stage_statuses(real_redis, sub_ids)
    # cancel_remaining_subjobs cancels every sibling, so preprocess, which had
    # already finished in a forked horse, ends up CANCELED as well.
    assert statuses["run_transcribe_align"] == RQJobStatus.FAILED
    assert statuses["run_assign_speakers"] == RQJobStatus.CANCELED
    assert statuses["run_postprocess"] == RQJobStatus.CANCELED
    pending = {RQJobStatus.QUEUED, RQJobStatus.DEFERRED, RQJobStatus.READY_TO_ENQUEUE, RQJobStatus.STARTED}
    assert not {stage for stage, status in statuses.items() if status in pending}, statuses
    assert real_redis.get("test:finalize_failure") == b"job-prod-fail"
    assert real_redis.get("test:finalize_success") is None
