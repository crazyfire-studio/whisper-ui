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
