"""Contract tests for background research jobs (hybrid polling)."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta

import pytest

from mcps.research.agent import ResearchResponse
from mcps.research.jobs import (
    InMemoryJobStore,
    JobNotFoundError,
    JobStatus,
    ResearchJob,
    ResearchJobs,
    random_job_name,
)
from tests.research_fakes import FakeClock, FakeResearcher

_RESULT: ResearchResponse = {"answer": "x", "explanation": "e", "sources": ["s"]}


class ProgressRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float | None]] = []

    async def __call__(
        self, message: str, progress: float, total: float | None
    ) -> None:
        self.calls.append((message, progress, total))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def make_jobs(clock):
    created: list[ResearchJobs] = []

    def _make(
        fake: FakeResearcher, *, names: list[str] | None = None
    ) -> tuple[ResearchJobs, InMemoryJobStore]:
        store = InMemoryJobStore(ttl=timedelta(hours=1), now=clock.now)
        kwargs = {"make_name": iter(names).__next__} if names else {}
        jobs = ResearchJobs(
            fake,
            store,
            wait=timedelta(seconds=0.2),
            poll_interval=timedelta(seconds=0.01),
            now=clock.now,
            **kwargs,
        )
        created.append(jobs)
        return jobs, store

    yield _make
    for jobs in created:
        await jobs.aclose()


async def test_fast_job_returns_completed_result(make_jobs):
    fake = FakeResearcher()
    fake.release.set()
    jobs, _ = make_jobs(fake)

    response = await jobs.wait(await jobs.start("q"), from_start=True)

    assert response.status == JobStatus.COMPLETED
    assert response.result["answer"] == "answer:q"
    assert response.instructions is None
    assert response.progress is None


async def test_slow_job_returns_running_with_instructions(make_jobs):
    jobs, _ = make_jobs(FakeResearcher())

    response = await jobs.wait(await jobs.start("q"), from_start=True)

    assert response.status == JobStatus.RUNNING
    assert response.job_id in response.instructions
    assert "web_research_status" in response.instructions


def test_default_job_name_is_three_words():
    names = {random_job_name() for _ in range(50)}

    assert all(re.fullmatch(r"[a-z]+-[a-z]+-[a-z]+", name) for name in names)
    assert len(names) > 1


async def test_duplicate_name_regenerated(make_jobs):
    jobs, _ = make_jobs(FakeResearcher(), names=["a-b-c", "a-b-c", "d-e-f"])

    first = await jobs.start("q1")
    second = await jobs.start("q2")

    assert (first, second) == ("a-b-c", "d-e-f")


async def test_progress_forwarded_while_waiting(make_jobs):
    jobs, _ = make_jobs(FakeResearcher(before=["a"]))
    recorder = ProgressRecorder()

    response = await jobs.wait(
        await jobs.start("q"), progress=recorder, from_start=True
    )

    assert response.status == JobStatus.RUNNING
    assert recorder.calls == [("a", 1.0, None)]
    assert response.progress == "a"


async def test_record_keeps_last_progress_only(make_jobs):
    fake = FakeResearcher(before=["a"], after=["b", "c"])
    fake.release.set()
    jobs, store = make_jobs(fake)
    job_id = await jobs.start("q")

    response = await jobs.wait(job_id, from_start=True)

    record = await store.get(job_id)
    assert response.status == JobStatus.COMPLETED
    assert (record.progress, record.progress_seq) == ("c", 3)


async def test_status_wait_forwards_only_new_progress(make_jobs):
    fake = FakeResearcher(before=["a"], after=["b"])
    jobs, _ = make_jobs(fake)
    job_id = await jobs.start("q")
    first = await jobs.wait(job_id, from_start=True)
    fake.release.set()
    recorder = ProgressRecorder()

    second = await jobs.wait(job_id, progress=recorder, from_start=False)

    assert first.status == JobStatus.RUNNING
    assert second.status == JobStatus.COMPLETED
    assert recorder.calls == [("b", 2.0, None)]


async def test_research_error_marks_failed(make_jobs):
    fake = FakeResearcher(error=RuntimeError("boom"))
    fake.release.set()
    jobs, _ = make_jobs(fake)

    response = await jobs.wait(await jobs.start("q"), from_start=True)

    assert response.status == JobStatus.FAILED
    assert response.error == "RuntimeError: boom"


async def test_unknown_job_raises_not_found(make_jobs):
    jobs, _ = make_jobs(FakeResearcher())

    with pytest.raises(JobNotFoundError):
        await jobs.wait("no-such-job", from_start=False)


async def test_wait_relies_on_store_without_local_task(make_jobs, clock):
    jobs, store = make_jobs(FakeResearcher())
    await store.create(
        ResearchJob(
            job_id="x-y-z", query="q", created_at=clock.now(), updated_at=clock.now()
        )
    )

    async def finish_later() -> None:
        await asyncio.sleep(0.05)
        await store.complete("x-y-z", _RESULT)

    finisher = asyncio.create_task(finish_later())
    response = await jobs.wait("x-y-z", from_start=False)
    await finisher

    assert response.status == JobStatus.COMPLETED
    assert response.result == _RESULT


async def test_finished_job_expires_after_ttl(make_jobs, clock):
    fake = FakeResearcher()
    fake.release.set()
    jobs, store = make_jobs(fake)
    job_id = await jobs.start("q")
    await jobs.wait(job_id, from_start=True)

    clock.advance(timedelta(hours=2))

    assert await store.get(job_id) is None
    with pytest.raises(JobNotFoundError):
        await jobs.wait(job_id, from_start=False)


async def test_running_job_not_expired(make_jobs, clock):
    jobs, store = make_jobs(FakeResearcher())
    job_id = await jobs.start("q")

    clock.advance(timedelta(hours=2))

    record = await store.get(job_id)
    assert record is not None
    assert record.status == JobStatus.RUNNING


async def test_cancel_marks_failed(make_jobs):
    fake = FakeResearcher()
    jobs, _ = make_jobs(fake)
    job_id = await jobs.start("q")
    await asyncio.sleep(0)

    jobs.cancel(job_id)
    response = await jobs.wait(job_id, from_start=False)

    assert response.status == JobStatus.FAILED
    assert "cancelled" in response.error
    assert fake.cancelled


async def test_aclose_cancels_running_jobs(make_jobs):
    fake = FakeResearcher()
    jobs, store = make_jobs(fake)
    job_id = await jobs.start("q")
    await asyncio.sleep(0)

    await jobs.aclose()

    assert fake.cancelled
    assert (await store.get(job_id)).status == JobStatus.FAILED


async def test_store_record_round_trips_json(clock):
    store = InMemoryJobStore(ttl=timedelta(hours=1), now=clock.now)
    await store.create(
        ResearchJob(
            job_id="a-b-c", query="q", created_at=clock.now(), updated_at=clock.now()
        )
    )
    await store.update_progress("a-b-c", "step")
    await store.complete("a-b-c", _RESULT)

    got = await store.get("a-b-c")

    assert got is not None
    assert got.result is not None
    assert ResearchJob.model_validate_json(got.model_dump_json()) == got
    assert got.result["answer"] == "x"
