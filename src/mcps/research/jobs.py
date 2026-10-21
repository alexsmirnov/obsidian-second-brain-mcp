"""Background research jobs with hybrid polling.

``web_research`` starts a job and waits a bounded time for it; callers that
miss the deadline get a job name and read the job record with
``web_research_status``. Job state lives behind the async ``JobStore``
protocol so the in-memory store can be replaced by a persistent one.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel

from mcps.research.agent import ProgressReporter, Researcher, ResearchResponse

logger = logging.getLogger("mcps.research.jobs")

_NAME_ATTEMPTS = 10

_ADJECTIVES = (
    "amber", "ancient", "autumn", "azure", "bold", "brave", "bright", "brisk",
    "calm", "careful", "clever", "cobalt", "cosmic", "crimson", "crisp", "curious",
    "dapper", "daring", "dawn", "deep", "distant", "dusty", "eager", "early",
    "electric", "emerald", "fancy", "fearless", "fierce", "floral", "fluffy",
    "frosty", "gentle", "gilded", "glad", "golden", "graceful", "grand", "green",
    "hardy", "hazy", "hidden", "humble", "icy", "indigo", "jolly", "keen", "kind",
    "lively", "lone", "lucky", "lunar", "magic", "mellow", "merry", "mighty",
    "misty", "modest", "noble", "nimble", "oaken", "odd", "olive", "pale",
    "patient", "plain", "polite", "proud", "quick", "quiet", "radiant", "rapid",
    "red", "rustic", "sandy", "scarlet", "shiny", "silent", "silver", "sleepy",
    "smooth", "solar", "steady", "stormy", "sunny", "swift", "tidy", "tiny",
    "velvet", "violet", "warm", "wild", "windy", "wise", "witty", "young", "zesty",
)
_NOUNS = (
    "acorn", "anchor", "aurora", "badger", "beacon", "birch", "bridge", "brook",
    "canyon", "castle", "cedar", "cliff", "cloud", "comet", "coral", "crane",
    "creek", "dolphin", "dragon", "eagle", "ember", "falcon", "fern", "field",
    "finch", "fjord", "forest", "fox", "garden", "glacier", "grove", "harbor",
    "hawk", "heron", "hill", "island", "jaguar", "lagoon", "lantern", "lark",
    "leaf", "lighthouse", "lotus", "maple", "meadow", "meteor", "moon", "moss",
    "mountain", "oak", "ocean", "orchard", "otter", "owl", "panda", "pebble",
    "pine", "planet", "pond", "prairie", "quartz", "rabbit", "raven", "reef",
    "ridge", "river", "robin", "sail", "salmon", "shore", "sparrow", "spring",
    "star", "stone", "summit", "swan", "thicket", "thunder", "tiger", "tulip",
    "valley", "violin", "voyage", "walrus", "waterfall", "willow", "wolf",
    "wren", "yacht", "zephyr",
)

_RUNNING_INSTRUCTIONS = (
    "Research is still running. Call web_research_status with job_id='{job_id}' "
    "now - it waits up to ~{wait_seconds} s by itself, so do not sleep between "
    "calls. Repeat while status is 'running'."
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class JobStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchJob(BaseModel):
    """Persistable job record; only the last progress message is kept."""

    job_id: str
    query: str
    status: JobStatus = JobStatus.RUNNING
    created_at: datetime
    updated_at: datetime
    progress: str | None = None
    progress_seq: int = 0
    result: ResearchResponse | None = None
    error: str | None = None


class ResearchJobResponse(BaseModel):
    """Tool output of ``web_research`` and ``web_research_status``."""

    job_id: str
    status: JobStatus
    progress: str | None = None
    result: ResearchResponse | None = None
    error: str | None = None
    instructions: str | None = None


class JobNotFoundError(LookupError):
    """Raised for unknown or expired job ids."""


class JobStore(Protocol):
    async def create(self, job: ResearchJob) -> None: ...

    async def get(self, job_id: str) -> ResearchJob | None: ...

    async def update_progress(self, job_id: str, message: str) -> None: ...

    async def complete(self, job_id: str, result: ResearchResponse) -> None: ...

    async def fail(self, job_id: str, error: str) -> None: ...


class InMemoryJobStore:
    """Dict-backed store; finished jobs are dropped lazily after ``ttl``."""

    def __init__(
        self, ttl: timedelta, now: Callable[[], datetime] = _utcnow
    ) -> None:
        self._ttl = ttl
        self._now = now
        self._jobs: dict[str, ResearchJob] = {}

    def _purge_expired(self) -> None:
        now = self._now()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status != JobStatus.RUNNING and now - job.updated_at > self._ttl
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _touch(self, job_id: str, **changes: object) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        self._jobs[job_id] = job.model_copy(
            update={**changes, "updated_at": self._now()}
        )

    async def create(self, job: ResearchJob) -> None:
        self._purge_expired()
        self._jobs[job.job_id] = job.model_copy(deep=True)

    async def get(self, job_id: str) -> ResearchJob | None:
        self._purge_expired()
        job = self._jobs.get(job_id)
        return job.model_copy(deep=True) if job else None

    async def update_progress(self, job_id: str, message: str) -> None:
        job = self._jobs.get(job_id)
        if job is not None:
            self._touch(job_id, progress=message, progress_seq=job.progress_seq + 1)

    async def complete(self, job_id: str, result: ResearchResponse) -> None:
        self._touch(job_id, status=JobStatus.COMPLETED, result=result)

    async def fail(self, job_id: str, error: str) -> None:
        self._touch(job_id, status=JobStatus.FAILED, error=error)


def random_job_name() -> str:
    """Three random lowercase words, e.g. ``amber-falcon-river``."""
    return "-".join(
        (
            secrets.choice(_ADJECTIVES),
            secrets.choice(_ADJECTIVES),
            secrets.choice(_NOUNS),
        )
    )


class ResearchJobs:
    """Runs research as background tasks and serves bounded waits on them."""

    def __init__(
        self,
        researcher: Researcher,
        store: JobStore,
        *,
        wait: timedelta,
        poll_interval: timedelta = timedelta(seconds=0.5),
        make_name: Callable[[], str] = random_job_name,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._now = now
        self._researcher = researcher
        self._store = store
        self._wait = wait
        self._poll_interval = poll_interval
        self._make_name = make_name
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self, query: str) -> str:
        job_id = await self._unique_name()
        created = self._now()
        await self._store.create(
            ResearchJob(
                job_id=job_id, query=query, created_at=created, updated_at=created
            )
        )
        task = asyncio.create_task(self._run(job_id, query))
        self._tasks[job_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))
        return job_id

    async def _unique_name(self) -> str:
        for _ in range(_NAME_ATTEMPTS):
            name = self._make_name()
            if await self._store.get(name) is None:
                return name
        raise RuntimeError("Could not generate a unique research job name")

    async def _run(self, job_id: str, query: str) -> None:
        async def report(message: str, _progress: float, _total: float | None) -> None:
            await self._store.update_progress(job_id, message)

        logger.info("Research job %s started", job_id)
        try:
            result = await self._researcher(query, progress=report)
        except asyncio.CancelledError:
            await self._store.fail(job_id, "Research was cancelled")
            logger.info("Research job %s cancelled", job_id)
            raise
        except Exception as error:
            logger.exception("Research job %s failed", job_id)
            await self._store.fail(job_id, f"{type(error).__name__}: {error}")
        else:
            await self._store.complete(job_id, result)
            logger.info("Research job %s completed", job_id)

    async def wait(
        self,
        job_id: str,
        progress: ProgressReporter | None = None,
        *,
        from_start: bool,
    ) -> ResearchJobResponse:
        """Wait up to the configured budget for the job, forwarding progress.

        ``from_start`` forwards the current progress message too; otherwise
        only messages that arrive after this call began are forwarded.
        """
        record = await self._require(job_id)
        last_seq = 0 if from_start else record.progress_seq
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._wait.total_seconds()
        while True:
            message = record.progress
            if progress and message is not None and record.progress_seq > last_seq:
                last_seq = record.progress_seq
                await progress(message, float(last_seq), None)
            remaining = deadline - loop.time()
            if record.status != JobStatus.RUNNING or remaining <= 0:
                return self._to_response(record)
            await asyncio.sleep(min(self._poll_interval.total_seconds(), remaining))
            record = await self._require(job_id)

    async def _require(self, job_id: str) -> ResearchJob:
        record = await self._store.get(job_id)
        if record is None:
            raise JobNotFoundError(job_id)
        return record

    def _to_response(self, record: ResearchJob) -> ResearchJobResponse:
        running = record.status == JobStatus.RUNNING
        return ResearchJobResponse(
            job_id=record.job_id,
            status=record.status,
            progress=record.progress if running else None,
            result=record.result,
            error=record.error,
            instructions=(
                _RUNNING_INSTRUCTIONS.format(
                    job_id=record.job_id,
                    wait_seconds=round(self._wait.total_seconds()),
                )
                if running
                else None
            ),
        )

    def cancel(self, job_id: str) -> None:
        task = self._tasks.get(job_id)
        if task is not None:
            task.cancel()

    async def aclose(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
