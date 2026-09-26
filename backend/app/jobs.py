"""
Background job store for long-running analysis.

Why this exists
---------------
Segmenting a 1000x1000 slide takes 25-80 seconds depending on the CPU the
container gets. The reverse proxy in front of the app gives up after about
15 seconds and returns HTTP 504, so a synchronous `/api/predict` cannot work in
production: a 512x512 image (9 tiles) already timed out on the deployed host
while a 256x256 image (1 tile, ~5 s) succeeded.

So the browser does not wait for the work. It POSTs the slide, receives a job id
immediately, and polls until the job is done. The proxy only ever sees short
requests.

Everything is in-memory: this is a single-process demo service with no database,
and results are disposable. The store is bounded in both count and age so a long
uptime cannot leak memory.
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# Keep only the most recent few results; the UI needs the newest one.
MAX_COMPLETED_JOBS = 3
# Forget results after this long, whether or not the client came back for them.
JOB_TTL_SECONDS = 15 * 60
# Analyses are CPU-bound, so running more than one at a time only makes every
# request slower. Extra requests queue.
MAX_WORKERS = 1


@dataclass
class Job:
    id: str
    status: str = "pending"          # pending -> running -> done | error
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.created

    def summary(self, include_result: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "job_id": self.id,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed(), 1),
        }
        if self.status == "error":
            payload["error"] = self.error
        if include_result and self.status == "done":
            payload.update(self.result or {})
        return payload


class JobStore:
    def __init__(self,
                 max_completed: int = MAX_COMPLETED_JOBS,
                 ttl_seconds: int = JOB_TTL_SECONDS,
                 max_workers: int = MAX_WORKERS):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._max_completed = max_completed
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------ write
    def submit(self, work: Callable[[], Dict[str, Any]]) -> Job:
        """Queue `work` and return its job immediately."""
        job = Job(id=uuid.uuid4().hex[:16])
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        self._executor.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            job.status = "running"
            job.started = time.time()
        try:
            result = work()
            with self._lock:
                job.result = result
                job.status = "done"
                job.finished = time.time()
        except Exception as exc:                       # noqa: BLE001 - reported to the client
            with self._lock:
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "error"
                job.finished = time.time()
        finally:
            with self._lock:
                self._evict_locked()

    def _evict_locked(self) -> None:
        """Drop expired jobs, then the oldest finished ones beyond the cap."""
        now = time.time()
        for jid in [j.id for j in self._jobs.values()
                    if j.finished and now - j.finished > self._ttl]:
            self._jobs.pop(jid, None)

        finished = sorted((j for j in self._jobs.values() if j.finished),
                          key=lambda j: j.finished or 0)
        while len(finished) > self._max_completed:
            self._jobs.pop(finished.pop(0).id, None)

    # ------------------------------------------------------------------- read
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)
