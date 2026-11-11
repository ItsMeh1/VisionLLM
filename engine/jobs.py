"""Background job registry for training runs."""

from __future__ import annotations

import math
import threading
import time
import traceback
import uuid
from typing import Any, Dict, Optional


def _clean(value: Any) -> Any:
    """Make a payload JSON-safe: NaN/Infinity become null, numpy scalars become Python."""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


class JobState:
    def __init__(self, kind: str, run_id: Optional[str] = None, total_epochs: int = 0):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.run_id = run_id
        self.status = "queued"
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.events: list = []
        self.progress = {"epoch": 0, "epochs": total_epochs, "percent": 0}
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.lock = threading.Lock()
        self._stop = threading.Event()

    def emit(self, kind: str, **fields) -> None:
        with self.lock:
            self.events.append({"t": round(time.time() - self.created, 2), "type": kind, **fields})
            if len(self.events) > 400:
                self.events = self.events[-400:]
            if kind == "epoch":
                record = fields.get("record") or {}
                total = self.progress.get("epochs") or 0
                epoch = int(record.get("epoch") or 0)
                self.progress = {
                    "epoch": epoch,
                    "epochs": total,
                    "percent": round(100 * epoch / total, 1) if total else 0,
                }

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def cancel(self) -> None:
        self._stop.set()
        with self.lock:
            if self.status in ("queued", "running"):
                self.status = "cancelling"

    def snapshot(self) -> dict:
        with self.lock:
            result = None
            if self.result:
                result = {
                    k: v
                    for k, v in self.result.items()
                    if k not in ("history",)
                }
                metrics = result.get("metrics")
                if isinstance(metrics, dict):
                    result["metrics"] = {k: v for k, v in metrics.items() if k not in ("images", "ground_truth")}
            return _clean(
                {
                    "id": self.id,
                    "kind": self.kind,
                    "run_id": self.run_id,
                    "status": self.status,
                    "created": self.created,
                    "started": self.started,
                    "finished": self.finished,
                    "progress": self.progress,
                    "events": list(self.events),
                    "result": result,
                    "error": self.error,
                }
            )


class JobRegistry:
    def __init__(self) -> None:
        self.jobs: Dict[str, JobState] = {}
        self.lock = threading.Lock()

    def create(self, kind: str, run_id: Optional[str] = None, total_epochs: int = 0) -> JobState:
        job = JobState(kind, run_id, total_epochs)
        with self.lock:
            self.jobs[job.id] = job
            if len(self.jobs) > 60:
                oldest = sorted(self.jobs.values(), key=lambda j: j.created)[: len(self.jobs) - 60]
                for stale in oldest:
                    if stale.status not in ("running", "queued", "cancelling"):
                        self.jobs.pop(stale.id, None)
        return job

    def get(self, job_id: str) -> Optional[JobState]:
        return self.jobs.get(job_id)

    def start_training(self, job: JobState, runner, *args) -> None:
        def target() -> None:
            job.started = time.time()
            with job.lock:
                job.status = "running"
            try:
                job.result = runner(*args)
                with job.lock:
                    job.status = "cancelled" if job.result.get("status") == "cancelled" else "completed"
            except Exception as exc:
                with job.lock:
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"
                job.emit("error", message=f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
            finally:
                job.finished = time.time()

        thread = threading.Thread(target=target, name=f"visionllm-{job.kind}-{job.id}", daemon=True)
        thread.start()
