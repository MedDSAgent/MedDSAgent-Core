"""
Job lifecycle management for async tools.

JobManager tracks jobs submitted to a CodeWorker subprocess.
Each session has one JobManager (one worker = one job at a time).

Job flow:
    submit()  →  status: "running"
    wait()    →  status: "completed" | "failed" | "timed_out"
    cancel()  →  status: "cancelled"
"""

import uuid
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from medds_agent.workers import CodeWorker

logger = logging.getLogger(__name__)

# Possible job statuses
STATUS_RUNNING    = "running"
STATUS_COMPLETED  = "completed"
STATUS_FAILED     = "failed"
STATUS_TIMED_OUT  = "timed_out"
STATUS_CANCELLED  = "cancelled"


class ToolBusyError(Exception):
    """
    Raised when an async tool is asked to submit a new job while a previous
    job is still running.
    """


@dataclass
class Job:
    job_id: str
    tool_name: str
    status: str                         # one of the STATUS_* constants
    result: Optional[str] = None        # formatted output string (on success)
    error: Optional[str] = None         # error message (on failure)
    submitted_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class JobManager:
    """
    Manages async job lifecycle for a single CodeWorker.

    Since only one job can run at a time per worker, submit() raises
    ToolBusyError if a job is already running.

    The manager stores all jobs (including completed/failed/cancelled) so
    that job_wait/job_check can look up results even after completion.
    """

    def __init__(self, worker: "CodeWorker"):
        self.worker = worker
        self._jobs: Dict[str, Job] = {}
        self._running_job_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(self, tool_name: str, method: str, params: dict) -> str:
        """
        Submit a job to the worker subprocess.

        The call returns immediately with a job_id. The actual execution
        happens asynchronously inside the subprocess.

        Parameters
        ----------
        tool_name : str
            Name of the tool submitting the job (for display / error messages).
        method : str
            Handler method to call (e.g. "execute").
        params : dict
            Parameters to pass to the handler method.

        Returns
        -------
        str
            job_id

        Raises
        ------
        ToolBusyError
            If a job is already running.
        WorkerDeadError
            If the worker subprocess has died.
        """
        if self._running_job_id is not None:
            running = self._jobs[self._running_job_id]
            raise ToolBusyError(
                f"{tool_name} is busy — job '{self._running_job_id}' "
                f"(submitted by {running.tool_name}) is still running. "
                f"Use job_wait(job_id='{self._running_job_id}', max_sec=<seconds>) "
                f"to collect results, or job_cancel(job_id='{self._running_job_id}') to abort."
            )

        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, tool_name=tool_name, status=STATUS_RUNNING)
        self._jobs[job_id] = job
        self._running_job_id = job_id

        # The actual execution is triggered by sending the command to the worker.
        # We do this in a background thread so submit() returns immediately.
        # The result is stored in the Job object when the thread finishes.
        import threading

        def _run():
            try:
                data = self.worker.send_command_sync(method, params)
                job.result = data.get("output", "")
                job.status = STATUS_COMPLETED
            except Exception as e:
                job.error = str(e)
                job.status = STATUS_FAILED
            finally:
                job.completed_at = datetime.now()
                if self._running_job_id == job_id:
                    self._running_job_id = None

        t = threading.Thread(target=_run, daemon=True, name=f"job-{job_id}")
        t.start()

        logger.debug("JobManager: submitted job %s for tool %s", job_id, tool_name)
        return job_id

    # ------------------------------------------------------------------
    # Wait (async)
    # ------------------------------------------------------------------

    async def wait_async(self, job_id: str, timeout_sec: float) -> Job:
        """
        Async-wait for a job to complete, up to timeout_sec.

        Polls in short intervals rather than blocking a thread, so the
        event loop stays responsive during the wait.

        Parameters
        ----------
        job_id : str
        timeout_sec : float
            Maximum seconds to wait.  0 = instant status check (no wait).

        Returns
        -------
        Job
            The job object with its current status.
            If still running after timeout, status is STATUS_TIMED_OUT
            (the job itself is NOT cancelled — it keeps running in the
            background and can be collected later with another wait call).
        """
        job = self._get_or_raise(job_id)

        if job.status != STATUS_RUNNING:
            return job

        if timeout_sec <= 0:
            # Instant check — return current status without waiting
            if job.status == STATUS_RUNNING:
                # Return a view indicating timed-out without mutating the job
                view = Job(
                    job_id=job.job_id,
                    tool_name=job.tool_name,
                    status=STATUS_TIMED_OUT,
                    submitted_at=job.submitted_at,
                )
                return view
            return job

        poll_interval = 0.2  # seconds
        elapsed = 0.0

        while elapsed < timeout_sec:
            if job.status != STATUS_RUNNING:
                return job
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Timed out — job is still running in the background
        if job.status == STATUS_RUNNING:
            view = Job(
                job_id=job.job_id,
                tool_name=job.tool_name,
                status=STATUS_TIMED_OUT,
                submitted_at=job.submitted_at,
            )
            return view

        return job

    def wait_sync(self, job_id: str, timeout_sec: float) -> Job:
        """
        Blocking version of wait_async.
        Used by sync tool execute() paths (e.g. JobWaitTool).
        """
        import time

        job = self._get_or_raise(job_id)

        if job.status != STATUS_RUNNING:
            return job

        if timeout_sec <= 0:
            if job.status == STATUS_RUNNING:
                return Job(
                    job_id=job.job_id,
                    tool_name=job.tool_name,
                    status=STATUS_TIMED_OUT,
                    submitted_at=job.submitted_at,
                )
            return job

        poll_interval = 0.2
        elapsed = 0.0

        while elapsed < timeout_sec:
            if job.status != STATUS_RUNNING:
                return job
            time.sleep(poll_interval)
            elapsed += poll_interval

        if job.status == STATUS_RUNNING:
            return Job(
                job_id=job.job_id,
                tool_name=job.tool_name,
                status=STATUS_TIMED_OUT,
                submitted_at=job.submitted_at,
            )

        return job

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self, job_id: str) -> Job:
        """
        Cancel a running job by restarting the worker subprocess.

        This is a hard cancel — the subprocess is killed and restarted.
        State accumulated by the cancelled job is lost (variables defined
        in that execution will not be in the new worker's environment).

        Parameters
        ----------
        job_id : str

        Returns
        -------
        Job  (status = STATUS_CANCELLED)
        """
        job = self._get_or_raise(job_id)

        if job.status != STATUS_RUNNING:
            # Already finished — nothing to cancel
            return job

        logger.debug("JobManager: cancelling job %s, restarting worker", job_id)

        job.status = STATUS_CANCELLED
        job.completed_at = datetime.now()
        if self._running_job_id == job_id:
            self._running_job_id = None

        # Restart the worker so the next job gets a clean process
        try:
            self.worker.restart()
        except Exception as e:
            logger.warning("JobManager: worker restart failed after cancel: %s", e)

        return job

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> Optional[Job]:
        """Return the Job object or None if not found."""
        return self._jobs.get(job_id)

    def has_pending(self) -> bool:
        """True if any job is currently running."""
        return self._running_job_id is not None

    def get_running_job_id(self) -> Optional[str]:
        """Return the job_id of the currently running job, or None."""
        return self._running_job_id

    def cancel_all(self):
        """Cancel all running jobs. Called on session close."""
        if self._running_job_id:
            try:
                self.cancel(self._running_job_id)
            except Exception as e:
                logger.warning("JobManager: cancel_all failed: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_raise(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise ValueError(
                f"Job '{job_id}' not found. "
                "It may have expired or the job_id is incorrect."
            )
        return job
