import time
import logging
import threading
from pathlib import Path
from typing import Optional
from app.services.job_service import JobService
from app.services.inference_service import MLInferenceService
from app.config import settings

logger = logging.getLogger(__name__)


class InferenceWorker:
    """
    Production-grade background worker thread polling the database queue, atomically claiming jobs,
    executing ML inference, persisting predictions idempotently, and handling missing/corrupted files cleanly.
    """

    def __init__(self, poll_interval: Optional[float] = None):
        self.poll_interval = poll_interval if poll_interval is not None else settings.WORKER_POLL_INTERVAL
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        """Start background worker thread with reload guard."""
        with self._lock:
            if self._running or (self._thread and self._thread.is_alive()):
                logger.debug("InferenceWorker thread is already running.")
                return

            self._running = True
            self._thread = threading.Thread(target=self._run_loop, name="InferenceWorkerThread", daemon=True)
            self._thread.start()
            logger.info("InferenceWorker background thread started cleanly.")

    def stop(self, timeout: float = 5.0):
        """Signal background worker thread to stop and wait for clean thread join."""
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            logger.info("InferenceWorker background thread stopped cleanly.")

    def _run_loop(self):
        """Worker main polling loop with top-level exception shield preventing worker crash."""
        last_stale_check = 0.0

        while self._running:
            processed_any = False
            try:
                processed_any = self.process_next_job()
            except (KeyboardInterrupt, SystemExit):
                break
            except BaseException as e:
                logger.error(f"Top-level exception caught in InferenceWorker loop: {e}", exc_info=True)

            # Periodically recover stale processing jobs (every 60 seconds)
            now = time.time()
            if now - last_stale_check > 60.0:
                last_stale_check = now
                try:
                    from app.database import SessionLocal
                    db = SessionLocal()
                    try:
                        JobService.recover_stale_jobs(db)
                    finally:
                        db.close()
                except Exception as e:
                    logger.error(f"Error checking stale jobs in worker loop: {e}")

            if not processed_any:
                time.sleep(self.poll_interval)

    def process_next_job(self, db_override=None) -> bool:
        """
        Poll and process the next queued job from database.
        Returns True if a job was claimed and processed, False otherwise.
        """
        from app.database import SessionLocal
        db = db_override or SessionLocal()
        should_close = db_override is None

        try:
            job = JobService.claim_next_job(db)
            if not job:
                return False

            job_id = job.id
            storage_path = Path(job.storage_path)

            logger.info(f"[Job ID: {job_id}] Processing claimed job (Input image: '{job.stored_filename}')...")

            # Validate uploaded image file exists on disk
            if not storage_path.exists():
                error_msg = f"Uploaded image file missing from storage path: {storage_path}"
                logger.error(f"[Job ID: {job_id}] {error_msg}")
                JobService.handle_job_failure(db, job_id, error_msg)
                return True

            try:
                inference_service = MLInferenceService.get_instance()
                results = inference_service.predict(storage_path)

                JobService.save_job_predictions(db, job_id, results)
                logger.info(
                    f"[Job ID: {job_id}] Completed inference. Detected {len(results['predictions'])} object(s) in {results['inference_time_ms']} ms."
                )
                return True
            except Exception as e:
                error_msg = f"Inference execution failed: {str(e)}"
                logger.error(f"[Job ID: {job_id}] {error_msg}", exc_info=True)
                JobService.handle_job_failure(db, job_id, error_msg)
                return True
        finally:
            if should_close:
                db.close()
