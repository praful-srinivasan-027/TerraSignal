import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from app.models.inference_job import InferenceJob, JobStatus, validate_state_transition
from app.models.prediction import Prediction
from app.config import settings
from app.exceptions import JobNotFoundError, InvalidJobStateError

logger = logging.getLogger(__name__)


class JobService:
    """
    Production-grade Job Service enforcing database consistency, idempotent prediction persistence,
    atomic state transitions, and stale job lease recovery.
    """

    @staticmethod
    def create_job(
        db: Session,
        original_filename: str,
        stored_filename: str,
        storage_path: str,
        file_size: int,
        mime_type: str,
        user_id: Optional[int] = None,
    ) -> InferenceJob:
        """Create and persist a new inference job with initial status 'queued'."""
        job = InferenceJob(
            user_id=user_id,
            status=JobStatus.QUEUED.value,
            original_filename=original_filename,
            stored_filename=stored_filename,
            storage_path=str(storage_path),
            file_size=file_size,
            mime_type=mime_type,
            created_at=datetime.utcnow(),
        )
        try:
            db.add(job)
            db.commit()
            db.refresh(job)
            logger.info(f"[Job ID: {job.id}] Created inference job (User ID: {user_id}, Status: {job.status})")
            return job
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create inference job: {e}", exc_info=True)
            raise

    @staticmethod
    def get_job_by_id(db: Session, job_id: str) -> Optional[InferenceJob]:
        """Fetch inference job by unique ID, including related predictions."""
        return db.query(InferenceJob).filter(InferenceJob.id == job_id).first()

    @staticmethod
    def claim_next_job(db: Session) -> Optional[InferenceJob]:
        """
        Atomically claim the next queued inference job for worker processing.
        Prevents race conditions where multiple background workers attempt to claim the same job simultaneously.
        """
        try:
            candidate = (
                db.query(InferenceJob)
                .filter(InferenceJob.status == JobStatus.QUEUED.value)
                .order_by(InferenceJob.created_at.asc())
                .first()
            )

            if not candidate:
                return None

            now = datetime.utcnow()
            rows_updated = (
                db.query(InferenceJob)
                .filter(
                    InferenceJob.id == candidate.id,
                    InferenceJob.status == JobStatus.QUEUED.value,
                )
                .update(
                    {
                        InferenceJob.status: JobStatus.PROCESSING.value,
                        InferenceJob.started_at: now,
                    },
                    synchronize_session=False,
                )
            )

            if rows_updated == 1:
                db.commit()
                db.refresh(candidate)
                logger.info(f"[Job ID: {candidate.id}] Atomically claimed for worker processing.")
                return candidate
            else:
                db.rollback()
                return None
        except Exception as e:
            db.rollback()
            logger.error(f"Error during atomic job claim: {e}", exc_info=True)
            return None

    @staticmethod
    def save_job_predictions(
        db: Session,
        job_id: str,
        results: Dict[str, Any],
    ) -> InferenceJob:
        """
        Idempotently persist prediction bounding boxes and mark job status as 'completed'.
        Purges any existing prediction records for the job ID before inserting new ones,
        preventing duplicate rows if a job is retried or re-processed.
        """
        try:
            job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()
            if not job:
                raise JobNotFoundError(f"Job '{job_id}' not found.")

            # Validate state machine transition: PROCESSING -> COMPLETED
            job.transition_to(JobStatus.COMPLETED.value)

            # Idempotency purge: remove any existing predictions for this job ID
            db.query(Prediction).filter(Prediction.job_id == job_id).delete(synchronize_session=False)

            width = results.get("image_width", 0)
            height = results.get("image_height", 0)
            model_ver = results.get("model_version", settings.MODEL_VERSION)
            inference_time_ms = results.get("inference_time_ms", 0.0)

            for p in results.get("predictions", []):
                prediction_record = Prediction(
                    job_id=job.id,
                    class_id=p["class_id"],
                    class_name=p["class_name"],
                    confidence=p["confidence"],
                    x1=p["x1"],
                    y1=p["y1"],
                    x2=p["x2"],
                    y2=p["y2"],
                    image_width=width,
                    image_height=height,
                    model_version=model_ver,
                    created_at=datetime.utcnow(),
                )
                db.add(prediction_record)

            now = datetime.utcnow()
            job.completed_at = now
            job.inference_time_ms = inference_time_ms

            db.commit()
            db.refresh(job)
            logger.info(
                f"[Job ID: {job.id}] Inference completed in {inference_time_ms:.2f} ms with {len(results.get('predictions', []))} predictions."
            )
            return job
        except Exception as e:
            db.rollback()
            logger.error(f"[Job ID: {job_id}] Failed to save predictions: {e}", exc_info=True)
            raise

    @staticmethod
    def handle_job_failure(
        db: Session,
        job_id: str,
        error_message: str,
    ):
        """
        Handle inference failure. Atomically increments retry_count.
        Re-queues job if retry_count < MAX_RETRIES.
        Marks job permanently as 'failed' if max retries exceeded.
        """
        try:
            job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()
            if not job:
                return

            if job.status == JobStatus.COMPLETED.value:
                logger.warning(f"[Job ID: {job_id}] Ignoring failure handling for already completed job.")
                return

            job.retry_count += 1
            sanitized_error = str(error_message).split("\n")[0][:500]

            if job.retry_count < settings.MAX_RETRIES:
                job.transition_to(JobStatus.QUEUED.value)
                job.started_at = None
                logger.warning(
                    f"[Job ID: {job.id}] Transient failure (Attempt {job.retry_count}/{settings.MAX_RETRIES}). Re-queuing job. Error: {sanitized_error}"
                )
            else:
                job.transition_to(JobStatus.FAILED.value)
                job.completed_at = datetime.utcnow()
                job.error_message = f"Inference failed after {job.retry_count} retries. Error: {sanitized_error}"
                logger.error(f"[Job ID: {job.id}] Permanently failed. Error: {job.error_message}")

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"[Job ID: {job_id}] Error handling job failure: {e}", exc_info=True)

    @staticmethod
    def recover_stale_jobs(db: Session, stale_seconds: Optional[int] = None) -> int:
        """
        Detect and recover jobs stuck in 'processing' state beyond the lease timeout.
        Re-queues or fails stale jobs cleanly. Returns count of recovered jobs.
        """
        timeout = stale_seconds or settings.STALE_JOB_TIMEOUT_SECONDS
        cutoff = datetime.utcnow() - timedelta(seconds=timeout)

        try:
            stale_jobs = (
                db.query(InferenceJob)
                .filter(
                    InferenceJob.status == JobStatus.PROCESSING.value,
                    InferenceJob.started_at < cutoff,
                )
                .all()
            )

            recovered_count = len(stale_jobs)
            for job in stale_jobs:
                logger.warning(
                    f"[Job ID: {job.id}] Recovering stale processing job (Started: {job.started_at}, Lease timeout: >{timeout}s)"
                )
                JobService.handle_job_failure(
                    db, job.id, f"Stale job processing lease expired (Timeout > {timeout}s)"
                )

            return recovered_count
        except Exception as e:
            db.rollback()
            logger.error(f"Error during stale job recovery: {e}", exc_info=True)
            return 0
