import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from sqlalchemy.orm import Session
from app.models.inference_job import InferenceJob, JobStatus
from app.config import settings

logger = logging.getLogger(__name__)


class CleanupService:
    """Retention cleanup service for removing old processed upload files."""

    @staticmethod
    def cleanup_old_files(db: Session, retention_hours: Optional[int] = None) -> int:
        """
        Delete uploaded files associated with completed or failed jobs older than RETENTION_HOURS.
        Ensures files belonging to active ('queued' or 'processing') jobs are NEVER deleted.
        """
        hours = retention_hours if retention_hours is not None else settings.RETENTION_HOURS
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        old_jobs = (
            db.query(InferenceJob)
            .filter(
                InferenceJob.status.in_([JobStatus.COMPLETED.value, JobStatus.FAILED.value]),
                InferenceJob.completed_at < cutoff,
            )
            .all()
        )

        deleted_count = 0
        for job in old_jobs:
            if not job.storage_path:
                continue

            file_path = Path(job.storage_path)

            # Safety check: ensure file path is inside UPLOAD_DIR
            try:
                file_path.resolve().relative_to(settings.UPLOAD_DIR.resolve())
            except ValueError:
                logger.warning(f"Refusing to delete file outside upload directory: {file_path}")
                continue

            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Cleaned up retained upload file for job {job.id}: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to delete upload file {file_path}: {e}")

        return deleted_count
