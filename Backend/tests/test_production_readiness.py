import os
import time
import pytest
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy.orm import Session
from app.models.inference_job import InferenceJob, JobStatus
from app.models.prediction import Prediction
from app.services.job_service import JobService
from app.workers.inference_worker import InferenceWorker
from app.exceptions import InvalidJobStateError
from app.config import settings


def test_concurrent_worker_claiming(db_session, default_user, tmp_path):
    """
    Test 3 & 10: Concurrent worker atomic claiming race condition.
    Spawn 10 concurrent threads attempting to claim 5 queued jobs simultaneously.
    Assert each job is claimed by exactly 1 thread and zero duplicate claims occur.
    """
    # Create 5 queued jobs
    job_ids = set()
    for i in range(5):
        img_path = tmp_path / f"job_{i}.jpg"
        img_path.write_bytes(b"image bytes")
        job = JobService.create_job(
            db=db_session,
            original_filename=f"job_{i}.jpg",
            stored_filename=f"job_{i}.jpg",
            storage_path=str(img_path),
            file_size=10,
            mime_type="image/jpeg",
            user_id=default_user.id,
        )
        job_ids.add(job.id)

    claimed_jobs = []

    def _worker_claim_task():
        # Each worker thread gets its own DB session
        from app.database import SessionLocal
        thread_db = SessionLocal()
        try:
            claimed = JobService.claim_next_job(thread_db)
            return claimed.id if claimed else None
        finally:
            thread_db.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_worker_claim_task) for _ in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    non_none_claims = [r for r in results if r is not None]
    assert len(non_none_claims) == 5
    assert len(set(non_none_claims)) == 5
    assert set(non_none_claims) == job_ids


def test_invalid_state_transitions(db_session, default_user):
    """
    Test 4 & 10: Enforce strict job state machine rules.
    Verify that invalid transitions (COMPLETED -> PROCESSING, COMPLETED -> QUEUED, FAILED -> PROCESSING) raise InvalidJobStateError.
    """
    # Completed job
    completed_job = InferenceJob(
        id="completed-job-1",
        user_id=default_user.id,
        status=JobStatus.COMPLETED.value,
        original_filename="a.jpg",
        stored_filename="a.jpg",
        storage_path="/tmp/a.jpg",
        file_size=100,
        mime_type="image/jpeg",
    )
    db_session.add(completed_job)
    db_session.commit()

    with pytest.raises(InvalidJobStateError):
        completed_job.transition_to(JobStatus.PROCESSING.value)

    with pytest.raises(InvalidJobStateError):
        completed_job.transition_to(JobStatus.QUEUED.value)

    # Failed job
    failed_job = InferenceJob(
        id="failed-job-1",
        user_id=default_user.id,
        status=JobStatus.FAILED.value,
        original_filename="b.jpg",
        stored_filename="b.jpg",
        storage_path="/tmp/b.jpg",
        file_size=100,
        mime_type="image/jpeg",
    )
    db_session.add(failed_job)
    db_session.commit()

    with pytest.raises(InvalidJobStateError):
        failed_job.transition_to(JobStatus.PROCESSING.value)


def test_prediction_idempotency(db_session, default_user):
    """
    Test 7 & 10: Prediction persistence idempotency.
    Re-executing save_job_predictions() multiple times for the same job purges pre-existing rows
    and generates zero duplicate Prediction rows.
    """
    job = InferenceJob(
        id="idempotent-job-777",
        user_id=default_user.id,
        status=JobStatus.PROCESSING.value,
        original_filename="test.jpg",
        stored_filename="test.jpg",
        storage_path="/tmp/test.jpg",
        file_size=100,
        mime_type="image/jpeg",
    )
    db_session.add(job)
    db_session.commit()

    prediction_payload = {
        "predictions": [
            {
                "class_id": 21,
                "class_name": "Tomato leaf bacterial spot",
                "confidence": 0.95,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 100.0,
                "y2": 200.0,
            }
        ],
        "image_width": 640,
        "image_height": 480,
        "inference_time_ms": 25.0,
    }

    # Execute save_job_predictions twice
    JobService.save_job_predictions(db_session, job.id, prediction_payload)
    job.status = JobStatus.PROCESSING.value  # Reset to processing to test second save
    db_session.commit()

    JobService.save_job_predictions(db_session, job.id, prediction_payload)

    # Verify database contains exactly 1 prediction row (no duplicates)
    rows = db_session.query(Prediction).filter(Prediction.job_id == job.id).all()
    assert len(rows) == 1
    assert rows[0].confidence == 0.95


def test_stale_job_lease_recovery(db_session, default_user):
    """
    Test 5 & 10: Crash recovery for stale processing jobs.
    Jobs stuck in 'processing' status > STALE_JOB_TIMEOUT_SECONDS are recovered and re-queued.
    """
    stale_time = datetime.utcnow() - timedelta(seconds=settings.STALE_JOB_TIMEOUT_SECONDS + 300)
    stale_job = InferenceJob(
        id="stale-job-999",
        user_id=default_user.id,
        status=JobStatus.PROCESSING.value,
        started_at=stale_time,
        original_filename="stale.jpg",
        stored_filename="stale.jpg",
        storage_path="/tmp/stale.jpg",
        file_size=100,
        mime_type="image/jpeg",
        retry_count=0,
    )
    db_session.add(stale_job)
    db_session.commit()

    recovered_count = JobService.recover_stale_jobs(db_session, stale_seconds=settings.STALE_JOB_TIMEOUT_SECONDS)
    assert recovered_count == 1

    updated_job = JobService.get_job_by_id(db_session, stale_job.id)
    assert updated_job.status == JobStatus.QUEUED.value
    assert updated_job.retry_count == 1


def test_retry_limit_exhaustion(db_session, default_user):
    """
    Test 6 & 10: Retry limits and permanent job failure.
    Repeated transient failures up to MAX_RETRIES transition job to status 'failed'.
    """
    job = InferenceJob(
        id="retry-job-333",
        user_id=default_user.id,
        status=JobStatus.PROCESSING.value,
        original_filename="retry.jpg",
        stored_filename="retry.jpg",
        storage_path="/tmp/retry.jpg",
        file_size=100,
        mime_type="image/jpeg",
        retry_count=settings.MAX_RETRIES - 1,
    )
    db_session.add(job)
    db_session.commit()

    JobService.handle_job_failure(db_session, job.id, "CUDA out of memory error")

    updated_job = JobService.get_job_by_id(db_session, job.id)
    assert updated_job.status == JobStatus.FAILED.value
    assert updated_job.retry_count == settings.MAX_RETRIES
    assert "CUDA out of memory error" in updated_job.error_message


def test_missing_image_file_handling(db_session, default_user, tmp_path):
    """
    Test 8 & 10: Missing image file on disk during worker processing is handled gracefully.
    """
    missing_path = tmp_path / "non_existent_file.jpg"

    job = JobService.create_job(
        db=db_session,
        original_filename="missing.jpg",
        stored_filename="missing.jpg",
        storage_path=str(missing_path),
        file_size=100,
        mime_type="image/jpeg",
        user_id=default_user.id,
    )

    worker = InferenceWorker()
    processed = worker.process_next_job(db_session)
    assert processed is True

    updated_job = JobService.get_job_by_id(db_session, job.id)
    assert updated_job is not None
    assert updated_job.retry_count == 1


def test_db_rollback_on_transaction_failure(db_session):
    """
    Test 2 & 10: Verify database rollbacks leave session clean after error.
    """
    try:
        invalid_job = InferenceJob(id=None)  # Missing non-nullable fields
        db_session.add(invalid_job)
        db_session.commit()
    except Exception:
        db_session.rollback()

    # Verify session remains usable
    test_query = db_session.query(InferenceJob).all()
    assert isinstance(test_query, list)


def test_worker_thread_startup_shutdown():
    """
    Test 1 & 10: Worker thread clean startup and shutdown lifecycle.
    """
    worker = InferenceWorker(poll_interval=0.1)
    worker.start()
    assert worker._running is True
    assert worker._thread is not None and worker._thread.is_alive()

    worker.stop(timeout=2.0)
    assert worker._running is False
    assert not worker._thread.is_alive()
