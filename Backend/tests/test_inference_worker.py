import os
import pytest
from unittest.mock import MagicMock, patch
from app.models.inference_job import InferenceJob, JobStatus
from app.models.prediction import Prediction
from app.services.job_service import JobService
from app.services.cleanup_service import CleanupService
from app.workers.inference_worker import InferenceWorker
from app.config import settings


def test_worker_processes_queued_job(db_session, tmp_path):
    """Test 6, 7, 8: Background worker claims queued job, executes ML prediction, and persists predictions."""
    test_img = tmp_path / "test_image.jpg"
    test_img.write_bytes(b"fake image bytes")

    job = JobService.create_job(
        db=db_session,
        original_filename="test_image.jpg",
        stored_filename="test_image.jpg",
        storage_path=str(test_img),
        file_size=len(b"fake image bytes"),
        mime_type="image/jpeg",
    )
    job_id = job.id
    assert job.status == JobStatus.QUEUED.value

    # Mock MLInferenceService predict method
    mock_inference = MagicMock()
    mock_inference.predict.return_value = {
        "predictions": [
            {
                "class_id": 26,
                "class_name": "Tomato mold leaf",
                "confidence": 0.88,
                "x1": 50.0,
                "y1": 60.0,
                "x2": 250.0,
                "y2": 300.0,
            }
        ],
        "image_width": 640,
        "image_height": 480,
        "inference_time_ms": 15.2,
        "model_version": "test_v1",
    }

    worker = InferenceWorker()

    with patch("app.workers.inference_worker.MLInferenceService.get_instance", return_value=mock_inference):
        processed = worker.process_next_job(db_session)
        assert processed is True

    # Query updated job from database
    updated_job = JobService.get_job_by_id(db_session, job_id)
    assert updated_job is not None
    assert updated_job.status == JobStatus.COMPLETED.value
    assert updated_job.inference_time_ms == 15.2
    assert updated_job.completed_at is not None

    # Verify predictions persisted in DB
    predictions = db_session.query(Prediction).filter(Prediction.job_id == job_id).all()
    assert len(predictions) == 1
    assert predictions[0].class_id == 26
    assert predictions[0].class_name == "Tomato mold leaf"
    assert predictions[0].confidence == 0.88
    assert predictions[0].x1 == 50.0
    assert predictions[0].y1 == 60.0


def test_worker_handles_inference_failure_and_retries(db_session, tmp_path):
    """Test 12 & 13: Worker handles transient failure, increments retry count, and re-queues job."""
    test_img = tmp_path / "failing_img.jpg"
    test_img.write_bytes(b"corrupted image")

    job = JobService.create_job(
        db=db_session,
        original_filename="failing_img.jpg",
        stored_filename="failing_img.jpg",
        storage_path=str(test_img),
        file_size=100,
        mime_type="image/jpeg",
    )
    job_id = job.id

    mock_inference = MagicMock()
    mock_inference.predict.side_effect = RuntimeError("CUDA Out of Memory Transient Error")

    worker = InferenceWorker()

    with patch("app.workers.inference_worker.MLInferenceService.get_instance", return_value=mock_inference):
        processed = worker.process_next_job(db_session)
        assert processed is True

    updated_job = JobService.get_job_by_id(db_session, job_id)
    assert updated_job is not None
    assert updated_job.retry_count == 1
    assert updated_job.status == JobStatus.QUEUED.value


def test_worker_permanent_failure_after_max_retries(db_session, tmp_path):
    """Test 12b: Worker marks job permanently as 'failed' after MAX_RETRIES exhausted."""
    test_img = tmp_path / "perm_fail.jpg"
    test_img.write_bytes(b"bad bytes")

    job = JobService.create_job(
        db=db_session,
        original_filename="perm_fail.jpg",
        stored_filename="perm_fail.jpg",
        storage_path=str(test_img),
        file_size=100,
        mime_type="image/jpeg",
    )
    job.retry_count = settings.MAX_RETRIES - 1
    db_session.commit()
    job_id = job.id

    mock_inference = MagicMock()
    mock_inference.predict.side_effect = ValueError("Fatal invalid tensor dimension")

    worker = InferenceWorker()

    with patch("app.workers.inference_worker.MLInferenceService.get_instance", return_value=mock_inference):
        processed = worker.process_next_job(db_session)
        assert processed is True

    updated_job = JobService.get_job_by_id(db_session, job_id)
    assert updated_job is not None
    assert updated_job.status == JobStatus.FAILED.value
    assert "Fatal invalid tensor dimension" in updated_job.error_message


def test_atomic_claiming_prevents_duplicate_processing(db_session, tmp_path):
    """Test 14: Atomic job claiming ensures two workers cannot claim the same job simultaneously."""
    test_img = tmp_path / "single_job.jpg"
    test_img.write_bytes(b"data")

    job = JobService.create_job(
        db=db_session,
        original_filename="single_job.jpg",
        stored_filename="single_job.jpg",
        storage_path=str(test_img),
        file_size=10,
        mime_type="image/jpeg",
    )

    # First worker claims job
    claimed_job_1 = JobService.claim_next_job(db_session)
    assert claimed_job_1 is not None
    assert claimed_job_1.id == job.id
    assert claimed_job_1.status == JobStatus.PROCESSING.value

    # Second worker tries to claim next job
    claimed_job_2 = JobService.claim_next_job(db_session)
    assert claimed_job_2 is None
