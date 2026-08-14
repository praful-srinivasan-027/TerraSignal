import io
import pytest
from app.models.inference_job import InferenceJob, JobStatus
from app.models.prediction import Prediction
from app.config import settings


def test_post_predict_valid_image(client, sample_jpeg_bytes):
    """Test 1 & 5: POST /predict with valid JPEG image returns 202 Accepted and initial queued status."""
    response = client.post(
        "/predict",
        files={"file": ("test_leaf.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert len(data["job_id"]) > 0


def test_post_predict_no_image(client):
    """Test 2: POST /predict with missing image returns HTTP 422 or 400 Bad Request."""
    response = client.post("/predict", files={})
    assert response.status_code == 422 or response.status_code == 400


def test_post_predict_unsupported_file(client):
    """Test 3: POST /predict with unsupported file type (e.g. .txt) returns HTTP 400 Bad Request."""
    response = client.post(
        "/predict",
        files={"file": ("script.txt", b"print('hello world')", "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_post_predict_fake_image_mime(client):
    """Test 3b: POST /predict with fake image header returns HTTP 400."""
    response = client.post(
        "/predict",
        files={"file": ("fake.jpg", b"This is not an image binary", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "File binary validation failed" in response.json()["detail"] or "not a valid image" in response.json()["detail"]


def test_post_predict_oversized_file(client, monkeypatch):
    """Test 4: POST /predict with file exceeding MAX_UPLOAD_SIZE returns HTTP 400 Bad Request."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE", 100)

    oversized_data = b"X" * 200
    response = client.post(
        "/predict",
        files={"file": ("oversized.jpg", oversized_data, "image/jpeg")},
    )
    assert response.status_code == 400
    assert "exceeds maximum allowed limit" in response.json()["detail"]


def test_get_predict_queued_state(client, db_session, sample_jpeg_bytes):
    """Test 9: GET /predict/{job_id} returns status queued for a newly created job."""
    submit_res = client.post(
        "/predict",
        files={"file": ("leaf.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    job_id = submit_res.json()["job_id"]

    get_res = client.get(f"/predict/{job_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["job_id"] == job_id
    assert data["status"] == "queued"
    assert data["predictions"] is None


def test_get_predict_completed_state(client, db_session, default_user):
    """Test 10: GET /predict/{job_id} returns status completed, image dimensions, and predictions list."""
    job = InferenceJob(
        id="test-completed-job-123",
        user_id=default_user.id,
        status=JobStatus.COMPLETED.value,
        original_filename="sample.jpg",
        stored_filename="sample.jpg",
        storage_path="/tmp/sample.jpg",
        file_size=1000,
        mime_type="image/jpeg",
        inference_time_ms=18.5,
    )
    db_session.add(job)
    db_session.commit()

    prediction1 = Prediction(
        job_id=job.id,
        class_id=21,
        class_name="Tomato leaf bacterial spot",
        confidence=0.92,
        x1=100.5,
        y1=50.2,
        x2=300.0,
        y2=250.8,
        image_width=800,
        image_height=600,
        model_version="test_v1",
    )
    db_session.add(prediction1)
    db_session.commit()

    get_res = client.get(f"/predict/{job.id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["job_id"] == job.id
    assert data["status"] == "completed"
    assert data["inference_time_ms"] == 18.5
    assert data["image"] == {"width": 800, "height": 600}
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["class_id"] == 21
    assert data["predictions"][0]["class_name"] == "Tomato leaf bacterial spot"
    assert data["predictions"][0]["confidence"] == 0.92
    assert data["predictions"][0]["bbox"] == {"x1": 100.5, "y1": 50.2, "x2": 300.0, "y2": 250.8}


def test_get_predict_failed_state(client, db_session, default_user):
    """Test 12 & 13: GET /predict/{job_id} returns status failed and error information."""
    job = InferenceJob(
        id="test-failed-job-456",
        user_id=default_user.id,
        status=JobStatus.FAILED.value,
        original_filename="corrupted.jpg",
        stored_filename="corrupted.jpg",
        storage_path="/tmp/corrupted.jpg",
        file_size=500,
        mime_type="image/jpeg",
        error_message="Image decoding corrupted",
    )
    db_session.add(job)
    db_session.commit()

    get_res = client.get(f"/predict/{job.id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["job_id"] == job.id
    assert data["status"] == "failed"
    assert data["error"] == "Image decoding corrupted"


def test_get_predict_nonexistent_id(client):
    """Test 11: GET /predict/{job_id} for non-existent job ID returns HTTP 404 Not Found."""
    response = client.get("/predict/non-existent-uuid-99999")
    assert response.status_code == 404
    assert "Inference job 'non-existent-uuid-99999' not found" in response.json()["detail"]
