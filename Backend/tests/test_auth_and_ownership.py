import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models.user import User
from app.models.inference_job import InferenceJob
from app.utils.auth import create_access_token, get_password_hash


def test_user_registration_and_login(client):
    """Test user registration (/auth/register) and login (/auth/login) token generation."""
    # 1. Register new account
    reg_res = client.post("/auth/register", json={"username": "alice", "password": "securepassword123"})
    assert reg_res.status_code == 201
    reg_data = reg_res.json()
    assert reg_data["username"] == "alice"
    assert "id" in reg_data

    # 2. Login with credentials
    login_res = client.post("/auth/login", json={"username": "alice", "password": "securepassword123"})
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"


def test_unauthenticated_predict_submission_fails(sample_jpeg_bytes):
    """Test unauthenticated POST /predict returns HTTP 401 Unauthorized."""
    unauth_client = TestClient(app)
    response = unauth_client.post(
        "/predict",
        files={"file": ("test.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert response.status_code == 401
    assert "Could not validate credentials" in response.json()["detail"] or "token missing" in response.json()["detail"]


def test_unauthenticated_predict_status_check_fails():
    """Test unauthenticated GET /predict/{job_id} returns HTTP 401 Unauthorized."""
    unauth_client = TestClient(app)
    response = unauth_client.get("/predict/some-job-id-123")
    assert response.status_code == 401


def test_job_bound_to_submitting_user(db_session, sample_jpeg_bytes):
    """Test that POST /predict binds job to the authenticated user ID."""
    # Create user Alice
    alice = User(username="alice_owner", hashed_password=get_password_hash("pass"), is_active=True)
    db_session.add(alice)
    db_session.commit()

    token_alice = create_access_token({"sub": alice.username, "user_id": alice.id})
    alice_client = TestClient(app, headers={"Authorization": f"Bearer {token_alice}"})

    res = alice_client.post(
        "/predict",
        files={"file": ("alice_leaf.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    job = db_session.query(InferenceJob).filter(InferenceJob.id == job_id).first()
    assert job is not None
    assert job.user_id == alice.id


def test_cross_user_access_blocked(db_session, sample_jpeg_bytes):
    """
    Test job ownership isolation:
    User A creates job.
    User B attempting GET /predict/{job_id} returns HTTP 404 Not Found.
    User A attempting GET /predict/{job_id} returns HTTP 200 OK.
    """
    # Create User Alice and User Bob
    alice = User(username="alice", hashed_password=get_password_hash("pass"), is_active=True)
    bob = User(username="bob", hashed_password=get_password_hash("pass"), is_active=True)
    db_session.add_all([alice, bob])
    db_session.commit()

    token_alice = create_access_token({"sub": alice.username, "user_id": alice.id})
    token_bob = create_access_token({"sub": bob.username, "user_id": bob.id})

    alice_client = TestClient(app, headers={"Authorization": f"Bearer {token_alice}"})
    bob_client = TestClient(app, headers={"Authorization": f"Bearer {token_bob}"})

    # Alice submits inference job
    res = alice_client.post(
        "/predict",
        files={"file": ("leaf.jpg", sample_jpeg_bytes, "image/jpeg")},
    )
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    # Bob attempts to fetch Alice's job -> Must return HTTP 404
    bob_res = bob_client.get(f"/predict/{job_id}")
    assert bob_res.status_code == 404
    assert f"Inference job '{job_id}' not found" in bob_res.json()["detail"]

    # Alice fetches her own job -> Must return HTTP 200
    alice_res = alice_client.get(f"/predict/{job_id}")
    assert alice_res.status_code == 200
    assert alice_res.json()["job_id"] == job_id
