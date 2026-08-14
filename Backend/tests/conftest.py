import io
import os
import sys
import pytest
from pathlib import Path
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

# Add Backend root directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import app.database as database_module
from app.database import Base, get_db
from app.main import app
from app.models.user import User
from app.utils.auth import get_password_hash, create_access_token

# Shared in-memory SQLite database for testing across threads
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function", autouse=True)
def setup_test_database(monkeypatch):
    """Override application database engine & sessionmaker to use shared SQLite in-memory database."""
    monkeypatch.setattr(database_module, "engine", test_engine)
    monkeypatch.setattr(database_module, "SessionLocal", TestingSessionLocal)

    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def db_session(setup_test_database):
    """Provide DB session for test functions."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def default_user(db_session) -> User:
    """Create default test user in database."""
    user = User(
        username="default_test_user",
        hashed_password=get_password_hash("password123"),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def auth_headers(default_user) -> dict:
    """Provide Bearer token headers for default test user."""
    token = create_access_token(data={"sub": default_user.username, "user_id": default_user.id})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def client(db_session, auth_headers):
    """FastAPI TestClient configured with default authentication headers."""
    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, headers=auth_headers) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Generate valid 100x100 JPEG image bytes for testing."""
    img = Image.new("RGB", (100, 100), color="green")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Generate valid 100x100 PNG image bytes for testing."""
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
