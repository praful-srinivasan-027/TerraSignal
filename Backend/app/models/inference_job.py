from enum import Enum
from datetime import datetime
import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base
from app.exceptions import InvalidJobStateError


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Set of valid state transitions: { (from_status, to_status) }
VALID_STATE_TRANSITIONS = {
    (JobStatus.QUEUED.value, JobStatus.PROCESSING.value),
    (JobStatus.PROCESSING.value, JobStatus.COMPLETED.value),
    (JobStatus.PROCESSING.value, JobStatus.FAILED.value),
    (JobStatus.PROCESSING.value, JobStatus.QUEUED.value),
}


def validate_state_transition(current_status: str, new_status: str) -> bool:
    """
    Enforce state machine rules for InferenceJob transitions.
    Raises InvalidJobStateError for invalid status changes.
    """
    if current_status == new_status:
        return True

    if (current_status, new_status) not in VALID_STATE_TRANSITIONS:
        raise InvalidJobStateError(
            f"Invalid job state transition from '{current_status}' to '{new_status}'."
        )
    return True


class InferenceJob(Base):
    """Database model for persistent asynchronous ML inference jobs."""

    __tablename__ = "inference_jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    status = Column(String(20), nullable=False, default=JobStatus.QUEUED.value, index=True)

    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    storage_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(100), nullable=False)

    retry_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=True)
    inference_time_ms = Column(Float, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="jobs")
    predictions = relationship(
        "Prediction",
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="Prediction.id",
    )

    def transition_to(self, new_status: str):
        """Validate and apply a status transition to this job instance."""
        validate_state_transition(self.status, new_status)
        self.status = new_status

    def __repr__(self):
        return f"<InferenceJob(id={self.id}, user_id={self.user_id}, status={self.status})>"
