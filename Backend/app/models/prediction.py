from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class Prediction(Base):
    """Database model for persisted ML object detection predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("inference_jobs.id", ondelete="CASCADE"), nullable=False, index=True)

    class_id = Column(Integer, nullable=False)
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)

    # Exact bounding box coordinates preserved without rounding loss
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)

    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    model_version = Column(String(100), nullable=False)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship back to InferenceJob
    job = relationship("InferenceJob", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction(id={self.id}, job_id={self.job_id}, class_name={self.class_name}, conf={self.confidence:.2f})>"
