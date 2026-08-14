from typing import List, Optional
from pydantic import BaseModel, Field
from app.models.inference_job import JobStatus


class JobSubmitResponse(BaseModel):
    """Immediate response schema returned upon successful POST /predict submission."""

    job_id: str = Field(..., description="Unique persistent identifier for the inference job")
    status: str = Field(JobStatus.QUEUED.value, description="Initial state of the job")


class BoundingBox(BaseModel):
    """Bounding box coordinates schema matching YOLO predictions."""

    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")


class PredictionItem(BaseModel):
    """Individual object detection prediction item schema."""

    class_id: int = Field(..., description="Model class integer ID")
    class_name: str = Field(..., description="Human-readable class name")
    confidence: float = Field(..., description="Model prediction confidence score (0.0 to 1.0)")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")


class ImageMetadata(BaseModel):
    """Dimensions schema of the processed input image."""

    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class JobStatusResponse(BaseModel):
    """Response schema returned by GET /predict/{job_id} reflecting job lifecycle state."""

    job_id: str = Field(..., description="Unique persistent job identifier")
    status: str = Field(..., description="Current status: queued, processing, completed, or failed")

    predictions: Optional[List[PredictionItem]] = Field(
        None, description="List of detected object predictions (present when completed)"
    )
    image: Optional[ImageMetadata] = Field(
        None, description="Input image dimension metadata (present when completed)"
    )
    inference_time_ms: Optional[float] = Field(
        None, description="Total ML inference execution duration in milliseconds (present when completed)"
    )
    error: Optional[str] = Field(
        None, description="Sanitized error description (present when status is failed)"
    )
