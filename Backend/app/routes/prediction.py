import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.utils.file_validation import validate_and_save_upload
from app.utils.auth import get_current_user
from app.models.user import User
from app.services.job_service import JobService
from app.models.inference_job import JobStatus
from app.schemas.prediction import JobSubmitResponse, JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ML Prediction"])


@router.post(
    "/predict",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobSubmitResponse,
    summary="Submit image for asynchronous ML leaf disease inference",
    description=(
        "Accepts an uploaded image file (JPG, PNG, WebP), validates content and file size, "
        "persists an inference job bound to the authenticated user, and returns a job_id immediately."
    ),
)
def submit_prediction_job(
    file: UploadFile = File(..., description="Uploaded image file (max 10MB)"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobSubmitResponse:
    """Submit image for asynchronous inference job processing bound to current user."""
    original_filename, stored_filename, storage_path, file_size, mime_type = validate_and_save_upload(file)

    job = JobService.create_job(
        db=db,
        original_filename=original_filename,
        stored_filename=stored_filename,
        storage_path=str(storage_path),
        file_size=file_size,
        mime_type=mime_type,
        user_id=current_user.id,
    )

    logger.info(f"POST /predict -> Accepted job {job.id} for user '{current_user.username}' (ID: {current_user.id})")
    return JobSubmitResponse(job_id=job.id, status=job.status)


@router.get(
    "/predict/{job_id}",
    response_model=JobStatusResponse,
    summary="Get inference job status and prediction results",
    description=(
        "Returns current state of the inference job. Enforces strict user ownership: "
        "authenticated users can only access their own inference jobs."
    ),
)
def get_prediction_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> JobStatusResponse:
    """Fetch status and predictions for an inference job belonging to the current user."""
    job = JobService.get_job_by_id(db, job_id)

    # Ownership check: Return 404 if job doesn't exist OR belongs to another user
    if not job or (job.user_id is not None and job.user_id != current_user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inference job '{job_id}' not found.",
        )

    if job.status == JobStatus.QUEUED.value:
        return JobStatusResponse(job_id=job.id, status=JobStatus.QUEUED.value)

    if job.status == JobStatus.PROCESSING.value:
        return JobStatusResponse(job_id=job.id, status=JobStatus.PROCESSING.value)

    if job.status == JobStatus.FAILED.value:
        return JobStatusResponse(
            job_id=job.id,
            status=JobStatus.FAILED.value,
            error=job.error_message or "Inference execution failed.",
        )

    # Job is COMPLETED
    predictions_list = []
    width = 0
    height = 0

    if job.predictions:
        for p in job.predictions:
            width = p.image_width
            height = p.image_height
            predictions_list.append({
                "class_id": p.class_id,
                "class_name": p.class_name,
                "confidence": p.confidence,
                "bbox": {
                    "x1": p.x1,
                    "y1": p.y1,
                    "x2": p.x2,
                    "y2": p.y2,
                },
            })

    return JobStatusResponse(
        job_id=job.id,
        status=JobStatus.COMPLETED.value,
        predictions=predictions_list,
        image={"width": width, "height": height},
        inference_time_ms=job.inference_time_ms or 0.0,
    )
