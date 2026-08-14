import os
import uuid
from pathlib import Path
from typing import Tuple
from fastapi import UploadFile, HTTPException, status
from PIL import Image
from app.config import settings


def validate_and_save_upload(file: UploadFile) -> Tuple[str, str, Path, int, str]:
    """
    Validate uploaded image file format, MIME type, size limit, and actual binary bytes.
    Saves file to secure storage directory using collision-safe UUID filename.

    Returns:
        (original_filename, stored_filename, storage_path, file_size, mime_type)
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or filename is missing.",
        )

    original_filename = file.filename
    _, ext = os.path.splitext(original_filename)
    ext = ext.lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(list(settings.ALLOWED_EXTENSIONS))}",
        )

    # Read binary contents and check file size limit
    try:
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unable to read uploaded file data.",
        )

    file_size = len(contents)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if file_size > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {max_mb:.1f} MB.",
        )

    # Validate actual image binary header using Pillow
    try:
        import io
        img = Image.open(io.BytesIO(contents))
        img.verify()
        format_name = img.format.lower() if img.format else ""
        valid_formats = {"jpeg", "jpg", "png", "webp"}
        if format_name not in valid_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image format '{format_name}'. Expected valid JPG, PNG, or WebP image data.",
            )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File binary validation failed. Uploaded file is not a valid image.",
        )

    # Collision-safe storage filename
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    storage_path = settings.UPLOAD_DIR / stored_filename

    # Save binary contents to disk securely
    try:
        with open(storage_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded image to local storage.",
        )

    mime_type = file.content_type or f"image/{ext.lstrip('.')}"
    return original_filename, stored_filename, storage_path, file_size, mime_type
