class InferenceSystemError(Exception):
    """Base exception class for asynchronous inference system errors."""
    pass


class InvalidJobStateError(InferenceSystemError):
    """Raised when an illegal job state transition is attempted."""
    pass


class JobNotFoundError(InferenceSystemError):
    """Raised when a specified job ID does not exist in storage."""
    pass


class InferenceProcessingError(InferenceSystemError):
    """Raised when an internal ML model inference error occurs."""
    pass
