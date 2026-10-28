from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.logging import logger

class FlexIDException(Exception):
    """Base exception for all domain-specific FlexID errors."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class ModelNotFoundError(FlexIDException):
    def __init__(self, message: str = "Requested model or version was not found."):
        super().__init__(message, status_code=status.HTTP_404_NOT_FOUND)

class ModelInferenceError(FlexIDException):
    def __init__(self, message: str = "Inference computation failed."):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

class FeatureValidationError(FlexIDException):
    def __init__(self, message: str = "Traffic feature schema validation failed."):
        super().__init__(message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

class BatchSizeExceededError(FlexIDException):
    def __init__(self, max_allowed: int):
        message = f"Batch request exceeds maximum allowed size of {max_allowed} records."
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)

class ExperimentNotFoundError(FlexIDException):
    def __init__(self, experiment_id: str):
        message = f"Experiment '{experiment_id}' was not found."
        super().__init__(message, status_code=status.HTTP_404_NOT_FOUND)

class ScalerNotFoundError(FlexIDException):
    def __init__(self, message: str = "Fitted scaler artifact not found. Please train/preprocess data first."):
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

async def flex_id_exception_handler(request: Request, exc: FlexIDException) -> JSONResponse:
    logger.error("Domain exception: %s on %s %s", exc.message, request.method, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "path": request.url.path,
        }
    )

async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: %s on %s %s", str(exc), request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing your request.",
            "path": request.url.path,
        }
    )
