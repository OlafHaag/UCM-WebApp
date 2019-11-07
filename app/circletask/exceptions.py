class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class UploadError(Error):
    """Exception raised for errors during upload processing."""
    pass


class ModelCreationError(Error):
    """Exception raised when creating model instance fails."""
    pass
