# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Custom Exceptions'''
# ===================================================

class ModelFileMissing(Exception):
    """Exception raised when model related file is missing.

    Attributes:
        message: (str) Exception message 
    """
    def __init__(self):
        self.message = "Model file missing!!"


class NoFaceDetected(Exception):
    """Raised when no face is detected in an image

    Attributes:
        message: (str) Exception message
    """
    def __init__(self) -> None:
        self.message = "No face found in image!!"

class MultipleFacesDetected(Exception):
    """Raised when multiple faces are detected in an image

    Attributes:
        message: (str) Exception message
    """
    def __init__(self) -> None:
        self.message = "Multiple faces found in image!!"

class InvalidImage(Exception):
    """Raised when an invalid image is encountered based on array dimension
    Attributes:
        message: (str) Exception message
    """
    def __init__(self) -> None:
        self.message = "Invalid Image!!"

class DatabaseFileNotFound(Exception):
    """Raised when the persistent storage databse file
    doesn't exists at the given location.
    Attributes:
        message: (str) Exception message
    """
    def __init__(self) -> None:
        self.message = "Database file not found!!"

class InvalidCacheInitializationData(Exception):
    """Raised when a data structure other than
    a list is supplied for cache initialization.
    Attributes:
        message: (str) Exception message
    """
    def __init__(self) -> None:
        self.message = "Invalid data structure. Please suppply a list!!"


class NotADictionary(Exception):
    """Raised when input is not a dictionary
    Attributes:
        message: (str) Exception message
    """
    def __init__(self) -> None:
        self.message = "Invalid data structure. Please suppply a dict!!"