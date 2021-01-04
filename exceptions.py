class ModelFileMissing(Exception):
    """Exception raised when model related file is missing.

    Attributes:
        message: (str) Exception message 
    """
    def __init__(self):
        self.message = "Model file missing!!"

