"""
A package of useful tools for photogrammetry processing.
"""
from strenum import StrEnum


class ModelExecutionEngines(StrEnum):
    """
    An enum of common neural network execution engines.
    """

    CPU = "CPU"
    CUDA = "CUDA"
    TENSORRT = "TensorRT"
    DirectML = "DirectML"


from .agisoft_metashape import process_frames