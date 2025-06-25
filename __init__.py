import os

from .src import EasyColorCorrection, BatchColorCorrection, RawImageProcessor, ColorCorrectionViewer

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")
CSS_DIRECTORY = os.path.join(os.path.dirname(__file__), "css")

NODE_CLASS_MAPPINGS = {
    "EasyColorCorrection": EasyColorCorrection,
    "BatchColorCorrection": BatchColorCorrection,
    "RawImageProcessor": RawImageProcessor,
    "ColorCorrectionViewer": ColorCorrectionViewer,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyColorCorrection": "Color Corrector",
    "BatchColorCorrection": "Batch Color Corrector (beta)",
    "RawImageProcessor": "RAW Image Processor (beta)",
    "ColorCorrectionViewer": "Color Correction Viewer (beta)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "CSS_DIRECTORY",
]
