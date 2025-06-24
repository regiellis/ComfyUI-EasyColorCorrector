import os

from .src import EasyColorCorrection

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")
CSS_DIRECTORY = os.path.join(os.path.dirname(__file__), "css")

NODE_CLASS_MAPPINGS = {
    "EasyColorCorrection": EasyColorCorrection,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyColorCorrection": "ComfyUI-EasyColorCorrection",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "CSS_DIRECTORY",
]
