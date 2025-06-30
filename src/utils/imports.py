# STANDARD LIBRARY IMPORTS
import typing
import os

# THIRD-PARTY IMPORTS
import torch
import torch.nn.functional as F
import numpy as np

# ADVANCED COMPUTER VISION AND COLOR SCIENCE
try:
    import cv2
    from sklearn.cluster import KMeans
    from skimage import filters, segmentation, measure
    import colour
    from scipy import ndimage

    ADVANCED_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"EasyColorCorrection: Advanced features disabled - missing library: {e}")
    ADVANCED_LIBS_AVAILABLE = False

# RAW IMAGE PROCESSING
try:
    import rawpy

    RAW_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"EasyColorCorrection: RAW processing disabled - missing rawpy library: {e}")
    RAW_PROCESSING_AVAILABLE = False

# DEEP LEARNING COLORIZATION
try:
    from PIL import Image
    import torchvision.transforms as transforms
    import urllib.request
    import os
    from huggingface_hub import hf_hub_download, snapshot_download
    import timm
    
    DL_COLORIZATION_AVAILABLE = True
except ImportError as e:
    print(f"EasyColorCorrection: Deep learning colorization disabled - missing libraries: {e}")
    DL_COLORIZATION_AVAILABLE = False

# HDR/EXR IMAGE PROCESSING
try:
    import OpenEXR
    import Imath

    EXR_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(
        f"EasyColorCorrection: EXR processing disabled - missing OpenEXR library: {e}"
    )
    EXR_PROCESSING_AVAILABLE = False

try:
    import imageio

    IMAGEIO_AVAILABLE = True
except ImportError as e:
    print(
        f"EasyColorCorrection: Extended format support disabled - missing imageio library: {e}"
    )
    IMAGEIO_AVAILABLE = False