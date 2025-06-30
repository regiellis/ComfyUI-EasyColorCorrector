"""Image analysis and content detection functions."""

import typing
import numpy as np
import torch
from .imports import ADVANCED_LIBS_AVAILABLE
from .device_utils import get_preferred_device

if ADVANCED_LIBS_AVAILABLE:
    import cv2
    from sklearn.cluster import KMeans
    from skimage import filters
    from scipy import ndimage


def analyze_image_content(
    image_np: np.ndarray, device: torch.device = None
) -> typing.Dict[str, typing.Any]:
    """Advanced image analysis using computer vision with GPU acceleration."""
    if not ADVANCED_LIBS_AVAILABLE:
        return {
            "faces": [],
            "dominant_colors": [],
            "scene_type": "unknown",
            "lighting": "auto",
        }

    analysis = {}

    if device is None:
        device = get_preferred_device()
    
    # Face detection
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        analysis["faces"] = faces.tolist() if len(faces) > 0 else []
    except Exception:
        analysis["faces"] = []

    # Color analysis
    try:
        pixels = image_np.reshape(-1, 3)
        sample_size = min(10000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=5, init="k-means++")
        kmeans.fit(sample_pixels)
        dominant_colors = kmeans.cluster_centers_
        analysis["dominant_colors"] = dominant_colors.tolist()
    except Exception:
        analysis["dominant_colors"] = []

    # Scene type classification
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = filters.sobel(gray)
        edge_density = np.mean(edges)

        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        color_variance = np.std(hsv[:, :, 1])
        texture_contrast = np.std(gray)
        
        if edge_density < 0.08 and avg_saturation > 120 and color_variance > 40:
            analysis["scene_type"] = "anime"
        elif edge_density < 0.12 and avg_saturation > 80 and texture_contrast < 35:
            analysis["scene_type"] = "stylized_art"
        elif avg_saturation > 100 and color_variance > 50 and texture_contrast > 40:
            analysis["scene_type"] = "concept_art"
        elif edge_density > 0.25 and texture_contrast > 65 and avg_saturation > 90:
            analysis["scene_type"] = "detailed_illustration"
        elif len(analysis["faces"]) > 0 and edge_density < 0.15:
            analysis["scene_type"] = "portrait"
        elif edge_density > 0.15 and avg_saturation < 80:
            analysis["scene_type"] = "realistic_photo"
        else:
            analysis["scene_type"] = "general"
    except Exception:
        analysis["scene_type"] = "general"

    # Lighting analysis
    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        brightness = np.mean(l_channel)
        contrast = np.std(l_channel)

        if brightness < 85:
            analysis["lighting"] = "low_light"
        elif brightness > 170:
            analysis["lighting"] = "bright"
        elif contrast < 20:
            analysis["lighting"] = "flat"
        else:
            analysis["lighting"] = "good"
    except Exception:
        analysis["lighting"] = "auto"

    return analysis