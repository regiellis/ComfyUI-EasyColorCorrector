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


# --- Helper functions for color space conversion ---
def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """Converts an RGB image tensor to HSV. Expects input shape [B, H, W, C] with values 0-1."""
    cmax, cmax_indices = torch.max(rgb, dim=-1)
    cmin = torch.min(rgb, dim=-1)[0]
    delta = cmax - cmin

    h = torch.zeros_like(cmax)
    h[cmax_indices == 0] = (((rgb[..., 1] - rgb[..., 2]) / (delta + 1e-8)) % 6)[
        cmax_indices == 0
    ]
    h[cmax_indices == 1] = (((rgb[..., 2] - rgb[..., 0]) / (delta + 1e-8)) + 2)[
        cmax_indices == 1
    ]
    h[cmax_indices == 2] = (((rgb[..., 0] - rgb[..., 1]) / (delta + 1e-8)) + 4)[
        cmax_indices == 2
    ]

    h = h / 6.0
    h[delta == 0] = 0.0

    s = torch.where(
        cmax == 0, torch.tensor(0.0, device=rgb.device), delta / (cmax + 1e-8)
    )
    v = cmax

    return torch.stack([h, s, v], dim=-1)


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Converts an HSV image tensor to RGB. Expects input shape [B, H, W, C] with values 0-1."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).floor()
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    rgb = torch.zeros_like(hsv)

    mask0, mask1, mask2 = (i % 6) == 0, (i % 6) == 1, (i % 6) == 2
    mask3, mask4, mask5 = (i % 6) == 3, (i % 6) == 4, (i % 6) == 5

    rgb[mask0] = torch.stack([v, t, p], dim=-1)[mask0]
    rgb[mask1] = torch.stack([q, v, p], dim=-1)[mask1]
    rgb[mask2] = torch.stack([p, v, t], dim=-1)[mask2]
    rgb[mask3] = torch.stack([p, q, v], dim=-1)[mask3]
    rgb[mask4] = torch.stack([t, p, v], dim=-1)[mask4]
    rgb[mask5] = torch.stack([v, p, q], dim=-1)[mask5]

    rgb[s == 0] = torch.stack([v, v, v], dim=-1)[s == 0]

    return rgb


# --- ADVANCED ANALYSIS FUNCTIONS ---
def extract_color_palette(
    image_np: np.ndarray, num_colors: int = 6
) -> typing.List[str]:
    """Extract perceptually-spaced dominant colors using advanced color science."""
    if not ADVANCED_LIBS_AVAILABLE:
        return []

    try:
        pixels = image_np.reshape(-1, 3)
        sample_size = min(10000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]

        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=5)
        kmeans.fit(sample_pixels)
        dominant_colors = kmeans.cluster_centers_

        try:
            rgb_norm = sample_pixels.astype(np.float32) / 255.0
            lab_pixels = colour.sRGB_to_Lab(rgb_norm)

            lab_kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=5)
            lab_kmeans.fit(lab_pixels)
            lab_centers = lab_kmeans.cluster_centers_

            perceptual_rgb = colour.Lab_to_sRGB(lab_centers)
            perceptual_rgb = np.clip(perceptual_rgb * 255, 0, 255)

            blended_colors = dominant_colors * 0.6 + perceptual_rgb * 0.4
            final_colors = np.clip(blended_colors, 0, 255)

        except Exception:
            final_colors = dominant_colors

        brightness_values = np.sum(final_colors * [0.299, 0.587, 0.114], axis=1)
        sorted_indices = np.argsort(brightness_values)
        final_colors = final_colors[sorted_indices]

        hex_colors = []
        for color in final_colors:
            r, g, b = [int(c) for c in color]
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")

        return hex_colors
    except Exception:
        return []


def match_to_reference_colors(
    image_np: np.ndarray, reference_np: np.ndarray, strength: float = 0.5
) -> np.ndarray:
    """
    Match image colors to reference using highlights/midtones/shadows LUT approach.
    Works like professional color grading - matches colors in different luminance zones.
    """
    if not ADVANCED_LIBS_AVAILABLE:
        return image_np

    try:
        if reference_np is None or image_np is None:
            return image_np

        if reference_np.size == 0 or image_np.size == 0:
            return image_np

        strength = np.clip(strength, 0.0, 1.0)
        if strength == 0.0:
            return image_np

        # Convert to float for processing
        image_float = image_np.astype(np.float32) / 255.0
        reference_float = reference_np.astype(np.float32) / 255.0
        matched_image = image_float.copy()

        # Calculate luminance for zone segmentation
        # Use proper luminance weights: 0.299*R + 0.587*G + 0.114*B
        img_luminance = (
            0.299 * image_float[:, :, 0] + 
            0.587 * image_float[:, :, 1] + 
            0.114 * image_float[:, :, 2]
        )
        ref_luminance = (
            0.299 * reference_float[:, :, 0] + 
            0.587 * reference_float[:, :, 1] + 
            0.114 * reference_float[:, :, 2]
        )

        # Define luminance zones with smooth transitions
        # Shadows: 0.0-0.33, Midtones: 0.33-0.66, Highlights: 0.66-1.0
        shadow_mask = np.clip(1.0 - img_luminance / 0.33, 0.0, 1.0)
        highlight_mask = np.clip((img_luminance - 0.66) / 0.34, 0.0, 1.0)
        midtone_mask = 1.0 - shadow_mask - highlight_mask
        midtone_mask = np.clip(midtone_mask, 0.0, 1.0)

        # For each zone, calculate average color shift from reference
        for zone_name, zone_mask in [
            ("shadows", shadow_mask),
            ("midtones", midtone_mask), 
            ("highlights", highlight_mask)
        ]:
            if np.sum(zone_mask) > 0:  # Only process if zone has pixels
                
                # Get pixels in this luminance zone
                img_zone_mask = zone_mask > 0.1  # Threshold to avoid near-zero weights
                ref_zone_mask = None
                
                # Find corresponding luminance zone in reference image
                if zone_name == "shadows":
                    ref_zone_mask = ref_luminance < 0.4
                elif zone_name == "midtones":
                    ref_zone_mask = (ref_luminance >= 0.3) & (ref_luminance <= 0.7)
                else:  # highlights
                    ref_zone_mask = ref_luminance > 0.6
                
                if np.sum(ref_zone_mask) > 0:  # Reference zone has pixels
                    
                    # Calculate average colors in this zone
                    for channel in range(3):
                        # Image zone average
                        img_zone_pixels = matched_image[:, :, channel][img_zone_mask]
                        img_zone_avg = np.mean(img_zone_pixels) if len(img_zone_pixels) > 0 else 0
                        
                        # Reference zone average  
                        ref_zone_pixels = reference_float[:, :, channel][ref_zone_mask]
                        ref_zone_avg = np.mean(ref_zone_pixels) if len(ref_zone_pixels) > 0 else 0
                        
                        # Calculate color shift for this zone
                        color_shift = ref_zone_avg - img_zone_avg
                        
                        # Apply shift with zone mask and strength
                        shift_amount = color_shift * strength * zone_mask[:, :, np.newaxis]
                        matched_image[:, :, channel] += shift_amount.squeeze()

        # Clamp values and convert back to uint8
        matched_image = np.clip(matched_image, 0.0, 1.0)
        return (matched_image * 255.0).astype(np.uint8)

    except Exception:
        # Return original image if anything fails
        return image_np


def get_preferred_device(use_gpu: bool = True):
    """Get the preferred device considering user settings."""
    if not use_gpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        analysis["faces"] = faces.tolist() if len(faces) > 0 else []
    except Exception:
        analysis["faces"] = []

    try:
        pixels = image_np.reshape(-1, 3)
        sample_size = min(10000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]

        try:
            sample_tensor = torch.from_numpy(sample_pixels.astype(np.float32)).to(
                device
            )
            _ = sample_tensor

            kmeans = KMeans(n_clusters=5, random_state=42, n_init=5, init="k-means++")
            kmeans.fit(sample_pixels)

            dominant_colors = kmeans.cluster_centers_
            analysis["dominant_colors"] = dominant_colors.tolist()
        except Exception:
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
            kmeans.fit(sample_pixels)
            dominant_colors = kmeans.cluster_centers_
            analysis["dominant_colors"] = dominant_colors.tolist()
    except Exception:
        analysis["dominant_colors"] = []

    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        try:
            gray_tensor = torch.from_numpy(gray.astype(np.float32)).to(device)
            sobel_x = (
                torch.tensor(
                    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            sobel_y = (
                torch.tensor(
                    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                    dtype=torch.float32,
                    device=device,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )

            gray_padded = torch.nn.functional.pad(
                gray_tensor.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
            )
            edges_x = torch.nn.functional.conv2d(gray_padded, sobel_x).squeeze()
            edges_y = torch.nn.functional.conv2d(gray_padded, sobel_y).squeeze()
            edges = torch.sqrt(edges_x**2 + edges_y**2)
            edge_density = torch.mean(edges).cpu().item()
        except Exception:
            edges = filters.sobel(gray)
            edge_density = np.mean(edges)

        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        try:
            hsv_tensor = torch.from_numpy(hsv.astype(np.float32)).to(device)
            avg_saturation = torch.mean(hsv_tensor[:, :, 1]).cpu().item()
            color_variance = torch.std(hsv_tensor[:, :, 1]).cpu().item()
            texture_contrast = (
                torch.std(torch.from_numpy(gray.astype(np.float32)).to(device))
                .cpu()
                .item()
            )
        except Exception:
            avg_saturation = np.mean(hsv[:, :, 1])
            color_variance = np.std(hsv[:, :, 1])
            texture_contrast = np.std(gray)
        if edge_density < 0.08 and avg_saturation > 120 and color_variance > 40:
            analysis["scene_type"] = "anime"
        elif edge_density < 0.12 and avg_saturation > 80 and texture_contrast < 35:
            analysis["scene_type"] = "stylized_art"
        elif avg_saturation > 100 and color_variance > 50 and texture_contrast > 40:
            analysis["scene_type"] = "concept_art"
        elif edge_density > 0.2 and texture_contrast > 50:
            analysis["scene_type"] = "detailed_illustration"
        elif len(analysis["faces"]) > 0 and edge_density < 0.15:
            analysis["scene_type"] = "portrait"
        elif edge_density > 0.15 and avg_saturation < 80:
            analysis["scene_type"] = "realistic_photo"
        else:
            analysis["scene_type"] = "general"
    except Exception:
        analysis["scene_type"] = "general"

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


def edge_aware_enhancement(image_np: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply edge-aware enhancement using advanced filtering."""
    if not ADVANCED_LIBS_AVAILABLE:
        return image_np

    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        edges_small = ndimage.gaussian_laplace(l_channel, sigma=0.5)
        edges_medium = ndimage.gaussian_laplace(l_channel, sigma=1.0)
        edges_large = ndimage.gaussian_laplace(l_channel, sigma=2.0)

        combined_edges = (
            np.abs(edges_small)
            + np.abs(edges_medium) * 0.5
            + np.abs(edges_large) * 0.25
        )

        edge_map = combined_edges / (np.max(combined_edges) + 1e-8)

        enhancement_mask = 1.0 - np.clip(edge_map * 2.0, 0.0, 1.0)

        enhancement_mask = ndimage.gaussian_filter(enhancement_mask, sigma=1.0)

        enhanced_l = l_channel.copy()

        mean_filtered = ndimage.uniform_filter(l_channel, size=5)
        local_contrast = l_channel - mean_filtered

        enhanced_l = l_channel + (local_contrast * strength * enhancement_mask)
        enhanced_l = np.clip(enhanced_l, 0, 255)

        lab[:, :, 0] = enhanced_l.astype(np.uint8)

        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced

    except Exception:
        return image_np


def intelligent_white_balance(
    image_np: np.ndarray, strength: float = 0.6
) -> np.ndarray:
    """Advanced white balance using professional color science.

    Args:
        strength: -1.0 to 1.0, where negative values enhance cool/blue tones,
                 positive values enhance warm/orange tones, 0.0 = neutral
    """
    if not ADVANCED_LIBS_AVAILABLE:
        return image_np

    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

        a_channel = lab[:, :, 1].astype(np.float32) - 128
        b_channel = lab[:, :, 2].astype(np.float32) - 128

        if strength >= 0.0:
            # Positive strength: traditional white balance (focus on temperature)
            a_shift = np.median(a_channel)
            b_shift = np.median(b_channel)
            # Reduce tint influence since we have separate tint control
            lab[:, :, 1] = np.clip(
                lab[:, :, 1] - a_shift * strength * 0.3, 0, 255
            )  # Reduced tint correction
            lab[:, :, 2] = np.clip(
                lab[:, :, 2] - b_shift * strength, 0, 255
            )  # Full temperature correction
        else:
            # Negative strength: enhance cool tones (blue/cyan)
            abs_strength = abs(strength)
            # Focus primarily on b channel (temperature) for cool adjustment
            cool_b_shift = -25 * abs_strength  # Toward blue (cooler temperature)
            # Minimal a channel adjustment since we have separate tint control
            cool_a_shift = -5 * abs_strength  # Slight toward green (cooler)

            lab[:, :, 1] = np.clip(lab[:, :, 1] + cool_a_shift, 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + cool_b_shift, 0, 255)

        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        try:
            rgb_norm = image_np.astype(np.float32) / 255.0

            grey_world_rgb = np.mean(rgb_norm.reshape(-1, 3), axis=0)

            xyz = colour.sRGB_to_XYZ(rgb_norm)

            illuminant_xy = colour.XYZ_to_xy(grey_world_rgb.reshape(1, -1))

            d65_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
                "D65"
            ]

            if np.linalg.norm(illuminant_xy - d65_xy) > 0.01:
                adapted_xyz = colour.chromatic_adaptation_VonKries(
                    xyz, illuminant_xy[0], d65_xy, transform="Bradford"
                )

                adapted_rgb = colour.XYZ_to_sRGB(adapted_xyz)
                adapted_rgb = np.clip(adapted_rgb, 0, 1)

                colour_corrected = (adapted_rgb * 255).astype(np.uint8)
                corrected = corrected * (1 - strength * 0.3) + colour_corrected * (
                    strength * 0.3
                )
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        except Exception:
            pass

        return corrected

    except Exception:
        return image_np


def enhance_faces(
    image_np: np.ndarray, faces: typing.List, strength: float = 0.3
) -> np.ndarray:
    """Intelligently enhance detected faces using advanced segmentation with heavy feathering."""
    if not ADVANCED_LIBS_AVAILABLE or not faces:
        return image_np

    try:
        enhanced = image_np.copy()

        for x, y, w, h in faces:
            pad = max(30, int(min(w, h) * 0.3))
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(image_np.shape[1], x + w + pad)
            y_end = min(image_np.shape[0], y + h + pad)

            face_region = enhanced[y_start:y_end, x_start:x_end]
            original_face_region = image_np[y_start:y_end, x_start:x_end].copy()

            segments = segmentation.slic(
                face_region, n_segments=50, compactness=10, sigma=1
            )

            face_lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)

            skin_segments = set()
            for segment_id in np.unique(segments):
                segment_mask = segments == segment_id
                segment_lab = face_lab[segment_mask]

                if len(segment_lab) > 10:
                    l_mean = np.mean(segment_lab[:, 0])
                    a_mean = np.mean(segment_lab[:, 1])
                    b_mean = np.mean(segment_lab[:, 2])

                    l_valid = 40 < l_mean < 200
                    a_valid = 120 < a_mean < 142
                    b_valid = 130 < b_mean < 165

                    is_dark_skin = l_mean < 90
                    dark_skin_valid = (
                        is_dark_skin
                        and 25 < l_mean < 90
                        and 118 < a_mean < 148
                        and 128 < b_mean < 168
                    )

                    is_very_light = l_mean > 160
                    light_skin_valid = (
                        is_very_light
                        and 160 < l_mean < 220
                        and 118 < a_mean < 135
                        and 130 < b_mean < 160
                    )

                    if (
                        (l_valid and a_valid and b_valid)
                        or dark_skin_valid
                        or light_skin_valid
                    ):
                        skin_segments.add(segment_id)

            skin_mask = np.zeros(segments.shape, dtype=bool)
            for segment_id in skin_segments:
                skin_mask |= segments == segment_id

            skin_mask_float = skin_mask.astype(float)

            skin_mask_float = ndimage.gaussian_filter(skin_mask_float, sigma=1.5)

            skin_mask_float = ndimage.gaussian_filter(skin_mask_float, sigma=4.0)

            skin_mask_float = ndimage.gaussian_filter(skin_mask_float, sigma=8.0)

            face_center_x = (x + w // 2) - x_start
            face_center_y = (y + h // 2) - y_start
            face_radius = max(w, h) * 0.6

            coords_y, coords_x = np.ogrid[
                : face_region.shape[0], : face_region.shape[1]
            ]

            distance_from_center = np.sqrt(
                (coords_x - face_center_x) ** 2 + (coords_y - face_center_y) ** 2
            )

            falloff_mask = 1.0 / (
                1.0 + np.exp((distance_from_center - face_radius) / (face_radius * 0.3))
            )

            final_mask = skin_mask_float * falloff_mask

            kernel = np.ones((5, 5), np.float32) / 25
            final_mask = cv2.filter2D(final_mask, -1, kernel)

            final_mask = ndimage.gaussian_filter(final_mask, sigma=6.0)

            # CRITICAL FIX: Prevent mask from affecting bright highlights (>85% luminance)
            # This prevents green tints in bright non-skin areas
            face_gray = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            highlight_protection = np.where(face_gray > 0.85, 0.0, 1.0)
            final_mask = final_mask * highlight_protection

            final_mask = np.clip(final_mask, 0.0, 1.0)

            enhanced_face = face_region.copy()

            face_lab = cv2.cvtColor(enhanced_face, cv2.COLOR_RGB2LAB)
            l_channel = face_lab[:, :, 0].astype(np.float32)
            a_channel = face_lab[:, :, 1].astype(np.float32)
            b_channel = face_lab[:, :, 2].astype(np.float32)

            very_dark_skin = (l_channel < 60) & (a_channel > 132)
            dark_skin = (l_channel < 90) & (l_channel >= 60) & (a_channel > 128)
            very_light_skin = l_channel > 180

            brightness_adjustment = np.where(
                very_dark_skin,
                strength * 0.18,
                np.where(
                    dark_skin,
                    strength * 0.22,
                    np.where(
                        very_light_skin,
                        strength * 0.15,
                        strength * 0.25,
                    ),
                ),
            )
            l_enhanced = l_channel * (1.0 + brightness_adjustment * final_mask)
            face_lab[:, :, 0] = np.clip(l_enhanced, 0, 255).astype(np.uint8)

            green_reduction = np.where(
                very_dark_skin,
                strength * 0.28,
                np.where(
                    dark_skin,
                    strength * 0.18,
                    np.where(
                        very_light_skin,
                        0.0,  # No green reduction for very light skin to avoid green tint
                        strength * 0.12,
                    ),
                ),
            )
            a_enhanced = a_channel * (1.0 - green_reduction * final_mask)
            face_lab[:, :, 1] = np.clip(a_enhanced, 0, 255).astype(np.uint8)

            warming_adjustment = np.where(
                very_dark_skin,
                strength * 0.15,
                np.where(
                    dark_skin,
                    strength * 0.12,
                    np.where(
                        very_light_skin,
                        strength * 0.05,
                        strength * 0.08,
                    ),
                ),
            )
            b_enhanced = b_channel * (1.0 + warming_adjustment * final_mask)
            face_lab[:, :, 2] = np.clip(b_enhanced, 0, 255).astype(np.uint8)

            enhanced_face = cv2.cvtColor(face_lab, cv2.COLOR_LAB2RGB)

            final_mask_3d = np.stack([final_mask, final_mask, final_mask], axis=2)
            blended_face = (
                original_face_region * (1.0 - final_mask_3d)
                + enhanced_face * final_mask_3d
            ).astype(np.uint8)

            enhanced[y_start:y_end, x_start:x_end] = blended_face

        return enhanced

    except Exception:
        return image_np


def generate_histogram_image(image_np: np.ndarray) -> np.ndarray:
    """Generate a well-scaled histogram visualization as an image."""
    try:
        hist_r = cv2.calcHist([image_np], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image_np], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image_np], [2], None, [256], [0, 256])

        hist_height = 512
        hist_width = 768
        hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

        hist_image.fill(20)

        margin = 40
        draw_height = hist_height - (margin * 2)
        draw_width = hist_width - (margin * 2)

        hist_r = cv2.normalize(hist_r, hist_r, 0, draw_height, cv2.NORM_MINMAX)
        hist_g = cv2.normalize(hist_g, hist_g, 0, draw_height, cv2.NORM_MINMAX)
        hist_b = cv2.normalize(hist_b, hist_b, 0, draw_height, cv2.NORM_MINMAX)

        bin_width = max(1, draw_width // 256)

        for i in range(256):
            x = margin + (i * draw_width // 256)

            if hist_r[i][0] > 0:
                cv2.rectangle(
                    hist_image,
                    (x, hist_height - margin),
                    (x + bin_width, hist_height - margin - int(hist_r[i][0])),
                    (80, 80, 255),
                    -1,
                )

            if hist_g[i][0] > 0:
                cv2.rectangle(
                    hist_image,
                    (x, hist_height - margin),
                    (x + bin_width, hist_height - margin - int(hist_g[i][0])),
                    (80, 255, 80),
                    -1,
                )

            if hist_b[i][0] > 0:
                cv2.rectangle(
                    hist_image,
                    (x, hist_height - margin),
                    (x + bin_width, hist_height - margin - int(hist_b[i][0])),
                    (255, 80, 80),
                    -1,
                )

        grid_spacing = draw_width // 8
        for i in range(1, 8):
            x = margin + (i * grid_spacing)
            cv2.line(
                hist_image, (x, margin), (x, hist_height - margin), (60, 60, 60), 1
            )

        grid_spacing = draw_height // 4
        for i in range(1, 4):
            y = margin + (i * grid_spacing)
            cv2.line(hist_image, (margin, y), (hist_width - margin, y), (60, 60, 60), 1)

        cv2.rectangle(
            hist_image,
            (margin, margin),
            (hist_width - margin, hist_height - margin),
            (100, 100, 100),
            2,
        )

        hist_image_rgb = cv2.cvtColor(hist_image, cv2.COLOR_BGR2RGB)
        return hist_image_rgb

    except Exception:
        return np.zeros((512, 768, 3), dtype=np.uint8)


def generate_palette_image(hex_colors: typing.List[str]) -> np.ndarray:
    """Generate a clean color palette visualization as an image."""
    try:
        if not hex_colors:
            return np.zeros((120, 600, 3), dtype=np.uint8)

        palette_height = 120
        palette_width = 600
        palette_image = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)

        swatch_width = palette_width // len(hex_colors)

        for i, hex_color in enumerate(hex_colors):
            hex_color = hex_color.lstrip("#")
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            except (ValueError, IndexError):
                r, g, b = 128, 128, 128

            x_start = i * swatch_width
            x_end = (i + 1) * swatch_width
            palette_image[:, x_start:x_end] = [r, g, b]

        return palette_image

    except Exception:
        return np.zeros((120, 600, 3), dtype=np.uint8)


class EasyColorCorrection:
    PRESETS: typing.Dict[str, typing.Dict[str, float]] = {
        "Natural Portrait": {
            "warmth": 0.08,
            "vibrancy": 0.12,
            "contrast": 0.08,
            "brightness": 0.03,
        },
        "Warm Portrait": {
            "warmth": 0.18,
            "vibrancy": 0.15,
            "contrast": 0.06,
            "brightness": 0.05,
        },
        "Cool Portrait": {
            "warmth": -0.12,
            "vibrancy": 0.08,
            "contrast": 0.10,
            "brightness": 0.02,
        },
        "High Key Portrait": {
            "warmth": 0.05,
            "vibrancy": 0.08,
            "contrast": -0.05,
            "brightness": 0.20,
        },
        "Dramatic Portrait": {
            "warmth": 0.02,
            "vibrancy": 0.20,
            "contrast": 0.25,
            "brightness": -0.05,
        },
        "Epic Fantasy": {
            "warmth": 0.1,
            "vibrancy": 0.4,
            "contrast": 0.3,
            "brightness": 0.05,
        },
        "Sci-Fi Chrome": {
            "warmth": -0.2,
            "vibrancy": 0.3,
            "contrast": 0.35,
            "brightness": 0.1,
        },
        "Dark Fantasy": {
            "warmth": -0.1,
            "vibrancy": 0.25,
            "contrast": 0.4,
            "brightness": -0.15,
        },
        "Vibrant Concept": {
            "warmth": 0.05,
            "vibrancy": 0.5,
            "contrast": 0.25,
            "brightness": 0.08,
        },
        "Matte Painting": {
            "warmth": 0.08,
            "vibrancy": 0.3,
            "contrast": 0.2,
            "brightness": 0.03,
        },
        "Digital Art": {
            "warmth": 0.0,
            "vibrancy": 0.45,
            "contrast": 0.28,
            "brightness": 0.05,
        },
        "Anime Bright": {
            "warmth": 0.12,
            "vibrancy": 0.45,
            "contrast": 0.2,
            "brightness": 0.12,
        },
        "Anime Moody": {
            "warmth": -0.05,
            "vibrancy": 0.35,
            "contrast": 0.25,
            "brightness": -0.05,
        },
        "Cyberpunk": {
            "warmth": -0.15,
            "vibrancy": 0.45,
            "contrast": 0.25,
            "brightness": -0.03,
        },
        "Pastel Dreams": {
            "warmth": 0.12,
            "vibrancy": -0.08,
            "contrast": -0.08,
            "brightness": 0.12,
        },
        "Neon Nights": {
            "warmth": -0.18,
            "vibrancy": 0.40,
            "contrast": 0.20,
            "brightness": -0.05,
        },
        "Comic Book": {
            "warmth": 0.05,
            "vibrancy": 0.5,
            "contrast": 0.35,
            "brightness": 0.08,
        },
        "Cinematic": {
            "warmth": 0.12,
            "vibrancy": 0.15,
            "contrast": 0.18,
            "brightness": 0.02,
        },
        "Teal & Orange": {
            "warmth": -0.08,
            "vibrancy": 0.25,
            "contrast": 0.15,
            "brightness": 0.0,
        },
        "Film Noir": {
            "warmth": -0.05,
            "vibrancy": -0.80,
            "contrast": 0.35,
            "brightness": -0.08,
        },
        "Vintage Film": {
            "warmth": 0.15,
            "vibrancy": -0.10,
            "contrast": 0.12,
            "brightness": 0.03,
        },
        "Bleach Bypass": {
            "warmth": -0.02,
            "vibrancy": -0.25,
            "contrast": 0.30,
            "brightness": 0.05,
        },
        "Golden Hour": {
            "warmth": 0.25,
            "vibrancy": 0.18,
            "contrast": 0.08,
            "brightness": 0.08,
        },
        "Blue Hour": {
            "warmth": -0.20,
            "vibrancy": 0.15,
            "contrast": 0.12,
            "brightness": 0.02,
        },
        "Sunny Day": {
            "warmth": 0.15,
            "vibrancy": 0.20,
            "contrast": 0.10,
            "brightness": 0.08,
        },
        "Overcast": {
            "warmth": -0.08,
            "vibrancy": 0.05,
            "contrast": 0.08,
            "brightness": 0.05,
        },
        "Sepia": {
            "warmth": 0.30,
            "vibrancy": -0.35,
            "contrast": 0.08,
            "brightness": 0.03,
        },
        "Black & White": {
            "warmth": 0.0,
            "vibrancy": -1.0,
            "contrast": 0.15,
            "brightness": 0.0,
        },
        "Faded": {
            "warmth": 0.05,
            "vibrancy": -0.15,
            "contrast": -0.12,
            "brightness": 0.08,
        },
        "Moody": {
            "warmth": -0.08,
            "vibrancy": 0.12,
            "contrast": 0.20,
            "brightness": -0.08,
        },
    }

    @classmethod
    def INPUT_TYPES(cls) -> typing.Dict:
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mode": (["Auto", "Preset", "Manual", "Colorize"], {"default": "Auto"}),
            },
            "optional": {
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "🎨 Optional reference image for color matching (concept art/mood boards)"
                    },
                ),
                "reference_strength": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Strength of color matching to reference image",
                    },
                ),
                "extract_palette": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "📊 Extract and display dominant color palette",
                    },
                ),
                "lock_input_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "🔒 Lock input image to prevent upstream nodes from reprocessing when adjusting color parameters",
                    },
                ),
                "ai_analysis": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "🤖 Enable AI-powered face detection, scene analysis & content-aware enhancement",
                    },
                ),
                "adjust_for_skin_tone": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "AI face detection + skin tone preservation (requires ai_analysis)",
                    },
                ),
                "white_balance_strength": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "White balance adjustment: -1.0 = cooler/blue, +1.0 = warmer/orange",
                    },
                ),
                "enhancement_strength": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 2.5,
                        "step": 0.05,
                        "tooltip": "Overall strength of AI-powered enhancements",
                    },
                ),
                "pop_factor": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Extra pop factor for artistic content (anime, detailed photos)",
                    },
                ),
                "effect_strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Strength of overall color correction effect",
                    },
                ),
                "warmth": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Green/Magenta warmth adjustment",
                    },
                ),
                "vibrancy": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Saturation boost for less saturated areas",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0.5,
                        "max": 0.5,
                        "step": 0.02,
                        "tooltip": "Contrast adjustment for overall image",
                    },
                ),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -0.5,
                        "max": 0.5,
                        "step": 0.02,
                        "tooltip": "Brightness adjustment for overall image",
                    },
                ),
                "tint": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Tint adjustment: -1.0 = green, +1.0 = magenta (Manual mode only)",
                    },
                ),
                "preset": (list(cls.PRESETS.keys()), {}),
                "variation": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Variation strength for preset adjustments",
                    },
                ),
                "lift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Lift adjustment for shadows in the image",
                    },
                ),
                "gamma": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Gamma adjustment for midtones in the image",
                    },
                ),
                "gain": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Gain adjustment for highlights in the image",
                    },
                ),
                "noise": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Noise reduction strength (0.0 = no noise reduction, 1.0 = maximum noise reduction)",
                    },
                ),
                # Colorize mode specific parameters
                "colorize_strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "🎨 Overall colorization strength (only for photos, not art/anime)",
                    },
                ),
                "skin_warmth": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "🧑 Warmth applied to detected skin tones",
                    },
                ),
                "sky_saturation": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "🌅 Saturation boost for sky and blue regions",
                    },
                ),
                "vegetation_green": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "🌿 Green enhancement for vegetation and foliage",
                    },
                ),
                "sepia_tone": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "🟤 Sepia tone blend for vintage photo colorization",
                    },
                ),
                "colorize_mode": (
                    ["auto", "portrait", "landscape", "vintage"],
                    {
                        "default": "auto",
                        "tooltip": "🎯 Colorization style: auto-detect, portrait focus, landscape focus, or vintage look",
                    },
                ),
                "use_gpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "🚀 Use GPU acceleration for processing (disable to force CPU)",
                    },
                ),
                "mask": ("MASK", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "palette_data", "histogram", "palette_image")
    FUNCTION = "run"
    CATEGORY = "itsjustregi / Easy Color Corrector"
    DISPLAY_NAME = "Easy Color Corrector"

    def run(
        self,
        image: torch.Tensor,
        mode: str,
        reference_image: typing.Optional[torch.Tensor] = None,
        reference_strength: float = 0.3,
        extract_palette: bool = False,
        lock_input_image: bool = False,
        ai_analysis: bool = True,
        adjust_for_skin_tone: bool = False,
        white_balance_strength: float = 0.0,
        enhancement_strength: float = 0.2,
        pop_factor: float = 0.7,
        effect_strength: float = 0.6,
        warmth: float = 0.0,
        vibrancy: float = 0.0,
        contrast: float = 0.0,
        brightness: float = 0.0,
        tint: float = 0.0,
        preset: str = "Anime",
        variation: float = 0.0,
        lift: float = 0.0,
        gamma: float = 0.0,
        gain: float = 0.0,
        noise: float = 0.0,
        colorize_strength: float = 0.8,
        skin_warmth: float = 0.3,
        sky_saturation: float = 0.6,
        vegetation_green: float = 0.5,
        sepia_tone: float = 0.0,
        colorize_mode: str = "auto",
        use_gpu: bool = True,
        mask: typing.Optional[torch.Tensor] = None,
    ) -> tuple:

        original_image = image.clone()
        _, height, width, _ = image.shape

        # Handle GPU processing based on user choice
        device = get_preferred_device(use_gpu)
        print(f"🔧 EasyColorCorrection Debug: use_gpu={use_gpu}, CUDA available={torch.cuda.is_available()}")
        print(f"🔧 Input image device: {image.device}, target device: {device}")
        
        if use_gpu and torch.cuda.is_available():
            if not str(image.device).startswith("cuda"):
                print(f"🚀 Moving image to GPU: {device}")
                image = image.to(device)
                original_image = original_image.to(device)
                if mask is not None:
                    mask = mask.to(device)
                    print(f"🚀 Moved mask to GPU: {device}")
                if reference_image is not None:
                    reference_image = reference_image.to(device)
                    print(f"🚀 Moved reference image to GPU: {device}")
                print(f"✅ Image now on device: {image.device}")
            else:
                print(f"✅ Image already on GPU: {image.device}")
        elif use_gpu and not torch.cuda.is_available():
            print("❌ GPU requested but CUDA not available - using CPU")
        else:
            print(f"💻 CPU processing selected - device: {device}")
        
        # Cache for locked input - avoid reprocessing upstream when enabled
        if lock_input_image:
            # Check if the input image has changed (cache invalidation)
            image_changed = False
            if (
                not hasattr(self, "_cached_original_image")
                or self._cached_original_image is None
            ):
                image_changed = True
            else:
                # Compare image tensors to detect changes from upstream
                try:
                    if not torch.equal(self._cached_original_image, original_image):
                        image_changed = True
                except:
                    # Different shapes or other comparison issues = definitely changed
                    image_changed = True

            if image_changed:
                self._cached_original_image = original_image.clone()
                self._cached_analysis = None
                print("🔄 Locked Input: New image detected, updating cache")

            processed_image = self._cached_original_image.clone()
        else:
            processed_image = image.clone()
            # Clear cache when input is not locked
            if hasattr(self, "_cached_original_image"):
                self._cached_original_image = None
                self._cached_analysis = None

        # --- SHARED AI ANALYSIS (available to all modes) ---
        analysis = None
        if ai_analysis and ADVANCED_LIBS_AVAILABLE:
            # Only create image_np when AI analysis is actually needed
            image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
            # Use cached analysis when input is locked
            if (
                lock_input_image
                and hasattr(self, "_cached_analysis")
                and self._cached_analysis is not None
            ):
                analysis = self._cached_analysis
            else:
                analysis = analyze_image_content(
                    image_np, get_preferred_device(use_gpu)
                )
                if lock_input_image:
                    self._cached_analysis = analysis

            print(
                f"🤖 AI Analysis{'(cached)' if lock_input_image and hasattr(self, '_cached_analysis') else ''} for {mode} Mode: {analysis['scene_type']} scene, {analysis['lighting']} lighting, {len(analysis['faces'])} faces detected"
            )
        else:
            # Provide fallback analysis for non-AI modes
            analysis = {
                "scene_type": "general",
                "lighting": "auto",
                "faces": [],
                "dominant_colors": [],
            }
            if not ai_analysis:
                print(f"🔧 {mode} Mode: AI analysis disabled, using fallback values")

        # --- AUTO MODE ---
        if mode == "Auto":
            if ai_analysis and ADVANCED_LIBS_AVAILABLE:
                if white_balance_strength != 0.0:
                    wb_corrected = intelligent_white_balance(
                        image_np, white_balance_strength
                    )
                    processed_image = (
                        torch.from_numpy(wb_corrected.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .to(processed_image.device)
                    )

                if enhancement_strength > 0.2:
                    edge_enhanced = edge_aware_enhancement(
                        image_np, enhancement_strength * 0.3
                    )
                    processed_image = (
                        torch.from_numpy(edge_enhanced.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .to(processed_image.device)
                    )
                    image_np = edge_enhanced
                if analysis["faces"] and adjust_for_skin_tone:
                    # Warning for overdriven enhancement
                    if enhancement_strength > 1.0:
                        print(f"⚠️ WARNING: Enhancement strength ({enhancement_strength:.1f}) > 1.0 with skin tone adjustment may cause color artifacts in highlights. Recommended: ≤ 1.0")
                    face_enhanced = enhance_faces(
                        image_np, analysis["faces"], enhancement_strength * 0.5
                    )
                    processed_image = (
                        torch.from_numpy(face_enhanced.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .to(processed_image.device)
                    )
            else:
                analysis = {
                    "scene_type": "general",
                    "lighting": "auto",
                    "faces": [],
                    "dominant_colors": [],
                }
                print("🔧 Basic Auto Mode: AI analysis disabled")
                # Use consistent white balance regardless of AI setting
                if white_balance_strength != 0.0 and ADVANCED_LIBS_AVAILABLE:
                    image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
                    wb_corrected = intelligent_white_balance(image_np, white_balance_strength)
                    processed_image = (
                        torch.from_numpy(wb_corrected.astype(np.float32) / 255.0)
                        .unsqueeze(0)
                        .to(processed_image.device)
                    )
                    print(f"🌡️ Applied white balance: {white_balance_strength:.2f} ({'warmer' if white_balance_strength > 0 else 'cooler' if white_balance_strength < 0 else 'neutral'})")
            hsv_enhanced = rgb_to_hsv(processed_image)
            h_enh, s_enh, v_enh = (
                hsv_enhanced[..., 0],
                hsv_enhanced[..., 1],
                hsv_enhanced[..., 2],
            )

            scene_type = analysis["scene_type"]
            lighting = analysis["lighting"]
            if scene_type == "anime":
                contrast_boost = 0.18 * enhancement_strength
                saturation_boost = 0.55 * enhancement_strength
                vibrancy_strength = 0.45 * pop_factor
            elif scene_type == "concept_art":
                contrast_boost = 0.25 * enhancement_strength
                saturation_boost = 0.6 * enhancement_strength
                vibrancy_strength = 0.5 * pop_factor
            elif scene_type == "stylized_art":
                contrast_boost = 0.22 * enhancement_strength
                saturation_boost = 0.5 * enhancement_strength
                vibrancy_strength = 0.4 * pop_factor
            elif scene_type == "detailed_illustration":
                contrast_boost = 0.2 * enhancement_strength
                saturation_boost = 0.45 * enhancement_strength
                vibrancy_strength = 0.35 * pop_factor
            elif scene_type == "portrait":
                contrast_boost = 0.12 * enhancement_strength
                saturation_boost = 0.3 * enhancement_strength
                vibrancy_strength = 0.25 * pop_factor
            elif scene_type == "realistic_photo":
                contrast_boost = 0.15 * enhancement_strength
                saturation_boost = 0.35 * enhancement_strength
                vibrancy_strength = 0.3 * pop_factor
            else:
                contrast_boost = 0.16 * enhancement_strength
                saturation_boost = 0.42 * enhancement_strength
                vibrancy_strength = 0.35 * pop_factor

            if lighting == "low_light":
                contrast_boost *= 1.3
                v_enh = v_enh * (1.0 + 0.15 * enhancement_strength)
            elif lighting == "bright":
                contrast_boost *= 0.8
                v_enh = v_enh * (1.0 - 0.1 * enhancement_strength)
            elif lighting == "flat":
                contrast_boost *= 1.5
            v_min, v_max = torch.quantile(v_enh, 0.01), torch.quantile(v_enh, 0.99)
            if v_max > v_min:
                v_enh = torch.clamp((v_enh - v_min) / (v_max - v_min), 0.0, 1.0)

            v_enh = 0.5 + (v_enh - 0.5) * (1.0 + contrast_boost)
            if analysis["dominant_colors"] and ADVANCED_LIBS_AVAILABLE:
                avg_dom_sat = np.mean(
                    [
                        np.max(color) - np.min(color)
                        for color in analysis["dominant_colors"]
                    ]
                )
                saturation_factor = max(0.5, 1.0 - avg_dom_sat / 255.0)
                saturation_boost *= saturation_factor

            s_enh = s_enh * (1.0 + saturation_boost)

            saturation_mask = 1.0 - s_enh
            s_enh = s_enh + (vibrancy_strength * saturation_mask * s_enh)
            if pop_factor > 0.3 and scene_type in [
                "anime",
                "concept_art",
                "stylized_art",
                "detailed_illustration",
            ]:
                highlight_mask = torch.clamp((v_enh - 0.7) * 3.33, 0.0, 1.0)
                if scene_type == "concept_art":
                    glow_strength = 0.25 * pop_factor * enhancement_strength
                elif scene_type in ["anime", "stylized_art"]:
                    glow_strength = 0.2 * pop_factor * enhancement_strength
                else:
                    glow_strength = 0.15 * pop_factor * enhancement_strength

                v_enh = v_enh + (glow_strength * highlight_mask)
            s_enh = torch.clamp(s_enh, 0.0, 1.0)
            v_enh = torch.clamp(v_enh, 0.0, 1.0)
            processed_image = hsv_to_rgb(torch.stack([h_enh, s_enh, v_enh], dim=-1))
            processed_image = torch.clamp(processed_image, 0.0, 1.0)

        # --- PRESET MODE ---
        if mode == "Preset":
            p_vals = self.PRESETS.get(preset, {}).copy()
            if analysis:
                scene_type = analysis["scene_type"]
                lighting = analysis["lighting"]

                if scene_type == "concept_art":
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.4
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.3
                    p_vals["brightness"] = p_vals.get("brightness", 0.0) + 0.05
                elif scene_type == "anime":
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.3
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.2
                elif scene_type == "stylized_art":
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.25
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.15
                elif scene_type == "detailed_illustration":
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.2
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.1
                elif scene_type == "portrait" and analysis["faces"]:
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 0.8
                    p_vals["warmth"] = p_vals.get("warmth", 0.0) + 0.05
                elif scene_type == "realistic_photo":
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.1

                if lighting == "low_light":
                    p_vals["brightness"] = p_vals.get("brightness", 0.0) + 0.1
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.4
                elif lighting == "bright":
                    p_vals["brightness"] = p_vals.get("brightness", 0.0) - 0.05
                elif lighting == "flat":
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.5

                print(
                    f"🎨 AI-Enhanced Preset: {preset} adapted for {scene_type}/{lighting}"
                )
            if analysis and analysis["dominant_colors"]:
                # Reduce variation if image is already very colorful
                avg_color_range = np.mean(
                    [
                        np.max(color) - np.min(color)
                        for color in analysis["dominant_colors"]
                    ]
                )
                variation_factor = max(0.3, 1.0 - avg_color_range / 200.0)
                v_factor = variation * 0.05 * variation_factor
            else:
                v_factor = variation * 0.05

            # Apply preset values with intelligent variation (only for Preset mode)
            if mode == "Preset":
                warmth = p_vals.get("warmth", 0.0) + (torch.randn(1).item() * v_factor)
                vibrancy = p_vals.get("vibrancy", 0.0) + (
                    torch.randn(1).item() * v_factor
                )
                contrast = p_vals.get("contrast", 0.0) + (
                    torch.randn(1).item() * v_factor
                )
                brightness = p_vals.get("brightness", 0.0) + (
                    torch.randn(1).item() * v_factor
                )
            # For Manual mode, use the direct parameter values (warmth, vibrancy, contrast, brightness are already set as function parameters)

        # --- ADVANCED COLOR PROCESSING (Preset and Manual modes) ---
        if mode != "Auto":
            # === INTELLIGENT WHITE BALANCE (if enabled) ===
            if white_balance_strength != 0.0 and ADVANCED_LIBS_AVAILABLE:
                # Create image_np if not already available (when AI analysis is disabled)
                if not ai_analysis or 'image_np' not in locals():
                    image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
                wb_corrected = intelligent_white_balance(
                    image_np, white_balance_strength
                )
                processed_image = (
                    torch.from_numpy(wb_corrected.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .to(processed_image.device)
                )
                print(f"🌡️ Applied white balance: {white_balance_strength:.2f} ({'warmer' if white_balance_strength > 0 else 'cooler' if white_balance_strength < 0 else 'neutral'})")

            # === FACE-AWARE PROCESSING ===
            if (
                analysis
                and analysis["faces"]
                and adjust_for_skin_tone
                and ADVANCED_LIBS_AVAILABLE
            ):
                # Warning for overdriven enhancement
                if enhancement_strength > 1.0:
                    print(f"⚠️ WARNING: Enhancement strength ({enhancement_strength:.1f}) > 1.0 with skin tone adjustment may cause color artifacts in highlights. Recommended: ≤ 1.0")
                face_enhanced = enhance_faces(
                    image_np, analysis["faces"], enhancement_strength * 0.3
                )
                processed_image = (
                    torch.from_numpy(face_enhanced.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .to(processed_image.device)
                )

            hsv_image = rgb_to_hsv(processed_image)
            h, s, v = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]

            # Warmth will be handled in LAB color space later for proper temperature control
            # (removing the simple hue shift approach)

            if vibrancy != 0.0:
                saturation_mask = 1.0 - s
                s = s * (1.0 + vibrancy) + (vibrancy * 0.3 * saturation_mask * s)

            if brightness != 0.0:
                v = v + brightness * (1.0 - v * 0.5)

            if contrast != 0.0:
                v = 0.5 + (v - 0.5) * (1.0 + contrast)

            processed_image = hsv_to_rgb(torch.stack([h, s, v], dim=-1))
            processed_image = torch.clamp(processed_image, 0.0, 1.0)

            # === TEMPERATURE & TINT PROCESSING (LAB color space) ===
            if (warmth != 0.0 or tint != 0.0) and mode == "Manual":
                # Convert to numpy for LAB processing
                image_np_for_color = (processed_image[0].cpu().numpy() * 255).astype(
                    np.uint8
                )

                if ADVANCED_LIBS_AVAILABLE:
                    try:
                        # Convert to LAB color space
                        lab = cv2.cvtColor(image_np_for_color, cv2.COLOR_RGB2LAB)

                        # Apply temperature adjustment to 'b' channel (blue-yellow axis)
                        if warmth != 0.0:
                            # 'b' channel: values around 128 are neutral, <128 is blue, >128 is yellow
                            temperature_shift = (
                                warmth * 35
                            )  # Scale factor for visible temperature effect
                            lab[:, :, 2] = np.clip(
                                lab[:, :, 2] + temperature_shift, 0, 255
                            )

                        # Apply tint adjustment to 'a' channel (green-magenta axis)
                        if tint != 0.0:
                            # 'a' channel: values around 128 are neutral, <128 is green, >128 is magenta
                            tint_shift = (
                                tint * 30
                            )  # Scale factor for visible tint effect
                            lab[:, :, 1] = np.clip(lab[:, :, 1] + tint_shift, 0, 255)

                        # Convert back to RGB
                        color_corrected_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                        processed_image = (
                            torch.from_numpy(
                                color_corrected_rgb.astype(np.float32) / 255.0
                            )
                            .unsqueeze(0)
                            .to(processed_image.device)
                        )
                        processed_image = torch.clamp(processed_image, 0.0, 1.0)

                        adjustments = []
                        if warmth != 0.0:
                            adjustments.append(
                                f"temperature: {warmth:.2f} ({'warmer' if warmth > 0 else 'cooler'})"
                            )
                        if tint != 0.0:
                            adjustments.append(
                                f"tint: {tint:.2f} ({'magenta' if tint > 0 else 'green'})"
                            )
                        print(
                            f"🌡️ Applied professional color adjustments: {', '.join(adjustments)}"
                        )

                    except Exception as e:
                        print(
                            f"⚠️ Temperature/tint processing failed, using original: {e}"
                        )
                else:
                    print(
                        "⚠️ Professional temperature/tint requires advanced libraries (OpenCV), skipping"
                    )

            # --- MANUAL MODE ---
            if mode == "Manual":
                ai_guidance = ""
                if analysis:
                    scene_type = analysis["scene_type"]
                    if scene_type in [
                        "concept_art",
                        "anime",
                        "stylized_art",
                        "detailed_illustration",
                    ]:
                        ai_guidance = f" + AI Guidance ({scene_type})"
                    else:
                        ai_guidance = " + AI Guidance"
                print(f"🎛️ Professional Manual Mode{ai_guidance}")

                if analysis and analysis["scene_type"] in [
                    "concept_art",
                    "anime",
                    "stylized_art",
                    "detailed_illustration",
                ]:
                    shadows_mask = 1.0 - torch.clamp(v * 2.5, 0.0, 1.0)
                    midtones_mask = 1.0 - torch.abs(v - 0.5) * 1.8
                    highlights_mask = torch.clamp((v - 0.6) * 2.5, 0.0, 1.0)
                else:
                    shadows_mask = 1.0 - torch.clamp(v * 3.0, 0.0, 1.0)
                    midtones_mask = 1.0 - torch.abs(v - 0.5) * 2.0
                    highlights_mask = torch.clamp((v - 0.66) * 3.0, 0.0, 1.0)

                # === 3-WAY COLOR CORRECTION (Lift/Gamma/Gain) ===
                corrections_applied = []
                if lift != 0.0:
                    # Shadows (lift) - increased strength for noticeable effect
                    lift_strength = 0.8  # Increased for more visible control
                    v = v + (lift * lift_strength * shadows_mask)
                    corrections_applied.append(f"lift: {lift:.2f}")

                if gamma != 0.0:
                    # Midtones (gamma) - increased strength for noticeable effect  
                    gamma_exp = 1.0 / (
                        1.0 + gamma * 1.2
                    )  # Increased for more visible control
                    v_gamma = torch.pow(torch.clamp(v, 0.001, 1.0), gamma_exp)
                    v = torch.lerp(v, v_gamma, midtones_mask)
                    corrections_applied.append(f"gamma: {gamma:.2f}")

                if gain != 0.0:
                    # Highlights (gain) - increased strength for noticeable effect
                    gain_strength = 0.8  # Increased for more visible control
                    v = v + (gain * gain_strength * highlights_mask)
                    corrections_applied.append(f"gain: {gain:.2f}")
                
                if corrections_applied:
                    print(f"🎛️ 3-way color correction applied: {', '.join(corrections_applied)}")

                # Convert modified HSV back to RGB after lift/gamma/gain processing
                processed_image = hsv_to_rgb(torch.stack([h, s, v], dim=-1))
                processed_image = torch.clamp(processed_image, 0.0, 1.0)

                if noise > 0.0:
                    mono_noise = torch.randn(
                        processed_image.shape[:-1], device=processed_image.device
                    ).unsqueeze(-1)
                    luminance_mask = 1.0 - torch.abs(v - 0.5) * 2.0
                    luminance_mask = torch.clamp(luminance_mask, 0.0, 1.0).unsqueeze(-1)

                    rgb_temp = hsv_to_rgb(torch.stack([h, s, v], dim=-1))
                    rgb_temp += mono_noise * noise * 0.15 * luminance_mask
                    rgb_temp = torch.clamp(rgb_temp, 0.0, 1.0)

                    hsv_temp = rgb_to_hsv(rgb_temp)
                    h, s, v = hsv_temp[..., 0], hsv_temp[..., 1], hsv_temp[..., 2]

        # --- COLORIZE MODE ---
        elif mode == "Colorize":
            # Smart detection to avoid colorizing art/anime
            if analysis:
                scene_type = analysis["scene_type"]
                if scene_type in [
                    "anime",
                    "concept_art",
                    "stylized_art",
                    "detailed_illustration",
                ]:
                    print(
                        f"🚫 Colorize Mode: Skipping colorization for {scene_type} content"
                    )
                    # Return original image for art content
                    processed_image = original_image.clone()
                else:
                    print(
                        f"🎨 Colorize Mode: Processing {scene_type} for intelligent colorization"
                    )
                    processed_image = self._apply_colorization(
                        original_image,
                        processed_image,
                        analysis,
                        colorize_strength,
                        skin_warmth,
                        sky_saturation,
                        vegetation_green,
                        sepia_tone,
                        colorize_mode,
                    )
            else:
                print(
                    "🎨 Colorize Mode: Applying general colorization (no AI analysis)"
                )
                processed_image = self._apply_colorization(
                    original_image,
                    processed_image,
                    None,
                    colorize_strength,
                    skin_warmth,
                    sky_saturation,
                    vegetation_green,
                    sepia_tone,
                    colorize_mode,
                )

            # --- SKIN TONE PROTECTION ---
            if analysis and analysis["faces"] and adjust_for_skin_tone:
                hsv_original = rgb_to_hsv(original_image)
                h_orig, s_orig, v_orig = (
                    hsv_original[..., 0],
                    hsv_original[..., 1],
                    hsv_original[..., 2],
                )

                skin_hue_mask = ((h_orig >= 0.0) & (h_orig <= 0.14)) | (
                    (h_orig >= 0.9) & (h_orig <= 1.0)
                )
                skin_sat_mask = (s_orig >= 0.15) & (s_orig <= 0.8)
                skin_val_mask = v_orig >= 0.2
                skin_mask = (skin_hue_mask & skin_sat_mask & skin_val_mask).float()

                h = torch.lerp(h, h_orig, skin_mask * 0.8)
                s = torch.lerp(s, s_orig, skin_mask * 0.6)
                v = torch.lerp(v, v_orig, skin_mask * 0.3)

            s = torch.clamp(s, 0.0, 1.0)
            v = torch.clamp(v, 0.0, 1.0)
            processed_hsv = torch.stack([h, s, v], dim=-1)
            processed_image = hsv_to_rgb(processed_hsv)

        if mode in ["Auto", "Preset"]:
            processed_image = torch.lerp(
                original_image, processed_image, effect_strength
            )

        if mode == "Manual" and noise > 0.0:
            mono_noise = torch.randn(
                processed_image.shape[:-1], device=image.device
            ).unsqueeze(-1)
            hsv_for_noise = rgb_to_hsv(processed_image)
            v_for_noise = hsv_for_noise[..., 2]
            luminance_mask = 1.0 - torch.abs(v_for_noise - 0.5) * 2.0
            luminance_mask = torch.clamp(luminance_mask, 0.0, 1.0).unsqueeze(-1)

            processed_image += mono_noise * noise * 0.2 * luminance_mask

        processed_image = torch.clamp(processed_image, 0.0, 1.0)

        # --- COLOR REFERENCE MATCHING ---
        if reference_image is not None and reference_strength > 0.0:
            image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
            ref_np = (reference_image[0].cpu().numpy() * 255).astype(np.uint8)

            matched_image = match_to_reference_colors(
                image_np, ref_np, reference_strength
            )
            processed_image = (
                torch.from_numpy(matched_image.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(processed_image.device)
            )
            processed_image = torch.clamp(processed_image, 0.0, 1.0)
            print(
                f"🎨 Applied color reference matching with {reference_strength:.1f} strength"
            )

        # --- COLOR PALETTE EXTRACTION ---
        palette_data = ""
        hex_colors = []
        if extract_palette:
            image_for_palette = (processed_image[0].cpu().numpy() * 255).astype(
                np.uint8
            )
            hex_colors = extract_color_palette(image_for_palette, num_colors=6)
            if hex_colors:
                palette_data = ",".join(hex_colors)
                print(f"🎨 Extracted color palette: {palette_data}")

        if mask is not None:
            if mask.shape[1:] != (height, width):
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            mask = mask.unsqueeze(-1)
            processed_image = torch.lerp(original_image, processed_image, mask)

        # --- GENERATE VISUALIZATION IMAGES ---
        final_image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)

        if extract_palette:
            histogram_image_np = generate_histogram_image(final_image_np)
            histogram_tensor = (
                torch.from_numpy(histogram_image_np.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(processed_image.device)
            )
        else:
            histogram_tensor = torch.zeros(
                (1, 512, 768, 3), device=processed_image.device
            )

        if extract_palette and hex_colors:
            palette_image_np = generate_palette_image(hex_colors)
            palette_image_tensor = (
                torch.from_numpy(palette_image_np.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(processed_image.device)
            )
        else:
            palette_image_tensor = torch.zeros(
                (1, 120, 600, 3), device=processed_image.device
            )

        return (processed_image, palette_data, histogram_tensor, palette_image_tensor)

    def _apply_colorization(
        self,
        original_image,
        processed_image,
        analysis,
        colorize_strength,
        skin_warmth,
        sky_saturation,
        vegetation_green,
        sepia_tone,
        colorize_mode,
    ):
        """
        Apply intelligent colorization to grayscale or desaturated photos.
        Uses GPU-optimized tensor operations for efficient processing.
        """
        device = original_image.device

        # Convert to HSV for color manipulation
        hsv_image = rgb_to_hsv(processed_image)
        h, s, v = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]

        # Check if image is grayscale or very desaturated
        avg_saturation = torch.mean(s).item()
        if avg_saturation > 0.3:
            print(
                f"⚠️ Image already has color (avg saturation: {avg_saturation:.2f}), applying gentle enhancement"
            )
            colorize_strength *= 0.3  # Reduce strength for already colored images

        # Create region masks based on luminance and edge detection
        luminance = torch.mean(processed_image, dim=-1, keepdim=True)

        # Sky detection (upper regions with high luminance)
        height = processed_image.shape[1]
        sky_region = torch.zeros_like(luminance, device=device)
        sky_upper_third = height // 3
        sky_region[:, :sky_upper_third, :, :] = 1.0

        # Enhance sky detection with luminance
        sky_luminance_mask = (luminance > 0.7).float()
        sky_mask = sky_region * sky_luminance_mask

        # Vegetation detection (mid-luminance areas, typically green regions)
        vegetation_mask = ((luminance > 0.2) & (luminance < 0.8)).float()
        vegetation_mask = vegetation_mask * (1.0 - sky_mask)  # Exclude sky areas

        # Skin tone detection (mid-luminance warm areas)
        skin_mask = torch.zeros_like(luminance, device=device)
        if analysis and analysis.get("faces"):
            # Use luminance-based approximation for skin areas
            skin_mask = ((luminance > 0.25) & (luminance < 0.85)).float()
            skin_mask = skin_mask * (1.0 - sky_mask) * (1.0 - vegetation_mask)

        # Apply colorization based on mode
        if colorize_mode == "vintage":
            # Vintage sepia-toned colorization
            base_hue = 0.08  # Warm sepia hue
            h = torch.full_like(h, base_hue)
            s = s + sepia_tone * 0.4 * colorize_strength

        elif colorize_mode == "portrait":
            # Portrait-focused colorization
            # Warm skin tones
            skin_hue = 0.08  # Warm skin hue
            h = torch.where(skin_mask > 0.3, skin_hue, h)
            s = torch.where(skin_mask > 0.3, s + skin_warmth * colorize_strength, s)

            # Subtle sky blues
            sky_hue = 0.58  # Blue hue
            h = torch.where(sky_mask > 0.5, sky_hue, h)
            s = torch.where(
                sky_mask > 0.5, s + sky_saturation * 0.3 * colorize_strength, s
            )

        elif colorize_mode == "landscape":
            # Landscape-focused colorization
            # Green vegetation
            vegetation_hue = 0.25  # Green hue
            h = torch.where(vegetation_mask > 0.4, vegetation_hue, h)
            s = torch.where(
                vegetation_mask > 0.4, s + vegetation_green * colorize_strength, s
            )

            # Blue skies
            sky_hue = 0.58  # Blue hue
            h = torch.where(sky_mask > 0.5, sky_hue, h)
            s = torch.where(sky_mask > 0.5, s + sky_saturation * colorize_strength, s)

        else:  # auto mode
            # Intelligent auto colorization
            # Sky areas -> blue
            sky_hue = 0.58
            h = torch.where(sky_mask > 0.5, sky_hue, h)
            s = torch.where(sky_mask > 0.5, s + sky_saturation * colorize_strength, s)

            # Vegetation areas -> green
            vegetation_hue = 0.25
            h = torch.where(vegetation_mask > 0.4, vegetation_hue, h)
            s = torch.where(
                vegetation_mask > 0.4, s + vegetation_green * colorize_strength, s
            )

            # Skin areas -> warm tones
            if torch.sum(skin_mask) > 0:
                skin_hue = 0.08
                h = torch.where(skin_mask > 0.3, skin_hue, h)
                s = torch.where(skin_mask > 0.3, s + skin_warmth * colorize_strength, s)

        # Apply sepia tone if specified
        if sepia_tone > 0:
            sepia_hue = 0.08  # Warm sepia
            h = torch.lerp(h, torch.full_like(h, sepia_hue), sepia_tone)
            s = s + sepia_tone * 0.3

        # Ensure values stay in valid range
        h = h % 1.0  # Wrap hue
        s = torch.clamp(s, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        # Convert back to RGB
        colorized_hsv = torch.stack([h, s, v], dim=-1)
        colorized_rgb = hsv_to_rgb(colorized_hsv)

        # Blend with original based on colorize_strength
        final_image = torch.lerp(processed_image, colorized_rgb, colorize_strength)

        return torch.clamp(final_image, 0.0, 1.0)


class BatchColorCorrection:
    """
    Batch Color Corrector node for processing video frame sequences from VHS upload nodes.
    Processes multiple frames efficiently while maintaining consistency across the sequence.
    """

    # Share presets with main EasyColorCorrection class
    PRESETS = EasyColorCorrection.PRESETS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["Auto", "Preset", "Manual"], {"default": "Auto"}),
                "frames_per_batch": (
                    "INT",
                    {
                        "default": 16, 
                        "min": 1, 
                        "max": 64, 
                        "step": 1,
                        "tooltip": "🎯 Batch Size Guide:\n1-4: Best Quality but Slow (high memory per frame)\n8-16: Balanced Speed & Quality (recommended)\n32-64: Fastest but Resource Heavy (requires more VRAM)"
                    },
                ),
                "use_gpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "⚠️ GPU: Faster processing but uses significant VRAM (2-8GB+ for large batches). CPU: Slower but uses system RAM instead of VRAM.",
                    },
                ),
            },
            "optional": {
                "ai_analysis": ("BOOLEAN", {"default": True}),
                "preset": (
                    [
                        "Natural",
                        "Warm",
                        "Cool",
                        "High Key",
                        "Dramatic",
                        "Epic Fantasy",
                        "Sci-Fi Chrome",
                        "Dark Fantasy",
                        "Vibrant Concept",
                        "Matte Painting",
                        "Digital Art",
                        "Anime Bright",
                        "Anime Moody",
                        "Cyberpunk",
                        "Pastel Dreams",
                        "Neon Nights",
                        "Comic Book",
                        "Cinematic",
                        "Teal & Orange",
                        "Film Noir",
                        "Vintage Film",
                        "Bleach Bypass",
                        "Golden Hour",
                        "Blue Hour",
                        "Sunny Day",
                        "Overcast",
                        "Sepia",
                        "Black & White",
                        "Faded",
                        "Moody",
                    ],
                    {"default": "Natural"},
                ),
                "effect_strength": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "enhancement_strength": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.1},
                ),
                "adjust_for_skin_tone": ("BOOLEAN", {"default": True}),
                "white_balance_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "warmth": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "vibrancy": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "brightness": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "lift": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "gamma": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "gain": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "noise": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "extract_palette": ("BOOLEAN", {"default": False}),
                "reference_image": ("IMAGE",),
                "reference_strength": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = (
        "images",
        "palette_data",
        "histogram",
        "palette_image",
        "frame_count",
    )
    FUNCTION = "batch_color_correct"
    CATEGORY = "itsjustregi / Easy Color Corrector"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always update for video previews

    DISPLAY_NAME = "Batch Color Corrector"

    def batch_color_correct(
        self,
        images,
        mode="Auto",
        frames_per_batch=16,
        use_gpu=True,
        ai_analysis=True,
        preset="Natural",
        effect_strength=0.4,
        enhancement_strength=0.8,
        adjust_for_skin_tone=True,
        white_balance_strength=0.6,
        warmth=0.0,
        vibrancy=0.0,
        brightness=0.0,
        contrast=0.0,
        lift=0.0,
        gamma=0.0,
        gain=0.0,
        noise=0.0,
        extract_palette=False,
        reference_image=None,
        reference_strength=0.5,
        mask=None,
    ):
        """
        GPU-optimized batch processing for video frame sequences.
        Processes multiple frames efficiently while keeping tensors on GPU.
        """

        # Get batch dimensions and device
        total_frames = images.shape[0]
        frame_height = images.shape[1]
        frame_width = images.shape[2]
        device = images.device

        # Debug device and user preferences
        print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
        print(f"🔧 User GPU preference: {use_gpu}")
        print(f"🔧 Initial device: {device}")
        print(f"🔧 Input tensor device: {images.device} | dtype: {images.dtype}")

        # Handle GPU processing based on user choice
        if use_gpu and torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
            print(f"🚀 GPU Memory Before: {gpu_memory_before:.2f} GB")

            if not str(device).startswith("cuda"):
                print("🚀 User enabled GPU - moving tensors to GPU...")
                images = images.cuda()
                device = images.device
                print(f"✅ Images moved to: {device}")
                if mask is not None:
                    mask = mask.cuda()
                    print(f"✅ Mask moved to: {device}")
            else:
                print(f"✅ Tensors already on GPU: {device}")

        elif use_gpu and not torch.cuda.is_available():
            print("❌ User requested GPU but CUDA not available - falling back to CPU")
            gpu_memory_before = 0

        else:
            print("💻 User selected CPU processing - keeping tensors on CPU")
            gpu_memory_before = 0

        print(
            f"🎬 GPU Batch Color Corrector: Processing {total_frames} frames ({frame_width}x{frame_height}) on {device}"
        )
        print(f"📊 Processing in batches of {frames_per_batch} frames")

        # Prepare output containers on GPU
        processed_frames = []
        all_palette_data = []
        all_histograms = []
        all_palette_images = []

        # Process frames in GPU-optimized batches with interruption support
        try:
            for batch_start in range(0, total_frames, frames_per_batch):
                # Check for interruption requests
                import comfy.model_management as model_management

                if model_management.interrupt_processing:
                    print("🛑 Batch processing interrupted by user")
                    # Return partially processed results
                    if processed_frames:
                        partial_images = torch.cat(processed_frames, dim=0)
                        partial_count = partial_images.shape[0]
                        print(
                            f"⚠️ Partial result: {partial_count}/{total_frames} frames processed"
                        )
                        return (
                            partial_images,
                            "",
                            torch.zeros((1, 512, 768, 3), device=device),
                            torch.zeros((1, 120, 600, 3), device=device),
                            partial_count,
                        )
                    else:
                        return (
                            images,
                            "",
                            torch.zeros((1, 512, 768, 3), device=device),
                            torch.zeros((1, 120, 600, 3), device=device),
                            0,
                        )

                batch_end = min(batch_start + frames_per_batch, total_frames)

                print(
                    f"🔄 GPU Processing batch {batch_start//frames_per_batch + 1}/{(total_frames + frames_per_batch - 1)//frames_per_batch}: frames {batch_start}-{batch_end-1}"
                )

                # Extract current batch - keep on GPU
                batch_frames = images[
                    batch_start:batch_end
                ]  # Shape: (batch_size, H, W, C)
                batch_masks = mask[batch_start:batch_end] if mask is not None else None

                # Process entire batch on GPU using vectorized operations
                batch_processed = self._process_batch_gpu(
                    batch_frames=batch_frames,
                    batch_masks=batch_masks,
                    mode=mode,
                    ai_analysis=ai_analysis,
                    preset=preset,
                    effect_strength=effect_strength,
                    enhancement_strength=enhancement_strength,
                    adjust_for_skin_tone=adjust_for_skin_tone,
                    white_balance_strength=white_balance_strength,
                    warmth=warmth,
                    vibrancy=vibrancy,
                    brightness=brightness,
                    contrast=contrast,
                    lift=lift,
                    gamma=gamma,
                    gain=gain,
                    noise=noise,
                    device=device,
                )

                processed_frames.append(batch_processed)

                # Only extract palette from middle frame to avoid CPU overhead
                if extract_palette and batch_start <= total_frames // 2 < batch_end:
                    middle_frame_idx = (total_frames // 2) - batch_start
                    middle_frame = batch_processed[
                        middle_frame_idx : middle_frame_idx + 1
                    ]

                    # Generate GPU-based simplified palette without CPU transfer
                    palette_data = "GPU_BATCH_MODE"  # Simplified for performance
                    histogram_tensor = torch.zeros((1, 512, 768, 3), device=device)
                    palette_img_tensor = torch.zeros((1, 120, 600, 3), device=device)

                    all_palette_data.append(palette_data)
                    all_histograms.append(histogram_tensor)
                    all_palette_images.append(palette_img_tensor)

        except KeyboardInterrupt:
            print("🛑 Batch processing interrupted by KeyboardInterrupt")
            if processed_frames:
                partial_images = torch.cat(processed_frames, dim=0)
                partial_count = partial_images.shape[0]
                print(
                    f"⚠️ Partial result: {partial_count}/{total_frames} frames processed"
                )
                return (
                    partial_images,
                    "",
                    torch.zeros((1, 512, 768, 3), device=device),
                    torch.zeros((1, 120, 600, 3), device=device),
                    partial_count,
                )
            else:
                return (
                    images,
                    "",
                    torch.zeros((1, 512, 768, 3), device=device),
                    torch.zeros((1, 120, 600, 3), device=device),
                    0,
                )
        except Exception as e:
            print(f"❌ Error during batch processing: {e}")
            if processed_frames:
                partial_images = torch.cat(processed_frames, dim=0)
                partial_count = partial_images.shape[0]
                print(
                    f"⚠️ Partial result after error: {partial_count}/{total_frames} frames processed"
                )
                return (
                    partial_images,
                    "",
                    torch.zeros((1, 512, 768, 3), device=device),
                    torch.zeros((1, 120, 600, 3), device=device),
                    partial_count,
                )
            else:
                return (
                    images,
                    "",
                    torch.zeros((1, 512, 768, 3), device=device),
                    torch.zeros((1, 120, 600, 3), device=device),
                    0,
                )

        # Combine all processed frames on GPU
        if processed_frames:
            final_images = torch.cat(processed_frames, dim=0)

            # Use middle frame data for representation
            representative_palette = all_palette_data[0] if all_palette_data else ""
            representative_histogram = (
                all_histograms[0]
                if all_histograms
                else torch.zeros((1, 512, 768, 3), device=device)
            )
            representative_palette_img = (
                all_palette_images[0]
                if all_palette_images
                else torch.zeros((1, 120, 600, 3), device=device)
            )

            # Memory cleanup and final GPU status
            if use_gpu and torch.cuda.is_available():
                gpu_memory_after = torch.cuda.memory_allocated(device) / 1024**3  # GB
                print(
                    f"🚀 GPU Memory After: {gpu_memory_after:.2f} GB (Delta: {gpu_memory_after - gpu_memory_before:.2f} GB)"
                )

                # Force garbage collection to free memory
                import gc

                gc.collect()
                torch.cuda.empty_cache()

                gpu_memory_final = torch.cuda.memory_allocated(device) / 1024**3  # GB
                print(f"🧹 GPU Memory After Cleanup: {gpu_memory_final:.2f} GB")

            print(
                f"✅ GPU Batch processing complete: {total_frames} frames processed on {device}"
            )
            if representative_palette:
                print(f"🎨 Representative palette: {representative_palette}")

            return (
                final_images,
                representative_palette,
                representative_histogram,
                representative_palette_img,
                total_frames,
            )
        else:
            # Fallback if no frames processed
            return (
                images,
                "",
                torch.zeros((1, 512, 768, 3), device=device),
                torch.zeros((1, 120, 600, 3), device=device),
                0,
            )

    def _process_batch_gpu(
        self,
        batch_frames,
        batch_masks,
        mode,
        ai_analysis,
        preset,
        effect_strength,
        enhancement_strength,
        adjust_for_skin_tone,
        white_balance_strength,
        warmth,
        vibrancy,
        brightness,
        contrast,
        lift,
        gamma,
        gain,
        noise,
        device,
    ):
        """
        GPU-optimized batch processing that processes multiple frames simultaneously.
        """
        batch_size = batch_frames.shape[0]

        # Debug GPU utilization in core processing
        print(f"🔥 Processing batch of {batch_size} frames on {device}")
        print(f"🎯 Batch tensor device: {batch_frames.device}")
        print(
            f"🎛️ Parameters: warmth={warmth}, vibrancy={vibrancy}, brightness={brightness}, contrast={contrast}"
        )

        # Ensure batch is on correct GPU device
        if str(batch_frames.device) != str(device):
            print(f"⚠️ Moving batch from {batch_frames.device} to {device}")
            batch_frames = batch_frames.to(device)
            if batch_masks is not None:
                batch_masks = batch_masks.to(device)

        original_batch = batch_frames.clone()

        # Process all frames in batch simultaneously using vectorized operations
        processed_batch = batch_frames.clone()

        # AI Analysis (only on first frame to save computation)
        analysis = None
        if ai_analysis:
            # Use GPU-based analysis instead of CPU-heavy OpenCV operations
            analysis = self._analyze_image_gpu(batch_frames[0], device)

        # Apply preset modifications if in Preset mode
        if mode == "Preset":
            # Map simplified batch preset names to full preset names
            preset_mapping = {
                "Natural": "Natural Portrait",
                "Warm": "Warm Portrait",
                "Cool": "Cool Portrait",
                "High Key": "High Key Portrait",
                "Dramatic": "Dramatic Portrait",
            }

            full_preset_name = preset_mapping.get(preset, preset)
            if full_preset_name in self.PRESETS:
                preset_values = self.PRESETS[full_preset_name]
                warmth += preset_values.get("warmth", 0.0)
                vibrancy += preset_values.get("vibrancy", 0.0)
                brightness += preset_values.get("brightness", 0.0)
                contrast += preset_values.get("contrast", 0.0)
                print(f"🎨 Applied batch preset: {preset} -> {full_preset_name}")
            else:
                print(f"⚠️ Preset '{preset}' not found in PRESETS dictionary")

            if analysis and analysis["scene_type"] in [
                "concept_art",
                "anime",
                "stylized_art",
            ]:
                vibrancy *= 1.4
                contrast *= 1.25

        # Auto mode specific processing
        if mode == "Auto":
            print(
                f"🤖 Batch Auto Mode: Applying intelligent enhancements to {batch_size} frames"
            )

            # Apply white balance to entire batch if enabled
            if white_balance_strength > 0.0:
                if ai_analysis and ADVANCED_LIBS_AVAILABLE:
                    # AI-based white balance on first frame, apply to all
                    first_frame_np = (batch_frames[0].cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    wb_corrected_first = intelligent_white_balance(
                        first_frame_np, white_balance_strength
                    )

                    # Calculate the correction factors from first frame
                    original_mean = torch.mean(batch_frames[0], dim=(0, 1))
                    corrected_mean = torch.mean(
                        torch.from_numpy(
                            wb_corrected_first.astype(np.float32) / 255.0
                        ).to(device),
                        dim=(0, 1),
                    )
                    wb_factors = corrected_mean / (original_mean + 1e-6)

                    # Apply same factors to entire batch
                    processed_batch = processed_batch * wb_factors.view(1, 1, 1, 3)
                    processed_batch = torch.clamp(processed_batch, 0.0, 1.0)
                    print(
                        f"🔧 Applied AI white balance to batch (factors: {wb_factors})"
                    )
                else:
                    # Tensor-based white balance for entire batch
                    B, H, W, C = processed_batch.shape
                    flat_batch = processed_batch.view(B, -1, C)
                    percentile_40 = torch.quantile(
                        flat_batch, 0.40, dim=1, keepdim=True
                    )
                    percentile_60 = torch.quantile(
                        flat_batch, 0.60, dim=1, keepdim=True
                    )
                    midtone_mean = (percentile_40 + percentile_60) / 2.0
                    avg_gray = torch.mean(midtone_mean, dim=-1, keepdim=True)
                    scale = avg_gray / (midtone_mean + 1e-6)
                    scale = torch.lerp(
                        torch.ones_like(scale), scale, white_balance_strength
                    )
                    scale = scale.view(B, 1, 1, C)
                    processed_batch = processed_batch * scale
                    processed_batch = torch.clamp(processed_batch, 0.0, 1.0)
                    print(f"🔧 Applied tensor white balance to batch")

            # Apply enhancement based on scene analysis
            if enhancement_strength > 0.2:
                hsv_temp = rgb_to_hsv(processed_batch)
                h_temp, s_temp, v_temp = (
                    hsv_temp[..., 0],
                    hsv_temp[..., 1],
                    hsv_temp[..., 2],
                )

                # Initialize variables with defaults
                scene_type = "general"
                lighting = "auto"
                
                if analysis:
                    scene_type = analysis["scene_type"]
                    lighting = analysis["lighting_condition"]

                    # Scene-specific enhancements
                    if scene_type == "anime":
                        contrast_boost = 0.18 * enhancement_strength
                        saturation_boost = 0.55 * enhancement_strength
                        v_temp = 0.5 + (v_temp - 0.5) * (1.0 + contrast_boost)
                        s_temp = s_temp * (1.0 + saturation_boost)
                        print(
                            f"🎨 Applied anime enhancement (contrast: {contrast_boost:.3f}, saturation: {saturation_boost:.3f})"
                        )
                    elif scene_type == "concept_art":
                        contrast_boost = 0.25 * enhancement_strength
                        saturation_boost = 0.40 * enhancement_strength
                        v_temp = 0.5 + (v_temp - 0.5) * (1.0 + contrast_boost)
                        s_temp = s_temp * (1.0 + saturation_boost)
                        print(f"🎨 Applied concept art enhancement")
                    elif scene_type == "portrait":
                        warmth += 0.05 * enhancement_strength
                        contrast_boost = 0.12 * enhancement_strength
                        v_temp = 0.5 + (v_temp - 0.5) * (1.0 + contrast_boost)
                        print(f"🎨 Applied portrait enhancement")
                    else:
                        # General enhancement
                        contrast_boost = 0.15 * enhancement_strength
                        saturation_boost = 0.20 * enhancement_strength
                        v_temp = 0.5 + (v_temp - 0.5) * (1.0 + contrast_boost)
                        s_temp = s_temp * (1.0 + saturation_boost)
                        print(f"🎨 Applied general enhancement")

                    # Lighting adjustments
                    if lighting == "low_light":
                        brightness += 0.1 * enhancement_strength
                        contrast += 0.4 * enhancement_strength
                        print(f"💡 Applied low-light enhancement")
                    elif lighting == "bright":
                        brightness -= 0.05 * enhancement_strength
                        print(f"💡 Applied bright lighting adjustment")
                    elif lighting == "flat":
                        contrast += 0.5 * enhancement_strength
                        print(f"💡 Applied flat lighting enhancement")

                s_temp = torch.clamp(s_temp, 0.0, 1.0)
                v_temp = torch.clamp(v_temp, 0.0, 1.0)
                processed_batch = hsv_to_rgb(
                    torch.stack([h_temp, s_temp, v_temp], dim=-1)
                )

            # Face enhancement for Auto mode (if faces detected and adjust_for_skin_tone enabled)
            if (
                analysis
                and analysis.get("faces")
                and adjust_for_skin_tone
                and ADVANCED_LIBS_AVAILABLE
            ):
                print(
                    f"👤 Applying face enhancement to batch with {len(analysis['faces'])} faces detected"
                )
                # Warning for overdriven enhancement
                if enhancement_strength > 1.0:
                    print(f"⚠️ WARNING: Enhancement strength ({enhancement_strength:.1f}) > 1.0 with skin tone adjustment may cause color artifacts in highlights. Recommended: ≤ 1.0")
                face_enhanced_frames = []

                for i in range(batch_size):
                    frame_np = (processed_batch[i].cpu().numpy() * 255).astype(np.uint8)
                    enhanced_frame_np = enhance_faces(
                        frame_np, analysis["faces"], enhancement_strength * 0.5
                    )
                    enhanced_frame_tensor = torch.from_numpy(
                        enhanced_frame_np.astype(np.float32) / 255.0
                    ).to(device)
                    face_enhanced_frames.append(enhanced_frame_tensor)

                processed_batch = torch.stack(face_enhanced_frames, dim=0)
                print(f"✅ Applied face enhancement to {batch_size} frames")

        # Convert to HSV for batch processing
        hsv_batch = rgb_to_hsv(processed_batch)
        h, s, v = hsv_batch[..., 0], hsv_batch[..., 1], hsv_batch[..., 2]

        # Track changes to confirm processing is happening
        original_mean = torch.mean(processed_batch).item()
        print(f"📊 Original batch mean: {original_mean:.4f}")

        # Apply color corrections to entire batch
        if warmth != 0.0:
            h = (h + warmth * 0.1) % 1.0

        if vibrancy != 0.0:
            saturation_mask = 1.0 - s
            s = s * (1.0 + vibrancy) + (vibrancy * 0.3 * saturation_mask * s)

        if brightness != 0.0:
            v = v + brightness * (1.0 - v * 0.5)

        if contrast != 0.0:
            v = 0.5 + (v - 0.5) * (1.0 + contrast)

        # Manual mode 3-way color correction (applied to entire batch)
        if mode == "Manual":
            # Create masks for entire batch
            shadows_mask = 1.0 - torch.clamp(v * 3.0, 0.0, 1.0)
            midtones_mask = 1.0 - torch.abs(v - 0.5) * 2.0
            highlights_mask = torch.clamp((v - 0.66) * 3.0, 0.0, 1.0)

            if lift != 0.0:
                v = v + (lift * 0.8 * shadows_mask)

            if gamma != 0.0:
                gamma_exp = 1.0 / (1.0 + gamma * 1.2)
                v_gamma = torch.pow(torch.clamp(v, 0.001, 1.0), gamma_exp)
                v = torch.lerp(v, v_gamma, midtones_mask)

            if gain != 0.0:
                v = v + (gain * 0.8 * highlights_mask)

        # Add noise to entire batch if specified
        if noise > 0.0:
            mono_noise = torch.randn(
                (batch_size, processed_batch.shape[1], processed_batch.shape[2], 1),
                device=device,
            )
            luminance_mask = 1.0 - torch.abs(v - 0.5) * 2.0
            luminance_mask = torch.clamp(luminance_mask, 0.0, 1.0).unsqueeze(-1)

            rgb_temp = hsv_to_rgb(torch.stack([h, s, v], dim=-1))
            rgb_temp += mono_noise * noise * 0.15 * luminance_mask
            rgb_temp = torch.clamp(rgb_temp, 0.0, 1.0)

            hsv_temp = rgb_to_hsv(rgb_temp)
            h, s, v = hsv_temp[..., 0], hsv_temp[..., 1], hsv_temp[..., 2]

        # Clamp and convert back to RGB
        s = torch.clamp(s, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)
        processed_hsv = torch.stack([h, s, v], dim=-1)
        processed_batch = hsv_to_rgb(processed_hsv)

        # Apply effect strength to entire batch
        if mode in ["Auto", "Preset"]:
            processed_batch = torch.lerp(
                original_batch, processed_batch, effect_strength
            )

        # Apply masks to entire batch if provided
        if batch_masks is not None:
            if batch_masks.shape[1:] != (
                processed_batch.shape[1],
                processed_batch.shape[2],
            ):
                batch_masks = F.interpolate(
                    batch_masks.unsqueeze(1),
                    size=(processed_batch.shape[1], processed_batch.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            batch_masks = batch_masks.unsqueeze(-1)
            processed_batch = torch.lerp(original_batch, processed_batch, batch_masks)

        processed_batch = torch.clamp(processed_batch, 0.0, 1.0)

        # Confirm processing happened
        final_mean = torch.mean(processed_batch).item()
        change = abs(final_mean - original_mean)
        print(f"📊 Final batch mean: {final_mean:.4f} | Change: {change:.6f}")
        if change > 0.001:
            print("✅ Color correction applied successfully")
        else:
            print("⚠️ Minimal/no changes detected")

        return processed_batch

    def _analyze_image_gpu(self, image_tensor, device):
        """
        GPU-based image analysis without CPU bottlenecks.
        Replaces CPU-heavy OpenCV operations with GPU tensor operations.
        """
        # Convert to HSV for analysis
        hsv = rgb_to_hsv(image_tensor.unsqueeze(0))[0]  # Remove batch dim
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Analyze brightness distribution (replaces histogram analysis)
        brightness_mean = torch.mean(v).item()
        brightness_std = torch.std(v).item()

        # Determine lighting condition based on brightness stats
        if brightness_mean < 0.3:
            lighting_condition = "low_light"
        elif brightness_mean > 0.8:
            lighting_condition = "bright"
        elif brightness_std < 0.15:
            lighting_condition = "flat"
        else:
            lighting_condition = "optimal"

        # Analyze saturation for scene type detection
        saturation_mean = torch.mean(s).item()
        saturation_std = torch.std(s).item()

        # Simple scene classification based on color statistics
        if saturation_mean > 0.6 and saturation_std > 0.25:
            scene_type = "concept_art"
        elif saturation_mean > 0.5:
            scene_type = "stylized_art"
        elif saturation_mean < 0.3:
            scene_type = "portrait"
        else:
            scene_type = "realistic_photo"

        # Edge detection using Sobel filters on GPU
        gray = torch.mean(image_tensor, dim=-1, keepdim=True)

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device
        ).view(1, 1, 3, 3)

        # Apply convolution for edge detection
        gray_padded = F.pad(
            gray.permute(2, 0, 1).unsqueeze(0), (1, 1, 1, 1), mode="reflect"
        )
        edges_x = F.conv2d(gray_padded, sobel_x)
        edges_y = F.conv2d(gray_padded, sobel_y)
        edges = torch.sqrt(edges_x**2 + edges_y**2)

        edge_density = torch.mean(edges).item()

        return {
            "scene_type": scene_type,
            "lighting_condition": lighting_condition,
            "brightness_mean": brightness_mean,
            "saturation_mean": saturation_mean,
            "edge_density": edge_density,
            "has_faces": False,  # Simplified - no CPU face detection
            "skin_tone_areas": [],  # Simplified - avoid CPU processing
        }


class RawImageProcessor:
    """
    Advanced Image Processor for camera raw, HDR, and lossless formats.
    Handles DNG, ARW, CR2, NEF, EXR, HDR, TIFF 16-bit, and other high-quality formats.
    Outputs processed IMAGE data compatible with EasyColorCorrection.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "format_type": (
                    ["auto", "raw", "exr", "hdr", "tiff16"],
                    {"default": "auto"},
                ),
            },
            "optional": {
                # RAW-specific controls
                "white_balance": (
                    [
                        "auto",
                        "camera",
                        "daylight",
                        "cloudy",
                        "shade",
                        "tungsten",
                        "fluorescent",
                        "flash",
                    ],
                    {"default": "auto"},
                ),
                "demosaic_algorithm": (
                    ["AHD", "VNG", "PPG", "AAHD"],
                    {"default": "AHD"},
                ),
                # HDR/EXR tone mapping controls
                "tone_mapping": (
                    ["none", "reinhard", "drago", "aces"],
                    {"default": "aces"},
                ),
                "hdr_exposure": (
                    "FLOAT",
                    {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1},
                ),
                "hdr_gamma": (
                    "FLOAT",
                    {"default": 2.2, "min": 0.5, "max": 4.0, "step": 0.1},
                ),
                # Universal controls
                "exposure": (
                    "FLOAT",
                    {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1},
                ),
                "highlights": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "shadows": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "brightness": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "contrast": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "saturation": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.1},
                ),
                "noise_reduction": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "output_colorspace": (
                    ["sRGB", "Adobe RGB", "ProPhoto RGB"],
                    {"default": "sRGB"},
                ),
                "output_gamma": (["sRGB", "linear", "1.8", "2.2"], {"default": "sRGB"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "process_raw_image"
    CATEGORY = "itsjustregi / Easy Color Corrector"

    def process_raw_image(
        self,
        file_path,
        format_type="auto",
        white_balance="auto",
        demosaic_algorithm="AHD",
        tone_mapping="aces",
        hdr_exposure=0.0,
        hdr_gamma=2.2,
        exposure=0.0,
        highlights=0.0,
        shadows=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        noise_reduction=0.0,
        output_colorspace="sRGB",
        output_gamma="sRGB",
    ):
        """
        Process advanced image formats (RAW, EXR, HDR, TIFF 16-bit) with professional controls.
        """
        if not file_path or not file_path.strip():
            raise ValueError("File path is required")

        # Auto-detect format based on file extension
        file_extension = file_path.lower().split(".")[-1]
        if format_type == "auto":
            if file_extension in ["dng", "arw", "cr2", "nef", "orf", "rw2", "raf"]:
                format_type = "raw"
            elif file_extension == "exr":
                format_type = "exr"
            elif file_extension == "hdr":
                format_type = "hdr"
            elif file_extension in ["tiff", "tif"]:
                format_type = "tiff16"
            else:
                format_type = "raw"  # Default fallback

        try:
            if format_type == "raw":
                rgb_array, metadata_info = self._process_raw_format(
                    file_path,
                    white_balance,
                    demosaic_algorithm,
                    exposure,
                    highlights,
                    shadows,
                    brightness,
                    noise_reduction,
                    output_colorspace,
                    output_gamma,
                )

            elif format_type == "exr":
                rgb_array, metadata_info = self._process_exr_format(
                    file_path,
                    tone_mapping,
                    hdr_exposure,
                    hdr_gamma,
                    exposure,
                    highlights,
                    shadows,
                    brightness,
                    contrast,
                    saturation,
                )

            elif format_type == "hdr":
                rgb_array, metadata_info = self._process_hdr_format(
                    file_path,
                    tone_mapping,
                    hdr_exposure,
                    hdr_gamma,
                    exposure,
                    highlights,
                    shadows,
                    brightness,
                    contrast,
                    saturation,
                )

            elif format_type == "tiff16":
                rgb_array, metadata_info = self._process_tiff16_format(
                    file_path,
                    exposure,
                    highlights,
                    shadows,
                    brightness,
                    contrast,
                    saturation,
                )

            else:
                raise ValueError(f"Unsupported format type: {format_type}")

        except Exception as e:
            error_msg = f"Error processing {format_type.upper()} file: {str(e)}"
            print(error_msg)
            # Return a black image as fallback
            rgb_array = np.zeros((512, 512, 3), dtype=np.uint8)
            metadata_info = {"error": error_msg}

        # Convert to PyTorch tensor format expected by ComfyUI
        if rgb_array.dtype != np.uint8:
            # Normalize HDR data to 0-1 range before converting to uint8
            if rgb_array.max() > 1.0:
                rgb_array = rgb_array / rgb_array.max()
            rgb_array = (rgb_array * 255).astype(np.uint8)

        # Convert to float and normalize to 0-1 range
        image_tensor = torch.from_numpy(rgb_array.astype(np.float32) / 255.0)

        # Add batch dimension: (H, W, C) -> (1, H, W, C)
        image_tensor = image_tensor.unsqueeze(0)

        # Format metadata as string
        metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata_info.items()])

        print(
            f"✅ {format_type.upper()} Image Processed: {rgb_array.shape[1]}x{rgb_array.shape[0]} from {file_path}"
        )
        if metadata_str:
            print(f"📷 Metadata: {metadata_str}")

        return (image_tensor, metadata_str)

    def _process_raw_format(
        self,
        file_path,
        white_balance,
        demosaic_algorithm,
        exposure,
        highlights,
        shadows,
        brightness,
        noise_reduction,
        output_colorspace,
        output_gamma,
    ):
        """Process RAW camera formats using rawpy."""
        if not RAW_PROCESSING_AVAILABLE:
            raise ValueError(
                "RAW processing not available. Install rawpy: pip install rawpy"
            )

        with rawpy.imread(file_path) as raw:
            params = rawpy.Params()

            # Demosaic algorithm
            demosaic_map = {
                "AHD": rawpy.DemosaicAlgorithm.AHD,
                "VNG": rawpy.DemosaicAlgorithm.VNG,
                "PPG": rawpy.DemosaicAlgorithm.PPG,
                "AAHD": rawpy.DemosaicAlgorithm.AAHD,
            }
            params.demosaic_algorithm = demosaic_map.get(
                demosaic_algorithm, rawpy.DemosaicAlgorithm.AHD
            )

            # White balance
            if white_balance == "auto":
                params.use_auto_wb = True
            elif white_balance == "camera":
                params.use_camera_wb = True
            else:
                wb_presets = {
                    "daylight": [1.0, 1.0, 1.0, 1.0],
                    "cloudy": [1.2, 1.0, 0.8, 1.0],
                    "shade": [1.4, 1.0, 0.7, 1.0],
                    "tungsten": [0.6, 1.0, 1.8, 1.0],
                    "fluorescent": [0.8, 1.0, 1.4, 1.0],
                    "flash": [1.1, 1.0, 0.9, 1.0],
                }
                if white_balance in wb_presets:
                    params.user_wb = wb_presets[white_balance]

            # Exposure and processing
            params.exp_correc = True
            params.exp_shift = exposure
            params.highlight_mode = (
                rawpy.HighlightMode.Clip
                if highlights == 0
                else rawpy.HighlightMode.Reconstruct
            )
            params.bright = 1.0 + brightness
            params.output_color = (
                rawpy.ColorSpace.sRGB
                if output_colorspace == "sRGB"
                else rawpy.ColorSpace.Adobe
            )
            params.gamma = (1.0, 1.0) if output_gamma == "linear" else (2.2, 4.5)

            if noise_reduction > 0:
                params.median_filter_passes = int(noise_reduction * 5)

            rgb_array = raw.postprocess(params)

            metadata_info = {
                "format": "RAW",
                "width": rgb_array.shape[1],
                "height": rgb_array.shape[0],
                "demosaic": demosaic_algorithm,
                "white_balance": white_balance,
            }

        return rgb_array, metadata_info

    def _process_exr_format(
        self,
        file_path,
        tone_mapping,
        hdr_exposure,
        hdr_gamma,
        exposure,
        highlights,
        shadows,
        brightness,
        contrast,
        saturation,
    ):
        """Process EXR HDR format using OpenEXR."""
        if not EXR_PROCESSING_AVAILABLE:
            # Fallback to imageio if OpenEXR not available
            if IMAGEIO_AVAILABLE:
                return self._process_with_imageio(
                    file_path,
                    "EXR",
                    tone_mapping,
                    hdr_exposure,
                    hdr_gamma,
                    exposure,
                    highlights,
                    shadows,
                    brightness,
                    contrast,
                    saturation,
                )
            else:
                raise ValueError("EXR processing requires OpenEXR or imageio library")

        exr_file = OpenEXR.InputFile(file_path)
        header = exr_file.header()

        # Get image dimensions
        dw = header["displayWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Read RGB channels
        channels = exr_file.channels(
            ["R", "G", "B"], Imath.PixelType(Imath.PixelType.FLOAT)
        )

        # Convert to numpy arrays
        r_channel = np.frombuffer(channels[0], dtype=np.float32).reshape(height, width)
        g_channel = np.frombuffer(channels[1], dtype=np.float32).reshape(height, width)
        b_channel = np.frombuffer(channels[2], dtype=np.float32).reshape(height, width)

        # Combine channels
        rgb_array = np.stack([r_channel, g_channel, b_channel], axis=2)

        # Apply HDR exposure
        if hdr_exposure != 0:
            rgb_array = rgb_array * (2**hdr_exposure)

        # Apply tone mapping
        rgb_array = self._apply_tone_mapping(rgb_array, tone_mapping, hdr_gamma)

        metadata_info = {
            "format": "EXR",
            "width": width,
            "height": height,
            "tone_mapping": tone_mapping,
            "hdr_exposure": hdr_exposure,
        }

        return rgb_array, metadata_info

    def _process_hdr_format(
        self,
        file_path,
        tone_mapping,
        hdr_exposure,
        hdr_gamma,
        exposure,
        highlights,
        shadows,
        brightness,
        contrast,
        saturation,
    ):
        """Process HDR format using imageio."""
        if not IMAGEIO_AVAILABLE:
            raise ValueError("HDR processing requires imageio library")

        # Read HDR image
        rgb_array = imageio.imread(file_path, format="HDR-FI")

        # Apply HDR exposure
        if hdr_exposure != 0:
            rgb_array = rgb_array * (2**hdr_exposure)

        # Apply tone mapping
        rgb_array = self._apply_tone_mapping(rgb_array, tone_mapping, hdr_gamma)

        metadata_info = {
            "format": "HDR",
            "width": rgb_array.shape[1],
            "height": rgb_array.shape[0],
            "tone_mapping": tone_mapping,
            "hdr_exposure": hdr_exposure,
        }

        return rgb_array, metadata_info

    def _process_tiff16_format(
        self, file_path, exposure, highlights, shadows, brightness, contrast, saturation
    ):
        """Process 16-bit TIFF format."""
        if not IMAGEIO_AVAILABLE:
            raise ValueError("TIFF 16-bit processing requires imageio library")

        # Read 16-bit TIFF
        rgb_array = imageio.imread(file_path)

        # Convert to float and normalize
        if rgb_array.dtype == np.uint16:
            rgb_array = rgb_array.astype(np.float32) / 65535.0
        elif rgb_array.dtype == np.uint8:
            rgb_array = rgb_array.astype(np.float32) / 255.0

        # Apply basic adjustments
        if exposure != 0:
            rgb_array = rgb_array * (2**exposure)

        rgb_array = np.clip(rgb_array, 0, 1)

        metadata_info = {
            "format": "TIFF 16-bit",
            "width": rgb_array.shape[1],
            "height": rgb_array.shape[0],
            "bit_depth": "16-bit" if rgb_array.dtype == np.uint16 else "8-bit",
        }

        return rgb_array, metadata_info

    def _process_with_imageio(
        self,
        file_path,
        format_name,
        tone_mapping,
        hdr_exposure,
        hdr_gamma,
        exposure,
        highlights,
        shadows,
        brightness,
        contrast,
        saturation,
    ):
        """Fallback processing using imageio."""
        rgb_array = imageio.imread(file_path)

        # Convert to float if needed
        if rgb_array.dtype == np.uint8:
            rgb_array = rgb_array.astype(np.float32) / 255.0
        elif rgb_array.dtype == np.uint16:
            rgb_array = rgb_array.astype(np.float32) / 65535.0

        # Apply HDR exposure if it's an HDR format
        if hdr_exposure != 0:
            rgb_array = rgb_array * (2**hdr_exposure)

        # Apply tone mapping for HDR formats
        if format_name in ["EXR", "HDR"]:
            rgb_array = self._apply_tone_mapping(rgb_array, tone_mapping, hdr_gamma)

        metadata_info = {
            "format": f"{format_name} (imageio fallback)",
            "width": rgb_array.shape[1],
            "height": rgb_array.shape[0],
        }

        return rgb_array, metadata_info

    def _apply_tone_mapping(self, rgb_array, tone_mapping, hdr_gamma):
        """Apply tone mapping to HDR image data."""
        if tone_mapping == "none":
            return np.clip(rgb_array, 0, 1)

        elif tone_mapping == "reinhard":
            # Simple Reinhard tone mapping
            return rgb_array / (1.0 + rgb_array)

        elif tone_mapping == "drago":
            # Drago tone mapping approximation
            luminance = np.dot(rgb_array, [0.299, 0.587, 0.114])
            max_lum = np.max(luminance)
            if max_lum > 0:
                scale = np.log10(max_lum + 1) / np.log10(
                    2.0
                    + 8.0 * ((luminance / max_lum) ** (np.log10(0.5) / np.log10(0.85)))
                )
                return rgb_array * scale[..., np.newaxis]
            return rgb_array

        elif tone_mapping == "aces":
            # ACES tone mapping curve approximation
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            return np.clip(
                (rgb_array * (a * rgb_array + b))
                / (rgb_array * (c * rgb_array + d) + e),
                0,
                1,
            )

        # Apply gamma correction
        if hdr_gamma != 2.2:
            rgb_array = np.power(np.clip(rgb_array, 0, 1), 1.0 / hdr_gamma)

        return np.clip(rgb_array, 0, 1)


class ColorCorrectionViewer:
    """
    Video Viewer for batch color-corrected image sequences.
    Provides playback controls with adjustable framerate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": (
                    "FLOAT",
                    {"default": 24.0, "min": 1.0, "max": 60.0, "step": 0.1},
                ),
            },
            "optional": {
                "auto_play": ("BOOLEAN", {"default": True}),
                "loop": ("BOOLEAN", {"default": True}),
                "frame_skip": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "view_sequence"
    CATEGORY = "itsjustregi / Easy Color Corrector"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always update for video display

    def view_sequence(self, images, fps=24.0, auto_play=True, loop=True, frame_skip=1):
        """
        Display image sequence with video playback controls.
        """
        if images is None or images.shape[0] == 0:
            print("⚠️ No images provided to viewer")
            return {"ui": {"text": "No images provided"}}

        total_frames = images.shape[0]

        # Apply frame skipping if specified
        if frame_skip > 1:
            selected_indices = torch.arange(0, total_frames, frame_skip)
            images = images[selected_indices]
            total_frames = images.shape[0]

        duration = total_frames / fps

        # For ComfyUI OUTPUT_NODE, we need to save images to ComfyUI's output directory
        import os
        import time
        from PIL import Image
        import folder_paths
        
        # Create subfolder first so we can reference it
        output_dir = folder_paths.get_output_directory()
        subfolder = f"colorviewer_{int(time.time())}"
        full_output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(full_output_dir, exist_ok=True)

        print(f"🎬 Color Corrector Viewer: {total_frames} frames at {fps} FPS")
        print(f"⏱️ Duration: {duration:.2f} seconds")
        print(f"🔄 Auto-play: {auto_play}, Loop: {loop}")
        if frame_skip > 1:
            print(f"⏭️ Frame skip: every {frame_skip} frames")
        
        # Store cleanup info for the deletion endpoint
        self._last_subfolder = subfolder
        
        results = []
        
        for i in range(total_frames):
            # Convert tensor to numpy and ensure proper format
            img_tensor = images[i]  # Single image
            img_np = img_tensor.cpu().numpy()
            
            # Ensure values are in 0-255 range and uint8
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # Save image to ComfyUI output directory
            img_pil = Image.fromarray(img_np, 'RGB')
            filename = f"frame_{i:04d}.png"
            img_path = os.path.join(full_output_dir, filename)
            img_pil.save(img_path)
            
            # Add image info in ComfyUI format
            results.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "output"
            })

        # Create a simple GIF for video preview (following VideoHelperSuite pattern)
        if total_frames > 1:
            # Create animated GIF from frames
            gif_filename = f"preview_{int(time.time())}.gif"
            gif_path = os.path.join(full_output_dir, gif_filename)
            
            try:
                # Convert images to PIL Images for GIF creation
                pil_images = []
                for filename_info in results:
                    img_path = os.path.join(full_output_dir, filename_info["filename"])
                    pil_img = Image.open(img_path)
                    pil_images.append(pil_img)
                
                # Create animated GIF
                if pil_images:
                    duration_ms = int(1000 / fps)  # Convert to milliseconds
                    pil_images[0].save(
                        gif_path,
                        save_all=True,
                        append_images=pil_images[1:],
                        duration=duration_ms,
                        loop=0 if loop else 1
                    )
                    
                    # Use VideoHelperSuite format for video preview
                    preview = {
                        "filename": gif_filename,
                        "subfolder": subfolder,
                        "type": "output",
                        "format": "gif",
                        "frame_rate": fps,
                        "frame_count": total_frames,
                        "duration": duration
                    }
                    
                    print(f"🎬 Created animated preview: {gif_filename}")
                    
                    return {
                        "ui": {
                            "gifs": [preview],  # Use VideoHelperSuite's "gifs" format
                            "images": results,   # Keep individual frames for compatibility
                        }
                    }
                    
            except Exception as e:
                print(f"⚠️ Failed to create GIF preview: {e}")
        
        # Fallback for single images or if GIF creation fails
        return {
            "ui": {
                "images": results,
                "frame_count": [total_frames],
                "duration": [duration], 
                "fps": [fps],
                "subfolder": [subfolder]
            }
        }


class ColorPaletteExtractor:
    """
    Extracts color palette from an image for use as reference input.
    Creates the same palette format used by the main EasyColorCorrection node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "palette_size": ("INT", {"default": 8, "min": 3, "max": 16, "step": 1}),
                "extract_palette": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("palette_data", "histogram", "palette_image")
    FUNCTION = "extract_palette"
    CATEGORY = "itsjustregi / Easy Color Corrector"

    def extract_palette(self, image, palette_size=8, extract_palette=True):
        """
        Extract color palette from input image.
        Returns palette data string, histogram visualization, and palette image.
        """
        if not extract_palette:
            # Return empty/default outputs when disabled
            empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return ("No palette extracted", empty_image, empty_image)

        try:
            # Convert image to numpy for processing
            image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            
            # Extract hex color palette
            hex_colors = extract_color_palette(image_np, num_colors=palette_size)
            
            # Create palette data string (same format as main node)
            if hex_colors:
                palette_data = ", ".join(hex_colors)
            else:
                palette_data = "No palette extracted"
            
            # Generate histogram image
            if ADVANCED_LIBS_AVAILABLE:
                histogram_image = generate_histogram_image(image_np)
                histogram_tensor = torch.from_numpy(histogram_image.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                # Create simple fallback histogram
                histogram_tensor = torch.zeros((1, 512, 768, 3), dtype=torch.float32)
                print("⚠️ Histogram generation disabled - missing OpenCV")
            
            # Generate palette image
            if hex_colors:
                palette_image = generate_palette_image(hex_colors)
                palette_tensor = torch.from_numpy(palette_image.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                # Create empty palette image if no colors extracted
                palette_tensor = torch.zeros((1, 120, 600, 3), dtype=torch.float32)
            
            print(f"🎨 Extracted {len(hex_colors)}-color palette: {palette_data}")
            
            return (palette_data, histogram_tensor, palette_tensor)
            
        except Exception as e:
            print(f"❌ ColorPaletteExtractor error: {e}")
            import traceback
            traceback.print_exc()
            # Return safe fallback
            empty_histogram = torch.zeros((1, 512, 768, 3), dtype=torch.float32)
            empty_palette = torch.zeros((1, 120, 600, 3), dtype=torch.float32)
            return ("Palette extraction failed", empty_histogram, empty_palette)
    
    @classmethod
    def cleanup_images(cls, subfolder):
        """Clean up generated images from a specific subfolder"""
        try:
            import folder_paths
            import shutil
            
            output_dir = folder_paths.get_output_directory()
            full_path = os.path.join(output_dir, subfolder)
            
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
                print(f"🗑️ Cleaned up {subfolder} directory")
                return True
            else:
                print(f"⚠️ Directory {subfolder} not found")
                return False
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False


# Export all classes
__all__ = [
    "EasyColorCorrection",
    "BatchColorCorrection",
    "RawImageProcessor",
    "ColorCorrectionViewer",
]
