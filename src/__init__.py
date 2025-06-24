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
    """Match image colors to reference image palette with robust error handling."""
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

        matched_image = image_np.copy()

        for channel in range(3):
            img_hist, _ = np.histogram(
                image_np[:, :, channel].flatten(), bins=256, range=(0, 256)
            )
            ref_hist, _ = np.histogram(
                reference_np[:, :, channel].flatten(), bins=256, range=(0, 256)
            )

            img_cdf = np.cumsum(img_hist).astype(np.float64)
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)

            img_cdf = img_cdf / (img_cdf[-1] + 1e-8)
            ref_cdf = ref_cdf / (ref_cdf[-1] + 1e-8)

            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                closest_idx = np.argmin(np.abs(ref_cdf - img_cdf[i]))
                lookup_table[i] = closest_idx

            original_channel = image_np[:, :, channel]
            matched_channel = lookup_table[original_channel]

            blended_channel = (
                original_channel * (1.0 - strength) + matched_channel * strength
            )
            matched_image[:, :, channel] = np.clip(blended_channel, 0, 255).astype(
                np.uint8
            )

        if ADVANCED_LIBS_AVAILABLE and strength > 0.3:
            try:
                lab_image = cv2.cvtColor(matched_image, cv2.COLOR_RGB2LAB).astype(
                    np.float32
                )
                lab_reference = cv2.cvtColor(reference_np, cv2.COLOR_RGB2LAB).astype(
                    np.float32
                )

                for i in range(1, 3):
                    img_mean = np.mean(lab_image[:, :, i])
                    ref_mean = np.mean(lab_reference[:, :, i])

                    color_shift = (
                        (ref_mean - img_mean) * strength * 0.3
                    )
                    lab_image[:, :, i] = lab_image[:, :, i] + color_shift

                lab_image[:, :, 0] = np.clip(lab_image[:, :, 0], 0, 100)
                lab_image[:, :, 1] = np.clip(
                    lab_image[:, :, 1], -127, 127
                )
                lab_image[:, :, 2] = np.clip(
                    lab_image[:, :, 2], -127, 127
                )

                lab_image_uint8 = lab_image.astype(np.uint8)
                matched_image = cv2.cvtColor(lab_image_uint8, cv2.COLOR_LAB2RGB)

            except Exception:
                pass

        matched_image = np.clip(matched_image, 0, 255).astype(np.uint8)
        return matched_image

    except Exception:
        # Return original image if anything fails
        return image_np


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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    """Advanced white balance using professional color science."""
    if not ADVANCED_LIBS_AVAILABLE:
        return image_np

    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)

        a_channel = lab[:, :, 1].astype(np.float32) - 128
        b_channel = lab[:, :, 2].astype(np.float32) - 128

        a_shift = np.median(a_channel)
        b_shift = np.median(b_channel)

        lab[:, :, 1] = np.clip(lab[:, :, 1] - a_shift * strength, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - b_shift * strength, 0, 255)

        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        try:
            rgb_norm = image_np.astype(np.float32) / 255.0

            grey_world_rgb = np.mean(rgb_norm.reshape(-1, 3), axis=0)

            xyz = colour.sRGB_to_XYZ(rgb_norm)

            illuminant_xy = colour.XYZ_to_xy(grey_world_rgb.reshape(1, -1))

            d65_xy = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
                "D65"
            ]

            if (
                np.linalg.norm(illuminant_xy - d65_xy) > 0.01
            ):
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

            final_mask = np.clip(final_mask, 0.0, 1.0)

            enhanced_face = face_region.copy()

            face_lab = cv2.cvtColor(enhanced_face, cv2.COLOR_RGB2LAB)
            l_channel = face_lab[:, :, 0].astype(np.float32)
            a_channel = face_lab[:, :, 1].astype(np.float32)
            b_channel = face_lab[:, :, 2].astype(np.float32)

            very_dark_skin = (l_channel < 60) & (
                a_channel > 132
            )
            dark_skin = (
                (l_channel < 90) & (l_channel >= 60) & (a_channel > 128)
            )
            very_light_skin = (
                l_channel > 180
            )

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
                        strength * 0.08,
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
                "mode": (["Auto", "Preset", "Manual"], {"default": "Auto"}),
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
                "realtime_preview": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable real-time preview when adjusting parameters",
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
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Strength of perceptual LAB-based white balance",
                    },
                ),
                "enhancement_strength": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.10,
                        "tooltip": "Overall strength of AI-powered enhancements",
                    },
                ),
                "pop_factor": (
                    "FLOAT",
                    {
                        "default": 0.30,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Extra pop factor for artistic content (anime, detailed photos)",
                    },
                ),
                "effect_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.10,
                        "tooltip": "Strength of overall color correction effect",
                    },
                ),
                "warmth": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Green/Magenta warmth adjustment",
                    },
                ),
                "vibrancy": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Saturation boost for less saturated areas",
                    },
                ),
                "contrast": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Contrast adjustment for overall image",
                    },
                ),
                "brightness": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.10,
                        "tooltip": "Brightness adjustment for overall image",
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
                "mask": ("MASK", {}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "palette_data", "histogram", "palette_image")
    FUNCTION = "run"
    CATEGORY = "itsjustregi / Easy Color Correction"
    DISPLAY_NAME = "Easy Color Correction"

    def run(
        self,
        image: torch.Tensor,
        mode: str,
        reference_image: typing.Optional[torch.Tensor] = None,
        reference_strength: float = 0.3,
        extract_palette: bool = False,
        realtime_preview: bool = False,
        ai_analysis: bool = True,
        adjust_for_skin_tone: bool = False,
        white_balance_strength: float = 0.6,
        enhancement_strength: float = 0.8,
        pop_factor: float = 0.7,
        effect_strength: float = 1.0,
        warmth: float = 0.0,
        vibrancy: float = 0.0,
        contrast: float = 0.0,
        brightness: float = 0.0,
        preset: str = "Anime",
        variation: float = 0.0,
        lift: float = 0.0,
        gamma: float = 0.0,
        gain: float = 0.0,
        noise: float = 0.0,
        mask: typing.Optional[torch.Tensor] = None,
    ) -> tuple:

        _ = realtime_preview

        original_image = image.clone()
        _, height, width, _ = image.shape

        processed_image = image.clone()

        # --- AUTO MODE ---
        if mode == "Auto":
            if ai_analysis and ADVANCED_LIBS_AVAILABLE:
                image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
                analysis = analyze_image_content(image_np, processed_image.device)
                print(
                    f"🤖 AI Analysis: {analysis['scene_type']} scene, {analysis['lighting']} lighting, {len(analysis['faces'])} faces detected"
                )
                if white_balance_strength > 0.0:
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
                if white_balance_strength > 0.0:
                    B, C = processed_image.shape[0], processed_image.shape[3]
                    flat_image = processed_image.view(B, -1, C)
                    percentile_40 = torch.quantile(
                        flat_image, 0.40, dim=1, keepdim=True
                    )
                    percentile_60 = torch.quantile(
                        flat_image, 0.60, dim=1, keepdim=True
                    )
                    midtone_mean = (percentile_40 + percentile_60) / 2.0
                    avg_gray = torch.mean(midtone_mean, dim=-1, keepdim=True)
                    scale = avg_gray / (midtone_mean + 1e-6)
                    scale = torch.lerp(
                        torch.ones_like(scale), scale, white_balance_strength
                    )
                    scale = scale.view(B, 1, 1, C)
                    color_deviation = torch.abs(scale - 1.0).max()
                    if color_deviation > 0.05:
                        processed_image = processed_image * scale
                        processed_image = torch.clamp(processed_image, 0.0, 1.0)
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

        # --- SHARED AI ANALYSIS ---
        analysis = None
        if ai_analysis and ADVANCED_LIBS_AVAILABLE:
            image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
            analysis = analyze_image_content(image_np, processed_image.device)
            print(
                f"🤖 AI Analysis for {mode} Mode: {analysis['scene_type']} scene, {analysis['lighting']} lighting, {len(analysis['faces'])} faces detected"
            )

        # --- PRESET MODE ---
        elif mode == "Preset":
            p_vals = self.PRESETS.get(
                preset, {}
            ).copy()
            if analysis:
                scene_type = analysis["scene_type"]
                lighting = analysis["lighting"]

                if scene_type == "concept_art":
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.4
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.3
                    p_vals["brightness"] = (
                        p_vals.get("brightness", 0.0) + 0.05
                    )
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
                    p_vals["warmth"] = (
                        p_vals.get("warmth", 0.0) + 0.05
                    )
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

            # Apply preset values with intelligent variation
            warmth = p_vals.get("warmth", 0.0) + (torch.randn(1).item() * v_factor)
            vibrancy = p_vals.get("vibrancy", 0.0) + (torch.randn(1).item() * v_factor)
            contrast = p_vals.get("contrast", 0.0) + (torch.randn(1).item() * v_factor)
            brightness = p_vals.get("brightness", 0.0) + (
                torch.randn(1).item() * v_factor
            )

        # --- ADVANCED COLOR PROCESSING (Preset and Manual modes) ---
        if mode != "Auto":
            # === INTELLIGENT WHITE BALANCE (if enabled) ===
            if white_balance_strength > 0.0 and ai_analysis and ADVANCED_LIBS_AVAILABLE:
                wb_corrected = intelligent_white_balance(
                    image_np, white_balance_strength
                )
                processed_image = (
                    torch.from_numpy(wb_corrected.astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .to(processed_image.device)
                )

            # === FACE-AWARE PROCESSING ===
            if (
                analysis
                and analysis["faces"]
                and adjust_for_skin_tone
                and ADVANCED_LIBS_AVAILABLE
            ):
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

            if warmth != 0.0:
                h = (h + warmth * 0.1) % 1.0

            if vibrancy != 0.0:
                saturation_mask = 1.0 - s
                s = s * (1.0 + vibrancy) + (vibrancy * 0.3 * saturation_mask * s)

            if brightness != 0.0:
                v = v + brightness * (1.0 - v * 0.5)

            if contrast != 0.0:
                v = 0.5 + (v - 0.5) * (1.0 + contrast)

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
                    shadows_mask = 1.0 - torch.clamp(
                        v * 2.5, 0.0, 1.0
                    )
                    midtones_mask = 1.0 - torch.abs(v - 0.5) * 1.8
                    highlights_mask = torch.clamp(
                        (v - 0.6) * 2.5, 0.0, 1.0
                    )
                else:
                    shadows_mask = 1.0 - torch.clamp(v * 3.0, 0.0, 1.0)
                    midtones_mask = 1.0 - torch.abs(v - 0.5) * 2.0
                    highlights_mask = torch.clamp((v - 0.66) * 3.0, 0.0, 1.0)

                if lift != 0.0:
                    if analysis and analysis["scene_type"] in [
                        "concept_art",
                        "anime",
                        "stylized_art",
                    ]:
                        lift_strength = 0.5
                    else:
                        lift_strength = 0.4
                    v = v + (lift * lift_strength * shadows_mask)

                if gamma != 0.0:
                    if analysis and analysis["scene_type"] in [
                        "concept_art",
                        "detailed_illustration",
                    ]:
                        gamma_exp = 1.0 / (1.0 + gamma * 1.0)
                    else:
                        gamma_exp = 1.0 / (1.0 + gamma * 0.8)
                    v_gamma = torch.pow(torch.clamp(v, 0.001, 1.0), gamma_exp)
                    v = torch.lerp(v, v_gamma, midtones_mask)

                if gain != 0.0:
                    if analysis and analysis["scene_type"] in [
                        "concept_art",
                        "anime",
                        "stylized_art",
                    ]:
                        gain_strength = 0.5
                    else:
                        gain_strength = 0.4
                    v = v + (gain * gain_strength * highlights_mask)

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
                v = torch.lerp(
                    v, v_orig, skin_mask * 0.3
                )

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
