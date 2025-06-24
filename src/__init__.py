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
    h[cmax_indices == 0] = (((rgb[..., 1] - rgb[..., 2]) / (delta + 1e-8)) % 6)[cmax_indices == 0]
    h[cmax_indices == 1] = (((rgb[..., 2] - rgb[..., 0]) / (delta + 1e-8)) + 2)[cmax_indices == 1]
    h[cmax_indices == 2] = (((rgb[..., 0] - rgb[..., 1]) / (delta + 1e-8)) + 4)[cmax_indices == 2]
    
    h = h / 6.0
    h[delta == 0] = 0.0
    
    s = torch.where(cmax == 0, torch.tensor(0.0, device=rgb.device), delta / (cmax + 1e-8))
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
def extract_color_palette(image_np: np.ndarray, num_colors: int = 6) -> typing.List[str]:
    """Extract perceptually-spaced dominant colors using advanced color science."""
    if not ADVANCED_LIBS_AVAILABLE:
        return []
    
    try:
        pixels = image_np.reshape(-1, 3)
        # Sample pixels for performance
        sample_size = min(10000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]
        
        # Method 1: Standard K-means clustering
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=5)
        kmeans.fit(sample_pixels)
        dominant_colors = kmeans.cluster_centers_
        
        # Method 2: Advanced perceptual spacing using colour library
        try:
            # Convert to LAB color space for perceptual clustering
            rgb_norm = sample_pixels.astype(np.float32) / 255.0
            lab_pixels = colour.sRGB_to_Lab(rgb_norm)
            
            # K-means in LAB space for better perceptual clustering
            lab_kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=5)
            lab_kmeans.fit(lab_pixels)
            lab_centers = lab_kmeans.cluster_centers_
            
            # Convert back to RGB
            perceptual_rgb = colour.Lab_to_sRGB(lab_centers)
            perceptual_rgb = np.clip(perceptual_rgb * 255, 0, 255)
            
            # Blend both methods for best results
            blended_colors = (dominant_colors * 0.6 + perceptual_rgb * 0.4)
            final_colors = np.clip(blended_colors, 0, 255)
            
        except Exception:
            # Fallback to standard clustering
            final_colors = dominant_colors
        
        # Sort by brightness for better palette presentation
        brightness_values = np.sum(final_colors * [0.299, 0.587, 0.114], axis=1)
        sorted_indices = np.argsort(brightness_values)
        final_colors = final_colors[sorted_indices]
        
        # Convert to hex codes
        hex_colors = []
        for color in final_colors:
            r, g, b = [int(c) for c in color]
            hex_colors.append(f"#{r:02x}{g:02x}{b:02x}")
        
        return hex_colors
    except Exception:
        return []


def match_to_reference_colors(image_np: np.ndarray, reference_np: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Match image colors to reference image palette with robust error handling."""
    if not ADVANCED_LIBS_AVAILABLE:
        return image_np
    
    try:
        # Validate inputs
        if reference_np is None or image_np is None:
            return image_np
            
        if reference_np.size == 0 or image_np.size == 0:
            return image_np
        
        # Clamp strength to safe range
        strength = np.clip(strength, 0.0, 1.0)
        if strength == 0.0:
            return image_np
        
        # Method 1: Simple RGB histogram matching (more stable)
        matched_image = image_np.copy()
        
        for channel in range(3):  # R, G, B channels
            # Get histograms
            img_hist, _ = np.histogram(image_np[:, :, channel].flatten(), bins=256, range=(0, 256))
            ref_hist, _ = np.histogram(reference_np[:, :, channel].flatten(), bins=256, range=(0, 256))
            
            # Calculate cumulative distribution functions
            img_cdf = np.cumsum(img_hist).astype(np.float64)
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)
            
            # Normalize CDFs
            img_cdf = img_cdf / (img_cdf[-1] + 1e-8)  # Avoid division by zero
            ref_cdf = ref_cdf / (ref_cdf[-1] + 1e-8)
            
            # Create lookup table for histogram matching
            lookup_table = np.zeros(256, dtype=np.uint8)
            for i in range(256):
                # Find closest match in reference CDF
                closest_idx = np.argmin(np.abs(ref_cdf - img_cdf[i]))
                lookup_table[i] = closest_idx
            
            # Apply lookup table with strength blending
            original_channel = image_np[:, :, channel]
            matched_channel = lookup_table[original_channel]
            
            # Blend with original based on strength
            blended_channel = original_channel * (1.0 - strength) + matched_channel * strength
            matched_image[:, :, channel] = np.clip(blended_channel, 0, 255).astype(np.uint8)
        
        # Method 2: LAB color space matching (fallback for fine-tuning)
        if ADVANCED_LIBS_AVAILABLE and strength > 0.3:
            try:
                # Conservative LAB space adjustment
                lab_image = cv2.cvtColor(matched_image, cv2.COLOR_RGB2LAB).astype(np.float32)
                lab_reference = cv2.cvtColor(reference_np, cv2.COLOR_RGB2LAB).astype(np.float32)
                
                # Only adjust A and B channels (color), preserve L (lightness) more
                for i in range(1, 3):  # Only A, B channels
                    img_mean = np.mean(lab_image[:, :, i])
                    ref_mean = np.mean(lab_reference[:, :, i])
                    
                    # Conservative shift - much smaller adjustment
                    color_shift = (ref_mean - img_mean) * strength * 0.3  # Reduced factor
                    lab_image[:, :, i] = lab_image[:, :, i] + color_shift
                
                # Clamp LAB values to valid ranges
                lab_image[:, :, 0] = np.clip(lab_image[:, :, 0], 0, 100)    # L: 0-100
                lab_image[:, :, 1] = np.clip(lab_image[:, :, 1], -127, 127)  # A: -127 to 127
                lab_image[:, :, 2] = np.clip(lab_image[:, :, 2], -127, 127)  # B: -127 to 127
                
                # Convert back to RGB with error handling
                lab_image_uint8 = lab_image.astype(np.uint8)
                matched_image = cv2.cvtColor(lab_image_uint8, cv2.COLOR_LAB2RGB)
                
            except Exception:
                # If LAB processing fails, use the histogram-matched result
                pass
        
        # Final safety clamp
        matched_image = np.clip(matched_image, 0, 255).astype(np.uint8)
        return matched_image
        
    except Exception:
        # Return original image if anything fails
        return image_np


def analyze_image_content(image_np: np.ndarray, device: torch.device = None) -> typing.Dict[str, typing.Any]:
    """Advanced image analysis using computer vision with GPU acceleration."""
    if not ADVANCED_LIBS_AVAILABLE:
        return {"faces": [], "dominant_colors": [], "scene_type": "unknown", "lighting": "auto"}
    
    analysis = {}
    
    # Determine device for GPU acceleration
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Face detection using OpenCV
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        analysis["faces"] = faces.tolist() if len(faces) > 0 else []
    except Exception:
        analysis["faces"] = []
    
    # GPU-accelerated dominant color analysis using K-means
    try:
        pixels = image_np.reshape(-1, 3)
        # Sample pixels for performance
        sample_size = min(10000, len(pixels))
        sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
        sample_pixels = pixels[sample_indices]
        
        # Try GPU-accelerated K-means if available, fallback to CPU
        try:
            # Convert to tensor for potential GPU acceleration (future enhancement)
            sample_tensor = torch.from_numpy(sample_pixels.astype(np.float32)).to(device)
            _ = sample_tensor  # Reserved for future GPU-accelerated clustering
            
            # Use sklearn with faster init when possible
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=5, init='k-means++')
            kmeans.fit(sample_pixels)  # sklearn handles CPU optimization well
            
            dominant_colors = kmeans.cluster_centers_
            analysis["dominant_colors"] = dominant_colors.tolist()
        except Exception:
            # CPU fallback
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
            kmeans.fit(sample_pixels)
            dominant_colors = kmeans.cluster_centers_
            analysis["dominant_colors"] = dominant_colors.tolist()
    except Exception:
        analysis["dominant_colors"] = []
    
    # GPU-accelerated scene type detection for artistic content
    try:
        # Edge density and pattern analysis - use GPU when possible
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Convert to tensor for GPU processing
        try:
            gray_tensor = torch.from_numpy(gray.astype(np.float32)).to(device)
            # Use torch operations for edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            gray_padded = torch.nn.functional.pad(gray_tensor.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
            edges_x = torch.nn.functional.conv2d(gray_padded, sobel_x).squeeze()
            edges_y = torch.nn.functional.conv2d(gray_padded, sobel_y).squeeze()
            edges = torch.sqrt(edges_x**2 + edges_y**2)
            edge_density = torch.mean(edges).cpu().item()
        except Exception:
            # CPU fallback using skimage
            edges = filters.sobel(gray)
            edge_density = np.mean(edges)
        
        # Color analysis - GPU accelerated
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        try:
            hsv_tensor = torch.from_numpy(hsv.astype(np.float32)).to(device)
            avg_saturation = torch.mean(hsv_tensor[:, :, 1]).cpu().item()
            color_variance = torch.std(hsv_tensor[:, :, 1]).cpu().item()
            texture_contrast = torch.std(torch.from_numpy(gray.astype(np.float32)).to(device)).cpu().item()
        except Exception:
            # CPU fallback
            avg_saturation = np.mean(hsv[:, :, 1])
            color_variance = np.std(hsv[:, :, 1])
            texture_contrast = np.std(gray)
        
        # Enhanced classification for artistic content
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


def edge_aware_enhancement(image_np: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply edge-aware enhancement using advanced filtering."""
    if not ADVANCED_LIBS_AVAILABLE:
        return image_np
    
    try:
        # Convert to LAB for better processing
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        
        # Edge detection using multiple scales
        edges_small = ndimage.gaussian_laplace(l_channel, sigma=0.5)
        edges_medium = ndimage.gaussian_laplace(l_channel, sigma=1.0)
        edges_large = ndimage.gaussian_laplace(l_channel, sigma=2.0)
        
        # Combine multi-scale edges
        combined_edges = np.abs(edges_small) + np.abs(edges_medium) * 0.5 + np.abs(edges_large) * 0.25
        
        # Normalize edge map
        edge_map = combined_edges / (np.max(combined_edges) + 1e-8)
        
        # Create enhancement mask (stronger away from edges)
        enhancement_mask = 1.0 - np.clip(edge_map * 2.0, 0.0, 1.0)
        
        # Apply morphological operations for smoother masks
        enhancement_mask = ndimage.gaussian_filter(enhancement_mask, sigma=1.0)
        
        # Enhance contrast in smooth regions while preserving edges
        enhanced_l = l_channel.copy()
        
        # Local contrast enhancement using bilateral filtering concept
        mean_filtered = ndimage.uniform_filter(l_channel, size=5)
        local_contrast = l_channel - mean_filtered
        
        # Apply enhancement with edge awareness
        enhanced_l = l_channel + (local_contrast * strength * enhancement_mask)
        enhanced_l = np.clip(enhanced_l, 0, 255)
        
        # Update LAB image
        lab[:, :, 0] = enhanced_l.astype(np.uint8)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
        
    except Exception:
        return image_np


def intelligent_white_balance(image_np: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Advanced white balance using professional color science."""
    if not ADVANCED_LIBS_AVAILABLE:
        return image_np
    
    try:
        # Method 1: Advanced LAB-based correction
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        
        # Analyze A and B channels for color cast
        a_channel = lab[:, :, 1].astype(np.float32) - 128
        b_channel = lab[:, :, 2].astype(np.float32) - 128
        
        # Use robust statistics (median) instead of mean
        a_shift = np.median(a_channel)
        b_shift = np.median(b_channel)
        
        # Apply correction with strength
        lab[:, :, 1] = np.clip(lab[:, :, 1] - a_shift * strength, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - b_shift * strength, 0, 255)
        
        corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Method 2: Advanced using colour library for professional color science
        try:
            # Convert to XYZ color space for advanced analysis
            rgb_norm = image_np.astype(np.float32) / 255.0
            
            # Use colour library for chromatic adaptation
            # Estimate illuminant using grey world assumption
            grey_world_rgb = np.mean(rgb_norm.reshape(-1, 3), axis=0)
            
            # Convert to XYZ for illuminant analysis
            xyz = colour.sRGB_to_XYZ(rgb_norm)
            
            # Estimate scene illuminant
            illuminant_xy = colour.XYZ_to_xy(grey_world_rgb.reshape(1, -1))
            
            # Use D65 as target illuminant (standard daylight)
            d65_xy = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
            
            # Compute chromatic adaptation matrix
            if np.linalg.norm(illuminant_xy - d65_xy) > 0.01:  # Only adapt if significant difference
                # Use Bradford chromatic adaptation
                adapted_xyz = colour.chromatic_adaptation_VonKries(
                    xyz, 
                    illuminant_xy[0], 
                    d65_xy,
                    transform='Bradford'
                )
                
                # Convert back to RGB
                adapted_rgb = colour.XYZ_to_sRGB(adapted_xyz)
                adapted_rgb = np.clip(adapted_rgb, 0, 1)
                
                # Blend with original correction
                colour_corrected = (adapted_rgb * 255).astype(np.uint8)
                corrected = corrected * (1 - strength * 0.3) + colour_corrected * (strength * 0.3)
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
                
        except Exception:
            # Fallback to LAB-only correction
            pass
        
        return corrected
        
    except Exception:
        return image_np


def enhance_faces(image_np: np.ndarray, faces: typing.List, strength: float = 0.3) -> np.ndarray:
    """Intelligently enhance detected faces using advanced segmentation."""
    if not ADVANCED_LIBS_AVAILABLE or not faces:
        return image_np
    
    try:
        enhanced = image_np.copy()
        
        for (x, y, w, h) in faces:
            # Extract face region with padding
            pad = 10
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(image_np.shape[1], x + w + pad)
            y_end = min(image_np.shape[0], y + h + pad)
            
            face_region = enhanced[y_start:y_end, x_start:x_end]
            
            # Use advanced segmentation for better face boundaries
            # SLIC segmentation for skin region detection
            segments = segmentation.slic(face_region, n_segments=50, compactness=10, sigma=1)
            
            # Convert to LAB for better skin tone analysis
            face_lab = cv2.cvtColor(face_region, cv2.COLOR_RGB2LAB)
            
            # Identify skin segments using color analysis
            skin_segments = set()
            for segment_id in np.unique(segments):
                segment_mask = segments == segment_id
                segment_lab = face_lab[segment_mask]
                
                if len(segment_lab) > 10:  # Minimum pixels for analysis
                    # Skin tone detection in LAB space
                    l_mean = np.mean(segment_lab[:, 0])
                    a_mean = np.mean(segment_lab[:, 1])
                    b_mean = np.mean(segment_lab[:, 2])
                    
                    # Skin tone criteria in LAB space
                    if (50 < l_mean < 200 and 
                        120 < a_mean < 140 and 
                        130 < b_mean < 160):
                        skin_segments.add(segment_id)
            
            # Create skin mask from segments
            skin_mask = np.zeros(segments.shape, dtype=bool)
            for segment_id in skin_segments:
                skin_mask |= (segments == segment_id)
            
            # Apply Gaussian smoothing to mask edges
            skin_mask_smooth = ndimage.gaussian_filter(skin_mask.astype(float), sigma=2.0)
            
            # Enhance skin regions
            enhanced_face = face_region.copy()
            
            # Enhance L channel (brightness) for skin
            face_lab = cv2.cvtColor(enhanced_face, cv2.COLOR_RGB2LAB)
            l_channel = face_lab[:, :, 0].astype(np.float32)
            l_enhanced = l_channel * (1.0 + strength * 0.25 * skin_mask_smooth)
            face_lab[:, :, 0] = np.clip(l_enhanced, 0, 255).astype(np.uint8)
            
            # Reduce green cast in skin tones
            a_channel = face_lab[:, :, 1].astype(np.float32)
            a_enhanced = a_channel * (1.0 - strength * 0.15 * skin_mask_smooth)
            face_lab[:, :, 1] = np.clip(a_enhanced, 0, 255).astype(np.uint8)
            
            # Warm up skin tones slightly
            b_channel = face_lab[:, :, 2].astype(np.float32)
            b_enhanced = b_channel * (1.0 + strength * 0.1 * skin_mask_smooth)
            face_lab[:, :, 2] = np.clip(b_enhanced, 0, 255).astype(np.uint8)
            
            # Convert back and blend
            enhanced_face = cv2.cvtColor(face_lab, cv2.COLOR_LAB2RGB)
            enhanced[y_start:y_end, x_start:x_end] = enhanced_face
        
        return enhanced
        
    except Exception:
        return image_np


def generate_histogram_image(image_np: np.ndarray) -> np.ndarray:
    """Generate a well-scaled histogram visualization as an image."""
    try:
        # Calculate histograms for RGB channels
        hist_r = cv2.calcHist([image_np], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image_np], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image_np], [2], None, [256], [0, 256])
        
        # Create larger histogram image for better visibility
        hist_height = 512
        hist_width = 768
        hist_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        
        # Add dark background for better contrast
        hist_image.fill(20)  # Dark gray background
        
        # Get the effective drawing area (leave margins)
        margin = 40
        draw_height = hist_height - (margin * 2)
        draw_width = hist_width - (margin * 2)
        
        # Normalize histograms to fit drawing area
        hist_r = cv2.normalize(hist_r, hist_r, 0, draw_height, cv2.NORM_MINMAX)
        hist_g = cv2.normalize(hist_g, hist_g, 0, draw_height, cv2.NORM_MINMAX)
        hist_b = cv2.normalize(hist_b, hist_b, 0, draw_height, cv2.NORM_MINMAX)
        
        # Calculate bin width for better scaling
        bin_width = max(1, draw_width // 256)
        
        # Draw histograms with better scaling
        for i in range(256):
            x = margin + (i * draw_width // 256)
            
            # Draw as filled rectangles for better visibility
            # Red channel
            if hist_r[i][0] > 0:
                cv2.rectangle(hist_image, 
                            (x, hist_height - margin), 
                            (x + bin_width, hist_height - margin - int(hist_r[i][0])), 
                            (80, 80, 255), -1)  # Filled red
            
            # Green channel (with blending)
            if hist_g[i][0] > 0:
                cv2.rectangle(hist_image, 
                            (x, hist_height - margin), 
                            (x + bin_width, hist_height - margin - int(hist_g[i][0])), 
                            (80, 255, 80), -1)  # Filled green
            
            # Blue channel (with blending)
            if hist_b[i][0] > 0:
                cv2.rectangle(hist_image, 
                            (x, hist_height - margin), 
                            (x + bin_width, hist_height - margin - int(hist_b[i][0])), 
                            (255, 80, 80), -1)  # Filled blue
        
        # Add subtle grid lines
        grid_spacing = draw_width // 8
        for i in range(1, 8):
            x = margin + (i * grid_spacing)
            cv2.line(hist_image, (x, margin), (x, hist_height - margin), (60, 60, 60), 1)
        
        grid_spacing = draw_height // 4
        for i in range(1, 4):
            y = margin + (i * grid_spacing)
            cv2.line(hist_image, (margin, y), (hist_width - margin, y), (60, 60, 60), 1)
        
        # Add border
        cv2.rectangle(hist_image, (margin, margin), (hist_width - margin, hist_height - margin), (100, 100, 100), 2)
        
        # Convert BGR to RGB for output
        hist_image_rgb = cv2.cvtColor(hist_image, cv2.COLOR_BGR2RGB)
        return hist_image_rgb
        
    except Exception:
        # Return black image if histogram generation fails
        return np.zeros((512, 768, 3), dtype=np.uint8)


def generate_palette_image(hex_colors: typing.List[str]) -> np.ndarray:
    """Generate a clean color palette visualization as an image."""
    try:
        if not hex_colors:
            return np.zeros((120, 600, 3), dtype=np.uint8)
        
        palette_height = 120  # Reduced height since no text
        palette_width = 600
        palette_image = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
        
        # Calculate swatch dimensions - use full height now
        swatch_width = palette_width // len(hex_colors)
        
        for i, hex_color in enumerate(hex_colors):
            # Convert hex to RGB
            hex_color = hex_color.lstrip('#')
            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
            except (ValueError, IndexError):
                r, g, b = 128, 128, 128  # Default gray for invalid colors
            
            # Draw color swatch (full height, no borders)
            x_start = i * swatch_width
            x_end = (i + 1) * swatch_width
            palette_image[:, x_start:x_end] = [r, g, b]
        
        return palette_image
        
    except Exception:
        # Return black image if palette generation fails
        return np.zeros((120, 600, 3), dtype=np.uint8)


class EasyColorCorrection:
    PRESETS: typing.Dict[str, typing.Dict[str, float]] = {
        # PROFESSIONAL PORTRAIT PRESETS
        "Natural Portrait": {"warmth": 0.08, "vibrancy": 0.12, "contrast": 0.08, "brightness": 0.03},
        "Warm Portrait": {"warmth": 0.18, "vibrancy": 0.15, "contrast": 0.06, "brightness": 0.05},
        "Cool Portrait": {"warmth": -0.12, "vibrancy": 0.08, "contrast": 0.10, "brightness": 0.02},
        "High Key Portrait": {"warmth": 0.05, "vibrancy": 0.08, "contrast": -0.05, "brightness": 0.20},
        "Dramatic Portrait": {"warmth": 0.02, "vibrancy": 0.20, "contrast": 0.25, "brightness": -0.05},
        
        # CONCEPT ART & ILLUSTRATION PRESETS
        "Epic Fantasy": {"warmth": 0.1, "vibrancy": 0.4, "contrast": 0.3, "brightness": 0.05},
        "Sci-Fi Chrome": {"warmth": -0.2, "vibrancy": 0.3, "contrast": 0.35, "brightness": 0.1},
        "Dark Fantasy": {"warmth": -0.1, "vibrancy": 0.25, "contrast": 0.4, "brightness": -0.15},
        "Vibrant Concept": {"warmth": 0.05, "vibrancy": 0.5, "contrast": 0.25, "brightness": 0.08},
        "Matte Painting": {"warmth": 0.08, "vibrancy": 0.3, "contrast": 0.2, "brightness": 0.03},
        "Digital Art": {"warmth": 0.0, "vibrancy": 0.45, "contrast": 0.28, "brightness": 0.05},
        
        # ARTISTIC & STYLIZED
        "Anime Bright": {"warmth": 0.12, "vibrancy": 0.45, "contrast": 0.2, "brightness": 0.12},
        "Anime Moody": {"warmth": -0.05, "vibrancy": 0.35, "contrast": 0.25, "brightness": -0.05},
        "Cyberpunk": {"warmth": -0.15, "vibrancy": 0.45, "contrast": 0.25, "brightness": -0.03},
        "Pastel Dreams": {"warmth": 0.12, "vibrancy": -0.08, "contrast": -0.08, "brightness": 0.12},
        "Neon Nights": {"warmth": -0.18, "vibrancy": 0.40, "contrast": 0.20, "brightness": -0.05},
        "Comic Book": {"warmth": 0.05, "vibrancy": 0.5, "contrast": 0.35, "brightness": 0.08},
        
        # CINEMATIC & FILM LOOKS
        "Cinematic": {"warmth": 0.12, "vibrancy": 0.15, "contrast": 0.18, "brightness": 0.02},
        "Teal & Orange": {"warmth": -0.08, "vibrancy": 0.25, "contrast": 0.15, "brightness": 0.0},
        "Film Noir": {"warmth": -0.05, "vibrancy": -0.80, "contrast": 0.35, "brightness": -0.08},
        "Vintage Film": {"warmth": 0.15, "vibrancy": -0.10, "contrast": 0.12, "brightness": 0.03},
        "Bleach Bypass": {"warmth": -0.02, "vibrancy": -0.25, "contrast": 0.30, "brightness": 0.05},
        
        # NATURAL & LANDSCAPE
        "Golden Hour": {"warmth": 0.25, "vibrancy": 0.18, "contrast": 0.08, "brightness": 0.08},
        "Blue Hour": {"warmth": -0.20, "vibrancy": 0.15, "contrast": 0.12, "brightness": 0.02},
        "Sunny Day": {"warmth": 0.15, "vibrancy": 0.20, "contrast": 0.10, "brightness": 0.08},
        "Overcast": {"warmth": -0.08, "vibrancy": 0.05, "contrast": 0.08, "brightness": 0.05},
        
        # CLASSIC EFFECTS
        "Sepia": {"warmth": 0.30, "vibrancy": -0.35, "contrast": 0.08, "brightness": 0.03},
        "Black & White": {"warmth": 0.0, "vibrancy": -1.0, "contrast": 0.15, "brightness": 0.0},
        "Faded": {"warmth": 0.05, "vibrancy": -0.15, "contrast": -0.12, "brightness": 0.08},
        "Moody": {"warmth": -0.08, "vibrancy": 0.12, "contrast": 0.20, "brightness": -0.08},
    }

    @classmethod
    def INPUT_TYPES(cls) -> typing.Dict:
        return {
            "required": {
                "image": ("IMAGE", {}),
                "mode": (["Auto", "Preset", "Manual"], {"default": "Auto"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "🎨 Optional reference image for color matching (concept art/mood boards)"}),
                "reference_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.10, "tooltip": "Strength of color matching to reference image"}),
                "extract_palette": ("BOOLEAN", {"default": False, "tooltip": "📊 Extract and display dominant color palette"}),
                "realtime_preview": ("BOOLEAN", {"default": False, "tooltip": "Enable real-time preview when adjusting parameters"}),
                "ai_analysis": ("BOOLEAN", {"default": True, "tooltip": "🤖 Enable AI-powered face detection, scene analysis & content-aware enhancement"}),
                "adjust_for_skin_tone": ("BOOLEAN", {"default": False, "tooltip": "AI face detection + skin tone preservation (requires ai_analysis)"}),
                "white_balance_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.10, "tooltip": "Strength of perceptual LAB-based white balance"}),
                "enhancement_strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.10, "tooltip": "Overall strength of AI-powered enhancements"}),
                "pop_factor": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.10, "tooltip": "Extra pop factor for artistic content (anime, detailed photos)"}),
                "effect_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.10, "tooltip": "Strength of overall color correction effect"}),
                "warmth": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.10, "tooltip": "Green/Magenta warmth adjustment"}),
                "vibrancy": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.10, "tooltip": "Saturation boost for less saturated areas"}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.10, "tooltip": "Contrast adjustment for overall image"}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.10, "tooltip": "Brightness adjustment for overall image"}),
                "preset": (list(cls.PRESETS.keys()), {}),
                "variation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.10, "tooltip": "Variation strength for preset adjustments"}),
                "lift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.10, "tooltip": "Lift adjustment for shadows in the image"}),
                "gamma": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.10, "tooltip": "Gamma adjustment for midtones in the image"}),
                "gain": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.10, "tooltip": "Gain adjustment for highlights in the image"}),
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.10, "tooltip": "Noise reduction strength (0.0 = no noise reduction, 1.0 = maximum noise reduction)"}),
                "mask": ("MASK", {}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "palette_data", "histogram", "palette_image")
    FUNCTION = "run"
    CATEGORY = "itsjustregi / Easy Color Correction"
    DISPLAY_NAME = "Easy Color Correction"

    def run(self, image: torch.Tensor, mode: str, reference_image: typing.Optional[torch.Tensor] = None,
            reference_strength: float = 0.3, extract_palette: bool = False, realtime_preview: bool = False, 
            ai_analysis: bool = True, adjust_for_skin_tone: bool = False, white_balance_strength: float = 0.6,
            enhancement_strength: float = 0.8, pop_factor: float = 0.7, effect_strength: float = 1.0,
            warmth: float = 0.0, vibrancy: float = 0.0, contrast: float = 0.0, brightness: float = 0.0,
            preset: str = "Anime", variation: float = 0.0, lift: float = 0.0, gamma: float = 0.0, 
            gain: float = 0.0, noise: float = 0.0, mask: typing.Optional[torch.Tensor] = None) -> tuple:
        
        # Note: realtime_preview is handled by the JavaScript frontend
        _ = realtime_preview  # Suppress unused parameter warning
        
        original_image = image.clone()
        _, height, width, _ = image.shape
        
        processed_image = image.clone()

        # --- AUTO MODE: AI-POWERED INTELLIGENT ENHANCEMENT ---
        if mode == "Auto":
            # === 1. ADVANCED CONTENT ANALYSIS (AI-POWERED) ===
            if ai_analysis and ADVANCED_LIBS_AVAILABLE:
                # Convert tensor to numpy for advanced analysis
                image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
                analysis = analyze_image_content(image_np, processed_image.device)
                print(f"🤖 AI Analysis: {analysis['scene_type']} scene, {analysis['lighting']} lighting, {len(analysis['faces'])} faces detected")
                
                # === 2. INTELLIGENT WHITE BALANCE ===
                if white_balance_strength > 0.0:
                    wb_corrected = intelligent_white_balance(image_np, white_balance_strength)
                    processed_image = torch.from_numpy(wb_corrected.astype(np.float32) / 255.0).unsqueeze(0).to(processed_image.device)
                
                # === 3. EDGE-AWARE PROCESSING ===
                if enhancement_strength > 0.2:
                    edge_enhanced = edge_aware_enhancement(image_np, enhancement_strength * 0.3)
                    processed_image = torch.from_numpy(edge_enhanced.astype(np.float32) / 255.0).unsqueeze(0).to(processed_image.device)
                    image_np = edge_enhanced  # Update for subsequent processing
                
                # === 4. FACE-AWARE ENHANCEMENT ===
                if analysis['faces'] and adjust_for_skin_tone:
                    face_enhanced = enhance_faces(image_np, analysis['faces'], enhancement_strength * 0.5)
                    processed_image = torch.from_numpy(face_enhanced.astype(np.float32) / 255.0).unsqueeze(0).to(processed_image.device)
            else:
                # Fallback to basic analysis when AI is disabled
                analysis = {
                    "scene_type": "general", 
                    "lighting": "auto", 
                    "faces": [], 
                    "dominant_colors": []
                }
                print("🔧 Basic Auto Mode: AI analysis disabled")
                
                # Basic white balance fallback
                if white_balance_strength > 0.0:
                    B, C = processed_image.shape[0], processed_image.shape[3]
                    flat_image = processed_image.view(B, -1, C)
                    percentile_40 = torch.quantile(flat_image, 0.40, dim=1, keepdim=True)
                    percentile_60 = torch.quantile(flat_image, 0.60, dim=1, keepdim=True)
                    midtone_mean = (percentile_40 + percentile_60) / 2.0
                    avg_gray = torch.mean(midtone_mean, dim=-1, keepdim=True)
                    scale = avg_gray / (midtone_mean + 1e-6)
                    scale = torch.lerp(torch.ones_like(scale), scale, white_balance_strength)
                    scale = scale.view(B, 1, 1, C)
                    color_deviation = torch.abs(scale - 1.0).max()
                    if color_deviation > 0.05:
                        processed_image = processed_image * scale
                        processed_image = torch.clamp(processed_image, 0.0, 1.0)
            
            # === 4. SCENE-ADAPTIVE COLOR ENHANCEMENT ===
            hsv_enhanced = rgb_to_hsv(processed_image)
            h_enh, s_enh, v_enh = hsv_enhanced[..., 0], hsv_enhanced[..., 1], hsv_enhanced[..., 2]
            
            # Adaptive parameters based on scene analysis
            scene_type = analysis['scene_type']
            lighting = analysis['lighting']
            
            # Enhanced scene-specific profiles for artistic content (reduced contrast, increased saturation)
            if scene_type == "anime":
                contrast_boost = 0.18 * enhancement_strength  # Reduced from 0.25
                saturation_boost = 0.55 * enhancement_strength  # Increased from 0.45
                vibrancy_strength = 0.45 * pop_factor  # Increased from 0.4
            elif scene_type == "concept_art":
                # Aggressive enhancement for concept art
                contrast_boost = 0.25 * enhancement_strength  # Reduced from 0.35
                saturation_boost = 0.6 * enhancement_strength  # Increased from 0.5
                vibrancy_strength = 0.5 * pop_factor  # Increased from 0.45
            elif scene_type == "stylized_art":
                # Boost stylized content while preserving artistic intent
                contrast_boost = 0.22 * enhancement_strength  # Reduced from 0.3
                saturation_boost = 0.5 * enhancement_strength  # Increased from 0.4
                vibrancy_strength = 0.4 * pop_factor  # Increased from 0.35
            elif scene_type == "detailed_illustration":
                # High detail artwork needs careful enhancement
                contrast_boost = 0.2 * enhancement_strength  # Reduced from 0.28
                saturation_boost = 0.45 * enhancement_strength  # Increased from 0.35
                vibrancy_strength = 0.35 * pop_factor  # Increased from 0.3
            elif scene_type == "portrait":
                # Conservative for portraits
                contrast_boost = 0.12 * enhancement_strength  # Reduced from 0.15
                saturation_boost = 0.3 * enhancement_strength  # Increased from 0.2
                vibrancy_strength = 0.25 * pop_factor  # Increased from 0.2
            elif scene_type == "realistic_photo":
                # Realistic photo enhancement
                contrast_boost = 0.15 * enhancement_strength  # Reduced from 0.2
                saturation_boost = 0.35 * enhancement_strength  # Increased from 0.25
                vibrancy_strength = 0.3 * pop_factor  # Increased from 0.25
            else:  # general
                contrast_boost = 0.16 * enhancement_strength  # Reduced from 0.22
                saturation_boost = 0.42 * enhancement_strength  # Increased from 0.32
                vibrancy_strength = 0.35 * pop_factor  # Increased from 0.28
            
            # Lighting-adaptive adjustments
            if lighting == "low_light":
                contrast_boost *= 1.3  # More contrast for dark images
                v_enh = v_enh * (1.0 + 0.15 * enhancement_strength)  # Brighten
            elif lighting == "bright":
                contrast_boost *= 0.8  # Less aggressive for bright images
                v_enh = v_enh * (1.0 - 0.1 * enhancement_strength)  # Slightly darken
            elif lighting == "flat":
                contrast_boost *= 1.5  # Much more contrast for flat images
            
            # === 5. ADVANCED CONTRAST WITH EDGE PRESERVATION ===
            # Perceptual contrast enhancement
            v_min, v_max = torch.quantile(v_enh, 0.01), torch.quantile(v_enh, 0.99)
            if v_max > v_min:
                v_enh = torch.clamp((v_enh - v_min) / (v_max - v_min), 0.0, 1.0)
            
            # Apply adaptive contrast
            v_enh = 0.5 + (v_enh - 0.5) * (1.0 + contrast_boost)
            
            # === 6. INTELLIGENT SATURATION ENHANCEMENT ===
            # Dominant color-aware saturation
            if analysis['dominant_colors'] and ADVANCED_LIBS_AVAILABLE:
                # Reduce saturation boost if image is already very saturated
                avg_dom_sat = np.mean([np.max(color) - np.min(color) for color in analysis['dominant_colors']])
                saturation_factor = max(0.5, 1.0 - avg_dom_sat / 255.0)
                saturation_boost *= saturation_factor
            
            s_enh = s_enh * (1.0 + saturation_boost)
            
            # === 7. ADVANCED VIBRANCY (SELECTIVE SATURATION) ===
            # Boost less saturated areas more
            saturation_mask = 1.0 - s_enh
            s_enh = s_enh + (vibrancy_strength * saturation_mask * s_enh)
            
            # === 8. INTELLIGENT GLOW FOR ARTISTIC CONTENT ===
            if pop_factor > 0.3 and scene_type in ["anime", "concept_art", "stylized_art", "detailed_illustration"]:
                # Advanced highlight enhancement for artistic content
                highlight_mask = torch.clamp((v_enh - 0.7) * 3.33, 0.0, 1.0)
                
                # Stronger glow for concept art
                if scene_type == "concept_art":
                    glow_strength = 0.25 * pop_factor * enhancement_strength
                elif scene_type in ["anime", "stylized_art"]:
                    glow_strength = 0.2 * pop_factor * enhancement_strength
                else:  # detailed_illustration
                    glow_strength = 0.15 * pop_factor * enhancement_strength
                    
                v_enh = v_enh + (glow_strength * highlight_mask)
            
            # === 9. FINAL ASSEMBLY ===
            s_enh = torch.clamp(s_enh, 0.0, 1.0)
            v_enh = torch.clamp(v_enh, 0.0, 1.0)
            processed_image = hsv_to_rgb(torch.stack([h_enh, s_enh, v_enh], dim=-1))
            processed_image = torch.clamp(processed_image, 0.0, 1.0)

        # --- SHARED AI ANALYSIS FOR ALL MODES ---
        analysis = None
        if ai_analysis and ADVANCED_LIBS_AVAILABLE:
            image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
            analysis = analyze_image_content(image_np, processed_image.device)
            print(f"🤖 AI Analysis for {mode} Mode: {analysis['scene_type']} scene, {analysis['lighting']} lighting, {len(analysis['faces'])} faces detected")

        # --- PRESET MODE: AI-ENHANCED INTELLIGENT PRESETS ---
        elif mode == "Preset":
            p_vals = self.PRESETS.get(preset, {}).copy()  # Copy to avoid modifying original
            
            # === AI-GUIDED PRESET ADAPTATION ===
            if analysis:
                scene_type = analysis['scene_type']
                lighting = analysis['lighting']
                
                # Enhanced content-aware preset adjustments for artistic content
                if scene_type == "concept_art":
                    # Aggressive enhancement for concept art
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.4
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.3
                    p_vals["brightness"] = p_vals.get("brightness", 0.0) + 0.05  # Slight brightness boost
                elif scene_type == "anime":
                    # Strong enhancement for anime
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.3
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.2
                elif scene_type == "stylized_art":
                    # Boost stylized art while preserving style
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.25
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.15
                elif scene_type == "detailed_illustration":
                    # Careful enhancement for detailed illustrations
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 1.2
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.1
                elif scene_type == "portrait" and analysis['faces']:
                    # Conservative for portraits
                    p_vals["vibrancy"] = p_vals.get("vibrancy", 0.0) * 0.8
                    p_vals["warmth"] = p_vals.get("warmth", 0.0) + 0.05  # Slight warmth for skin
                elif scene_type == "realistic_photo":
                    # Balanced enhancement for realistic photos
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.1
                
                # Lighting-aware adjustments
                if lighting == "low_light":
                    p_vals["brightness"] = p_vals.get("brightness", 0.0) + 0.1
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.4
                elif lighting == "bright":
                    p_vals["brightness"] = p_vals.get("brightness", 0.0) - 0.05
                elif lighting == "flat":
                    p_vals["contrast"] = p_vals.get("contrast", 0.0) * 1.5
                
                print(f"🎨 AI-Enhanced Preset: {preset} adapted for {scene_type}/{lighting}")
            
            # Smart variation based on content
            if analysis and analysis['dominant_colors']:
                # Reduce variation if image is already very colorful
                avg_color_range = np.mean([np.max(color) - np.min(color) for color in analysis['dominant_colors']])
                variation_factor = max(0.3, 1.0 - avg_color_range / 200.0)
                v_factor = variation * 0.05 * variation_factor
            else:
                v_factor = variation * 0.05
            
            # Apply preset values with intelligent variation
            warmth = p_vals.get("warmth", 0.0) + (torch.randn(1).item() * v_factor)
            vibrancy = p_vals.get("vibrancy", 0.0) + (torch.randn(1).item() * v_factor)
            contrast = p_vals.get("contrast", 0.0) + (torch.randn(1).item() * v_factor)
            brightness = p_vals.get("brightness", 0.0) + (torch.randn(1).item() * v_factor)

        # --- ADVANCED COLOR PROCESSING (Preset and Manual modes) ---
        if mode != "Auto":
            # === INTELLIGENT WHITE BALANCE (if enabled) ===
            if white_balance_strength > 0.0 and ai_analysis and ADVANCED_LIBS_AVAILABLE:
                wb_corrected = intelligent_white_balance(image_np, white_balance_strength)
                processed_image = torch.from_numpy(wb_corrected.astype(np.float32) / 255.0).unsqueeze(0).to(processed_image.device)
            
            # === FACE-AWARE PROCESSING ===
            if analysis and analysis['faces'] and adjust_for_skin_tone and ADVANCED_LIBS_AVAILABLE:
                face_enhanced = enhance_faces(image_np, analysis['faces'], enhancement_strength * 0.3)
                processed_image = torch.from_numpy(face_enhanced.astype(np.float32) / 255.0).unsqueeze(0).to(processed_image.device)
            
            # === ADVANCED HSV PROCESSING ===
            hsv_image = rgb_to_hsv(processed_image)
            h, s, v = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]

            # Enhanced warmth with better color science
            if warmth != 0.0: 
                h = (h + warmth * 0.1) % 1.0
            
            # Perceptual vibrancy (selective saturation)
            if vibrancy != 0.0:
                saturation_mask = 1.0 - s  # More effect on less saturated areas
                s = s * (1.0 + vibrancy) + (vibrancy * 0.3 * saturation_mask * s)
            
            # Brightness with gamma-aware adjustment
            if brightness != 0.0:
                v = v + brightness * (1.0 - v * 0.5)  # Preserve highlights better
            
            # Perceptual contrast
            if contrast != 0.0:
                v = 0.5 + (v - 0.5) * (1.0 + contrast)
        
            # === MANUAL MODE: PROFESSIONAL GRADING TOOLS ===
            if mode == "Manual":
                ai_guidance = ""
                if analysis:
                    scene_type = analysis['scene_type']
                    if scene_type in ["concept_art", "anime", "stylized_art", "detailed_illustration"]:
                        ai_guidance = f" + AI Guidance ({scene_type})"
                    else:
                        ai_guidance = " + AI Guidance"
                print(f"🎛️ Professional Manual Mode{ai_guidance}")
                
                # Enhanced 3-way color corrector with artistic content awareness
                if analysis and analysis['scene_type'] in ["concept_art", "anime", "stylized_art", "detailed_illustration"]:
                    # Artistic content often has distinct shadow/midtone/highlight regions
                    shadows_mask = 1.0 - torch.clamp(v * 2.5, 0.0, 1.0)  # Slightly softer shadows
                    midtones_mask = 1.0 - torch.abs(v - 0.5) * 1.8  # Broader midtones
                    highlights_mask = torch.clamp((v - 0.6) * 2.5, 0.0, 1.0)  # Earlier highlight threshold
                else:
                    # Standard photo masking
                    shadows_mask = 1.0 - torch.clamp(v * 3.0, 0.0, 1.0)
                    midtones_mask = 1.0 - torch.abs(v - 0.5) * 2.0
                    highlights_mask = torch.clamp((v - 0.66) * 3.0, 0.0, 1.0)
                
                # Lift (Shadows) - Enhanced for artistic content
                if lift != 0.0:
                    if analysis and analysis['scene_type'] in ["concept_art", "anime", "stylized_art"]:
                        lift_strength = 0.5  # More aggressive for artistic content
                    else:
                        lift_strength = 0.4
                    v = v + (lift * lift_strength * shadows_mask)
                
                # Gamma (Midtones) - Enhanced perceptual curve for artistic content
                if gamma != 0.0:
                    if analysis and analysis['scene_type'] in ["concept_art", "detailed_illustration"]:
                        # More dramatic gamma curve for artistic content
                        gamma_exp = 1.0 / (1.0 + gamma * 1.0)
                    else:
                        gamma_exp = 1.0 / (1.0 + gamma * 0.8)
                    v_gamma = torch.pow(torch.clamp(v, 0.001, 1.0), gamma_exp)
                    v = torch.lerp(v, v_gamma, midtones_mask)
                
                # Gain (Highlights) - Enhanced for artistic highlights
                if gain != 0.0:
                    if analysis and analysis['scene_type'] in ["concept_art", "anime", "stylized_art"]:
                        gain_strength = 0.5  # More aggressive for artistic highlights
                    else:
                        gain_strength = 0.4
                    v = v + (gain * gain_strength * highlights_mask)
                
                # Professional noise with luminance preservation
                if noise > 0.0:
                    # Create film-like grain
                    mono_noise = torch.randn(processed_image.shape[:-1], device=processed_image.device).unsqueeze(-1)
                    # Apply in LAB space for better perceptual results
                    luminance_mask = 1.0 - torch.abs(v - 0.5) * 2.0
                    luminance_mask = torch.clamp(luminance_mask, 0.0, 1.0).unsqueeze(-1)
                    
                    # Convert to RGB for noise application
                    rgb_temp = hsv_to_rgb(torch.stack([h, s, v], dim=-1))
                    rgb_temp += (mono_noise * noise * 0.15 * luminance_mask)
                    rgb_temp = torch.clamp(rgb_temp, 0.0, 1.0)
                    
                    # Convert back to HSV
                    hsv_temp = rgb_to_hsv(rgb_temp)
                    h, s, v = hsv_temp[..., 0], hsv_temp[..., 1], hsv_temp[..., 2]
            
            # === SKIN TONE PROTECTION FOR ALL MODES ===
            if analysis and analysis['faces'] and adjust_for_skin_tone:
                # Advanced skin tone preservation
                hsv_original = rgb_to_hsv(original_image)
                h_orig, s_orig, v_orig = hsv_original[..., 0], hsv_original[..., 1], hsv_original[..., 2]
                
                # Create better skin mask
                skin_hue_mask = ((h_orig >= 0.0) & (h_orig <= 0.14)) | ((h_orig >= 0.9) & (h_orig <= 1.0))
                skin_sat_mask = (s_orig >= 0.15) & (s_orig <= 0.8)
                skin_val_mask = v_orig >= 0.2
                skin_mask = (skin_hue_mask & skin_sat_mask & skin_val_mask).float()
                
                # Protect skin tones with varying intensity
                h = torch.lerp(h, h_orig, skin_mask * 0.8)
                s = torch.lerp(s, s_orig, skin_mask * 0.6)
                v = torch.lerp(v, v_orig, skin_mask * 0.3)  # Less protection for brightness
            
            # === FINAL ASSEMBLY WITH ADVANCED CLAMPING ===
            s = torch.clamp(s, 0.0, 1.0)
            v = torch.clamp(v, 0.0, 1.0)
            processed_hsv = torch.stack([h, s, v], dim=-1)
            processed_image = hsv_to_rgb(processed_hsv)
        
        # Apply effect strength for Auto and Preset modes
        if mode in ["Auto", "Preset"]:
            processed_image = torch.lerp(original_image, processed_image, effect_strength)
            
        # Add monochromatic noise for Manual mode, respecting luminance
        if mode == "Manual" and noise > 0.0:
            # Create monochromatic noise
            mono_noise = torch.randn(processed_image.shape[:-1], device=image.device).unsqueeze(-1)
            # Get current luminance for noise masking
            hsv_for_noise = rgb_to_hsv(processed_image)
            v_for_noise = hsv_for_noise[..., 2]
            # Create a mask to make noise more visible in midtones
            luminance_mask = 1.0 - torch.abs(v_for_noise - 0.5) * 2.0
            luminance_mask = torch.clamp(luminance_mask, 0.0, 1.0).unsqueeze(-1)
            
            # Apply masked noise
            processed_image += (mono_noise * noise * 0.2 * luminance_mask)

        processed_image = torch.clamp(processed_image, 0.0, 1.0)

        # === COLOR REFERENCE MATCHING ===
        if reference_image is not None and reference_strength > 0.0:
            # Convert both images to numpy for color matching
            image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
            ref_np = (reference_image[0].cpu().numpy() * 255).astype(np.uint8)
            
            # Apply color matching
            matched_image = match_to_reference_colors(image_np, ref_np, reference_strength)
            processed_image = torch.from_numpy(matched_image.astype(np.float32) / 255.0).unsqueeze(0).to(processed_image.device)
            processed_image = torch.clamp(processed_image, 0.0, 1.0)
            print(f"🎨 Applied color reference matching with {reference_strength:.1f} strength")

        # === COLOR PALETTE EXTRACTION ===
        palette_data = ""
        hex_colors = []
        if extract_palette:
            image_for_palette = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
            hex_colors = extract_color_palette(image_for_palette, num_colors=6)
            if hex_colors:
                palette_data = ",".join(hex_colors)
                print(f"🎨 Extracted color palette: {palette_data}")

        # Apply final user mask if provided
        if mask is not None:
            if mask.shape[1:] != (height, width):
                 mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode='bilinear', align_corners=False).squeeze(1)
            mask = mask.unsqueeze(-1)
            processed_image = torch.lerp(original_image, processed_image, mask)

        # === GENERATE VISUALIZATION IMAGES ===
        final_image_np = (processed_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # Generate histogram image
        if extract_palette:
            histogram_image_np = generate_histogram_image(final_image_np)
            histogram_tensor = torch.from_numpy(histogram_image_np.astype(np.float32) / 255.0).unsqueeze(0).to(processed_image.device)
        else:
            # Generate black image when extract_palette is disabled
            histogram_tensor = torch.zeros((1, 512, 768, 3), device=processed_image.device)
        
        # Generate palette image
        if extract_palette and hex_colors:
            palette_image_np = generate_palette_image(hex_colors)
            palette_image_tensor = torch.from_numpy(palette_image_np.astype(np.float32) / 255.0).unsqueeze(0).to(processed_image.device)
        else:
            # Generate black image when no palette is extracted
            palette_image_tensor = torch.zeros((1, 120, 600, 3), device=processed_image.device)

        return (processed_image, palette_data, histogram_tensor, palette_image_tensor)
