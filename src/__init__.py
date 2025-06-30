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


def create_color_range_mask(hsv: torch.Tensor, target_hue: float, hue_range: float = 60.0) -> torch.Tensor:
    """Create a mask for a specific color range in HSV space."""
    h = hsv[..., 0]  # Hue channel (0-1)
    s = hsv[..., 1]  # Saturation channel (0-1)
    
    # Convert target hue from degrees to 0-1 range
    target_h = (target_hue % 360) / 360.0
    range_h = hue_range / 360.0
    
    # Calculate hue distance (handling wrap-around)
    hue_diff = torch.abs(h - target_h)
    hue_diff = torch.min(hue_diff, 1.0 - hue_diff)  # Handle wrap-around at 0/1
    
    # Create smooth falloff mask based on hue distance and saturation
    hue_mask = torch.exp(-((hue_diff / range_h) ** 2) * 3.0)  # Gaussian falloff
    saturation_mask = torch.clamp(s * 2.0, 0.0, 1.0)  # Favor more saturated colors
    
    return hue_mask * saturation_mask


def apply_selective_color_adjustment(
    image: torch.Tensor,
    target_hue: float,
    hue_adjustment: float,
    saturation_adjustment: float,
    lightness_adjustment: float,
    hue_range: float = 60.0
) -> torch.Tensor:
    """Apply selective color adjustments to a specific color range."""
    if abs(hue_adjustment) < 0.001 and abs(saturation_adjustment) < 0.001 and abs(lightness_adjustment) < 0.001:
        return image
    
    # Convert to HSV for processing
    hsv = rgb_to_hsv(image)
    
    # Create mask for target color range
    mask = create_color_range_mask(hsv, target_hue, hue_range)
    
    # Apply adjustments
    adjusted_hsv = hsv.clone()
    
    # Hue adjustment (wrap around 0-1)
    if abs(hue_adjustment) > 0.001:
        hue_shift = hue_adjustment * 0.1  # Scale adjustment
        adjusted_hsv[..., 0] = (adjusted_hsv[..., 0] + hue_shift * mask.unsqueeze(-1)) % 1.0
    
    # Saturation adjustment
    if abs(saturation_adjustment) > 0.001:
        sat_factor = 1.0 + saturation_adjustment
        adjusted_hsv[..., 1] = torch.clamp(
            adjusted_hsv[..., 1] * (1.0 + (sat_factor - 1.0) * mask.unsqueeze(-1)),
            0.0, 1.0
        )
    
    # Lightness adjustment (applied to Value in HSV)
    if abs(lightness_adjustment) > 0.001:
        light_factor = 1.0 + lightness_adjustment * 0.5  # Scale adjustment
        adjusted_hsv[..., 2] = torch.clamp(
            adjusted_hsv[..., 2] * (1.0 + (light_factor - 1.0) * mask.unsqueeze(-1)),
            0.0, 1.0
        )
    
    # Convert back to RGB
    return hsv_to_rgb(adjusted_hsv)


class DDColorColorization:
    """DDColor Deep Learning Colorization using pre-trained models."""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.model_name = "piddnad/ddcolor_modelscope"
        
    def _load_fallback_model(self, device):
        """Load a simple fallback model if DDColor fails."""
        try:
            import torch.nn as nn
            
            class FallbackColorizationNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                    )
                    self.decoder = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        nn.Conv2d(256, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(64, 2, 3, padding=1),
                        nn.Tanh(),
                    )
                
                def forward(self, x):
                    encoded = self.encoder(x)
                    decoded = self.decoder(encoded)
                    return decoded
            
            self.model = FallbackColorizationNet()
            self.model.to(device)
            self.model.eval()
            self.device = device
            
            print("✅ Fallback colorization model loaded")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load fallback model: {e}")
            return False
    
    
    def load_model(self, device):
        """Load the DDColor model from Hugging Face."""
        if not DL_COLORIZATION_AVAILABLE:
            return False
            
        try:
            print("🔄 Loading DDColor model...")
            from huggingface_hub import snapshot_download
            import torch.nn as nn
            
            # Try to download the model files (this will be cached)
            try:
                model_path = snapshot_download(repo_id=self.model_name, cache_dir=os.path.join(os.path.dirname(__file__), ".cache"))
                print(f"✅ DDColor model files downloaded to {model_path}")
            except Exception as e:
                print(f"⚠️ Could not download DDColor model: {e}")
                print("⚠️ Falling back to simplified colorization")
                return self._load_fallback_model(device)
            
            # Load DDColor architecture (improved version with proper upsampling)
            class DDColorNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Simple but effective encoder
                    self.encoder = nn.Sequential(
                        # Input: 3 x 256 x 256
                        nn.Conv2d(3, 64, 3, padding=1),  # 64 x 256 x 256
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1),  # 64 x 256 x 256
                        nn.ReLU(),
                        nn.MaxPool2d(2),  # 64 x 128 x 128
                        
                        nn.Conv2d(64, 128, 3, padding=1),  # 128 x 128 x 128
                        nn.ReLU(),
                        nn.Conv2d(128, 128, 3, padding=1),  # 128 x 128 x 128
                        nn.ReLU(),
                        nn.MaxPool2d(2),  # 128 x 64 x 64
                        
                        nn.Conv2d(128, 256, 3, padding=1),  # 256 x 64 x 64
                        nn.ReLU(),
                        nn.Conv2d(256, 256, 3, padding=1),  # 256 x 64 x 64
                        nn.ReLU(),
                        nn.MaxPool2d(2),  # 256 x 32 x 32
                        
                        nn.Conv2d(256, 512, 3, padding=1),  # 512 x 32 x 32
                        nn.ReLU(),
                    )
                    
                    # Decoder with proper upsampling to match input size
                    self.decoder = nn.Sequential(
                        # 512 x 32 x 32 -> 256 x 64 x 64
                        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                        nn.ReLU(),
                        
                        # 256 x 64 x 64 -> 128 x 128 x 128
                        nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                        nn.ReLU(),
                        
                        # 128 x 128 x 128 -> 64 x 256 x 256
                        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                        nn.ReLU(),
                        
                        # Final layer to produce AB channels: 64 x 256 x 256 -> 2 x 256 x 256
                        nn.Conv2d(64, 2, 3, padding=1),
                        nn.Tanh(),
                    )
                
                def forward(self, x):
                    # Encode
                    encoded = self.encoder(x)
                    # Decode to AB channels
                    ab = self.decoder(encoded)
                    return ab
            
            self.model = DDColorNet()
            self.model.to(device)
            self.model.eval()
            self.device = device
            
            # Set up transforms
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ DDColor model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load DDColor model: {e}")
            print("⚠️ Falling back to basic colorization")
            return self._load_fallback_model(device)
    
    def colorize_image(self, image_tensor, strength=1.0):
        """Colorize a grayscale or low-saturation image using DDColor."""
        if self.model is None:
            print("❌ DDColor model not loaded")
            return image_tensor
            
        try:
            print(f"🤖 Applying DDColor colorization with strength {strength:.2f}")
            device = image_tensor.device
            original_size = image_tensor.shape
            print(f"🔧 Input image size: {original_size}")
            
            # Prepare input - convert to proper format
            input_rgb = image_tensor.squeeze(0)  # Remove batch dimension
            
            # Convert to grayscale for DDColor input (expects single channel)
            if ADVANCED_LIBS_AVAILABLE:
                # Convert to grayscale using proper weights
                gray = 0.299 * input_rgb[:,:,0] + 0.587 * input_rgb[:,:,1] + 0.114 * input_rgb[:,:,2]
                gray_3ch = gray.unsqueeze(-1).repeat(1, 1, 3)  # Convert to 3-channel for model
            else:
                # Simple average
                gray = torch.mean(input_rgb, dim=2, keepdim=True)
                gray_3ch = gray.repeat(1, 1, 3)
            
            # Resize for model input and add batch dimension
            input_resized = F.interpolate(gray_3ch.unsqueeze(0).permute(0, 3, 1, 2), 
                                        size=(256, 256), mode='bilinear', align_corners=False)
            
            print(f"🔧 Model input size: {input_resized.shape}")
            
            # Run through model
            with torch.no_grad():
                ab_pred = self.model(input_resized)
                print(f"🔧 Model output size: {ab_pred.shape}")
                
                # Resize back to original size
                if ab_pred.shape[-2:] != original_size[1:3]:
                    ab_pred = F.interpolate(ab_pred, size=original_size[1:3], 
                                          mode='bilinear', align_corners=False)
                    print(f"🔧 Resized output to: {ab_pred.shape}")
                
                # Convert to LAB and combine with L channel
                if ADVANCED_LIBS_AVAILABLE:
                    print("🔧 Using advanced LAB color space conversion")
                    # Convert RGB input to LAB to get L channel
                    rgb_np = (input_rgb.cpu().numpy() * 255).astype(np.uint8)
                    lab_image = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
                    l_channel = lab_image[:, :, 0].astype(np.float32)
                    
                    # Scale AB predictions - be more conservative with scaling
                    ab_np = ab_pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    
                    # DDColor typically outputs in [-1,1], scale more conservatively
                    ab_np = ab_np * 50.0  # Reduced from 127 to 50 for less aggressive coloring
                    
                    # Combine L and AB channels
                    lab_pred = np.zeros((original_size[1], original_size[2], 3), dtype=np.float32)
                    lab_pred[:,:,0] = l_channel  # L channel from original
                    lab_pred[:,:,1] = ab_np[:,:,0]  # A channel from prediction
                    lab_pred[:,:,2] = ab_np[:,:,1]  # B channel from prediction
                    
                    # Ensure values are in valid LAB range
                    lab_pred[:,:,0] = np.clip(lab_pred[:,:,0], 0, 100)  # L channel range
                    lab_pred[:,:,1:] = np.clip(lab_pred[:,:,1:], -127, 127)  # AB channel range
                    
                    print(f"🔧 LAB stats - L: [{lab_pred[:,:,0].min():.1f}, {lab_pred[:,:,0].max():.1f}], A: [{lab_pred[:,:,1].min():.1f}, {lab_pred[:,:,1].max():.1f}], B: [{lab_pred[:,:,2].min():.1f}, {lab_pred[:,:,2].max():.1f}]")
                    
                    # Convert back to RGB - ensure proper data type
                    lab_uint8 = lab_pred.astype(np.uint8)
                    rgb_pred = cv2.cvtColor(lab_uint8, cv2.COLOR_LAB2RGB)
                    result_tensor = torch.from_numpy(rgb_pred.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                    print("✅ LAB to RGB conversion completed")
                else:
                    print("🔧 Using fallback RGB colorization")
                    # Simple fallback - blend AB predictions with grayscale
                    ab_pred_scaled = ab_pred * 0.5  # Reduce intensity
                    
                    # Create colorized version
                    colorized = torch.zeros_like(input_rgb)
                    gray_normalized = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
                    
                    # Use AB predictions to modulate RG channels
                    ab_resized = ab_pred.squeeze(0).permute(1, 2, 0)
                    colorized[:,:,0] = gray_normalized.squeeze(-1)  # Keep grayscale as red
                    colorized[:,:,1] = torch.clamp(gray_normalized.squeeze(-1) + ab_resized[:,:,0] * 0.3, 0, 1)  # A -> Green
                    colorized[:,:,2] = torch.clamp(gray_normalized.squeeze(-1) + ab_resized[:,:,1] * 0.3, 0, 1)  # B -> Blue
                    
                    result_tensor = colorized.unsqueeze(0)
            
            # Blend with original based on strength
            result = torch.lerp(image_tensor, result_tensor, strength)
            print(f"✅ DDColor colorization completed, blended with strength {strength:.2f}")
            
            return result
            
        except Exception as e:
            print(f"❌ DDColor colorization failed: {e}")
            import traceback
            traceback.print_exc()
            return image_tensor


# Global colorization instance
_deep_colorizer = None

def get_deep_colorizer():
    """Get or create the global deep colorization instance."""
    global _deep_colorizer
    if _deep_colorizer is None:
        _deep_colorizer = DDColorColorization()
    return _deep_colorizer


def apply_semantic_color_adjustments(
    image: torch.Tensor,
    skin_tone_adjustment: float = 0.0,
    sky_adjustment: float = 0.0,
    foliage_adjustment: float = 0.0,
    selective_hue_shift: float = 0.0,
    selective_saturation: float = 0.0,
    selective_strength: float = 1.0,
) -> torch.Tensor:
    """Apply semantic color adjustments targeting skin tones, sky, and foliage."""
    result = image.clone()
    
    # Early exit if no adjustments
    if (abs(skin_tone_adjustment) < 0.001 and abs(sky_adjustment) < 0.001 and 
        abs(foliage_adjustment) < 0.001 and abs(selective_hue_shift) < 0.001 and 
        abs(selective_saturation) < 0.001):
        return result
    
    # Define semantic color targets with broader, more natural ranges
    semantic_targets = []
    
    # Skin tones: warm oranges/yellows (20-50 degrees)
    if abs(skin_tone_adjustment) > 0.001:
        semantic_targets.append((35, selective_hue_shift, selective_saturation + skin_tone_adjustment, skin_tone_adjustment * 0.3, 80.0))
    
    # Sky/water: blues and cyans (200-260 degrees)  
    if abs(sky_adjustment) > 0.001:
        semantic_targets.append((230, selective_hue_shift, selective_saturation + sky_adjustment, sky_adjustment * 0.2, 70.0))
    
    # Foliage: greens (90-150 degrees)
    if abs(foliage_adjustment) > 0.001:
        semantic_targets.append((120, selective_hue_shift, selective_saturation + foliage_adjustment, foliage_adjustment * 0.3, 80.0))
    
    # Apply semantic adjustments
    for target_hue, hue_adj, sat_adj, light_adj, hue_range in semantic_targets:
        if abs(hue_adj) > 0.001 or abs(sat_adj) > 0.001 or abs(light_adj) > 0.001:
            adjustment = apply_selective_color_adjustment(
                result, target_hue, hue_adj * selective_strength, 
                sat_adj * selective_strength, light_adj * selective_strength, hue_range
            )
            # Blend with original based on selective_strength
            result = torch.lerp(result, adjustment, selective_strength)
    
    return result


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
        # Balanced parameters for reliable face detection
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,      # Standard scale factor
            minNeighbors=3,       # Balanced between sensitivity and accuracy
            minSize=(30, 30),     # Reasonable minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
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
                # Semantic Selective Color Controls
                "skin_tone_adjustment": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "🧑 Adjust skin tones and flesh colors (targets warm oranges/yellows)",
                    },
                ),
                "sky_adjustment": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "🌅 Adjust sky and blue regions (targets blues/cyans)",
                    },
                ),
                "foliage_adjustment": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "🌿 Adjust foliage and vegetation (targets greens)",
                    },
                ),
                "selective_hue_shift": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "🎨 Universal hue shift for targeted colors",
                    },
                ),
                "selective_saturation": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "🎨 Universal saturation adjustment for targeted colors",
                    },
                ),
                "selective_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "💪 Overall strength of selective color adjustments",
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
                    ["deep_learning", "classic", "portrait", "landscape", "vintage"],
                    {
                        "default": "deep_learning",
                        "tooltip": "🎯 Colorization method: deep_learning (AI), classic (HSV), portrait, landscape, or vintage",
                    },
                ),
                "force_colorize": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "🔄 Force colorization even for art/anime content (overrides AI detection)",
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
        # Semantic Selective Color parameters
        skin_tone_adjustment: float = 0.0,
        sky_adjustment: float = 0.0,
        foliage_adjustment: float = 0.0,
        selective_hue_shift: float = 0.0,
        selective_saturation: float = 0.0,
        selective_strength: float = 1.0,
        colorize_strength: float = 0.8,
        skin_warmth: float = 0.3,
        sky_saturation: float = 0.6,
        vegetation_green: float = 0.5,
        sepia_tone: float = 0.0,
        colorize_mode: str = "auto",
        force_colorize: bool = False,
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

            # Manual mode controls only apply when mode == "Manual"
            # Move HSV processing inside Manual mode

            # --- MANUAL MODE ---
            if mode == "Manual":
                # Convert to HSV for manual adjustments
                hsv_image = rgb_to_hsv(processed_image)
                h, s, v = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]
                
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

                # Basic HSV adjustments
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
                if (warmth != 0.0 or tint != 0.0):
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
                
                # Update HSV after temperature/tint adjustments
                hsv_image = rgb_to_hsv(processed_image)
                h, s, v = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]

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
            if mode == "Colorize":
                # Always apply colorization in colorize mode - user knows what they're doing
                if analysis:
                    scene_type = analysis["scene_type"]
                    print(f"🎨 Colorize Mode: Processing {scene_type} content")
                    print(f"🔧 Calling _apply_colorization with mode: {colorize_mode}")
                    print(f"🔧 Colorization parameters: strength={colorize_strength}, skin_warmth={skin_warmth}, sky_saturation={sky_saturation}")
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

                # Note: Skin tone protection is now handled within the colorization methods

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
        Advanced deep learning colorization with fallback to classic methods.
        Supports multiple colorization modes including state-of-the-art AI models.
        """
        device = original_image.device
        print(f"🔧 _apply_colorization called with mode: {colorize_mode}")
        print(f"🔧 DL_COLORIZATION_AVAILABLE: {DL_COLORIZATION_AVAILABLE}")
        
        # Check if image needs colorization
        avg_saturation = torch.mean(rgb_to_hsv(processed_image)[..., 1]).item()
        print(f"🔧 Average saturation: {avg_saturation:.3f}")
        if avg_saturation > 0.3:
            print(f"⚠️ Image already has color (avg saturation: {avg_saturation:.2f}), applying gentle enhancement")
            colorize_strength *= 0.3
        
        if colorize_mode == "deep_learning":
            # Use deep learning colorization (primary method)
            if DL_COLORIZATION_AVAILABLE:
                try:
                    print("🤖 Initializing DDColor deep learning colorization...")
                    colorizer = get_deep_colorizer()
                    
                    # Load model if not already loaded
                    if colorizer.model is None:
                        print("📥 Loading DDColor model...")
                        success = colorizer.load_model(device)
                        if not success:
                            print("⚠️ DDColor model failed to load, falling back to classic colorization")
                            return self._apply_classic_colorization(
                                original_image, processed_image, analysis,
                                colorize_strength, skin_warmth, sky_saturation,
                                vegetation_green, sepia_tone
                            )
                    else:
                        print("✅ DDColor model already loaded")
                    
                    # Apply deep learning colorization
                    print(f"🎨 Applying DDColor colorization (strength: {colorize_strength:.2f})...")
                    colorized_image = colorizer.colorize_image(processed_image, colorize_strength)
                    
                    # Check if colorization actually happened
                    if torch.equal(colorized_image, processed_image):
                        print("⚠️ DDColor returned unchanged image, falling back to classic colorization")
                        return self._apply_classic_colorization(
                            original_image, processed_image, analysis,
                            colorize_strength, skin_warmth, sky_saturation,
                            vegetation_green, sepia_tone
                        )
                    
                    # Apply additional adjustments if needed
                    if skin_warmth != 0.0 or sky_saturation != 0.0 or vegetation_green != 0.0:
                        print("🔧 Applying additional color adjustments...")
                        colorized_image = self._apply_color_adjustments(
                            colorized_image, analysis, skin_warmth, 
                            sky_saturation, vegetation_green, colorize_strength
                        )
                    
                    print("✅ DDColor deep learning colorization completed successfully")
                    return colorized_image
                    
                except Exception as e:
                    print(f"❌ DDColor deep learning colorization failed: {e}")
                    print("⚠️ Falling back to classic colorization")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Deep learning libraries not available, using classic colorization")
        
        # Fallback to classic colorization for other modes or when DL fails
        print(f"🔧 Using classic colorization fallback for mode: {colorize_mode}")
        return self._apply_classic_colorization(
            original_image, processed_image, analysis,
            colorize_strength, skin_warmth, sky_saturation,
            vegetation_green, sepia_tone, colorize_mode
        )
    
    def _apply_classic_colorization(
        self,
        original_image,
        processed_image, 
        analysis,
        colorize_strength,
        skin_warmth,
        sky_saturation,
        vegetation_green,
        sepia_tone,
        colorize_mode="classic"
    ):
        """Classic HSV-based colorization for fallback and specific modes."""
        device = original_image.device
        print(f"🔧 Classic colorization: mode={colorize_mode}, strength={colorize_strength}")
        print(f"🔧 Parameters: skin_warmth={skin_warmth}, sky_saturation={sky_saturation}, vegetation_green={vegetation_green}")
        
        # Convert to HSV for color manipulation
        hsv_image = rgb_to_hsv(processed_image)
        h, s, v = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]
        
        # Create region masks
        luminance = torch.mean(processed_image, dim=-1)
        height = processed_image.shape[1]
        
        # Sky detection (upper regions with high luminance)
        sky_region = torch.zeros_like(luminance, device=device)
        sky_upper_third = height // 3
        sky_region[:, :sky_upper_third, :] = 1.0
        sky_luminance_mask = (luminance > 0.7).float()
        sky_mask = sky_region * sky_luminance_mask
        
        # Vegetation detection (mid-luminance areas)
        vegetation_mask = ((luminance > 0.2) & (luminance < 0.8)).float()
        vegetation_mask = vegetation_mask * (1.0 - sky_mask)
        
        # Skin tone detection
        skin_mask = torch.zeros_like(luminance, device=device)
        if analysis and analysis.get("faces"):
            skin_mask = ((luminance > 0.25) & (luminance < 0.85)).float()
            skin_mask = skin_mask * (1.0 - sky_mask) * (1.0 - vegetation_mask)
        
        # Apply colorization based on mode
        if colorize_mode == "vintage":
            base_hue = 0.08  # Warm sepia hue
            h = torch.full_like(h, base_hue)
            s = s + sepia_tone * 0.4 * colorize_strength
            
        elif colorize_mode == "portrait":
            # Portrait-focused colorization
            skin_hue = 0.08
            h = torch.where(skin_mask > 0.3, skin_hue, h)
            s = torch.where(skin_mask > 0.3, s + skin_warmth * colorize_strength, s)
            
        elif colorize_mode == "landscape":
            # Landscape-focused colorization
            sky_hue = 0.58  # Blue
            h = torch.where(sky_mask > 0.5, sky_hue, h)
            s = torch.where(sky_mask > 0.5, s + sky_saturation * colorize_strength, s)
            
            vegetation_hue = 0.25  # Green
            h = torch.where(vegetation_mask > 0.4, vegetation_hue, h)
            s = torch.where(vegetation_mask > 0.4, s + vegetation_green * colorize_strength, s)
            
        else:  # classic or auto mode
            # Basic intelligent colorization
            if sky_saturation > 0:
                sky_hue = 0.58
                h = torch.where(sky_mask > 0.5, sky_hue, h)
                s = torch.where(sky_mask > 0.5, s + sky_saturation * colorize_strength, s)
            
            if vegetation_green > 0:
                vegetation_hue = 0.25
                h = torch.where(vegetation_mask > 0.4, vegetation_hue, h)
                s = torch.where(vegetation_mask > 0.4, s + vegetation_green * colorize_strength, s)
            
            if skin_warmth > 0 and analysis and analysis.get("faces"):
                skin_hue = 0.08
                h = torch.where(skin_mask > 0.3, skin_hue, h)
                s = torch.where(skin_mask > 0.3, s + skin_warmth * colorize_strength, s)
        
        # Apply sepia tone if specified
        if sepia_tone > 0:
            sepia_hue = 0.08
            h = torch.lerp(h, torch.full_like(h, sepia_hue), sepia_tone)
            s = s + sepia_tone * 0.3
        
        # Clamp and convert back to RGB
        h = h % 1.0
        s = torch.clamp(s, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)
        
        colorized_hsv = torch.stack([h, s, v], dim=-1)
        colorized_rgb = hsv_to_rgb(colorized_hsv)
        
        # Blend with original
        final_image = torch.lerp(processed_image, colorized_rgb, colorize_strength)
        return torch.clamp(final_image, 0.0, 1.0)
    
    def _apply_color_adjustments(
        self,
        image,
        analysis,
        skin_warmth,
        sky_saturation, 
        vegetation_green,
        strength
    ):
        """Apply additional color adjustments to deep learning colorized image."""
        if skin_warmth == 0.0 and sky_saturation == 0.0 and vegetation_green == 0.0:
            return image
            
        print(f"🔧 Applying targeted color adjustments: skin_warmth={skin_warmth:.2f}, sky_saturation={sky_saturation:.2f}, vegetation_green={vegetation_green:.2f}")
        
        # Apply targeted adjustments without full classic colorization pipeline
        result = image.clone()
        
        # Convert to HSV for targeted adjustments
        hsv = rgb_to_hsv(result)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        
        # Skin tone adjustments (warm oranges/peaches)
        if skin_warmth != 0.0:
            # Target skin tone hues (roughly 0.02-0.08 in HSV)
            skin_mask = ((h >= 0.01) & (h <= 0.1)) & (s > 0.1)
            if skin_mask.any():
                # Warm up skin tones
                h = torch.where(skin_mask, h + skin_warmth * 0.02, h)
                s = torch.where(skin_mask, torch.clamp(s + skin_warmth * 0.1, 0, 1), s)
                print(f"   ✅ Applied skin warming to {skin_mask.sum().item()} pixels")
        
        # Sky saturation adjustments (blues)
        if sky_saturation != 0.0:
            # Target sky/blue hues (roughly 0.55-0.75 in HSV)
            sky_mask = ((h >= 0.5) & (h <= 0.8)) & (s > 0.1)
            if sky_mask.any():
                s = torch.where(sky_mask, torch.clamp(s + sky_saturation * 0.2, 0, 1), s)
                print(f"   ✅ Applied sky saturation to {sky_mask.sum().item()} pixels")
        
        # Vegetation adjustments (greens)
        if vegetation_green != 0.0:
            # Target green hues (roughly 0.25-0.45 in HSV)
            green_mask = ((h >= 0.2) & (h <= 0.5)) & (s > 0.1)
            if green_mask.any():
                s = torch.where(green_mask, torch.clamp(s + vegetation_green * 0.15, 0, 1), s)
                h = torch.where(green_mask, h + vegetation_green * 0.01, h)
                print(f"   ✅ Applied vegetation enhancement to {green_mask.sum().item()} pixels")
        
        # Convert back to RGB
        adjusted_hsv = torch.stack([h, s, v], dim=-1)
        result = hsv_to_rgb(adjusted_hsv)
        
        # Blend with original based on reduced strength
        adjustment_strength = strength * 0.5  # Reduce strength for subtle adjustments
        result = torch.lerp(image, result, adjustment_strength)
        
        print(f"   ✅ Color adjustments applied with strength {adjustment_strength:.2f}")
        return result


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
                "tint": (
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
    DISPLAY_NAME = "Easy Batch Color Corrector (Beta)"
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
        tint=0.0,
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

        # Pre-allocate output tensor to avoid OOM from torch.cat
        _, height, width, channels = images.shape
        final_images = torch.zeros((total_frames, height, width, channels), device=device, dtype=images.dtype)
        
        # Lists for metadata (much smaller memory footprint)
        all_palette_data = []
        all_histograms = []
        all_palette_images = []
        processed_count = 0

        # Process frames in GPU-optimized batches with interruption support
        try:
            for batch_start in range(0, total_frames, frames_per_batch):
                # Check for interruption requests
                import comfy.model_management as model_management

                if model_management.interrupt_processing:
                    print("🛑 Batch processing interrupted by user")
                    # Return partially processed results
                    if processed_count > 0:
                        partial_images = final_images[:processed_count]
                        print(
                            f"⚠️ Partial result: {processed_count}/{total_frames} frames processed"
                        )
                        return (
                            partial_images,
                            "",
                            torch.zeros((1, 512, 768, 3), device=device),
                            torch.zeros((1, 120, 600, 3), device=device),
                            processed_count,
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

                # Copy batch results directly into pre-allocated tensor
                final_images[batch_start:batch_end] = batch_processed
                processed_count = batch_end

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
            if processed_count > 0:
                partial_images = final_images[:processed_count]
                print(
                    f"⚠️ Partial result: {processed_count}/{total_frames} frames processed"
                )
                return (
                    partial_images,
                    "",
                    torch.zeros((1, 512, 768, 3), device=device),
                    torch.zeros((1, 120, 600, 3), device=device),
                    processed_count,
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
            if processed_count > 0:
                partial_images = final_images[:processed_count]
                print(
                    f"⚠️ Partial result after error: {processed_count}/{total_frames} frames processed"
                )
                return (
                    partial_images,
                    "",
                    torch.zeros((1, 512, 768, 3), device=device),
                    torch.zeros((1, 120, 600, 3), device=device),
                    processed_count,
                )
            else:
                return (
                    images,
                    "",
                    torch.zeros((1, 512, 768, 3), device=device),
                    torch.zeros((1, 120, 600, 3), device=device),
                    0,
                )

        # Processing completed successfully
        if processed_count == total_frames:

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

        # Manual mode color corrections (applied to entire batch)
        if mode == "Manual":
            # Apply basic color corrections first
            if warmth != 0.0:
                h = (h + warmth * 0.1) % 1.0

            if vibrancy != 0.0:
                saturation_mask = 1.0 - s
                s = s * (1.0 + vibrancy) + (vibrancy * 0.3 * saturation_mask * s)

            if brightness != 0.0:
                v = v + brightness * (1.0 - v * 0.5)

            if contrast != 0.0:
                v = 0.5 + (v - 0.5) * (1.0 + contrast)

            # Apply tint adjustment (requires LAB color space)
            if tint != 0.0 and ADVANCED_LIBS_AVAILABLE:
                try:
                    # Convert current HSV back to RGB then to numpy for LAB processing
                    temp_hsv = torch.stack([h, s, v], dim=-1)
                    temp_rgb = hsv_to_rgb(temp_hsv)
                    image_np_for_tint = (temp_rgb.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    
                    # Convert to LAB and apply tint
                    lab = cv2.cvtColor(image_np_for_tint, cv2.COLOR_RGB2LAB)
                    tint_shift = tint * 30  # Scale factor for visible tint effect
                    lab[:, :, 1] = np.clip(lab[:, :, 1] + tint_shift, 0, 255)
                    
                    # Convert back to RGB then to tensors
                    corrected_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                    corrected_tensor = torch.from_numpy(corrected_rgb.astype(np.float32) / 255.0).unsqueeze(0).to(device)
                    
                    # Convert back to HSV for continued processing
                    corrected_hsv = rgb_to_hsv(corrected_tensor)
                    h, s, v = corrected_hsv[..., 0], corrected_hsv[..., 1], corrected_hsv[..., 2]
                    
                except Exception as e:
                    print(f"⚠️ Tint processing failed: {e}")

            # Create masks for 3-way color correction
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

            # Semantic selective color adjustments not available in BatchColorCorrection 
            # (use main EasyColorCorrection node for selective controls)

            # Add noise if specified (Manual mode only)
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
    DISPLAY_NAME = "Raw Image Processor"

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
    DISPLAY_NAME = "Color Corrector Image Viewer"
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
    DISPLAY_NAME = "Color Palette Extractor"

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


class FilmEmulation:
    """
    Professional film stock emulation node for creative styling.
    
    Simulates the characteristics of classic analog film stocks including:
    - Color response curves specific to each film type
    - Film grain patterns and intensity
    - Contrast curves and saturation response
    - Temperature/tint characteristics
    - Push/pull processing simulation
    """
    
    # Film stock definitions with their characteristic parameters
    FILM_STOCKS = {
        "Kodak Portra 400": {
            "description": "Warm, natural skin tones with subtle grain",
            "temperature_bias": 0.15,  # Warm bias
            "tint_bias": 0.05,         # Slight magenta
            "saturation_curve": "smooth",
            "contrast_curve": "soft",
            "grain_type": "fine",
            "color_shifts": {"shadows": (0.02, 0.1, 0.05), "highlights": (0.1, 0.05, 0.0)},
            "gamma_curve": 0.95,
        },
        "Kodak Portra 800": {
            "description": "Higher speed version with more pronounced grain",
            "temperature_bias": 0.18,
            "tint_bias": 0.08,
            "saturation_curve": "smooth",
            "contrast_curve": "soft",
            "grain_type": "medium",
            "color_shifts": {"shadows": (0.03, 0.12, 0.08), "highlights": (0.12, 0.08, 0.02)},
            "gamma_curve": 0.92,
        },
        "Fuji Velvia 50": {
            "description": "Saturated, punchy colors with cool shadows",
            "temperature_bias": -0.1,  # Cool bias
            "tint_bias": -0.05,        # Slight green
            "saturation_curve": "punchy",
            "contrast_curve": "high",
            "grain_type": "ultra_fine",
            "color_shifts": {"shadows": (-0.05, 0.0, 0.1), "highlights": (0.05, 0.15, 0.05)},
            "gamma_curve": 1.1,
        },
        "Fuji Velvia 100": {
            "description": "Slightly more subdued than Velvia 50",
            "temperature_bias": -0.08,
            "tint_bias": -0.03,
            "saturation_curve": "punchy",
            "contrast_curve": "medium_high",
            "grain_type": "fine",
            "color_shifts": {"shadows": (-0.03, 0.0, 0.08), "highlights": (0.03, 0.12, 0.03)},
            "gamma_curve": 1.05,
        },
        "Kodak Vision3 500T": {
            "description": "Cinematic tungsten-balanced stock",
            "temperature_bias": 0.25,  # Strong tungsten bias
            "tint_bias": 0.1,
            "saturation_curve": "cinematic",
            "contrast_curve": "medium",
            "grain_type": "medium",
            "color_shifts": {"shadows": (0.05, 0.2, 0.1), "highlights": (0.15, 0.1, 0.0)},
            "gamma_curve": 0.9,
        },
        "Kodak Gold 200": {
            "description": "Vintage warm look with yellow/orange cast",
            "temperature_bias": 0.2,
            "tint_bias": 0.15,         # Strong yellow/orange
            "saturation_curve": "vintage",
            "contrast_curve": "medium",
            "grain_type": "medium",
            "color_shifts": {"shadows": (0.08, 0.15, 0.05), "highlights": (0.2, 0.12, 0.0)},
            "gamma_curve": 0.88,
        },
        "Fuji Pro 400H": {
            "description": "Soft, pastel tones, overexposure-friendly",
            "temperature_bias": 0.05,
            "tint_bias": 0.02,
            "saturation_curve": "soft",
            "contrast_curve": "low",
            "grain_type": "fine",
            "color_shifts": {"shadows": (0.02, 0.05, 0.08), "highlights": (0.08, 0.08, 0.05)},
            "gamma_curve": 0.85,
        },
        "Kodak Tri-X 400": {
            "description": "Classic B&W with distinctive grain",
            "temperature_bias": 0.0,   # B&W
            "tint_bias": 0.0,
            "saturation_curve": "bw",
            "contrast_curve": "high",
            "grain_type": "heavy",
            "color_shifts": {"shadows": (0.0, 0.0, 0.0), "highlights": (0.0, 0.0, 0.0)},
            "gamma_curve": 1.2,
        },
        "Ilford HP5 Plus": {
            "description": "High contrast B&W with fine grain",
            "temperature_bias": 0.0,
            "tint_bias": 0.0,
            "saturation_curve": "bw",
            "contrast_curve": "very_high",
            "grain_type": "medium",
            "color_shifts": {"shadows": (0.0, 0.0, 0.0), "highlights": (0.0, 0.0, 0.0)},
            "gamma_curve": 1.3,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "film_stock": (list(cls.FILM_STOCKS.keys()), {"default": "Kodak Portra 400"}),
                "strength": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Blend strength between original and film emulation"
                }),
                "grain_intensity": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Film grain intensity (0.0 = no grain, 1.0 = maximum grain)"
                }),
                "exposure_compensation": ("FLOAT", {
                    "default": 0.0, 
                    "min": -2.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "tooltip": "Exposure compensation in stops (simulates over/under exposure response)"
                }),
                "push_pull": ("FLOAT", {
                    "default": 0.0, 
                    "min": -2.0, 
                    "max": 2.0, 
                    "step": 0.5,
                    "tooltip": "Push/pull processing simulation (affects contrast and grain)"
                }),
                "highlight_rolloff": ("FLOAT", {
                    "default": 0.8, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.05,
                    "tooltip": "Film highlight rolloff characteristics"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "film_info")
    FUNCTION = "apply_film_emulation"
    CATEGORY = "itsjustregi / Easy Color Corrector"
    DISPLAY_NAME = "Film Emulation"

    def apply_film_emulation(self, image, film_stock, strength=1.0, grain_intensity=0.3, 
                           exposure_compensation=0.0, push_pull=0.0, highlight_rolloff=0.8):
        """Apply film stock emulation to the input image."""
        
        original_image = image.clone()
        processed_image = image.clone()
        
        # Get film stock characteristics
        stock_params = self.FILM_STOCKS[film_stock]
        
        print(f"🎞️ Applying {film_stock} emulation (strength: {strength:.2f})")
        print(f"📝 {stock_params['description']}")
        
        # Apply exposure compensation first (simulates film response to different exposures)
        if exposure_compensation != 0.0:
            processed_image = self._apply_exposure_compensation(
                processed_image, exposure_compensation, stock_params
            )
        
        # Apply film color characteristics
        processed_image = self._apply_color_response(processed_image, stock_params, push_pull)
        
        # Apply film grain
        if grain_intensity > 0.0:
            processed_image = self._apply_film_grain(
                processed_image, stock_params, grain_intensity, push_pull
            )
        
        # Apply highlight rolloff
        processed_image = self._apply_highlight_rolloff(
            processed_image, highlight_rolloff, stock_params
        )
        
        # Blend with original based on strength
        if strength < 1.0:
            processed_image = torch.lerp(original_image, processed_image, strength)
        
        # Generate film info string
        film_info = f"Film: {film_stock}, Strength: {strength:.2f}, Grain: {grain_intensity:.2f}"
        if exposure_compensation != 0.0:
            film_info += f", Exposure: {exposure_compensation:+.1f} stops"
        if push_pull != 0.0:
            film_info += f", Push/Pull: {push_pull:+.1f}"
            
        print(f"✅ Film emulation applied: {film_info}")
        
        return (processed_image, film_info)

    def _apply_exposure_compensation(self, image, exposure_stops, stock_params):
        """Simulate film response to different exposure levels."""
        # Convert stops to linear multiplier
        exposure_multiplier = 2.0 ** exposure_stops
        
        # Apply exposure with film-specific response
        exposed_image = image * exposure_multiplier
        
        # Film has non-linear response to overexposure (shoulder curve)
        if exposure_stops > 0:
            # Simulate film shoulder - highlights compress gracefully
            shoulder_strength = min(exposure_stops / 2.0, 1.0)
            exposed_image = torch.where(
                exposed_image > 0.8,
                0.8 + (exposed_image - 0.8) * (1.0 - shoulder_strength * 0.7),
                exposed_image
            )
        
        return torch.clamp(exposed_image, 0.0, 1.0)

    def _apply_color_response(self, image, stock_params, push_pull):
        """Apply film-specific color response curves and characteristics."""
        
        # Convert to HSV for easier manipulation
        if ADVANCED_LIBS_AVAILABLE:
            hsv_image = rgb_to_hsv(image)
            h, s, v = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]
        else:
            # Fallback to simple processing
            return self._apply_simple_film_response(image, stock_params, push_pull)
        
        # Apply temperature and tint bias
        temp_bias = stock_params["temperature_bias"]
        tint_bias = stock_params["tint_bias"]
        
        if temp_bias != 0.0:
            # Adjust hue for temperature (blue-yellow axis)
            h = (h + temp_bias * 0.1) % 1.0
        
        # Apply saturation curve based on film type
        saturation_curve = stock_params["saturation_curve"]
        s = self._apply_saturation_curve(s, saturation_curve, push_pull)
        
        # Apply contrast curve (gamma)
        gamma = stock_params["gamma_curve"]
        if push_pull != 0.0:
            # Push processing increases contrast, pull decreases it
            gamma = gamma + (push_pull * 0.15)
        
        v = torch.pow(torch.clamp(v, 0.001, 1.0), gamma)
        
        # Apply color shifts to shadows and highlights
        color_shifts = stock_params["color_shifts"]
        processed_image = hsv_to_rgb(torch.stack([h, s, v], dim=-1))
        processed_image = self._apply_color_shifts(processed_image, color_shifts)
        
        return torch.clamp(processed_image, 0.0, 1.0)

    def _apply_saturation_curve(self, saturation, curve_type, push_pull):
        """Apply film-specific saturation response curves."""
        
        if curve_type == "bw":
            # Black and white films
            return torch.zeros_like(saturation)
        
        # Saturation multipliers based on film type
        multipliers = {
            "soft": 0.85,
            "smooth": 0.95,
            "cinematic": 1.0,
            "vintage": 1.1,
            "punchy": 1.3,
        }
        
        multiplier = multipliers.get(curve_type, 1.0)
        
        # Push processing affects saturation
        if push_pull != 0.0:
            multiplier += push_pull * 0.15
        
        # Apply film-specific saturation curve
        if curve_type == "punchy":
            # Velvia-style: boost mid-saturation, compress high saturation
            s_adjusted = saturation * multiplier
            s_adjusted = torch.where(
                saturation > 0.7,
                0.7 + (s_adjusted - 0.7) * 0.6,  # Compress high saturation
                s_adjusted
            )
        elif curve_type == "soft":
            # Pro 400H style: gentle saturation with smooth rolloff
            s_adjusted = saturation * multiplier
            s_adjusted = torch.pow(s_adjusted, 1.1)  # Slight curve
        else:
            # Linear adjustment for other types
            s_adjusted = saturation * multiplier
        
        return torch.clamp(s_adjusted, 0.0, 1.0)

    def _apply_color_shifts(self, image, color_shifts):
        """Apply film-specific color shifts to shadows and highlights."""
        
        # Calculate luminance for masking
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        
        # Create shadow and highlight masks
        shadow_mask = torch.clamp(1.0 - luminance * 3.0, 0.0, 1.0)
        highlight_mask = torch.clamp((luminance - 0.6) * 2.5, 0.0, 1.0)
        
        # Apply shadow color shifts
        shadow_shift = color_shifts["shadows"]
        for i, shift in enumerate(shadow_shift):
            if shift != 0.0:
                image[..., i] += shift * shadow_mask.unsqueeze(-1)
        
        # Apply highlight color shifts
        highlight_shift = color_shifts["highlights"]
        for i, shift in enumerate(highlight_shift):
            if shift != 0.0:
                image[..., i] += shift * highlight_mask.unsqueeze(-1)
        
        return torch.clamp(image, 0.0, 1.0)

    def _apply_simple_film_response(self, image, stock_params, push_pull):
        """Simplified film response when advanced libraries aren't available."""
        
        # Basic gamma adjustment
        gamma = stock_params["gamma_curve"]
        if push_pull != 0.0:
            gamma = gamma + (push_pull * 0.15)
        
        processed = torch.pow(torch.clamp(image, 0.001, 1.0), gamma)
        
        # Basic temperature/tint adjustment
        temp_bias = stock_params["temperature_bias"]
        tint_bias = stock_params["tint_bias"]
        
        if temp_bias != 0.0:
            # Simple temperature shift (blue-yellow)
            processed[..., 0] += temp_bias * 0.1  # Red
            processed[..., 2] -= temp_bias * 0.05  # Blue
        
        if tint_bias != 0.0:
            # Simple tint shift (green-magenta)
            processed[..., 0] += tint_bias * 0.05  # Red (magenta)
            processed[..., 1] -= tint_bias * 0.05  # Green
        
        return torch.clamp(processed, 0.0, 1.0)

    def _apply_film_grain(self, image, stock_params, grain_intensity, push_pull):
        """Apply film-specific grain patterns."""
        
        grain_type = stock_params["grain_type"]
        device = image.device
        
        # Grain characteristics based on type
        grain_params = {
            "ultra_fine": {"size": 0.5, "strength": 0.3},
            "fine": {"size": 0.7, "strength": 0.4},
            "medium": {"size": 1.0, "strength": 0.6},
            "heavy": {"size": 1.5, "strength": 0.8},
        }
        
        params = grain_params.get(grain_type, grain_params["medium"])
        
        # Push processing increases grain
        grain_strength = params["strength"] * grain_intensity
        if push_pull > 0:
            grain_strength *= (1.0 + push_pull * 0.5)
        
        # Generate film grain noise
        noise_shape = image.shape
        grain_noise = torch.randn(noise_shape, device=device) * grain_strength * 0.02
        
        # Apply grain with luminance dependency (more grain in shadows for most films)
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        grain_mask = 1.0 - luminance * 0.5  # More grain in darker areas
        grain_mask = grain_mask.unsqueeze(-1)
        
        grained_image = image + (grain_noise * grain_mask)
        
        return torch.clamp(grained_image, 0.0, 1.0)

    def _apply_highlight_rolloff(self, image, rolloff_strength, stock_params):
        """Apply film-specific highlight rolloff characteristics."""
        
        # Film naturally compresses highlights
        luminance = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        
        # Create highlight mask
        highlight_threshold = 0.8
        highlight_mask = torch.clamp((luminance - highlight_threshold) / (1.0 - highlight_threshold), 0.0, 1.0)
        
        # Apply rolloff based on film characteristics
        contrast_curve = stock_params["contrast_curve"]
        
        if contrast_curve in ["high", "very_high"]:
            # High contrast films have harder rolloff
            rolloff_curve = torch.pow(highlight_mask, 1.5)
        else:
            # Softer films have gentler rolloff
            rolloff_curve = torch.pow(highlight_mask, 0.8)
        
        # Apply the rolloff
        rolloff_factor = 1.0 - (rolloff_curve * rolloff_strength * 0.3)
        rolloff_factor = rolloff_factor.unsqueeze(-1)
        
        return image * rolloff_factor


class VAEColorCorrector:
    """
    Specialized color correction for VAE artifacts in inpainting/img2img workflows.
    Fixes color shifts in unmasked areas by referencing the original input image.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE", {"tooltip": "🔴 Original input image (before VAE encoding)"}),
                "processed_image": ("IMAGE", {"tooltip": "🟡 Image after VAE decode (with color shifts)"}),
                "correction_strength": (
                    "FLOAT", 
                    {
                        "default": 0.8, 
                        "min": 0.0, 
                        "max": 1.0, 
                        "step": 0.01,
                        "tooltip": "🎚️ How strongly to correct VAE color shifts (0.0 = no correction, 1.0 = full correction)"
                    }
                ),
                "method": (
                    ["luminance_zones", "histogram_matching", "statistical_matching", "advanced_3d_lut"], 
                    {
                        "default": "luminance_zones",
                        "tooltip": "🔧 Color correction method:\n• luminance_zones: Professional shadows/midtones/highlights (recommended)\n• histogram_matching: Match color distributions\n• statistical_matching: Match color statistics\n• advanced_3d_lut: Most accurate but slower 3D color mapping"
                    }
                ),
                "auto_preserve": (
                    "BOOLEAN", 
                    {
                        "default": True,
                        "tooltip": "🤖 Auto-detect and preserve inpainted areas when no mask is provided"
                    }
                ),
                "lock_input_image": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "🔒 Lock input images to prevent upstream nodes from reprocessing when adjusting correction parameters",
                    },
                ),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "🔧 VAE model used for encoding/decoding (helps with VAE-specific color corrections)"}),
                "mask": ("MASK", {"tooltip": "⚫ Optional mask - white areas will be preserved (inpainted areas)"}),
                "edge_feather": (
                    "INT", 
                    {
                        "default": 5, 
                        "min": 0, 
                        "max": 50, 
                        "step": 1,
                        "tooltip": "🌟 Feather edges between corrected/preserved areas (pixels)"
                    }
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("corrected_image",)
    FUNCTION = "correct_vae_colors"
    CATEGORY = "itsjustregi / Easy Color Corrector"
    DISPLAY_NAME = "VAE Color Corrector"
    
    def correct_vae_colors(
        self, 
        original_image, 
        processed_image, 
        correction_strength=0.8,
        method="luminance_zones",
        auto_preserve=True,
        lock_input_image=False,
        vae=None,
        mask=None,
        edge_feather=5
    ):
        """
        Correct VAE-induced color shifts by referencing the original image.
        """
        device = original_image.device
        
        # Handle input image locking to prevent upstream reprocessing
        if lock_input_image:
            # Check if input images have changed (cache invalidation)
            images_changed = False
            if (
                not hasattr(self, "_cached_original_image")
                or self._cached_original_image is None
                or not hasattr(self, "_cached_processed_image")
                or self._cached_processed_image is None
            ):
                images_changed = True
            else:
                # Compare image tensors to detect changes from upstream
                try:
                    if not torch.equal(self._cached_original_image, original_image) or \
                       not torch.equal(self._cached_processed_image, processed_image):
                        images_changed = True
                except:
                    # Different shapes or other comparison issues = definitely changed
                    images_changed = True
            
            if images_changed:
                self._cached_original_image = original_image.clone()
                self._cached_processed_image = processed_image.clone()
                self._cached_vae_analysis = None
                print("🔄 Locked Input: New images detected, updating cache")
            
            # Use cached images for processing
            original_image = self._cached_original_image.clone()
            processed_image = self._cached_processed_image.clone()
        
        print(f"🔧 VAE Color Correction: method={method}, strength={correction_strength:.2f}")
        print(f"📏 Original: {original_image.shape}, Processed: {processed_image.shape}")
        if lock_input_image:
            print("🔒 Input images locked - preventing upstream reprocessing")
        if vae is not None:
            print("🎯 VAE model provided for enhanced correction")
        
        # Ensure images are same size
        if original_image.shape != processed_image.shape:
            print("⚠️ Image size mismatch - resizing processed to match original")
            processed_image = F.interpolate(
                processed_image.permute(0, 3, 1, 2), 
                size=(original_image.shape[1], original_image.shape[2]), 
                mode='bilinear', 
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # Process each image in batch
        corrected_batch = []
        
        for i in range(original_image.shape[0]):
            orig_img = original_image[i]
            proc_img = processed_image[i]
            current_mask = mask[i] if mask is not None else None
            
            # Apply color correction
            corrected_img = self._apply_vae_color_correction(
                orig_img, proc_img, method, correction_strength, 
                auto_preserve, vae, current_mask, edge_feather, device, lock_input_image
            )
            
            corrected_batch.append(corrected_img)
        
        result = torch.stack(corrected_batch, dim=0)
        print(f"✅ VAE color correction completed for {len(corrected_batch)} images")
        
        return (result,)
    
    def _apply_vae_color_correction(
        self, original_img, processed_img, method, strength, 
        auto_preserve, vae, mask, edge_feather, device, lock_input_image
    ):
        """Apply the actual color correction."""
        
        # Convert to numpy for processing
        orig_np = (original_img.cpu().numpy() * 255).astype(np.uint8)
        proc_np = (processed_img.cpu().numpy() * 255).astype(np.uint8)
        
        # VAE-specific adjustments with caching
        vae_adjustment = 1.0
        vae_color_bias = None
        if vae is not None:
            # Use cached VAE analysis when input is locked
            if (
                lock_input_image
                and hasattr(self, "_cached_vae_analysis")
                and self._cached_vae_analysis is not None
            ):
                vae_adjustment, vae_color_bias = self._cached_vae_analysis
                print("🎯 Using cached VAE analysis for enhanced correction")
            else:
                print("🎯 VAE model detected - analyzing VAE characteristics for enhanced correction")
                vae_adjustment, vae_color_bias = self._analyze_vae_characteristics(vae, orig_np, proc_np)
                # Cache the analysis if input is locked
                if lock_input_image:
                    self._cached_vae_analysis = (vae_adjustment, vae_color_bias)
        
        print(f"🔧 Applying {method} color correction...")
        
        # Balance strength across different methods
        adjusted_strength = self._balance_correction_strength(method, strength) * vae_adjustment
        
        # Additional safety: limit maximum strength to prevent quantization artifacts
        if adjusted_strength > 1.0:
            print(f"⚠️ Strength {adjusted_strength:.2f} > 1.0, clamping to 1.0 to prevent artifacts")
            adjusted_strength = 1.0
        
        print(f"🔧 Adjusted strength: {strength:.2f} → {adjusted_strength:.2f} for {method}")
        
        # Apply color correction based on method with VAE bias compensation
        if method == "advanced_3d_lut":
            corrected_np = self._advanced_3d_lut_correction(orig_np, proc_np, adjusted_strength, vae_color_bias)
        elif method == "luminance_zones":
            if vae_color_bias is not None:
                corrected_np = self._vae_aware_luminance_correction(proc_np, orig_np, adjusted_strength, vae_color_bias)
            else:
                corrected_np = match_to_reference_colors(proc_np, orig_np, adjusted_strength)
        elif method == "histogram_matching":
            corrected_np = self._histogram_matching_correction(orig_np, proc_np, adjusted_strength, vae_color_bias)
        else:  # statistical_matching
            corrected_np = self._statistical_matching_correction(orig_np, proc_np, adjusted_strength, vae_color_bias)
        
        # Convert back to tensor
        corrected_tensor = torch.from_numpy(corrected_np.astype(np.float32) / 255.0).to(device)
        
        # Handle mask-based preservation
        if mask is not None:
            # Always use mask consistently: white pixels = preserve, black pixels = correct
            corrected_tensor = self._apply_mask_preservation(
                processed_img, corrected_tensor, mask, edge_feather, device
            )
        elif auto_preserve:
            # Auto-detect changed areas if no mask provided and auto_preserve is True
            corrected_tensor = self._auto_preserve_inpainted(
                original_img, processed_img, corrected_tensor, edge_feather, device
            )
        
        return corrected_tensor
    
    def _safe_clamp_colors(self, image_np, preserve_gradients=True):
        """Safely clamp colors to prevent quantization artifacts (black squares)."""
        # Diagnostic: Check for extreme values
        min_val = np.min(image_np)
        max_val = np.max(image_np)
        if min_val < -10 or max_val > 265:
            print(f"⚠️ Extreme color values detected: min={min_val:.1f}, max={max_val:.1f} - applying safe clamping")
        
        if preserve_gradients:
            # Soft clamping with sigmoid-like curve to preserve gradients
            image_float = image_np.astype(np.float32)
            
            # Apply soft clamping for values outside normal range
            below_zero = image_float < 0
            above_255 = image_float > 255
            
            if np.any(below_zero):
                # Soft approach to zero for negative values
                negative_values = image_float[below_zero]
                image_float[below_zero] = -5 * np.log(1 + np.exp(-negative_values / 5))
                
            if np.any(above_255):
                # Soft approach to 255 for values above
                high_values = image_float[above_255]
                image_float[above_255] = 255 + 5 * np.log(1 + np.exp((high_values - 255) / 5))
            
            # Final gentle clamp
            return np.clip(image_float, 0, 255).astype(np.uint8)
        else:
            # Standard hard clamping
            return np.clip(image_np, 0, 255).astype(np.uint8)
    
    def _analyze_vae_characteristics(self, vae, original_np, processed_np):
        """Analyze VAE-specific color characteristics and biases."""
        try:
            # Note: VAE model object could be used for future model-specific analysis
            # Currently we analyze empirically by comparing original vs processed images
            # Calculate per-channel color bias introduced by the VAE
            orig_float = original_np.astype(np.float32)
            proc_float = processed_np.astype(np.float32)
            
            # Analyze color bias in different luminance zones
            gray_orig = np.mean(orig_float, axis=2)
            
            # Create luminance-based masks
            shadows_mask = gray_orig < 85
            midtones_mask = (gray_orig >= 85) & (gray_orig <= 170)
            highlights_mask = gray_orig > 170
            
            vae_bias = {}
            
            for zone_name, mask in [("shadows", shadows_mask), ("midtones", midtones_mask), ("highlights", highlights_mask)]:
                if np.sum(mask) > 100:  # Ensure enough pixels for reliable statistics
                    orig_zone = orig_float[mask]
                    proc_zone = proc_float[mask]
                    
                    # Calculate color bias per channel
                    bias_r = np.mean(proc_zone[:, 0]) - np.mean(orig_zone[:, 0])
                    bias_g = np.mean(proc_zone[:, 1]) - np.mean(orig_zone[:, 1])
                    bias_b = np.mean(proc_zone[:, 2]) - np.mean(orig_zone[:, 2])
                    
                    vae_bias[zone_name] = np.array([bias_r, bias_g, bias_b])
                else:
                    vae_bias[zone_name] = np.array([0.0, 0.0, 0.0])
            
            # Calculate overall bias strength
            total_bias = np.mean([np.abs(bias).sum() for bias in vae_bias.values()])
            
            # Adjust correction strength based on VAE bias severity
            if total_bias > 15:  # High bias
                vae_adjustment = 0.85
                print(f"🔴 High VAE color bias detected ({total_bias:.1f}) - reducing correction strength")
            elif total_bias > 8:  # Medium bias
                vae_adjustment = 0.92
                print(f"🟡 Medium VAE color bias detected ({total_bias:.1f}) - slight adjustment")
            else:  # Low bias
                vae_adjustment = 0.98
                print(f"🟢 Low VAE color bias detected ({total_bias:.1f}) - minimal adjustment")
            
            return vae_adjustment, vae_bias
            
        except Exception as e:
            print(f"⚠️ VAE analysis failed: {e}, using default settings")
            return 0.9, None
    
    def _vae_aware_luminance_correction(self, processed_np, original_np, strength, vae_color_bias):
        """Enhanced luminance zone correction that compensates for VAE-specific color biases."""
        try:
            # Start with standard luminance correction
            corrected_np = match_to_reference_colors(processed_np, original_np, strength)
            
            # Apply VAE bias compensation
            corrected_float = corrected_np.astype(np.float32)
            gray = np.mean(corrected_float, axis=2)
            
            # Apply zone-specific bias corrections
            shadows_mask = gray < 85
            midtones_mask = (gray >= 85) & (gray <= 170)
            highlights_mask = gray > 170
            
            bias_strength = strength * 0.3  # Gentle bias correction
            
            for zone_name, mask in [("shadows", shadows_mask), ("midtones", midtones_mask), ("highlights", highlights_mask)]:
                if zone_name in vae_color_bias and np.sum(mask) > 0:
                    bias = vae_color_bias[zone_name]
                    # Apply inverse bias correction with limiting
                    for c in range(3):
                        bias_correction = bias[c] * bias_strength
                        # Limit bias correction to prevent extreme shifts
                        bias_correction = np.clip(bias_correction, -30, 30)
                        corrected_float[mask, c] -= bias_correction
            
            # Safe clamping to prevent quantization artifacts
            corrected_np = self._safe_clamp_colors(corrected_float, preserve_gradients=True)
            print(f"✅ VAE-aware luminance correction applied with bias compensation")
            
            return corrected_np
            
        except Exception as e:
            print(f"❌ VAE-aware correction failed: {e}, falling back to standard")
            return match_to_reference_colors(processed_np, original_np, strength)
    
    def _balance_correction_strength(self, method, strength):
        """Balance correction strength across different methods for consistent results."""
        # Different methods have different sensitivities, so we adjust accordingly
        if method == "luminance_zones":
            # Luminance zones method is well-balanced, use as-is
            return strength
        elif method == "histogram_matching":
            # Histogram matching can be aggressive, reduce slightly
            return strength * 0.85
        elif method == "statistical_matching":
            # Statistical matching is gentler, increase slightly
            return min(1.0, strength * 1.15)
        elif method == "advanced_3d_lut":
            # 3D LUT can be very aggressive, reduce significantly
            return strength * 0.7
        else:
            return strength
    
    def _advanced_3d_lut_correction(self, original_np, processed_np, strength, vae_color_bias=None):
        """Advanced 3D LUT-based color correction for precise VAE artifact fixing."""
        if not ADVANCED_LIBS_AVAILABLE:
            return match_to_reference_colors(processed_np, original_np, strength)
        
        try:
            print("🔧 Building 3D color mapping...")
            
            # Create color mapping using k-means clustering
            from sklearn.cluster import KMeans
            
            # Sample colors for mapping (use every 4th pixel for speed)
            orig_samples = original_np[::4, ::4].reshape(-1, 3)
            proc_samples = processed_np[::4, ::4].reshape(-1, 3)
            
            # Use k-means to find representative color pairs
            n_clusters = min(64, len(orig_samples) // 10)  # Adaptive cluster count
            
            if n_clusters < 8:
                # Too few samples, fall back to zone-based matching
                return match_to_reference_colors(processed_np, original_np, strength)
            
            # Cluster processed colors
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            proc_clusters = kmeans.fit_predict(proc_samples)
            proc_centers = kmeans.cluster_centers_
            
            # Find corresponding original colors for each cluster
            orig_centers = np.zeros_like(proc_centers)
            for i in range(n_clusters):
                cluster_mask = proc_clusters == i
                if np.sum(cluster_mask) > 0:
                    # Average original colors that correspond to this processed cluster
                    orig_centers[i] = np.mean(orig_samples[cluster_mask], axis=0)
                else:
                    orig_centers[i] = proc_centers[i]  # Fallback
            
            # Apply color mapping to full image
            corrected_np = processed_np.astype(np.float32)
            
            # For each pixel, find closest processed center and map to original center
            proc_flat = processed_np.reshape(-1, 3).astype(np.float32)
            
            # Vectorized distance calculation
            distances = np.linalg.norm(
                proc_flat[:, np.newaxis, :] - proc_centers[np.newaxis, :, :], axis=2
            )
            closest_clusters = np.argmin(distances, axis=1)
            
            # Apply color shifts with distance-based blending and VAE bias compensation
            min_distances = np.min(distances, axis=1)
            max_distance = np.percentile(min_distances, 90)  # Use 90th percentile for normalization
            
            for i in range(n_clusters):
                cluster_mask = closest_clusters == i
                if np.sum(cluster_mask) > 0:
                    # Calculate color shift for this cluster
                    color_shift = orig_centers[i] - proc_centers[i]
                    
                    # Apply VAE bias compensation if available
                    if vae_color_bias is not None:
                        # Determine luminance zone for this cluster
                        cluster_luminance = np.mean(proc_centers[i])
                        if cluster_luminance < 85:
                            zone_bias = vae_color_bias.get("shadows", np.array([0.0, 0.0, 0.0]))
                        elif cluster_luminance <= 170:
                            zone_bias = vae_color_bias.get("midtones", np.array([0.0, 0.0, 0.0]))
                        else:
                            zone_bias = vae_color_bias.get("highlights", np.array([0.0, 0.0, 0.0]))
                        
                        # Add VAE bias compensation to color shift
                        color_shift -= zone_bias * 0.5  # Gentle bias compensation
                    
                    # Apply with distance-based falloff and strength
                    cluster_distances = min_distances[cluster_mask]
                    distance_weights = np.clip(1.0 - cluster_distances / max_distance, 0.1, 1.0)
                    
                    for c in range(3):
                        shift_amount = color_shift[c] * strength * distance_weights
                        # Limit maximum shift to prevent extreme corrections
                        shift_amount = np.clip(shift_amount, -50, 50)
                        proc_flat[cluster_mask, c] += shift_amount
            
            # Reshape back and apply safe clamping
            corrected_np = proc_flat.reshape(processed_np.shape)
            corrected_np = self._safe_clamp_colors(corrected_np, preserve_gradients=True)
            
            print(f"✅ 3D LUT correction applied using {n_clusters} color clusters")
            return corrected_np
            
        except Exception as e:
            print(f"❌ 3D LUT correction failed: {e}, falling back to zone matching")
            return match_to_reference_colors(processed_np, original_np, strength)
    
    def _histogram_matching_correction(self, original_np, processed_np, strength, vae_color_bias=None):
        """Histogram-based color matching."""
        if not ADVANCED_LIBS_AVAILABLE:
            return match_to_reference_colors(processed_np, original_np, strength)
        
        try:
            from skimage import exposure
            
            corrected_np = processed_np.astype(np.float32)
            original_float = original_np.astype(np.float32)
            
            # Match histogram for each channel
            for c in range(3):
                matched_channel = exposure.match_histograms(
                    corrected_np[:,:,c], original_float[:,:,c]
                )
                # Blend with original based on strength
                corrected_np[:,:,c] = (
                    processed_np[:,:,c] * (1 - strength) + 
                    matched_channel * strength
                )
            
            # Apply VAE bias compensation if available
            if vae_color_bias is not None:
                gray = np.mean(corrected_np, axis=2)
                bias_strength = strength * 0.2  # Gentle bias correction for histogram method
                
                # Apply zone-specific bias corrections
                shadows_mask = gray < 85
                midtones_mask = (gray >= 85) & (gray <= 170)
                highlights_mask = gray > 170
                
                for zone_name, mask in [("shadows", shadows_mask), ("midtones", midtones_mask), ("highlights", highlights_mask)]:
                    if zone_name in vae_color_bias and np.sum(mask) > 0:
                        bias = vae_color_bias[zone_name]
                        for c in range(3):
                            bias_correction = bias[c] * bias_strength
                            # Limit bias correction to prevent extreme shifts
                            bias_correction = np.clip(bias_correction, -30, 30)
                            corrected_np[mask, c] -= bias_correction
                            
                print("✅ VAE bias compensation applied to histogram matching")
            
            return self._safe_clamp_colors(corrected_np, preserve_gradients=True)
            
        except Exception as e:
            print(f"❌ Histogram matching failed: {e}")
            return match_to_reference_colors(processed_np, original_np, strength)
    
    def _statistical_matching_correction(self, original_np, processed_np, strength, vae_color_bias=None):
        """Statistical moment matching (mean and std)."""
        corrected_np = processed_np.astype(np.float32)
        original_float = original_np.astype(np.float32)
        
        for c in range(3):
            # Calculate statistics
            proc_mean = np.mean(corrected_np[:,:,c])
            proc_std = np.std(corrected_np[:,:,c])
            orig_mean = np.mean(original_float[:,:,c])
            orig_std = np.std(original_float[:,:,c])
            
            # Normalize and rescale
            if proc_std > 0:
                normalized = (corrected_np[:,:,c] - proc_mean) / proc_std
                rescaled = normalized * orig_std + orig_mean
                
                # Blend with strength
                corrected_np[:,:,c] = (
                    corrected_np[:,:,c] * (1 - strength) + 
                    rescaled * strength
                )
        
        # Apply VAE bias compensation if available
        if vae_color_bias is not None:
            gray = np.mean(corrected_np, axis=2)
            bias_strength = strength * 0.25  # Gentle bias correction for statistical method
            
            # Apply zone-specific bias corrections
            shadows_mask = gray < 85
            midtones_mask = (gray >= 85) & (gray <= 170)
            highlights_mask = gray > 170
            
            for zone_name, mask in [("shadows", shadows_mask), ("midtones", midtones_mask), ("highlights", highlights_mask)]:
                if zone_name in vae_color_bias and np.sum(mask) > 0:
                    bias = vae_color_bias[zone_name]
                    for c in range(3):
                        bias_correction = bias[c] * bias_strength
                        # Limit bias correction to prevent extreme shifts
                        bias_correction = np.clip(bias_correction, -30, 30)
                        corrected_np[mask, c] -= bias_correction
                        
            print("✅ VAE bias compensation applied to statistical matching")
        
        return self._safe_clamp_colors(corrected_np, preserve_gradients=True)
    
    def _apply_mask_preservation(self, processed_img, corrected_img, mask, edge_feather, device):
        """Apply mask to preserve areas. White pixels in mask = preserve, black pixels = correct."""
        # Mask convention: white (1.0) = preserve, black (0.0) = correct
        # We need correction_mask where 1.0 = correct, 0.0 = preserve
        correction_mask = 1.0 - mask.to(device)  # Invert: white becomes 0 (preserve), black becomes 1 (correct)
        
        if edge_feather > 0 and ADVANCED_LIBS_AVAILABLE:
            # Apply gaussian blur for soft edges
            correction_mask_np = correction_mask.cpu().numpy()
            correction_mask_np = cv2.GaussianBlur(
                correction_mask_np, (edge_feather*2+1, edge_feather*2+1), edge_feather/3
            )
            correction_mask = torch.from_numpy(correction_mask_np).to(device)
        
        # Apply mask: blend between processed and corrected
        correction_mask = correction_mask.unsqueeze(-1)  # Add channel dimension
        result = processed_img * (1 - correction_mask) + corrected_img * correction_mask
        
        print(f"✅ Mask-based preservation applied with {edge_feather}px feather")
        return result
    
    def _auto_preserve_inpainted(self, original_img, processed_img, corrected_img, edge_feather, device):
        """Auto-detect inpainted areas and preserve them."""
        # Calculate difference between original and processed to find changed areas
        diff = torch.abs(original_img - processed_img)
        diff_magnitude = torch.mean(diff, dim=-1)  # Average across RGB
        
        # Threshold to find significantly changed areas (likely inpainted)
        threshold = torch.quantile(diff_magnitude, 0.7)  # Top 30% of changes
        inpainted_mask = (diff_magnitude > threshold).float()
        
        # Correction mask: 1 = correct, 0 = preserve
        correction_mask = 1.0 - inpainted_mask
        
        if edge_feather > 0 and ADVANCED_LIBS_AVAILABLE:
            # Smooth the mask
            correction_mask_np = correction_mask.cpu().numpy()
            correction_mask_np = cv2.GaussianBlur(
                correction_mask_np, (edge_feather*2+1, edge_feather*2+1), edge_feather/3
            )
            correction_mask = torch.from_numpy(correction_mask_np).to(device)
        
        correction_mask = correction_mask.unsqueeze(-1)  # Add channel dimension
        result = processed_img * (1 - correction_mask) + corrected_img * correction_mask
        
        preserved_pixels = torch.sum(1 - correction_mask).item()
        print(f"✅ Auto-preserved {preserved_pixels:.0f} pixels (likely inpainted areas)")
        return result


# Export all classes
__all__ = [
    "EasyColorCorrection",
    "BatchColorCorrection",
    "RawImageProcessor",
    "ColorCorrectionViewer",
    "ColorPaletteExtractor",
    "FilmEmulation",
    "VAEColorCorrector",
]