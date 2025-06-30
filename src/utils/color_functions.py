"""Color space conversion and color manipulation functions."""

import torch
import numpy as np
from .imports import ADVANCED_LIBS_AVAILABLE

if ADVANCED_LIBS_AVAILABLE:
    import cv2


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
    
    hsv = rgb_to_hsv(image)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    # Create mask for target color range
    mask = create_color_range_mask(hsv, target_hue, hue_range)
    mask = mask.unsqueeze(-1)  # Add channel dimension for broadcasting
    
    # Apply adjustments
    if abs(hue_adjustment) > 0.001:
        h_adjusted = (h + hue_adjustment / 360.0) % 1.0
        h = torch.lerp(h, h_adjusted, mask.squeeze(-1))
    
    if abs(saturation_adjustment) > 0.001:
        s_adjusted = torch.clamp(s + saturation_adjustment, 0.0, 1.0)
        s = torch.lerp(s, s_adjusted, mask.squeeze(-1))
    
    if abs(lightness_adjustment) > 0.001:
        v_adjusted = torch.clamp(v + lightness_adjustment, 0.0, 1.0)
        v = torch.lerp(v, v_adjusted, mask.squeeze(-1))
    
    # Convert back to RGB
    adjusted_hsv = torch.stack([h, s, v], dim=-1)
    return hsv_to_rgb(adjusted_hsv)


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
        abs(foliage_adjustment) < 0.001 and abs(selective_hue_shift) < 0.001):
        return result
    
    # Convert to HSV for adjustments
    hsv = rgb_to_hsv(result)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    # Skin tone adjustments (orange-red hues, ~15-45 degrees)
    if abs(skin_tone_adjustment) > 0.001:
        result = apply_selective_color_adjustment(
            result, 30.0, 0.0, skin_tone_adjustment * 0.3, skin_tone_adjustment * 0.1, 40.0
        )
    
    # Sky adjustments (blue hues, ~200-240 degrees) 
    if abs(sky_adjustment) > 0.001:
        result = apply_selective_color_adjustment(
            result, 220.0, 0.0, sky_adjustment * 0.4, 0.0, 50.0
        )
    
    # Foliage adjustments (green hues, ~80-140 degrees)
    if abs(foliage_adjustment) > 0.001:
        result = apply_selective_color_adjustment(
            result, 110.0, 0.0, foliage_adjustment * 0.3, foliage_adjustment * 0.05, 60.0
        )
    
    # Custom selective adjustments
    if abs(selective_hue_shift) > 0.001 or abs(selective_saturation) > 0.001:
        # Apply to mid-range colors (avoiding extreme darks/lights)
        hsv_current = rgb_to_hsv(result)
        h_curr, s_curr, v_curr = hsv_current[..., 0], hsv_current[..., 1], hsv_current[..., 2]
        
        # Create mask for mid-tones with decent saturation
        mask = ((v_curr > 0.2) & (v_curr < 0.8) & (s_curr > 0.1)).float()
        mask = mask.unsqueeze(-1)
        
        if abs(selective_hue_shift) > 0.001:
            h_adjusted = (h_curr + selective_hue_shift / 360.0) % 1.0
            h_curr = torch.lerp(h_curr, h_adjusted, mask.squeeze(-1) * selective_strength)
        
        if abs(selective_saturation) > 0.001:
            s_adjusted = torch.clamp(s_curr + selective_saturation, 0.0, 1.0)
            s_curr = torch.lerp(s_curr, s_adjusted, mask.squeeze(-1) * selective_strength)
        
        adjusted_hsv = torch.stack([h_curr, s_curr, v_curr], dim=-1)
        result = hsv_to_rgb(adjusted_hsv)
    
    return result