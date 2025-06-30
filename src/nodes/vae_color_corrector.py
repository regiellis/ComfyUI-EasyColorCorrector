"""VAE Color Corrector node for fixing VAE-induced color shifts in inpainting workflows."""

import torch
import torch.nn.functional as F
import numpy as np

from ..utils import ADVANCED_LIBS_AVAILABLE, match_to_reference_colors

if ADVANCED_LIBS_AVAILABLE:
    import cv2
    from sklearn.cluster import KMeans
    from skimage import exposure


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
                        "default": "advanced_3d_lut",
                        "tooltip": "🔧 Color correction method:\n• luminance_zones: Professional shadows/midtones/highlights\n• histogram_matching: Match color distributions\n• statistical_matching: Match color statistics\n• advanced_3d_lut: Most accurate 3D color mapping"
                    }
                ),
                "preserve_inpainted": (
                    "BOOLEAN", 
                    {
                        "default": True,
                        "tooltip": "🎨 Only correct areas that existed in original (preserve new inpainted content)"
                    }
                ),
            },
            "optional": {
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
    CATEGORY = "EasyColorCorrection"
    
    def correct_vae_colors(
        self, 
        original_image, 
        processed_image, 
        correction_strength=0.8,
        method="advanced_3d_lut",
        preserve_inpainted=True,
        mask=None,
        edge_feather=5
    ):
        """
        Correct VAE-induced color shifts by referencing the original image.
        """
        device = original_image.device
        
        print(f"🔧 VAE Color Correction: method={method}, strength={correction_strength:.2f}")
        print(f"📏 Original: {original_image.shape}, Processed: {processed_image.shape}")
        
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
                preserve_inpainted, current_mask, edge_feather, device
            )
            
            corrected_batch.append(corrected_img)
        
        result = torch.stack(corrected_batch, dim=0)
        print(f"✅ VAE color correction completed for {len(corrected_batch)} images")
        
        return (result,)
    
    def _apply_vae_color_correction(
        self, original_img, processed_img, method, strength, 
        preserve_inpainted, mask, edge_feather, device
    ):
        """Apply the actual color correction."""
        
        # Convert to numpy for processing
        orig_np = (original_img.cpu().numpy() * 255).astype(np.uint8)
        proc_np = (processed_img.cpu().numpy() * 255).astype(np.uint8)
        
        print(f"🔧 Applying {method} color correction...")
        
        # Apply color correction based on method
        if method == "advanced_3d_lut":
            corrected_np = self._advanced_3d_lut_correction(orig_np, proc_np, strength)
        elif method == "luminance_zones":
            corrected_np = match_to_reference_colors(proc_np, orig_np, strength)
        elif method == "histogram_matching":
            corrected_np = self._histogram_matching_correction(orig_np, proc_np, strength)
        else:  # statistical_matching
            corrected_np = self._statistical_matching_correction(orig_np, proc_np, strength)
        
        # Convert back to tensor
        corrected_tensor = torch.from_numpy(corrected_np.astype(np.float32) / 255.0).to(device)
        
        # Handle mask-based preservation
        if preserve_inpainted and mask is not None:
            corrected_tensor = self._apply_mask_preservation(
                processed_img, corrected_tensor, mask, edge_feather, device
            )
        elif preserve_inpainted:
            # Auto-detect changed areas if no mask provided
            corrected_tensor = self._auto_preserve_inpainted(
                original_img, processed_img, corrected_tensor, edge_feather, device
            )
        
        return corrected_tensor
    
    def _advanced_3d_lut_correction(self, original_np, processed_np, strength):
        """Advanced 3D LUT-based color correction for precise VAE artifact fixing."""
        if not ADVANCED_LIBS_AVAILABLE:
            return match_to_reference_colors(processed_np, original_np, strength)
        
        try:
            print("🔧 Building 3D color mapping...")
            
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
            proc_flat = processed_np.reshape(-1, 3).astype(np.float32)
            
            # Vectorized distance calculation
            distances = np.linalg.norm(
                proc_flat[:, np.newaxis, :] - proc_centers[np.newaxis, :, :], axis=2
            )
            closest_clusters = np.argmin(distances, axis=1)
            
            # Apply color shifts with distance-based blending
            min_distances = np.min(distances, axis=1)
            max_distance = np.percentile(min_distances, 90)  # Use 90th percentile for normalization
            
            for i in range(n_clusters):
                cluster_mask = closest_clusters == i
                if np.sum(cluster_mask) > 0:
                    # Calculate color shift for this cluster
                    color_shift = orig_centers[i] - proc_centers[i]
                    
                    # Apply with distance-based falloff and strength
                    cluster_distances = min_distances[cluster_mask]
                    distance_weights = np.clip(1.0 - cluster_distances / max_distance, 0.1, 1.0)
                    
                    for c in range(3):
                        shift_amount = color_shift[c] * strength * distance_weights
                        proc_flat[cluster_mask, c] += shift_amount
            
            # Reshape back and clamp
            corrected_np = proc_flat.reshape(processed_np.shape)
            corrected_np = np.clip(corrected_np, 0, 255).astype(np.uint8)
            
            print(f"✅ 3D LUT correction applied using {n_clusters} color clusters")
            return corrected_np
            
        except Exception as e:
            print(f"❌ 3D LUT correction failed: {e}, falling back to zone matching")
            return match_to_reference_colors(processed_np, original_np, strength)
    
    def _histogram_matching_correction(self, original_np, processed_np, strength):
        """Histogram-based color matching."""
        if not ADVANCED_LIBS_AVAILABLE:
            return match_to_reference_colors(processed_np, original_np, strength)
        
        try:
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
            
            return np.clip(corrected_np, 0, 255).astype(np.uint8)
            
        except Exception as e:
            print(f"❌ Histogram matching failed: {e}")
            return match_to_reference_colors(processed_np, original_np, strength)
    
    def _statistical_matching_correction(self, original_np, processed_np, strength):
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
        
        return np.clip(corrected_np, 0, 255).astype(np.uint8)
    
    def _apply_mask_preservation(self, processed_img, corrected_img, mask, edge_feather, device):
        """Apply mask to preserve inpainted areas."""
        # Invert mask: 0 = preserve (inpainted), 1 = correct (original areas)
        correction_mask = 1.0 - mask.to(device)
        
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