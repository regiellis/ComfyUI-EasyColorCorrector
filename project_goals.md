# ComfyUI-EasyColorCorrection Specification

## Overview

**ComfyUI-EasyColorCorrection** is a ComfyUI custom node designed for flexible and efficient image color correction and post-processing within AI image generation workflows. The node leverages PyTorch for GPU-accelerated operations and supports both global and localized (masked) adjustments. It is intended to simplify color grading and correction for users working in ComfyUI, especially those focusing on structured prompts and style profiles[2].

---

## Current Features

### 1. Global Tonal Adjustments

- **Contrast:** Adjusts the difference between light and dark areas.
- **Gamma:** Controls mid-tone brightness.
- **Saturation:** Enhances or subdues image vibrancy.

### 2. Color Adjustments

- **Hue Rotation:** Rotates the color spectrum.
- **RGB Channel Offsets:** Allows individual adjustments to Red, Green, and Blue channels for precise color grading.

### 3. Creative Effects

- **Color Gel:** Applies a colored tint with adjustable strength. The gel color can be specified using hex codes (e.g., `#RRGGBB`) or RGB comma-separated values (e.g., `R,G,B`).

### 4. Sharpness

- **Sharpness:** Adjusts overall image sharpness using a convolution-based approach.

### 5. Black & White Conversion

- **Grayscale:** Converts the image to black and white.

### 6. Film Grain

- **Grain Strength:** Controls the intensity of added film grain.
- **Grain Contrast:** Adjusts the contrast of the grain.
- **Color Grain Mix:** Blends between monochromatic and colored grain.

### 7. Masking

- **Use Mask:** Applies adjustments only to the white areas of the mask.
- **Invert Mask Influence:** Inverts the mask effect, applying adjustments to black areas.
- **Mask Influence:** Controls the strength of the mask’s effect on adjustments (0–100%).

---

## Usage

- **Input:** Image tensor (B, H, W, C).
- **Parameters:** Adjustable via node UI for all features.
- **Output:** Adjusted image tensor and mask tensor (for masking operations).
- **Compatibility:** Works seamlessly with other ComfyUI nodes.

---

## Proposed New Features

### 1. Automatic Color Correction Based on Input Image

- **Description:** Analyzes the input image (e.g., using histograms, dominant color, or perceptual analysis) and suggests or applies adjustments to achieve a balanced or stylized look.
- **Implementation:** Add a new function to analyze the image and auto-adjust parameters, then integrate with the existing adjustment pipeline.

### 2. Genre-Based Presets as Overrides

- **Description:** Predefined parameter sets for popular genres/styles (e.g., “anime,” “film noir,” “vintage,” “digital art”) that can override manual settings.
- **Implementation:** Add a dropdown or toggle for preset selection, mapping each genre to a set of adjustment values. Override manual settings when a preset is active.

---

## Feature Comparison Table

| Feature                        | Current Status | Proposed Status                |
|---------------------------------|----------------|-------------------------------|
| Manual color correction         | Yes            | Yes (enhanced by presets)     |
| Automatic correction (by image) | No             | Yes (new feature)             |
| Genre/style presets             | No             | Yes (new feature)             |

---

## User Workflow Example

1. **Load Image:** Connect image tensor to the node.
2. **Adjust Parameters:** Manually set contrast, gamma, saturation, etc., or select a genre preset.
3. **Automatic Correction:** (Optional) Enable automatic analysis to apply suggested adjustments.
4. **Apply Mask:** (Optional) Use a mask for localized adjustments.
5. **Output:** Receive the adjusted image and mask tensors for further processing.

---

## Technical Notes

- **Backend:** PyTorch for GPU acceleration and tensor operations.
- **Compatibility:** Designed for ComfyUI, supports batch processing.
- **Extensibility:** New features can be added as modular functions within the existing codebase.

---

*Project: ComfyUI-EasyColorCorrection*  
*Version: 1.1.0 (example version, update as needed)*

---
