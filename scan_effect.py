"""
CamScanner-style scan effect pipeline.
Applied in this exact order:
1. Warm paper tone: blend with #FFFBEB at 12% opacity
2. Gaussian noise: std=6
3. Random slight rotation: -0.8 to +0.8 degrees
4. Vignette: darken edges 15%
5. Uneven brightness: right side 7% darker
6. Slight blur: GaussianBlur radius=0.5
7. Contrast boost: factor=1.12
"""

import base64
import io
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


WARM_COLOR = (255, 251, 235)  # #FFFBEB
WARM_OPACITY = 0.12
NOISE_STD = 6
MAX_ROTATION = 0.8
VIGNETTE_STRENGTH = 0.15
BRIGHTNESS_TAPER = 0.07
BLUR_RADIUS = 0.5
CONTRAST_FACTOR = 1.12


def _b64_to_pil(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _apply_warm_tone(img: np.ndarray) -> np.ndarray:
    """Blend with warm paper color at 12% opacity."""
    warm = np.array(WARM_COLOR, dtype=np.float32)
    result = img.astype(np.float32)
    result = result * (1 - WARM_OPACITY) + warm * WARM_OPACITY
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_noise(img: np.ndarray) -> np.ndarray:
    """Add Gaussian noise with std=6 to all channels."""
    noise = np.random.normal(0, NOISE_STD, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_rotation(img: Image.Image) -> Image.Image:
    """Rotate by random small angle and crop to original size."""
    angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
    if abs(angle) < 0.01:
        return img
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    return rotated


def _apply_vignette(img: np.ndarray) -> np.ndarray:
    """Darken edges by 15%, stronger in corners."""
    h, w = img.shape[:2]
    # Create vignette mask using Gaussian kernel
    sigma_x = w * 0.6
    sigma_y = h * 0.6

    x = np.arange(w) - w / 2
    y = np.arange(h) - h / 2
    X, Y = np.meshgrid(x, y)

    vignette = np.exp(-(X ** 2 / (2 * sigma_x ** 2) + Y ** 2 / (2 * sigma_y ** 2)))
    # Normalize: center = 1, edges = 1 - VIGNETTE_STRENGTH
    vignette = 1 - VIGNETTE_STRENGTH * (1 - vignette)
    vignette = np.clip(vignette, 0, 1)

    result = img.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= vignette
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_uneven_brightness(img: np.ndarray) -> np.ndarray:
    """Make right side 7% darker (simulates lighting angle)."""
    h, w = img.shape[:2]
    # Linear gradient: left=1.0, right=(1-0.07)
    gradient = np.linspace(1.0, 1.0 - BRIGHTNESS_TAPER, w, dtype=np.float32)
    gradient = gradient[np.newaxis, :, np.newaxis]  # (1, w, 1)

    result = img.astype(np.float32) * gradient
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_blur(img: Image.Image) -> Image.Image:
    """Apply slight Gaussian blur with radius=0.5."""
    return img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))


def _apply_contrast(img: Image.Image) -> Image.Image:
    """Boost contrast by factor 1.12."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(CONTRAST_FACTOR)


def apply_scan_effect(page_b64: str) -> str:
    """
    Apply the full CamScanner-style scan effect pipeline to a page image.
    Input/output: base64 PNG strings.
    """
    # Load image
    pil_img = _b64_to_pil(page_b64)

    # Step 1: Warm paper tone (PIL → numpy)
    arr = np.array(pil_img)
    arr = _apply_warm_tone(arr)

    # Step 2: Gaussian noise
    arr = _apply_noise(arr)

    # Back to PIL for rotation
    pil_img = Image.fromarray(arr)

    # Step 3: Random slight rotation
    pil_img = _apply_rotation(pil_img)

    # Step 4: Vignette (PIL → numpy)
    arr = np.array(pil_img)
    arr = _apply_vignette(arr)

    # Step 5: Uneven brightness
    arr = _apply_uneven_brightness(arr)

    # Back to PIL for blur and contrast
    pil_img = Image.fromarray(arr)

    # Step 6: Slight blur
    pil_img = _apply_blur(pil_img)

    # Step 7: Contrast boost
    pil_img = _apply_contrast(pil_img)

    return _pil_to_b64(pil_img)


def apply_scan_effect_all(pages_b64: list[str]) -> list[str]:
    """Apply scan effect to all pages."""
    return [apply_scan_effect(page_b64) for page_b64 in pages_b64]


def pages_to_pdf(pages_b64: list[str]) -> bytes:
    """
    Convert a list of base64 PNG page images into a single PDF bytes object.
    Uses reportlab for PDF creation.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage
    from reportlab.lib.units import mm
    import tempfile
    import os

    buf = io.BytesIO()

    # We'll use PIL + reportlab directly
    # Create a multi-page PDF by combining images
    pil_images = []
    for b64 in pages_b64:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        pil_images.append(img)

    if not pil_images:
        return b""

    # Save as PDF using PIL (append mode for multi-page)
    pdf_buf = io.BytesIO()
    pil_images[0].save(
        pdf_buf,
        format="PDF",
        save_all=True,
        append_images=pil_images[1:],
        resolution=150,
    )
    return pdf_buf.getvalue()
