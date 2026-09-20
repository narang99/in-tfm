"""DICOM -> model-ready image conversion, and undoing the processor's normalization for
display (as opposed to model input).
"""

from pathlib import Path

import numpy as np
import pydicom
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor


def dicom_to_uint8(ds: pydicom.Dataset, use_windowing: bool = True) -> UInt8[np.ndarray, "h w"]:
    arr = ds.pixel_array.astype(np.float64)

    # Apply Rescale Slope/Intercept (raw -> real-world values, e.g. HU for CT)
    slope = getattr(ds, "RescaleSlope", 1)
    if slope is None:
        slope = 1.0
    slope = float(slope)
    intercept = getattr(ds, "RescaleIntercept", 0)
    if intercept is None:
        intercept = 0.0
    intercept = float(intercept)
    arr = arr * slope + intercept

    if use_windowing and hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        center = ds.WindowCenter
        width = ds.WindowWidth
        # These can be multi-valued (DS lists); take the first value
        center = float(center[0]) if isinstance(center, pydicom.multival.MultiValue) else float(center)
        width = float(width[0]) if isinstance(width, pydicom.multival.MultiValue) else float(width)

        low = center - width / 2.0
        high = center + width / 2.0
        arr = np.clip(arr, low, high)
        arr = (arr - low) / (high - low)  # 0-1
    else:
        # Fallback: simple min-max normalization over the whole image
        arr_min = arr.min()
        arr_max = arr.max()
        arr = (arr - arr_min) / (arr_max - arr_min) if arr_max > arr_min else np.zeros_like(arr)

    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # Handle MONOCHROME1 (inverted grayscale, 0=white)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255 - arr

    return arr


def get_batch(processor: BaseImageProcessor, dcm_paths: list[Path]) -> BatchFeature:
    images = [Image.fromarray(dicom_to_uint8(pydicom.dcmread(d)), mode="L") for d in dcm_paths]
    return processor(images, return_tensors="pt")


def inv_tfm(
    processor: BaseImageProcessor, tensor: Float[torch.Tensor, "*batch c h w"]
) -> UInt8[np.ndarray, "h w c"]:
    """Undo the image processor's mean/std normalization, for display (not model input)."""
    mean = np.array(processor.image_mean)  # e.g. [0.485, 0.456, 0.406]
    std = np.array(processor.image_std)  # e.g. [0.229, 0.224, 0.225]
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]  # drop batch dim -> (C, H, W)

    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = img * std + mean  # undo normalization -> back to [0,1]
    img = np.clip(img, 0, 1)
    return (img * 255).round().astype(np.uint8)
