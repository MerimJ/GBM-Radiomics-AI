"""
Shared utilities for the CFB-GBM radiomics pipeline.
"""

from __future__ import annotations

import logging
import json
import time
import yaml
from pathlib import Path
from functools import wraps
from typing import Any

import numpy as np
import SimpleITK as sitk


# ── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                                datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── Config ───────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Path helpers ─────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def patient_nifti_dir(nifti_root: Path, patient_id: str,
                      timepoint: str = "baseline") -> Path:
    return ensure_dir(nifti_root / patient_id / timepoint)


# ── SimpleITK helpers ─────────────────────────────────────────────────────────

def load_nifti(path: str | Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def save_nifti(image: sitk.Image, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    sitk.WriteImage(image, str(path))


def resample_image(image: sitk.Image,
                   new_spacing: list[float],
                   interpolator=sitk.sitkBSpline,
                   default_pixel_value: float = 0.0) -> sitk.Image:
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def resample_mask_to_image(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(mask)


def image_to_array(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image)


def array_to_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    img = sitk.GetImageFromArray(array)
    img.CopyInformation(reference)
    return img


# ── Timing decorator ─────────────────────────────────────────────────────────

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        logging.getLogger(__name__).info(f"{func.__name__} completed in {elapsed:.1f}s")
        return result
    return wrapper


# ── JSON sidecar ─────────────────────────────────────────────────────────────

def write_json_sidecar(data: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json_sidecar(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Series classification ─────────────────────────────────────────────────────

def classify_series(series_description: str, config: dict) -> str:
    """Return sequence type label from SeriesDescription string."""
    desc = (series_description or "").lower()
    keywords = config.get("sequence_keywords", {})
    for seq_name, kw_list in keywords.items():
        if any(kw.lower() in desc for kw in kw_list):
            return seq_name
    return "unknown"
