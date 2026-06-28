"""
preprocess_mri.py — Step 5: Resample, register, normalise, and bias-correct MRI.

For each patient / timepoint:
  1. Load T1Gd (and FLAIR, T1, T2 if available).
  2. Resample to target isotropic spacing (default 1×1×1 mm).
  3. N4 bias field correction (optional).
  4. Register FLAIR to T1Gd space (if enabled).
  5. Z-score normalise inside non-zero voxels.
  6. Resample tumor mask to T1Gd space.
  7. Save preprocessed images.

Output:
    data/processed/{patient_id}/{timepoint}/t1gd.nii.gz
    data/processed/{patient_id}/{timepoint}/flair.nii.gz
    data/processed/{patient_id}/{timepoint}/tumor_mask.nii.gz

Usage:
    python src/preprocess_mri.py \
        --images data/nifti \
        --masks  data/masks \
        --output data/processed
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (get_logger, ensure_dir, load_config,
                   resample_image, resample_mask_to_image,
                   save_nifti, load_nifti)

logger = get_logger(__name__)

SEQUENCES = ["t1gd", "flair", "t1", "t2", "dwi", "adc"]


# ── Processing functions ──────────────────────────────────────────────────────

def n4_bias_correction(image: sitk.Image) -> sitk.Image:
    """Apply SimpleITK N4 bias field correction."""
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    image_float = sitk.Cast(image, sitk.sitkFloat32)
    corrected = corrector.Execute(image_float, mask)
    return sitk.Cast(corrected, image.GetPixelID())


def zscore_normalise(image: sitk.Image) -> sitk.Image:
    """Z-score normalisation over non-zero voxels."""
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    nonzero = arr[arr != 0]
    if len(nonzero) == 0:
        return sitk.Cast(image, sitk.sitkFloat32)
    mu = nonzero.mean()
    sigma = nonzero.std()
    if sigma < 1e-8:
        sigma = 1.0
    arr_norm = np.where(arr != 0, (arr - mu) / sigma, 0.0)
    result = sitk.GetImageFromArray(arr_norm)
    result.CopyInformation(image)
    return result


def register_to_fixed(moving: sitk.Image, fixed: sitk.Image) -> sitk.Image:
    """Rigid registration of moving image onto fixed image space."""
    fixed_f  = sitk.Cast(sitk.RescaleIntensity(fixed),  sitk.sitkFloat32)
    moving_f = sitk.Cast(sitk.RescaleIntensity(moving), sitk.sitkFloat32)

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY))
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    transform = reg.Execute(fixed_f, moving_f)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    return resampler.Execute(moving)


def ensure_binary_mask(mask: sitk.Image) -> sitk.Image:
    """Threshold mask to strict binary 0/1."""
    arr = sitk.GetArrayFromImage(mask)
    arr_bin = (arr > 0).astype(np.uint8)
    result = sitk.GetImageFromArray(arr_bin)
    result.CopyInformation(mask)
    return result


# ── Per-patient pipeline ──────────────────────────────────────────────────────

def process_patient(patient_id: str, timepoint: str,
                    nifti_dir: Path, mask_dir: Path,
                    out_dir: Path, cfg: dict) -> dict:
    spacing   = cfg["preprocessing"]["target_spacing"]
    do_n4     = cfg["preprocessing"]["n4_bias_correction"]
    do_reg    = cfg["preprocessing"]["register_flair_to_t1gd"]
    norm_mode = cfg["preprocessing"]["normalization"]

    out_pt = ensure_dir(out_dir / patient_id / timepoint)
    result = {"PatientID": patient_id, "Timepoint": timepoint}

    # --- Load T1Gd (anchor image) ---
    t1gd_src = nifti_dir / patient_id / timepoint / "t1gd.nii.gz"
    if not t1gd_src.exists():
        logger.warning(f"  {patient_id}: no T1Gd NIfTI, skipping")
        result["Status"] = "no_t1gd"
        return result

    t1gd = load_nifti(t1gd_src)
    t1gd = resample_image(t1gd, spacing, sitk.sitkBSpline)
    if do_n4:
        try:
            t1gd = n4_bias_correction(t1gd)
        except Exception as exc:
            logger.warning(f"  {patient_id}: N4 failed ({exc}), continuing without")
    if norm_mode == "zscore":
        t1gd = zscore_normalise(t1gd)

    save_nifti(t1gd, out_pt / "t1gd.nii.gz")
    result["T1Gd"] = True

    # --- Load and process other sequences ---
    for seq in ["flair", "t1", "t2", "dwi", "adc"]:
        src = nifti_dir / patient_id / timepoint / f"{seq}.nii.gz"
        if not src.exists():
            result[seq.upper()] = False
            continue
        try:
            img = load_nifti(src)
            img = resample_image(img, spacing, sitk.sitkBSpline)
            if do_n4 and seq in ("flair", "t1", "t2"):
                try:
                    img = n4_bias_correction(img)
                except Exception:
                    pass
            if seq == "flair" and do_reg:
                img = register_to_fixed(img, t1gd)
            if norm_mode == "zscore":
                img = zscore_normalise(img)
            save_nifti(img, out_pt / f"{seq}.nii.gz")
            result[seq.upper()] = True
        except Exception as exc:
            logger.warning(f"  {patient_id} {seq}: {exc}")
            result[seq.upper()] = False

    # --- Resample tumor mask ---
    mask_src = mask_dir / patient_id / timepoint / "tumor_mask.nii.gz"
    if mask_src.exists():
        try:
            mask = load_nifti(mask_src)
            mask = resample_mask_to_image(mask, t1gd)
            mask = ensure_binary_mask(mask)
            save_nifti(mask, out_pt / "tumor_mask.nii.gz")
            result["Mask"] = True
        except Exception as exc:
            logger.warning(f"  {patient_id}: mask resampling failed: {exc}")
            result["Mask"] = False
    else:
        result["Mask"] = False

    result["Status"] = "ok"
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Preprocess MRI NIfTIs")
    p.add_argument("--images", default="data/nifti")
    p.add_argument("--masks",  default="data/masks")
    p.add_argument("--output", default="data/processed")
    p.add_argument("--patients", nargs="+", default=None)
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    nifti_root = Path(args.images)
    mask_root  = Path(args.masks)
    out_root   = Path(args.output)

    # Discover patient / timepoint pairs
    pairs = []
    for pt_dir in sorted(nifti_root.iterdir()):
        if not pt_dir.is_dir():
            continue
        pid = pt_dir.name
        if args.patients and pid not in args.patients:
            continue
        for tp_dir in sorted(pt_dir.iterdir()):
            if tp_dir.is_dir():
                pairs.append((pid, tp_dir.name))

    logger.info(f"Preprocessing {len(pairs)} patient-timepoints...")

    import pandas as pd
    results = []
    for pid, tp in tqdm(pairs, desc="Preprocessing"):
        try:
            res = process_patient(pid, tp, nifti_root, mask_root, out_root, cfg)
        except Exception as exc:
            logger.error(f"  {pid}/{tp}: {exc}")
            res = {"PatientID": pid, "Timepoint": tp, "Status": "error", "Error": str(exc)}
        results.append(res)

    df = pd.DataFrame(results)
    log_path = out_root / "preprocessing_log.csv"
    ensure_dir(log_path.parent)
    df.to_csv(log_path, index=False)
    ok = (df["Status"] == "ok").sum()
    logger.info(f"Done: {ok}/{len(df)} successful. Log → {log_path}")


if __name__ == "__main__":
    main()
