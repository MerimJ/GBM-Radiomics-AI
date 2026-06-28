"""
convert_rtstruct_to_mask.py — Step 4: Convert DICOM RTSTRUCT contours to
binary NIfTI masks aligned with the reference MRI.

For each patient:
  1. Locate RTSTRUCT DICOM file.
  2. List all ROI names and identify tumor-related structures.
  3. Rasterise contours into a binary mask using rt_utils.
  4. Resample mask to the T1Gd NIfTI space (nearest-neighbour).
  5. Save mask and QC overlays.

Output:
    data/masks/{patient_id}/{timepoint}/tumor_mask.nii.gz
    data/processed/roi_name_mapping.csv
    results/figures/qc/{patient_id}_{timepoint}_mask_qc.png

Usage:
    python src/convert_rtstruct_to_mask.py \
        --input  data/raw/cfb_gbm \
        --nifti  data/nifti \
        --output data/masks
"""

import argparse
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (get_logger, ensure_dir, load_config,
                   resample_mask_to_image, save_nifti, load_nifti)

logger = get_logger(__name__)

# Keywords to identify tumor ROIs (case-insensitive)
TUMOR_KEYWORDS = [
    r"gtv", r"tumor", r"tumour", r"lesion", r"glioblastoma",
    r"enhancing", r"necrosis", r"core", r"gross\s*target",
    r"edema", r"oedema", r"infiltrat",
]


def is_tumor_roi(name: str) -> bool:
    name_lower = name.lower()
    return any(re.search(kw, name_lower) for kw in TUMOR_KEYWORDS)


def find_rtstruct_files(patient_dir: Path) -> list[Path]:
    candidates = []
    for f in patient_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            if getattr(ds, "Modality", "") == "RTSTRUCT":
                candidates.append(f)
        except Exception:
            pass
    return candidates


def list_roi_names(rtstruct_path: Path) -> list[str]:
    ds = pydicom.dcmread(str(rtstruct_path))
    names = []
    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            names.append(getattr(roi, "ROIName", ""))
    return names


def rtstruct_to_mask_rtutils(rtstruct_path: Path,
                              reference_series_dir: Path,
                              roi_names: list[str]) -> np.ndarray | None:
    """Use rt_utils to rasterise contours into a 3D array."""
    try:
        from rt_utils import RTStructBuilder
        rts = RTStructBuilder.create_from(
            dicom_series_path=str(reference_series_dir),
            rt_struct_path=str(rtstruct_path),
        )
        # Combine all selected ROIs with OR
        combined = None
        for name in roi_names:
            try:
                mask_3d = rts.get_roi_mask_by_name(name)  # bool 3-D ndarray
                combined = mask_3d if combined is None else (combined | mask_3d)
            except Exception as exc:
                logger.debug(f"  Skipping ROI '{name}': {exc}")
        return combined
    except Exception as exc:
        logger.warning(f"rt_utils failed: {exc}")
        return None


def save_qc_figure(t1gd_path: Path, mask_array: np.ndarray,
                   out_path: Path, patient_id: str) -> None:
    try:
        img = load_nifti(t1gd_path)
        img_arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
        # Find central slice of the mask
        z_slices = np.where(mask_array.sum(axis=(1, 2)) > 0)[0]
        z = int(z_slices[len(z_slices) // 2]) if len(z_slices) > 0 else img_arr.shape[0] // 2

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_arr[z], cmap="gray", origin="lower")
        axes[0].set_title("T1Gd")
        axes[1].imshow(img_arr[z], cmap="gray", origin="lower")
        axes[1].imshow(mask_array[z], cmap="Reds", alpha=0.4, origin="lower")
        axes[1].set_title("T1Gd + Tumor Mask")
        fig.suptitle(f"{patient_id} — slice {z}", fontsize=11)
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"QC figure failed for {patient_id}: {exc}")


def process_patient(patient_id: str,
                    raw_patient_dir: Path,
                    nifti_root: Path,
                    mask_root: Path,
                    qc_root: Path) -> dict:
    result = {"PatientID": patient_id, "Status": "skip",
              "SelectedROIs": "", "NumROIs": 0}

    # --- Locate RTSTRUCT ---
    rt_files = find_rtstruct_files(raw_patient_dir)
    if not rt_files:
        result["Status"] = "no_rtstruct"
        return result

    # Use first RTSTRUCT found (could extend for multiple timepoints)
    rt_path = rt_files[0]
    all_rois = list_roi_names(rt_path)
    tumor_rois = [r for r in all_rois if is_tumor_roi(r)]

    if not tumor_rois:
        logger.warning(f"  {patient_id}: no tumor ROIs found among {all_rois}")
        result["Status"] = "no_tumor_roi"
        result["AllROIs"] = str(all_rois)
        return result

    result["SelectedROIs"] = str(tumor_rois)
    result["NumROIs"] = len(tumor_rois)

    # --- Reference MRI (T1Gd) ---
    t1gd_path = nifti_root / patient_id / "baseline" / "t1gd.nii.gz"
    if not t1gd_path.exists():
        result["Status"] = "no_t1gd_nifti"
        return result

    ref_img = load_nifti(t1gd_path)

    # --- Find CT or MR series dir used as reference by RTSTRUCT ---
    # rt_utils needs the DICOM series the RTSTRUCT was built on
    # Try finding a DICOM MR or CT folder under the patient directory
    ref_series_dir = None
    for sub in raw_patient_dir.rglob("*"):
        if sub.is_dir():
            dcm_files = list(sub.glob("*.dcm"))
            if dcm_files:
                try:
                    ds0 = pydicom.dcmread(str(dcm_files[0]),
                                          stop_before_pixels=True, force=True)
                    if getattr(ds0, "Modality", "") in ("MR", "CT"):
                        ref_series_dir = sub
                        break
                except Exception:
                    pass

    if ref_series_dir is None:
        result["Status"] = "no_ref_series"
        return result

    # --- Rasterise ---
    mask_arr = rtstruct_to_mask_rtutils(rt_path, ref_series_dir, tumor_rois)
    if mask_arr is None or mask_arr.sum() == 0:
        result["Status"] = "mask_empty"
        return result

    # mask_arr is (Z, Y, X) bool
    mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    # Note: rt_utils aligns to the DICOM series space; we resample to T1Gd space
    mask_resampled = resample_mask_to_image(mask_sitk, ref_img)

    # --- Save ---
    out_dir = ensure_dir(mask_root / patient_id / "baseline")
    mask_out = out_dir / "tumor_mask.nii.gz"
    save_nifti(mask_resampled, mask_out)

    # QC
    mask_final_arr = sitk.GetArrayFromImage(mask_resampled)
    qc_out = qc_root / f"{patient_id}_baseline_mask_qc.png"
    save_qc_figure(t1gd_path, mask_final_arr, qc_out, patient_id)

    voxels = int(mask_final_arr.sum())
    result.update({"Status": "ok", "MaskVoxels": voxels,
                   "MaskPath": str(mask_out)})
    logger.info(f"  ✓ {patient_id}: mask saved ({voxels} voxels)")
    return result


def main():
    p = argparse.ArgumentParser(description="RTSTRUCT → NIfTI tumor mask")
    p.add_argument("--input",  default="data/raw/cfb_gbm")
    p.add_argument("--nifti",  default="data/nifti")
    p.add_argument("--output", default="data/masks")
    p.add_argument("--patients", nargs="+", default=None)
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    raw_root = Path(args.input)
    nifti_root = Path(args.nifti)
    mask_root = Path(args.output)
    qc_root = ensure_dir(Path(cfg["paths"]["qc_images"]))

    patient_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    if args.patients:
        patient_dirs = [d for d in patient_dirs if d.name in args.patients]

    logger.info(f"Processing {len(patient_dirs)} patients...")
    all_results = []
    for pd_dir in tqdm(patient_dirs, desc="Patients"):
        pid = pd_dir.name
        try:
            res = process_patient(pid, pd_dir, nifti_root, mask_root, qc_root)
        except Exception as exc:
            logger.error(f"  {pid}: unexpected error: {exc}")
            res = {"PatientID": pid, "Status": "error", "Error": str(exc)}
        all_results.append(res)

    df = pd.DataFrame(all_results)
    out_csv = Path(cfg["paths"]["processed"]) / "mask_conversion_log.csv"
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)

    ok = (df["Status"] == "ok").sum()
    logger.info(f"Masks created: {ok}/{len(df)}")
    logger.info(f"Log → {out_csv}")

    # ROI name mapping
    roi_rows = df[df["SelectedROIs"].notna() & (df["SelectedROIs"] != "")]
    roi_rows[["PatientID", "SelectedROIs", "NumROIs"]].to_csv(
        Path(cfg["paths"]["processed"]) / "roi_name_mapping.csv", index=False)


if __name__ == "__main__":
    main()
