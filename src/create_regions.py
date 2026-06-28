"""
create_regions.py — Step 6: Generate peritumoral rings and optional regions
from binary tumor masks.

Regions created for each patient:
    tumor_mask.nii.gz     — intratumoral (from RTSTRUCT or AI segmentation)
    ring_5mm.nii.gz       — 5 mm peritumoral shell
    ring_10mm.nii.gz      — 5–10 mm peritumoral shell
    edema_mask.nii.gz     — FLAIR abnormality region (optional)

Also generates QC images showing each region overlay on T1Gd and FLAIR.

Usage:
    python src/create_regions.py \
        --input  data/processed \
        --output data/processed/regions \
        --rings  5 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (get_logger, ensure_dir, load_config,
                   save_nifti, load_nifti, image_to_array, array_to_image)

logger = get_logger(__name__)


# ── Morphological dilation ────────────────────────────────────────────────────

def dilate_mask_mm(mask: sitk.Image, radius_mm: float) -> sitk.Image:
    """Dilate a binary mask by radius_mm using SimpleITK ball structuring element."""
    spacing = mask.GetSpacing()  # (x, y, z) in mm
    radii_voxels = [max(1, int(round(radius_mm / s))) for s in spacing]

    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelType(sitk.sitkBall)
    dilate.SetKernelRadius(radii_voxels)
    dilate.SetForegroundValue(1)
    return dilate.Execute(sitk.Cast(mask, sitk.sitkUInt8))


def create_ring(mask: sitk.Image, inner_mm: float, outer_mm: float) -> sitk.Image:
    """Create a ring between two dilation radii."""
    inner = dilate_mask_mm(mask, inner_mm)
    outer = dilate_mask_mm(mask, outer_mm)
    ring_arr = (image_to_array(outer) - image_to_array(inner)).clip(0, 1).astype(np.uint8)
    # Also exclude the original tumor mask
    ring_arr[image_to_array(mask) > 0] = 0
    return array_to_image(ring_arr, mask)


def create_flair_mask(flair: sitk.Image, tumor_mask: sitk.Image,
                      lower_percentile: float = 90.0) -> sitk.Image:
    """Threshold FLAIR at high intensity to approximate edema/FLAIR abnormality."""
    flair_arr = image_to_array(flair).astype(np.float32)
    mask_arr  = image_to_array(tumor_mask) > 0
    nonzero   = flair_arr[flair_arr != 0]
    if len(nonzero) == 0:
        return array_to_image(np.zeros_like(flair_arr, dtype=np.uint8), tumor_mask)
    threshold = np.percentile(nonzero, lower_percentile)
    flair_high = (flair_arr >= threshold).astype(np.uint8)
    flair_high[mask_arr] = 1  # include tumor core
    return array_to_image(flair_high, tumor_mask)


# ── QC visualisation ──────────────────────────────────────────────────────────

def _add_overlay(ax, base_arr, overlay_arr, title, cmap="Reds", alpha=0.4):
    ax.imshow(base_arr, cmap="gray", origin="lower")
    if overlay_arr is not None and overlay_arr.sum() > 0:
        ax.imshow(np.ma.masked_where(overlay_arr == 0, overlay_arr),
                  cmap=cmap, alpha=alpha, origin="lower")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_qc_montage(patient_id: str, t1gd: sitk.Image | None,
                    flair: sitk.Image | None,
                    masks: dict[str, sitk.Image],
                    out_path: Path) -> None:
    """4-column QC montage: T1Gd+tumor, FLAIR+tumor, T1Gd+ring5, FLAIR+ring5."""
    try:
        # Choose central slice from tumor mask
        tumor_arr = image_to_array(masks.get("tumor_mask",
                                             list(masks.values())[0]))
        z_idx = np.where(tumor_arr.sum(axis=(1, 2)) > 0)[0]
        z = int(z_idx[len(z_idx) // 2]) if len(z_idx) else tumor_arr.shape[0] // 2

        t1gd_sl  = image_to_array(t1gd)[z]  if t1gd  else None
        flair_sl = image_to_array(flair)[z] if flair else None

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        combos = [
            (t1gd_sl,  "tumor_mask", "T1Gd + Tumor",   "Reds"),
            (flair_sl, "tumor_mask", "FLAIR + Tumor",   "Reds"),
            (t1gd_sl,  "ring_5mm",   "T1Gd + Ring 5mm", "Blues"),
            (flair_sl, "ring_5mm",   "FLAIR + Ring 5mm","Blues"),
        ]
        for ax, (base, mask_key, title, cmap) in zip(axes, combos):
            m_arr = image_to_array(masks[mask_key])[z] if mask_key in masks else None
            _add_overlay(ax, base if base is not None else
                         np.zeros((10, 10)), m_arr, title, cmap)

        fig.suptitle(patient_id, fontsize=12)
        plt.tight_layout()
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"QC montage failed for {patient_id}: {exc}")


# ── Per-patient processing ─────────────────────────────────────────────────────

def process_patient(patient_id: str, timepoint: str,
                    proc_root: Path, out_root: Path,
                    rings: list[int], qc_root: Path) -> dict:
    pt_dir  = proc_root / patient_id / timepoint
    mask_path = pt_dir / "tumor_mask.nii.gz"

    if not mask_path.exists():
        return {"PatientID": patient_id, "Status": "no_mask"}

    mask = load_nifti(mask_path)
    out_dir = ensure_dir(out_root / patient_id / timepoint)

    # Copy tumor mask
    out_mask = out_dir / "tumor_mask.nii.gz"
    save_nifti(mask, out_mask)

    saved_masks = {"tumor_mask": mask}

    # Peritumoral rings
    radii = sorted(rings)
    prev_r = 0
    for r in radii:
        ring = create_ring(mask, prev_r, r)
        ring_path = out_dir / f"ring_{r}mm.nii.gz"
        save_nifti(ring, ring_path)
        saved_masks[f"ring_{r}mm"] = ring
        prev_r = r

    # FLAIR edema mask (optional)
    flair_path = pt_dir / "flair.nii.gz"
    t1gd_path  = pt_dir / "t1gd.nii.gz"
    flair_img  = load_nifti(flair_path) if flair_path.exists() else None
    t1gd_img   = load_nifti(t1gd_path)  if t1gd_path.exists() else None

    if flair_img is not None:
        edema = create_flair_mask(flair_img, mask)
        save_nifti(edema, out_dir / "edema_mask.nii.gz")
        saved_masks["edema_mask"] = edema

    # QC montage
    qc_path = qc_root / f"{patient_id}_{timepoint}_regions_qc.png"
    save_qc_montage(patient_id, t1gd_img, flair_img, saved_masks, qc_path)

    return {"PatientID": patient_id, "Status": "ok",
            "MaskVoxels": int(image_to_array(mask).sum())}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Create peritumoral rings")
    p.add_argument("--input",  default="data/processed")
    p.add_argument("--output", default="data/processed/regions")
    p.add_argument("--rings",  nargs="+", type=int, default=[5, 10])
    p.add_argument("--patients", nargs="+", default=None)
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    proc_root = Path(args.input)
    out_root  = Path(args.output)
    qc_root   = ensure_dir(Path(cfg["paths"]["qc_images"]))

    import pandas as pd

    pairs = []
    for pt_dir in sorted(proc_root.iterdir()):
        if not pt_dir.is_dir() or pt_dir.name in ("regions",):
            continue
        pid = pt_dir.name
        if args.patients and pid not in args.patients:
            continue
        for tp_dir in sorted(pt_dir.iterdir()):
            if tp_dir.is_dir():
                pairs.append((pid, tp_dir.name))

    logger.info(f"Creating regions for {len(pairs)} patient-timepoints...")
    results = []
    for pid, tp in tqdm(pairs, desc="Regions"):
        try:
            res = process_patient(pid, tp, proc_root, out_root, args.rings, qc_root)
        except Exception as exc:
            logger.error(f"  {pid}/{tp}: {exc}")
            res = {"PatientID": pid, "Status": "error", "Error": str(exc)}
        results.append(res)

    df = pd.DataFrame(results)
    log_path = Path(cfg["paths"]["processed"]) / "regions_log.csv"
    df.to_csv(log_path, index=False)
    ok = (df["Status"] == "ok").sum()
    logger.info(f"Done: {ok}/{len(df)} successful. Log → {log_path}")


if __name__ == "__main__":
    main()
