"""
create_regions_nifti.py — Create peritumoral rings from pre-existing NIfTI
GTV masks in the CFB-GBM dataset.

Input structure:
    data/raw/cfb_gbm/{patient_id}/{timepoint}/{patient_id}_{timepoint}_gtv.nii.gz

Output structure:
    data/processed/regions/{patient_id}/{timepoint}/
        tumor_mask.nii.gz
        ring_5mm.nii.gz
        ring_10mm.nii.gz

Also generates QC figures:
    results/figures/qc/{patient_id}_{timepoint}_qc.png

Usage:
    python src/create_regions_nifti.py \
        --input  data/raw/cfb_gbm \
        --output data/processed/regions
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (get_logger, ensure_dir, load_config,
                   load_nifti, save_nifti, image_to_array, array_to_image)

logger = get_logger(__name__)


def dilate_mask_mm(mask: sitk.Image, radius_mm: float) -> sitk.Image:
    spacing = mask.GetSpacing()
    radii = [max(1, int(round(radius_mm / s))) for s in spacing]
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetKernelType(sitk.sitkBall)
    dilate.SetKernelRadius(radii)
    dilate.SetForegroundValue(1)
    return dilate.Execute(sitk.Cast(mask, sitk.sitkUInt8))


def create_ring(mask: sitk.Image, inner_mm: float, outer_mm: float) -> sitk.Image:
    outer = dilate_mask_mm(mask, outer_mm)
    if inner_mm > 0:
        inner = dilate_mask_mm(mask, inner_mm)
        ring_arr = (image_to_array(outer) - image_to_array(inner)).clip(0, 1)
    else:
        ring_arr = image_to_array(outer) - image_to_array(mask)
        ring_arr = ring_arr.clip(0, 1)
    ring_arr = ring_arr.astype(np.uint8)
    return array_to_image(ring_arr, mask)


def save_qc_figure(t1gd: sitk.Image, flair: sitk.Image,
                   tumor: sitk.Image, ring5: sitk.Image,
                   patient_id: str, timepoint: str, out_path: Path) -> None:
    try:
        tumor_arr = image_to_array(tumor)
        z_idx = np.where(tumor_arr.sum(axis=(1, 2)) > 0)[0]
        z = int(z_idx[len(z_idx) // 2]) if len(z_idx) > 0 else tumor_arr.shape[0] // 2

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        panels = [
            (t1gd,  None,  "T1Gd"),
            (t1gd,  tumor, "T1Gd + GTV"),
            (flair, tumor, "FLAIR + GTV"),
            (t1gd,  ring5, "T1Gd + Ring 5mm"),
        ]
        cmaps = [None, "Reds", "Reds", "Blues"]
        for ax, (img, mask, title), cmap in zip(axes, panels, cmaps):
            if img is not None:
                ax.imshow(image_to_array(img)[z], cmap="gray", origin="lower")
            if mask is not None:
                m = image_to_array(mask)[z]
                ax.imshow(np.ma.masked_where(m == 0, m),
                          cmap=cmap, alpha=0.45, origin="lower")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

        fig.suptitle(f"Patient {patient_id} — {timepoint} — slice {z}", fontsize=11)
        plt.tight_layout()
        ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        logger.warning(f"QC figure failed for {patient_id}: {exc}")


def process_patient(patient_id: str, raw_root: Path,
                    out_root: Path, qc_root: Path,
                    rings: list[int]) -> dict:
    result = {"PatientID": patient_id, "Status": "skip"}
    pt_dir = raw_root / patient_id

    for tp_dir in sorted(pt_dir.iterdir()):
        if not tp_dir.is_dir():
            continue
        timepoint = tp_dir.name

        # Find GTV mask — pattern: {num}_{tp}_gtv.nii.gz
        gtv_files = list(tp_dir.glob("*_gtv.nii.gz"))
        if not gtv_files:
            logger.debug(f"  {patient_id}/{timepoint}: no GTV mask")
            continue

        gtv_path  = gtv_files[0]
        t1gd_files = list(tp_dir.glob("*_t1gd.nii.gz"))
        flair_files = list(tp_dir.glob("*_flair.nii.gz"))

        if not t1gd_files:
            logger.warning(f"  {patient_id}/{timepoint}: no T1Gd image")
            continue

        t1gd_path  = t1gd_files[0]
        flair_path = flair_files[0] if flair_files else None

        # Load
        gtv   = load_nifti(gtv_path)
        t1gd  = load_nifti(t1gd_path)
        flair = load_nifti(flair_path) if flair_path else None

        # Ensure mask is binary
        gtv_arr = (image_to_array(gtv) > 0).astype(np.uint8)
        gtv = array_to_image(gtv_arr, gtv)

        # Output directory
        out_dir = ensure_dir(out_root / patient_id / timepoint)

        # Save tumor mask
        save_nifti(gtv, out_dir / "tumor_mask.nii.gz")

        # Create rings
        ring_imgs = {}
        radii = sorted(rings)
        prev_r = 0
        for r in radii:
            ring = create_ring(gtv, prev_r, r)
            ring_path = out_dir / f"ring_{r}mm.nii.gz"
            save_nifti(ring, ring_path)
            ring_imgs[r] = ring
            prev_r = r

        # QC figure
        ring5 = ring_imgs.get(5) or ring_imgs.get(radii[0])
        qc_path = qc_root / f"{patient_id}_{timepoint}_qc.png"
        save_qc_figure(t1gd, flair, gtv, ring5,
                       patient_id, timepoint, qc_path)

        voxels = int(gtv_arr.sum())
        logger.info(f"  ✓ {patient_id}/{timepoint}: GTV={voxels} voxels, "
                    f"rings={rings}")
        result = {"PatientID": patient_id, "Timepoint": timepoint,
                  "Status": "ok", "GTVvoxels": voxels,
                  "GTVpath": str(gtv_path)}

    return result


def main():
    p = argparse.ArgumentParser(description="Create peritumoral rings from NIfTI GTV masks")
    p.add_argument("--input",    default="data/raw/cfb_gbm")
    p.add_argument("--output",   default="data/processed/regions")
    p.add_argument("--rings",    nargs="+", type=int, default=[5, 10])
    p.add_argument("--patients", nargs="+", default=None)
    p.add_argument("--config",   default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    raw_root = Path(args.input)
    out_root = Path(args.output)
    qc_root  = ensure_dir(Path(cfg["paths"]["qc_images"]))

    patient_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    if args.patients:
        patient_dirs = [d for d in patient_dirs if d.name in args.patients]

    logger.info(f"Processing {len(patient_dirs)} patients, rings={args.rings}")

    results = []
    for pt_dir in tqdm(patient_dirs, desc="Patients"):
        try:
            res = process_patient(pt_dir.name, raw_root, out_root,
                                  qc_root, args.rings)
        except Exception as exc:
            logger.error(f"  {pt_dir.name}: {exc}")
            res = {"PatientID": pt_dir.name, "Status": "error", "Error": str(exc)}
        results.append(res)

    df = pd.DataFrame(results)
    log_path = Path(cfg["paths"]["processed"]) / "regions_log.csv"
    ensure_dir(log_path.parent)
    df.to_csv(log_path, index=False)
    ok = (df["Status"] == "ok").sum()
    logger.info(f"Done: {ok}/{len(df)} successful. QC figures → {qc_root}")


if __name__ == "__main__":
    main()
