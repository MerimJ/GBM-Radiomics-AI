"""
extract_radiomics_nifti.py — Extract PyRadiomics features from pre-existing
NIfTI images using tumor mask + peritumoral rings.

Extracts features for:
  - T1Gd + tumor mask      → T1Gd_Tumor
  - FLAIR + tumor mask     → FLAIR_Tumor
  - T1Gd + ring_5mm        → T1Gd_Ring5
  - FLAIR + ring_5mm       → FLAIR_Ring5
  - T1Gd + ring_10mm       → T1Gd_Ring10
  - FLAIR + ring_10mm      → FLAIR_Ring10

Usage:
    python src/extract_radiomics_nifti.py \
        --input   data/raw/cfb_gbm \
        --regions data/processed/regions \
        --output  data/features/radiomics_25patients.csv
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)
logging.getLogger("radiomics").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

REGIONS = [
    {"label": "T1Gd_Tumor",  "image": "t1gd",  "mask": "tumor_mask.nii.gz"},
    {"label": "FLAIR_Tumor", "image": "flair",  "mask": "tumor_mask.nii.gz"},
    {"label": "T1Gd_Ring5",  "image": "t1gd",  "mask": "ring_5mm.nii.gz"},
    {"label": "FLAIR_Ring5", "image": "flair",  "mask": "ring_5mm.nii.gz"},
    {"label": "T1Gd_Ring10", "image": "t1gd",  "mask": "ring_10mm.nii.gz"},
    {"label": "FLAIR_Ring10","image": "flair",  "mask": "ring_10mm.nii.gz"},
]


def find_image(tp_dir: Path, seq: str) -> Path | None:
    matches = list(tp_dir.glob(f"*_{seq}.nii.gz"))
    return matches[0] if matches else None


def extract_features(image_path: Path, mask_path: Path,
                     params_file: Path) -> dict | None:
    try:
        from radiomics import featureextractor
        extractor = featureextractor.RadiomicsFeatureExtractor(str(params_file))
        result = extractor.execute(str(image_path), str(mask_path))
        return {k: float(v) for k, v in result.items()
                if not k.startswith("diagnostics_")}
    except Exception as exc:
        logger.warning(f"    Extraction failed: {exc}")
        return None


def process_patient(patient_id: str, raw_root: Path,
                    region_root: Path, params_file: Path,
                    timepoint: str = "t0") -> dict:
    row = {"PatientID": patient_id, "Timepoint": timepoint}
    tp_dir     = raw_root    / patient_id / timepoint
    region_dir = region_root / patient_id / timepoint

    if not tp_dir.exists() or not region_dir.exists():
        return row

    for region in REGIONS:
        label     = region["label"]
        img_path  = find_image(tp_dir, region["image"])
        mask_path = region_dir / region["mask"]

        if img_path is None:
            logger.debug(f"  {patient_id}: no {region['image']} image")
            continue
        if not mask_path.exists():
            logger.debug(f"  {patient_id}: no {region['mask']}")
            continue

        # Check mask has enough voxels
        import SimpleITK as sitk
        import numpy as np
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        if mask_arr.sum() < 50:
            logger.debug(f"  {patient_id} {label}: mask too small ({mask_arr.sum()} voxels)")
            continue

        logger.debug(f"  {patient_id}: extracting {label}...")
        feats = extract_features(img_path, mask_path, params_file)
        if feats:
            for k, v in feats.items():
                row[f"{label}_{k}"] = v

    n_feats = len([k for k in row if k not in ("PatientID", "Timepoint")])
    logger.info(f"  ✓ {patient_id}/{timepoint}: {n_feats} features extracted")
    return row


def main():
    p = argparse.ArgumentParser(description="Extract radiomics from NIfTI images")
    p.add_argument("--input",   default="data/raw/cfb_gbm")
    p.add_argument("--regions", default="data/processed/regions")
    p.add_argument("--output",  default="data/features/radiomics_25patients.csv")
    p.add_argument("--params",  default="config/pyradiomics_params.yaml")
    p.add_argument("--timepoint", default="t0")
    p.add_argument("--patients", nargs="+", default=None)
    args = p.parse_args()

    cfg = load_config("config/config.yaml")
    raw_root    = Path(args.input)
    region_root = Path(args.regions)
    params_file = Path(args.params)
    out_path    = Path(args.output)

    patient_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])
    if args.patients:
        patient_dirs = [d for d in patient_dirs if d.name in args.patients]

    logger.info(f"Extracting radiomics for {len(patient_dirs)} patients "
                f"× {len(REGIONS)} regions...")

    records = []
    for pt_dir in tqdm(patient_dirs, desc="Patients"):
        try:
            row = process_patient(pt_dir.name, raw_root, region_root,
                                  params_file, args.timepoint)
        except Exception as exc:
            logger.error(f"  {pt_dir.name}: {exc}")
            row = {"PatientID": pt_dir.name, "Timepoint": args.timepoint}
        records.append(row)

    df = pd.DataFrame(records)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)

    n_feats = df.shape[1] - 2
    logger.info(f"Done → {out_path}  ({len(df)} patients × {n_feats} features)")


if __name__ == "__main__":
    main()
