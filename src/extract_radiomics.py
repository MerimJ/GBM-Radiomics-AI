"""
extract_radiomics.py — Step 7: Extract PyRadiomics features for each patient
from multiple image–mask region combinations.

Feature naming convention:
    {Region}_{ImageType}_{FeatureClass}_{FeatureName}
    e.g.  T1Gd_Tumor_original_glcm_Contrast
          FLAIR_Ring5_wavelet-HHL_firstorder_Entropy

Output:
    data/features/radiomics_raw.csv

Usage:
    python src/extract_radiomics.py \
        --input  data/processed/regions \
        --images data/processed \
        --params config/pyradiomics_params.yaml \
        --output data/features/radiomics_raw.csv
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

# Suppress pyradiomics verbose output
logging.getLogger("radiomics").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


# ── Region → image / mask file mappings ──────────────────────────────────────

def build_region_map(cfg: dict) -> list[dict]:
    """Build list of (label, image_seq, mask_filename) from config."""
    regions = cfg["radiomics"]["regions_to_extract"]
    seq_file_map = {
        "T1Gd":  "t1gd.nii.gz",
        "FLAIR": "flair.nii.gz",
        "T1":    "t1.nii.gz",
        "T2":    "t2.nii.gz",
        "ADC":   "adc.nii.gz",
    }
    mask_file_map = {
        "tumor":    "tumor_mask.nii.gz",
        "ring5mm":  "ring_5mm.nii.gz",
        "ring10mm": "ring_10mm.nii.gz",
        "edema":    "edema_mask.nii.gz",
    }
    out = []
    for r in regions:
        out.append({
            "label":     r["label"],
            "image_seq": seq_file_map.get(r["image"], f"{r['image'].lower()}.nii.gz"),
            "mask_file": mask_file_map.get(r["mask"], f"{r['mask']}.nii.gz"),
        })
    return out


# ── PyRadiomics extraction ────────────────────────────────────────────────────

def extract_features(image_path: Path, mask_path: Path,
                     params_file: Path) -> dict | None:
    try:
        import radiomics
        from radiomics import featureextractor
        extractor = featureextractor.RadiomicsFeatureExtractor(str(params_file))
        result = extractor.execute(str(image_path), str(mask_path))
        # Filter out diagnostic keys
        features = {k: float(v) for k, v in result.items()
                    if not k.startswith("diagnostics_") and k != "Image type"}
        return features
    except Exception as exc:
        logger.warning(f"Extraction failed {image_path.parent.name}: {exc}")
        return None


# ── Per-patient extraction ────────────────────────────────────────────────────

def process_patient(patient_id: str, timepoint: str,
                    image_dir: Path, region_dir: Path,
                    region_map: list[dict], params_file: Path) -> dict:
    row = {"PatientID": patient_id, "Timepoint": timepoint}
    img_tp  = image_dir  / patient_id / timepoint
    reg_tp  = region_dir / patient_id / timepoint

    for region in region_map:
        label     = region["label"]
        img_path  = img_tp  / region["image_seq"]
        mask_path = reg_tp  / region["mask_file"]

        if not img_path.exists():
            logger.debug(f"  {patient_id}: image not found: {img_path.name}")
            continue
        if not mask_path.exists():
            logger.debug(f"  {patient_id}: mask not found: {mask_path.name}")
            continue

        feats = extract_features(img_path, mask_path, params_file)
        if feats is None:
            continue

        for feat_name, feat_val in feats.items():
            row[f"{label}_{feat_name}"] = feat_val

    return row


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Extract radiomic features")
    p.add_argument("--input",  default="data/processed/regions",
                   help="Directory with patient region masks")
    p.add_argument("--images", default="data/processed",
                   help="Directory with preprocessed MRI images")
    p.add_argument("--params", default="config/pyradiomics_params.yaml")
    p.add_argument("--output", default="data/features/radiomics_raw.csv")
    p.add_argument("--patients", nargs="+", default=None)
    p.add_argument("--config", default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    region_map  = build_region_map(cfg)
    params_file = Path(args.params)
    region_dir  = Path(args.input)
    image_dir   = Path(args.images)
    out_path    = Path(args.output)

    # Discover patient-timepoint pairs from region directory
    pairs = []
    for pt_dir in sorted(region_dir.iterdir()):
        if not pt_dir.is_dir():
            continue
        pid = pt_dir.name
        if args.patients and pid not in args.patients:
            continue
        for tp_dir in sorted(pt_dir.iterdir()):
            if tp_dir.is_dir():
                pairs.append((pid, tp_dir.name))

    logger.info(f"Extracting features for {len(pairs)} patient-timepoints "
                f"× {len(region_map)} regions...")

    records = []
    for pid, tp in tqdm(pairs, desc="Patients"):
        try:
            row = process_patient(pid, tp, image_dir, region_dir,
                                  region_map, params_file)
        except Exception as exc:
            logger.error(f"  {pid}/{tp}: {exc}")
            row = {"PatientID": pid, "Timepoint": tp}
        records.append(row)

    df = pd.DataFrame(records)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)

    n_feats = df.shape[1] - 2  # subtract PatientID, Timepoint
    logger.info(f"Extracted {n_feats} features for {len(df)} patients → {out_path}")


if __name__ == "__main__":
    main()
