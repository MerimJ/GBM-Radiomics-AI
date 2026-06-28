"""
inspect_dicom.py — Step 2: Scan downloaded DICOM folders and build a metadata
catalogue. Classifies each series by likely MRI sequence type and prints a
summary of dataset completeness.

Usage:
  python src/inspect_dicom.py \
      --input  data/raw/cfb_gbm \
      --output data/processed/series_metadata.csv
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pydicom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config, classify_series

logger = get_logger(__name__)

DICOM_TAGS = {
    "PatientID":           (0x0010, 0x0020),
    "StudyInstanceUID":    (0x0020, 0x000D),
    "SeriesInstanceUID":   (0x0020, 0x000E),
    "StudyDate":           (0x0008, 0x0020),
    "Modality":            (0x0008, 0x0060),
    "SeriesDescription":   (0x0008, 0x103E),
    "SliceThickness":      (0x0050, 0x0050),
    "PixelSpacing":        (0x0028, 0x0030),
    "Rows":                (0x0028, 0x0010),
    "Columns":             (0x0028, 0x0011),
    "ImageOrientationPatient": (0x0020, 0x0037),
    "Manufacturer":        (0x0008, 0x0070),
    "ManufacturerModelName": (0x0008, 0x1090),
    "MagneticFieldStrength": (0x0018, 0x0087),
}


def read_dicom_tag(ds, tag_tuple, default=""):
    try:
        val = ds[tag_tuple].value
        if isinstance(val, pydicom.sequence.Sequence):
            return str(val)
        return str(val) if val is not None else default
    except (KeyError, AttributeError):
        return default


def find_representative_dcm(series_dir: Path) -> Path | None:
    """Return one DICOM file from a series directory."""
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(series_dir.rglob("*.dcm"))
    if not dcm_files:
        # TCIA sometimes omits extension
        dcm_files = [f for f in series_dir.rglob("*")
                     if f.is_file() and not f.suffix.lower() in (".csv", ".json", ".txt")]
    return dcm_files[0] if dcm_files else None


def count_dicom_files(series_dir: Path) -> int:
    return len(list(series_dir.rglob("*.dcm"))) or len(
        [f for f in series_dir.rglob("*") if f.is_file()]
    )


def scan_dicom_root(root: Path, config: dict) -> pd.DataFrame:
    """Walk root directory, read one DICOM per series, return catalogue DataFrame."""
    records = []

    # Expect root / patient_id / modality / series_uid / *.dcm  OR
    #         root / patient_id / *.dcm  (flat per patient)
    # We detect the structure dynamically.

    # Collect all unique directories that contain DICOM files
    dirs_with_dicom: list[Path] = []
    logger.info(f"Scanning {root} for DICOM files...")
    for dcm_file in tqdm(list(root.rglob("*.dcm")), desc="Locating DICOMs"):
        parent = dcm_file.parent
        if parent not in dirs_with_dicom:
            dirs_with_dicom.append(parent)

    if not dirs_with_dicom:
        logger.warning("No .dcm files found. Trying files without extension...")
        # Fallback: look for files in leaf directories
        for leaf in root.rglob("*"):
            if leaf.is_dir() and not any(leaf.iterdir()):
                continue
            if leaf.is_dir():
                files = [f for f in leaf.iterdir() if f.is_file()
                         and f.suffix.lower() not in (".json", ".csv", ".txt", ".xml")]
                if files:
                    dirs_with_dicom.append(leaf)

    logger.info(f"Found {len(dirs_with_dicom)} series directories")

    for series_dir in tqdm(dirs_with_dicom, desc="Reading DICOM headers"):
        rep = find_representative_dcm(series_dir)
        if rep is None:
            continue
        try:
            ds = pydicom.dcmread(str(rep), stop_before_pixels=True, force=True)
        except Exception as exc:
            logger.warning(f"Cannot read {rep}: {exc}")
            continue

        row = {tag: read_dicom_tag(ds, val) for tag, val in DICOM_TAGS.items()}
        row["NumSlices"] = count_dicom_files(series_dir)
        row["SeriesDir"] = str(series_dir)

        # Pixel spacing
        ps = read_dicom_tag(ds, (0x0028, 0x0030))
        try:
            px, py = [float(v) for v in ps.replace("[", "").replace("]", "").split(",")]
        except Exception:
            px = py = None
        row["PixelSpacingX_mm"] = px
        row["PixelSpacingY_mm"] = py

        thick = read_dicom_tag(ds, (0x0050, 0x0050))
        row["SliceThickness_mm"] = float(thick) if thick else None

        row["SequenceType"] = classify_series(row["SeriesDescription"], config)

        records.append(row)

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    n_patients = df["PatientID"].nunique()
    has_t1gd   = df[df["SequenceType"] == "T1Gd"]["PatientID"].unique()
    has_flair  = df[df["SequenceType"] == "FLAIR"]["PatientID"].unique()
    has_rtstruct = df[df["Modality"] == "RTSTRUCT"]["PatientID"].unique()
    has_rtdose   = df[df["Modality"] == "RTDOSE"]["PatientID"].unique()

    # Follow-up: patients with multiple study dates
    followup = (df.groupby("PatientID")["StudyDate"].nunique() > 1)
    has_followup = followup[followup].index.tolist()

    # Usable for radiomics: T1Gd + RTSTRUCT
    usable = set(has_t1gd) & set(has_rtstruct)

    print("\n" + "=" * 60)
    print("  CFB-GBM Dataset Summary")
    print("=" * 60)
    print(f"  Total patients          : {n_patients}")
    print(f"  With T1Gd               : {len(has_t1gd)}")
    print(f"  With FLAIR              : {len(has_flair)}")
    print(f"  With RTSTRUCT           : {len(has_rtstruct)}")
    print(f"  With RTDOSE             : {len(has_rtdose)}")
    print(f"  With follow-up MRI      : {len(has_followup)}")
    print(f"  Usable for radiomics    : {len(usable)}  (T1Gd + RTSTRUCT)")
    print("=" * 60)

    # Per-modality counts
    print("\nSeries count by Modality:")
    for mod, cnt in df["Modality"].value_counts().items():
        print(f"  {mod:<20} {cnt}")

    print("\nSequence type breakdown (MR only):")
    mr = df[df["Modality"] == "MR"]
    for seq, cnt in mr["SequenceType"].value_counts().items():
        print(f"  {seq:<20} {cnt}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Inspect DICOM metadata")
    p.add_argument("--input",  default="data/raw/cfb_gbm")
    p.add_argument("--output", default="data/processed/series_metadata.csv")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    root = Path(args.input)
    if not root.exists():
        logger.error(f"Input directory not found: {root}")
        sys.exit(1)

    df = scan_dicom_root(root, cfg)
    ensure_dir(Path(args.output).parent)
    df.to_csv(args.output, index=False)
    logger.info(f"Metadata CSV → {args.output}  ({len(df)} series)")

    print_summary(df)


if __name__ == "__main__":
    main()
