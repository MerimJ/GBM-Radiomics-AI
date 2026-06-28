"""
convert_dicom_to_nifti.py — Step 3: Convert DICOM MRI series to NIfTI.

Reads the series metadata CSV produced by inspect_dicom.py, selects the best
series for each patient / modality, converts with dicom2nifti or SimpleITK,
and saves JSON sidecars with provenance metadata.

Output layout:
    data/nifti/{patient_id}/{timepoint}/t1gd.nii.gz
    data/nifti/{patient_id}/{timepoint}/flair.nii.gz
    data/nifti/{patient_id}/{timepoint}/t1.nii.gz
    data/nifti/{patient_id}/{timepoint}/t2.nii.gz
    data/nifti/{patient_id}/{timepoint}/t1gd.json   (sidecar)
    ...

Usage:
    python src/convert_dicom_to_nifti.py \
        --metadata data/processed/series_metadata.csv \
        --output   data/nifti \
        [--patients GBM-001 GBM-002]
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config, write_json_sidecar

logger = get_logger(__name__)

# Map sequence type → output filename (without extension)
SEQ_FILENAMES = {
    "T1Gd":  "t1gd",
    "T1":    "t1",
    "T2":    "t2",
    "FLAIR": "flair",
    "DWI":   "dwi",
    "ADC":   "adc",
}

MODALITY_FILTER = {"MR"}  # only convert MR for NIfTI


def assign_timepoints(group: pd.DataFrame) -> pd.DataFrame:
    """Assign a timepoint label (baseline, followup_1, …) by StudyDate order."""
    dates = sorted(group["StudyDate"].unique())
    date_map = {d: ("baseline" if i == 0 else f"followup_{i}")
                for i, d in enumerate(dates)}
    group = group.copy()
    group["Timepoint"] = group["StudyDate"].map(date_map)
    return group


def select_best_series(sub: pd.DataFrame) -> pd.Series | None:
    """From a set of series with the same sequence type, pick the one with most slices."""
    if sub.empty:
        return None
    return sub.loc[sub["NumSlices"].idxmax()]


def try_dicom2nifti(series_dir: Path, out_path: Path) -> bool:
    """Attempt conversion using dicom2nifti library."""
    try:
        import dicom2nifti
        import dicom2nifti.settings as settings
        settings.disable_validate_slicecount()
        settings.disable_validate_orientation()
        tmp_dir = out_path.parent / "_d2n_tmp"
        ensure_dir(tmp_dir)
        dicom2nifti.convert_directory(str(series_dir), str(tmp_dir), compression=True)
        nifti_files = list(tmp_dir.glob("*.nii.gz"))
        if nifti_files:
            nifti_files[0].rename(out_path)
            # Cleanup remaining tmp files
            for f in tmp_dir.iterdir():
                f.unlink(missing_ok=True)
            tmp_dir.rmdir()
            return True
    except Exception as exc:
        logger.debug(f"dicom2nifti failed: {exc}")
    return False


def try_sitk(series_dir: Path, out_path: Path) -> bool:
    """Attempt conversion using SimpleITK DICOM reader."""
    try:
        reader = sitk.ImageSeriesReader()
        dcm_names = reader.GetGDCMSeriesFileNames(str(series_dir))
        if not dcm_names:
            return False
        reader.SetFileNames(dcm_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()
        sitk.WriteImage(image, str(out_path))
        return True
    except Exception as exc:
        logger.debug(f"SimpleITK conversion failed: {exc}")
    return False


def build_sidecar(row: pd.Series, out_path: Path) -> dict:
    return {
        "PatientID":         row.get("PatientID", ""),
        "SeriesInstanceUID": row.get("SeriesInstanceUID", ""),
        "SeriesDescription": row.get("SeriesDescription", ""),
        "SequenceType":      row.get("SequenceType", ""),
        "StudyDate":         row.get("StudyDate", ""),
        "NumSlices":         int(row.get("NumSlices", 0)),
        "SourceDir":         str(row.get("SeriesDir", "")),
        "NIfTIPath":         str(out_path),
    }


def convert_patient(patient_id: str, timepoint: str,
                    series_rows: pd.DataFrame,
                    output_root: Path) -> dict[str, bool]:
    """Convert all available sequences for one patient / timepoint."""
    out_dir = ensure_dir(output_root / patient_id / timepoint)
    results = {}

    for seq_type, fname in SEQ_FILENAMES.items():
        subset = series_rows[series_rows["SequenceType"] == seq_type]
        row = select_best_series(subset)
        if row is None:
            continue

        series_dir = Path(row["SeriesDir"])
        out_nifti = out_dir / f"{fname}.nii.gz"

        if out_nifti.exists():
            logger.debug(f"  {seq_type}: already exists, skipping")
            results[seq_type] = True
            continue

        if not series_dir.exists():
            logger.warning(f"  {seq_type}: source dir not found: {series_dir}")
            results[seq_type] = False
            continue

        ok = try_dicom2nifti(series_dir, out_nifti)
        if not ok:
            ok = try_sitk(series_dir, out_nifti)

        if ok:
            write_json_sidecar(build_sidecar(row, out_nifti),
                               out_dir / f"{fname}.json")
            logger.debug(f"  ✓ {seq_type} → {out_nifti.name}")
        else:
            logger.warning(f"  ✗ {seq_type}: conversion failed for {patient_id}")
        results[seq_type] = ok

    return results


def main():
    p = argparse.ArgumentParser(description="Convert DICOM MRI to NIfTI")
    p.add_argument("--metadata", default="data/processed/series_metadata.csv")
    p.add_argument("--output",   default="data/nifti")
    p.add_argument("--patients", nargs="+", default=None)
    p.add_argument("--config",   default="config/config.yaml")
    args = p.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.metadata)

    # Filter to MR only
    df = df[df["Modality"].isin(MODALITY_FILTER)].copy()

    if args.patients:
        df = df[df["PatientID"].isin(args.patients)]

    patients = df["PatientID"].unique()
    logger.info(f"Converting {len(patients)} patients to NIfTI...")

    summary = []
    for pid in tqdm(patients, desc="Patients"):
        pid_df = df[df["PatientID"] == pid].copy()
        pid_df = assign_timepoints(pid_df)
        for tp, tp_df in pid_df.groupby("Timepoint"):
            results = convert_patient(pid, tp, tp_df, Path(args.output))
            summary.append({"PatientID": pid, "Timepoint": tp, **results})

    sumdf = pd.DataFrame(summary)
    out_csv = Path(cfg["paths"]["processed"]) / "nifti_conversion_log.csv"
    ensure_dir(out_csv.parent)
    sumdf.to_csv(out_csv, index=False)

    succeeded = sumdf.get("T1Gd", pd.Series(dtype=bool)).sum()
    logger.info(f"T1Gd conversion: {succeeded}/{len(sumdf)} patient-timepoints")
    logger.info(f"Conversion log → {out_csv}")


if __name__ == "__main__":
    main()
