"""
download_tcia.py — Step 1: Download metadata and images from TCIA CFB-GBM.

Two modes:
  --mode metadata   Download series metadata only (fast, first step).
  --mode images     Download selected DICOM series (slow, full images).

Usage:
  python src/download_tcia.py --mode metadata --output data/processed/series_metadata.csv
  python src/download_tcia.py --mode images   --metadata data/processed/series_metadata.csv \
      --output data/raw/cfb_gbm [--patients GBM-001 GBM-002] [--modalities MR RTSTRUCT]
"""

import argparse
import time
import sys
import zipfile
import io
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config

logger = get_logger(__name__)

COLLECTION = "CFB-GBM"
BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


# ── NBIA REST helpers ─────────────────────────────────────────────────────────

def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> list[dict]:
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json() if r.content else []
        except Exception as exc:
            logger.warning(f"Attempt {attempt}/{retries} failed for {url}: {exc}")
            if attempt < retries:
                time.sleep(5 * attempt)
    logger.error(f"All {retries} attempts failed for {url}")
    return []


def get_patients(collection: str) -> list[str]:
    data = _get("getPatient", {"Collection": collection})
    return [d["PatientId"] for d in data]


def get_studies(patient_id: str, collection: str) -> list[dict]:
    return _get("getPatientStudy",
                {"PatientID": patient_id, "Collection": collection})


def get_series(study_uid: str) -> list[dict]:
    return _get("getSeries", {"StudyInstanceUID": study_uid})


def get_series_size(series_uid: str) -> int:
    data = _get("getSeriesSize", {"SeriesInstanceUID": series_uid})
    return int(data[0].get("TotalSizeInBytes", 0)) if data else 0


def download_series(series_uid: str, out_dir: Path, retries: int = 3) -> bool:
    url = f"{BASE_URL}/getImage"
    params = {"SeriesInstanceUID": series_uid}
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=300, stream=True)
            r.raise_for_status()
            content = b"".join(r.iter_content(chunk_size=1 << 20))
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(out_dir)
            return True
        except Exception as exc:
            logger.warning(f"Download attempt {attempt}/{retries} failed for {series_uid}: {exc}")
            if attempt < retries:
                time.sleep(10 * attempt)
    return False


# ── Metadata download ─────────────────────────────────────────────────────────

def download_metadata(collection: str, output_csv: Path,
                      patient_subset: list[str] | None = None) -> pd.DataFrame:
    logger.info(f"Fetching patient list for collection '{collection}'...")
    all_patients = get_patients(collection)
    logger.info(f"  Found {len(all_patients)} patients")

    if patient_subset:
        patients = [p for p in all_patients if p in patient_subset]
        logger.info(f"  Filtered to {len(patients)} patients per --patients flag")
    else:
        patients = all_patients

    records = []
    for pid in tqdm(patients, desc="Patients"):
        studies = get_studies(pid, collection)
        for study in studies:
            study_uid = study.get("StudyInstanceUID", "")
            study_date = study.get("StudyDate", "")
            series_list = get_series(study_uid)
            for s in series_list:
                records.append({
                    "PatientID":         pid,
                    "StudyInstanceUID":  study_uid,
                    "StudyDate":         study_date,
                    "SeriesInstanceUID": s.get("SeriesInstanceUID", ""),
                    "Modality":          s.get("Modality", ""),
                    "SeriesDescription": s.get("SeriesDescription", ""),
                    "NumImages":         s.get("ImageCount", 0),
                    "BodyPartExamined":  s.get("BodyPartExamined", ""),
                    "Manufacturer":      s.get("Manufacturer", ""),
                    "ManufacturerModelName": s.get("ManufacturerModelName", ""),
                })

    df = pd.DataFrame(records)
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    logger.info(f"Metadata saved → {output_csv}  ({len(df)} series)")
    return df


# ── Image download ────────────────────────────────────────────────────────────

def download_images(metadata_csv: Path, output_dir: Path,
                    modality_filter: list[str] | None = None,
                    patient_filter: list[str] | None = None) -> None:
    df = pd.read_csv(metadata_csv)
    if modality_filter:
        df = df[df["Modality"].isin(modality_filter)]
    if patient_filter:
        df = df[df["PatientID"].isin(patient_filter)]

    logger.info(f"Downloading {len(df)} series for {df['PatientID'].nunique()} patients...")

    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Series"):
        pid    = row["PatientID"]
        mod    = row["Modality"]
        s_uid  = row["SeriesInstanceUID"]
        dest   = output_dir / pid / mod / s_uid
        if dest.exists() and any(dest.iterdir()):
            continue  # already downloaded
        dest.mkdir(parents=True, exist_ok=True)
        ok = download_series(s_uid, dest)
        if not ok:
            failed.append(s_uid)
            logger.error(f"FAILED: {pid} / {mod} / {s_uid}")

    if failed:
        fail_file = output_dir / "failed_series.txt"
        fail_file.write_text("\n".join(failed))
        logger.warning(f"{len(failed)} series failed. See {fail_file}")
    else:
        logger.info("All series downloaded successfully.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Download TCIA CFB-GBM data")
    p.add_argument("--mode", choices=["metadata", "images"], default="metadata")
    p.add_argument("--output",   default="data/processed/series_metadata.csv")
    p.add_argument("--metadata", default="data/processed/series_metadata.csv",
                   help="Metadata CSV (required for --mode images)")
    p.add_argument("--patients", nargs="+", default=None,
                   help="Limit to specific patient IDs")
    p.add_argument("--modalities", nargs="+", default=["MR", "RTSTRUCT", "CT", "RTDOSE"],
                   help="Modalities to download")
    p.add_argument("--config", default="config/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    subset = args.patients or cfg["tcia"].get("patient_subset")

    if args.mode == "metadata":
        download_metadata(
            collection=cfg["tcia"]["collection"],
            output_csv=Path(args.output),
            patient_subset=subset,
        )
    else:
        download_images(
            metadata_csv=Path(args.metadata),
            output_dir=Path(args.output) if args.mode == "images"
                       else Path(cfg["paths"]["raw_dicom"]),
            modality_filter=args.modalities,
            patient_filter=subset,
        )


if __name__ == "__main__":
    main()
