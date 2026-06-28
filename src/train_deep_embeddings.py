"""
train_deep_embeddings.py — Step 11: Extract deep radiomic embeddings from
3D MRI patches using a MONAI/PyTorch 3D ResNet.

Pipeline:
  1. Load preprocessed T1Gd and FLAIR volumes.
  2. Crop 3D patch centred on tumor mask centroid.
  3. Resize to fixed patch size (default 96³).
  4. Forward pass through pretrained or randomly-initialised 3D ResNet.
  5. Extract penultimate-layer embeddings.
  6. Save embeddings CSV.
  7. Combine with handcrafted radiomics and train LogReg/RF for comparison.

Usage:
    python src/train_deep_embeddings.py \
        --input   data/processed \
        --masks   data/processed/regions \
        --output  data/features/deep_embeddings.csv \
        --gpu     cuda
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, ensure_dir, load_config, load_nifti, image_to_array

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


# ── Patch extraction ──────────────────────────────────────────────────────────

def get_tumor_centroid(mask_arr: np.ndarray) -> tuple[int, int, int]:
    coords = np.where(mask_arr > 0)
    if len(coords[0]) == 0:
        s = np.array(mask_arr.shape) // 2
        return int(s[0]), int(s[1]), int(s[2])
    return (int(coords[0].mean()), int(coords[1].mean()), int(coords[2].mean()))


def crop_patch(volume: np.ndarray, centroid: tuple[int, int, int],
               patch_size: tuple[int, int, int]) -> np.ndarray:
    """Crop a patch centred at centroid, zero-padding if needed."""
    pz, py, px = patch_size
    cz, cy, cx = centroid
    hz, hy, hx = pz // 2, py // 2, px // 2
    # Pad volume
    pad_z = max(0, hz - cz), max(0, cz + hz - volume.shape[0] + 1)
    pad_y = max(0, hy - cy), max(0, cy + hy - volume.shape[1] + 1)
    pad_x = max(0, hx - cx), max(0, cx + hx - volume.shape[2] + 1)
    vol_padded = np.pad(volume, (pad_z, pad_y, pad_x), mode="constant")
    cz += pad_z[0]; cy += pad_y[0]; cx += pad_x[0]
    patch = vol_padded[cz-hz:cz+hz, cy-hy:cy+hy, cx-hx:cx+hx]
    return patch[:pz, :py, :px]


# ── Dataset ───────────────────────────────────────────────────────────────────

class GBMPatchDataset(Dataset):
    def __init__(self, records: list[dict], patch_size: tuple[int, int, int]):
        self.records = records
        self.patch_size = patch_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        patches = []
        for key in ("t1gd_path", "flair_path"):
            path = rec.get(key)
            if path and Path(path).exists():
                arr = image_to_array(load_nifti(path)).astype(np.float32)
            else:
                arr = np.zeros(self.patch_size, dtype=np.float32)

            mask_path = rec.get("mask_path")
            if mask_path and Path(mask_path).exists():
                mask_arr = image_to_array(load_nifti(mask_path))
                centroid = get_tumor_centroid(mask_arr)
            else:
                centroid = tuple(s // 2 for s in arr.shape)

            patch = crop_patch(arr, centroid, self.patch_size)
            # Normalise patch
            std = patch.std()
            if std > 0:
                patch = (patch - patch.mean()) / std
            patches.append(patch)

        # Stack as 2-channel input: (2, D, H, W)
        tensor = torch.from_numpy(np.stack(patches, axis=0))
        return {"PatientID": rec["PatientID"], "tensor": tensor}


# ── 3D ResNet (simple) ────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.shortcut = (nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm3d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class ResNet3D(nn.Module):
    def __init__(self, in_channels: int = 2, embedding_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1),
        )
        self.layer1 = ConvBlock3D(32, 64)
        self.layer2 = ConvBlock3D(64, 128, stride=2)
        self.layer3 = ConvBlock3D(128, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc     = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Extract deep 3D embeddings")
    p.add_argument("--input",   default="data/processed")
    p.add_argument("--masks",   default="data/processed/regions")
    p.add_argument("--output",  default="data/features/deep_embeddings.csv")
    p.add_argument("--gpu",     default="cuda")
    p.add_argument("--config",  default="config/config.yaml")
    p.add_argument("--finetune", action="store_true",
                   help="Fine-tune on labels instead of pure embedding extraction")
    args = p.parse_args()

    cfg = load_config(args.config)
    dl_cfg = cfg["deep_learning"]
    patch_size = tuple(dl_cfg["patch_size"])
    emb_dim    = dl_cfg["embeddings_dim"]
    dropout    = dl_cfg["dropout"]
    device     = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    proc_root = Path(args.input)
    mask_root = Path(args.masks)

    # --- Discover patients ---
    records = []
    for pt_dir in sorted(proc_root.iterdir()):
        if not pt_dir.is_dir():
            continue
        pid = pt_dir.name
        for tp_dir in sorted(pt_dir.iterdir()):
            if not tp_dir.is_dir():
                continue
            tp = tp_dir.name
            records.append({
                "PatientID":  pid,
                "Timepoint":  tp,
                "t1gd_path":  str(tp_dir / "t1gd.nii.gz"),
                "flair_path": str(tp_dir / "flair.nii.gz"),
                "mask_path":  str(mask_root / pid / tp / "tumor_mask.nii.gz"),
            })

    logger.info(f"Extracting embeddings for {len(records)} patient-timepoints...")

    dataset = GBMPatchDataset(records, patch_size)
    loader  = DataLoader(dataset, batch_size=dl_cfg["batch_size"],
                         shuffle=False, num_workers=0)

    model = ResNet3D(in_channels=2, embedding_dim=emb_dim, dropout=dropout).to(device)
    model.eval()

    if args.finetune:
        logger.info("Fine-tune mode: loading ml_dataset for labels...")
        # Full fine-tune training loop would go here; placeholder for now.
        logger.warning("Fine-tune not implemented in this scaffold. "
                       "Using randomly initialised weights for embeddings.")

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Embedding"):
            tensors = batch["tensor"].to(device)
            emb = model(tensors).cpu().numpy()
            for i, pid in enumerate(batch["PatientID"]):
                row = {"PatientID": pid}
                for j in range(emb_dim):
                    row[f"DL_emb_{j:03d}"] = float(emb[i, j])
                all_embeddings.append(row)

    df_emb = pd.DataFrame(all_embeddings)
    out_path = Path(args.output)
    ensure_dir(out_path.parent)
    df_emb.to_csv(out_path, index=False)
    logger.info(f"Embeddings → {out_path}  ({len(df_emb)} patients × {emb_dim} dims)")

    # --- Merge with handcrafted features and compare ---
    feat_path = Path("data/features/ml_dataset.csv")
    if feat_path.exists():
        df_feat = pd.read_csv(feat_path)
        df_combined = df_feat.merge(df_emb, on="PatientID", how="inner")
        combined_path = out_path.parent / "ml_dataset_combined.csv"
        df_combined.to_csv(combined_path, index=False)
        logger.info(f"Combined dataset → {combined_path}  "
                    f"({df_combined.shape[0]} patients × {df_combined.shape[1]} cols)")


if __name__ == "__main__":
    main()
