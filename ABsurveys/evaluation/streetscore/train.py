# train.py
# coding: utf-8

"""Fine-tuning street-perception ViT models on pairwise AB-comparison data."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import (
    DEFAULT_METRICS,
    HF_REPO_ID,
    IMAGE_TRANSFORM,
    TRAIN_IMAGE_TRANSFORM,
    Net,
    get_model_filename,
)
from losses import (
    crossentropy_loss,
    f1_score,
    pair_loss,
    pair_accuracy,
    mixed_loss,
)
from uncertainty import calculate_entropy


# ---------------------------------------------------------------------------
# Early-stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when the chosen validation accuracy has not improved.

    Args:
        patience: Number of epochs without meaningful improvement before stopping.
            Pass ``None`` to disable early stopping entirely.
        min_delta: Minimum absolute improvement in accuracy that counts as progress.
            Pass ``None`` to disable early stopping entirely.
    """

    def __init__(self, patience: int | None = 4, min_delta: float | None = 0.005) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._enabled = patience is not None and min_delta is not None
        self._best: float = -1.0
        self._wait: int = 0

    def step(self, val_acc: float) -> bool:
        """Call once per epoch with the current validation accuracy.

        Returns:
            bool: True when training should stop (always False when disabled).
        """
        if not self._enabled:
            return False

        if val_acc >= self._best + self.min_delta:
            self._best = val_acc
            self._wait = 0
        else:
            self._wait += 1

        if self._wait >= self.patience:
            print(
                f"\n  Early stopping: chosen val_acc has not improved by >{self.min_delta:.3f} "
                f"for {self.patience} consecutive epochs. Best val_acc={self._best:.4f}"
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Plotting / history helpers
# ---------------------------------------------------------------------------

def _save_history(history: list[dict], csv_path: str) -> None:
    """Append epoch rows to the history CSV (creates it on first call)."""
    df = pd.DataFrame(history)
    write_header = not os.path.isfile(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def _plot_curves(csv_path: str, jpg_path: str, metric: str, loss_name: str = "Loss", acc_name: str = "Accuracy") -> None:
    """Read history CSV and save a figure showing training curves."""
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)
    df["_step"] = range(len(df))

    # Detect run boundaries: wherever epoch does NOT increment by 1.
    boundaries = [0] + (
        df.index[df["epoch"].diff().fillna(0) <= 0].tolist()
    ) + [len(df)]
    boundaries = sorted(set(boundaries))

    has_acc = "train_acc" in df.columns
    has_score_stats = "val_score_mean" in df.columns
    has_entropy = "entropy" in df.columns

    # Render a clean, informative multi-panel plot
    ncols = 2 + int(has_score_stats) + int(has_entropy)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    
    ax_loss = axes[0]
    ax_acc  = axes[1]
    
    idx = 2
    ax_stats = axes[idx] if has_score_stats else None
    if has_score_stats:
        idx += 1
    ax_ent = axes[idx] if has_entropy else None

    fig.suptitle(f"Training curves — {metric}", fontsize=13)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for run_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        seg = df.iloc[start:end]
        color = colors[run_idx % len(colors)]
        run_label = f" run {run_idx + 1}" if len(boundaries) > 2 else ""

        # Loss Curve
        train_loss_seg = seg.dropna(subset=["train_loss"])
        if not train_loss_seg.empty:
            ax_loss.plot(train_loss_seg["_step"], train_loss_seg["train_loss"],
                         label=f"train{run_label}", color=color,
                         linestyle="-", marker="o", markersize=3)
        val_loss_seg = seg.dropna(subset=["val_loss"])
        if not val_loss_seg.empty:
            ax_loss.plot(val_loss_seg["_step"], val_loss_seg["val_loss"],
                         label=f"val{run_label}", color=color,
                         linestyle="--", marker="s", markersize=3)

        # Accuracy Curves
        if has_acc:
            train_acc_seg = seg.dropna(subset=["train_acc"])
            if not train_acc_seg.empty:
                ax_acc.plot(train_acc_seg["_step"], train_acc_seg["train_acc"],
                            label=f"train{run_label}", color=color,
                            linestyle="-", marker="o", markersize=3)
            val_acc_seg = seg.dropna(subset=["val_acc"])
            if not val_acc_seg.empty:
                ax_acc.plot(val_acc_seg["_step"], val_acc_seg["val_acc"],
                            label=f"val{run_label}", color=color,
                            linestyle="--", marker="s", markersize=3)

        # Score Stats Curve (Mean and SD)
        if ax_stats is not None:
            stats_seg = seg.dropna(subset=["val_score_mean", "val_score_std"])
            if not stats_seg.empty:
                ax_stats.plot(stats_seg["_step"], stats_seg["val_score_mean"],
                             label=f"mean{run_label}", color=color,
                             linestyle="-", marker="o", markersize=3)
                ax_stats.fill_between(stats_seg["_step"],
                                      stats_seg["val_score_mean"] - stats_seg["val_score_std"],
                                      stats_seg["val_score_mean"] + stats_seg["val_score_std"],
                                      color=color, alpha=0.15, label=f"std{run_label}")

        # Entropy Curve
        if ax_ent is not None:
            ent_seg = seg.dropna(subset=["entropy"])
            if not ent_seg.empty:
                ax_ent.plot(ent_seg["_step"], ent_seg["entropy"],
                            label=f"val{run_label}", color=color,
                            linestyle="--", marker="s", markersize=3)

    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel(f"Loss ({loss_name})")
    ax_loss.legend(fontsize=9); ax_loss.grid(True, alpha=0.3)

    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel(f"Accuracy ({acc_name})")
    ax_acc.legend(fontsize=8); ax_acc.grid(True, alpha=0.3)

    if ax_stats is not None:
        ax_stats.set_xlabel("Epoch")
        ax_stats.set_ylabel("Score (0-10)")
        ax_stats.set_title("Val Score Distribution")
        ax_stats.legend(fontsize=9); ax_stats.grid(True, alpha=0.3)

    if ax_ent is not None:
        ax_ent.set_xlabel("Epoch")
        ax_ent.set_ylabel("Softmax Entropy (bits)")
        ax_ent.set_title("Prediction Confidence")
        ax_ent.legend(fontsize=9); ax_ent.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(jpg_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ABPairDataset(Dataset):
    """Converts AB-pair human preference judgments into structured pairs of images.

    Each item returns (image_A, image_B, label, is_tie) where label is 1 if image A is preferred
    and 0 if image B is preferred.
    For equal/tie pairs, they are duplicated into two samples:
    one with label 1 (A is preferred) and is_tie=1,
    the other with label 0 (B is preferred) and is_tie=1.
    """

    def __init__(
        self,
        human_df: pd.DataFrame,
        img_df: pd.DataFrame,
        img_base_dir: str = "",
        transform=None,
        double_length: bool = False,
    ) -> None:
        self.transform = transform or IMAGE_TRANSFORM

        # Build img_id → (path, base_dir) lookup
        has_base_dir_col = "_base_dir" in img_df.columns
        self.img_lookup: dict[str, tuple[str, str]] = {}
        for _, row in img_df.iterrows():
            iid = str(row["img_id"])
            path = str(row["path"])
            base = str(row["_base_dir"]) if has_base_dir_col else img_base_dir
            self.img_lookup[iid] = (path, base)

        # Record valid AB comparison pairs
        self.samples: list[tuple[str, str, int, int]] = []
        tie_count = 0

        for _, row in human_df.iterrows():
            img_id_a = str(row["img_id_A"])
            img_id_b = str(row["img_id_B"])
            answer = row["answer"]

            if img_id_a not in self.img_lookup or img_id_b not in self.img_lookup:
                continue

            if answer == "A":
                self.samples.append((img_id_a, img_id_b, 1, 0))
            elif answer == "B":
                self.samples.append((img_id_a, img_id_b, 0, 0))
            elif answer == "=":
                tie_count += 1
                # Duplicate the pair as instructed (one with label 1 for A, one with label 0 for B)
                self.samples.append((img_id_a, img_id_b, 1, 1))
                self.samples.append((img_id_a, img_id_b, 0, 1))

        print(f"  [Dataset] Processed {len(human_df)} raw rows. Found {tie_count} tie ('=') pairs.")
        print(f"  [Dataset] Verified doubles for ties: each '=' pair is split into 2 samples (once with label 1/A preferred, once with label 0/B preferred). Total tie samples = {tie_count * 2}.")

        if double_length:
            orig_len = len(self.samples)
            self.samples = self.samples * 2
            print(f"  [Dataset] Doubled training dataset length from {orig_len} to {len(self.samples)} to allow multiple augmentation views per epoch.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        img_id_a, img_id_b, label, is_tie = self.samples[idx]

        # Process Image A
        path_a, base_a = self.img_lookup[img_id_a]
        if not os.path.isabs(path_a):
            path_a = os.path.join(base_a, path_a) if base_a else path_a
        img_a = Image.open(path_a)
        if img_a.mode != "RGB":
            img_a = img_a.convert("RGB")

        # Process Image B
        path_b, base_b = self.img_lookup[img_id_b]
        if not os.path.isabs(path_b):
            path_b = os.path.join(base_b, path_b) if base_b else path_b
        img_b = Image.open(path_b)
        if img_b.mode != "RGB":
            img_b = img_b.convert("RGB")

        return self.transform(img_a), self.transform(img_b), label, is_tie


class SingleImageDataset(Dataset):
    """Converts single-image label rows (output of build_single_image_df / resolve_pairs_image_level)
    into a Dataset compatible with the existing training loop.

    Used when loss_hyperparam == "crossentropy" to correctly handle cross-split
    pairs: only the image that belongs to this split is loaded; the other image
    from the pair is not needed and never loaded.

    To keep the training loop unchanged (it expects batches of image_A, image_B,
    label, is_tie), we synthesise a *dummy* second image by returning the same
    tensor for both slots.  The crossentropy loss path only uses the concatenated
    logits and the symmetric targets derived from label, so both slots contribute
    meaningfully:

        images = cat([image_A, image_B], dim=0)   → [img, img]
        targets = cat([label, 1-label], dim=0)     → [label, 1-label]

    This means each single-image row is seen twice per forward pass (once as its
    true label, once as the complement), which is exactly the binary cross-entropy
    convention and does not introduce any leakage.

    Args:
        single_img_df: Output of :func:`build_single_image_df` or
            :func:`resolve_pairs_image_level` — must have columns
            ``img_id``, ``label``, ``is_tie``.
        img_df: Image registry DataFrame with ``img_id`` and ``path`` columns.
        img_base_dir: Fallback base directory when ``_base_dir`` column is absent.
        transform: Image transform to apply (default: IMAGE_TRANSFORM).
        double_length: If True, duplicate the sample list for augmentation variety.
    """

    def __init__(
        self,
        single_img_df: pd.DataFrame,
        img_df: pd.DataFrame,
        img_base_dir: str = "",
        transform=None,
        double_length: bool = False,
    ) -> None:
        self.transform = transform or IMAGE_TRANSFORM

        has_base_dir_col = "_base_dir" in img_df.columns
        self.img_lookup: dict[str, tuple[str, str]] = {}
        for _, row in img_df.iterrows():
            iid = str(row["img_id"])
            path = str(row["path"])
            base = str(row["_base_dir"]) if has_base_dir_col else img_base_dir
            self.img_lookup[iid] = (path, base)

        # samples: (img_id, label, is_tie)
        self.samples: list[tuple[str, int, int]] = []
        tie_count = 0

        for _, row in single_img_df.iterrows():
            img_id = str(row["img_id"])
            if img_id not in self.img_lookup:
                continue
            label = int(row["label"])
            is_tie = int(row["is_tie"])
            if is_tie:
                tie_count += 1
            self.samples.append((img_id, label, is_tie))

        print(f"  [SingleImageDataset] {len(self.samples):,} single-image samples "
              f"({tie_count:,} from tie pairs).")

        if double_length:
            orig_len = len(self.samples)
            self.samples = self.samples * 2
            print(f"  [SingleImageDataset] Doubled length: {orig_len} → {len(self.samples)}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        img_id, label, is_tie = self.samples[idx]
        path, base = self.img_lookup[img_id]
        if not os.path.isabs(path):
            path = os.path.join(base, path) if base else path
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self.transform(img)
        # Return the same tensor for both A and B slots; the loop uses cat([A,B])
        # and targets = cat([label, 1-label]) — both logits see the true image,
        # which is correct for single-image cross-entropy.
        return tensor, tensor, label, is_tie


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _score_images(
    model: nn.Module,
    img_ids: list[str],
    img_lookup: dict,
    transform,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, float]:
    """Score unique images deterministically (0-10)."""
    model.eval()
    scores: dict[str, float] = {}

    for i in range(0, len(img_ids), batch_size):
        batch_ids = img_ids[i : i + batch_size]
        tensors = []
        valid_ids = []
        for iid in batch_ids:
            if iid not in img_lookup:
                continue
            path, base = img_lookup[iid]
            if not os.path.isabs(path):
                path = os.path.join(base, path) if base else path
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                tensors.append(transform(img))
                valid_ids.append(iid)
            except Exception:
                pass

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(preferred)
        for iid, p in zip(valid_ids, probs.cpu().numpy()):
            scores[iid] = float(p) * 10.0

    return scores


def _compute_unique_image_stats(scores: dict[str, float]) -> tuple[float, float, float]:
    """Compute mean score, standard deviation, and entropy for validation set.

    Returns:
        tuple[float, float, float]: (mean score, std score, mean entropy).
    """
    if not scores:
        return float("nan"), float("nan"), float("nan")
    score_vals = np.array(list(scores.values()))
    mean = float(np.mean(score_vals))
    std = float(np.std(score_vals))
    
    # Calculate Shannon Entropy
    p_vals = score_vals / 10.0
    p_clamped = np.clip(p_vals, 1e-9, 1.0 - 1e-9)
    entropies = -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))
    mean_entropy = float(np.mean(entropies))
    
    return mean, std, mean_entropy


def _evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_hyperparam: str = "pair",
    accuracy_hyperparam: str = "pair",
) -> tuple[float, float]:
    """Run model evaluation over a dataloader of image pairs.

    Returns:
        tuple[float, float]: (average active loss, active accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct_acc = 0
    total_samples = 0

    with torch.no_grad():
        for images_A, images_B, labels, is_tie in loader:
            images_A = images_A.to(device)
            images_B = images_B.to(device)
            labels = labels.to(device)
            is_tie = is_tie.to(device)

            # Combined forward pass
            images = torch.cat([images_A, images_B], dim=0)
            logits = model(images)
            logits_A, logits_B = torch.chunk(logits, 2, dim=0)

            # Standard targets for cross entropy (flattened)
            targets = torch.cat([labels, 1 - labels], dim=0).long()

            # Scores (scaled probabilities)
            scores_A = torch.softmax(logits_A, dim=1)[:, 1] * 10.0
            scores_B = torch.softmax(logits_B, dim=1)[:, 1] * 10.0

            # Compute specific loss
            if loss_hyperparam == "crossentropy":
                loss_val = crossentropy_loss(logits, targets)
            elif loss_hyperparam == "mixed":
                loss_val = mixed_loss(logits, scores_A, scores_B, labels, targets, is_tie=is_tie)
            else:
                loss_val = pair_loss(scores_A, scores_B, labels.float(), is_tie=is_tie)

            # Compute specific accuracy
            if accuracy_hyperparam in ("single", "crossentropy"):
                acc_val = f1_score(logits, targets)
                num_items = targets.size(0)
            else:
                acc_val = pair_accuracy(scores_A, scores_B, labels, is_tie=is_tie)
                num_items = labels.size(0)

            total_loss += loss_val.item() * labels.size(0)
            correct_acc += int(acc_val * num_items)
            total_samples += num_items

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = correct_acc / total_samples

    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# Helper for crossentropy val unique-image scoring
# ---------------------------------------------------------------------------

def _make_dummy_val_human_df(single_img_df: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal synthetic human_df so _score_images can collect unique val img_ids.

    In crossentropy / single-image mode the val human_df may not exist (cross-split
    pairs were split at the image level).  We synthesise a two-column DataFrame with
    img_id_A and img_id_B both set to each unique img_id so the unique-image
    collection logic in the training loop still works correctly.

    Args:
        single_img_df: Output of resolve_pairs_image_level for the val split.

    Returns:
        DataFrame with ``img_id_A`` and ``img_id_B`` columns (both equal to img_id).
    """
    unique_ids = single_img_df["img_id"].astype(str).unique().tolist()
    return pd.DataFrame({"img_id_A": unique_ids, "img_id_B": unique_ids})


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    human_df: pd.DataFrame,
    img_df: pd.DataFrame,
    metric: str,
    model_folder: str,
    val_human_df: pd.DataFrame | None = None,
    val_img_df: pd.DataFrame | None = None,
    # Single-image DataFrames for crossentropy mode (output of resolve_pairs_image_level)
    single_img_train_df: pd.DataFrame | None = None,
    single_img_val_df: pd.DataFrame | None = None,
    from_checkpoint: str | None = None,
    vit_weights: bool = True,
    freeze_vit: bool = False,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    num_workers: int = 4,
    device: torch.device | None = None,
    pretrained_model_dir: str = "models/default_models",
    early_stopping_patience: int | None = 4,
    early_stopping_min_delta: float | None = 0.005,
    # Added hyperparams
    loss_hyperparam: str = "pair",
    accuracy_hyperparam: str = "pair",
    save_last_epoch: bool = False,
) -> str:
    """Fine-tune a perception model on AB-survey data.

    Dataset selection
    -----------------
    When ``loss_hyperparam == "crossentropy"`` and ``single_img_train_df`` is
    provided, the training Dataset is built as a :class:`SingleImageDataset`
    from the single-image rows.  This correctly includes cross-split pairs (where
    one image belongs to a different split) because the crossentropy loss is
    computed per-image independently — there is no leakage risk.

    When ``loss_hyperparam`` is ``"pair"`` or ``"mixed"``, the Dataset is the
    standard :class:`ABPairDataset` where BOTH images of every pair must belong
    to the same split (pairs that span the boundary are dropped upstream in
    :func:`dataset.resolve_pairs_for_val_split`).

    Args:
        human_df: AB-pair human preference judgments for training (pair/mixed mode).
        img_df: Image registry.
        metric: Name of the metric (e.g. "walk").
        model_folder: Directory where models, history, and plots are written.
        val_human_df: AB-pair human preference judgments for validation (pair/mixed mode).
        val_img_df: Image registry for validation.
        single_img_train_df: Single-image label rows for crossentropy training.
            When provided and loss is "crossentropy", takes precedence over human_df.
        single_img_val_df: Single-image label rows for crossentropy validation.
            When provided and loss is "crossentropy", takes precedence over val_human_df.
        from_checkpoint: Path or metric name of pretrained model to load.
        vit_weights: Whether to initialize with ImageNet ViT weights.
        freeze_vit: Whether to freeze ViT backbone.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        num_workers: Number of dataloader workers.
        device: Torch device.
        pretrained_model_dir: Base directory for pretrained models.
        early_stopping_patience: Patience for early stopping.
        early_stopping_min_delta: Minimum progress delta for early stopping.
        loss_hyperparam: Loss function to optimize ("pair", "crossentropy", or "mixed").
        accuracy_hyperparam: Accuracy function to monitor ("pair" or "crossentropy"/"single").
        save_last_epoch: If True, always overwrite the checkpoint with the model from the
            most recent epoch, regardless of whether validation accuracy improved.
            The best-accuracy checkpoint logic still runs in parallel — the history CSV
            still records ``checkpoint=True`` only on accuracy improvements — but the
            .pth file on disk will always reflect the last completed epoch.
            Useful when you want the freshest weights rather than the most validated ones,
            e.g. when fine-tuning for a fixed number of epochs on a small dataset.

    Returns:
        str: Absolute path to the saved best model.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Training device: {device} | Optimizing Loss: '{loss_hyperparam}' | Monitoring Accuracy: '{accuracy_hyperparam}'")

    # ── Decide which dataset class to use ───────────────────────────────────
    use_single_image_mode = (
        loss_hyperparam == "crossentropy"
        and single_img_train_df is not None
        and len(single_img_train_df) > 0
    )

    if use_single_image_mode:
        print("  [Dataset mode] crossentropy + single_img_train_df → SingleImageDataset "
              "(cross-split pairs included via image-level resolution).")
        train_dataset = SingleImageDataset(
            single_img_df=single_img_train_df,
            img_df=img_df,
            transform=TRAIN_IMAGE_TRANSFORM,
            double_length=True,
        )
    else:
        train_dataset = ABPairDataset(
            human_df=human_df,
            img_df=img_df,
            transform=TRAIN_IMAGE_TRANSFORM,
            double_length=True,
        )
    print(f"  Train dataset size: {len(train_dataset):,} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    has_val = False
    val_loader = None
    val_img_lookup: dict = {}
    effective_val_human_df = val_human_df  # used for unique image scoring in val phase

    use_single_image_val = (
        loss_hyperparam == "crossentropy"
        and single_img_val_df is not None
        and len(single_img_val_df) > 0
    )

    if use_single_image_val:
        print("  [Val mode] crossentropy + single_img_val_df → SingleImageDataset for validation.")
        effective_val_img_df = val_img_df if val_img_df is not None else img_df
        val_dataset = SingleImageDataset(
            single_img_df=single_img_val_df,
            img_df=effective_val_img_df,
            transform=IMAGE_TRANSFORM,
        )
        if len(val_dataset) > 0:
            has_val = True
            print(f"  Val dataset size: {len(val_dataset):,} samples")
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            has_base_dir_col = "_base_dir" in effective_val_img_df.columns
            for _, row in effective_val_img_df.iterrows():
                iid = str(row["img_id"])
                path = str(row["path"])
                base = str(row["_base_dir"]) if has_base_dir_col else ""
                val_img_lookup[iid] = (path, base)
            # Build a synthetic val_human_df so unique-image scoring still works.
            # We derive it from single_img_val_df: unique img_ids scored individually.
            effective_val_human_df = _make_dummy_val_human_df(single_img_val_df)
        else:
            print("  No validation samples in SingleImageDataset — skipping validation.")
    elif val_human_df is not None and len(val_human_df) > 0:
        effective_val_img_df = val_img_df if val_img_df is not None else img_df
        val_dataset = ABPairDataset(
            human_df=val_human_df,
            img_df=effective_val_img_df,
            transform=IMAGE_TRANSFORM,
        )
        if len(val_dataset) > 0:
            has_val = True
            print(f"  Val dataset size: {len(val_dataset):,} pairs")
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            has_base_dir_col = "_base_dir" in effective_val_img_df.columns
            for _, row in effective_val_img_df.iterrows():
                iid = str(row["img_id"])
                path = str(row["path"])
                base = str(row["_base_dir"]) if has_base_dir_col else ""
                val_img_lookup[iid] = (path, base)

    if not has_val:
        print("  No validation set supplied — skipping validation and early stopping.")

    # Model instantiation / checkpoint loading
    model_path = os.path.join(model_folder, get_model_filename(metric))
    is_resuming = os.path.isfile(model_path)

    if is_resuming:
        print(f"Loading existing model {model_path} for resuming …")
        import model as _model_module
        sys.modules["Model_01"] = _model_module
        with torch.serialization.safe_globals([Net]):
            model = torch.load(model_path, map_location=device, weights_only=False)
        if freeze_vit:
            if isinstance(model, nn.DataParallel):
                model.module.freeze_backbone()
            else:
                model.freeze_backbone()
    elif from_checkpoint is not None:
        print(f"Loading checkpoint from {from_checkpoint} …")
        from_ckpt_path = os.path.join(
            pretrained_model_dir,
            f"{from_checkpoint.lower()}.pth",
        )
        if not os.path.isfile(from_ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {from_ckpt_path}")
        import model as _model_module
        sys.modules["Model_01"] = _model_module
        with torch.serialization.safe_globals([Net]):
            model = torch.load(from_ckpt_path, map_location=device, weights_only=False)
        if freeze_vit:
            if isinstance(model, nn.DataParallel):
                model.module.freeze_backbone()
            else:
                model.freeze_backbone()
    else:
        print(f"Building fresh model (vit_weights={vit_weights}, freeze_vit={freeze_vit}) …")
        model = Net(num_classes=2, vit_weights=vit_weights, freeze_vit=freeze_vit)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Prepare outputs
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(model_folder, get_model_filename(metric))
    csv_path  = os.path.join(model_folder, f"{metric}_history.csv")
    jpg_path  = os.path.join(model_folder, f"{metric}_curves.jpg")

    epoch_offset = 0
    best_accuracy: float = float("-inf")
    if is_resuming:
        try:
            hist_existing = pd.read_csv(csv_path)
            if not hist_existing.empty and "epoch" in hist_existing.columns:
                if "checkpoint" in hist_existing.columns:
                    # Find rows where checkpoint is True
                    is_ckpt_true = hist_existing["checkpoint"].astype(str).str.strip().str.lower() == "true"
                    ckpt_rows = hist_existing[is_ckpt_true]
                    if not ckpt_rows.empty:
                        # Find the row with the maximum epoch that has checkpoint == True
                        last_ckpt_row = ckpt_rows.loc[ckpt_rows["epoch"].idxmax()]
                        epoch_offset = int(last_ckpt_row["epoch"])
                        
                        # Truncate hist_existing up to the index of last_ckpt_row
                        idx_last_ckpt = last_ckpt_row.name
                        hist_existing_truncated = hist_existing.iloc[:idx_last_ckpt + 1]
                        hist_existing_truncated.to_csv(csv_path, index=False)
                        
                        if "val_acc" in last_ckpt_row and not pd.isna(last_ckpt_row["val_acc"]):
                            best_accuracy = float(last_ckpt_row["val_acc"])
                        
                        print(f"  Resuming from epoch {epoch_offset} (loaded checkpoint with best accuracy {best_accuracy:.4f})")
                        print(f"  Truncated history CSV to epoch {epoch_offset} (removed unsaved training progress after last checkpoint)")
                        _plot_curves(csv_path, jpg_path, metric, loss_hyperparam, accuracy_hyperparam)
                    else:
                        epoch_offset = int(hist_existing["epoch"].max())
                        print(f"  Resuming from epoch {epoch_offset} (no checkpoint marking found, keeping all history)")
                else:
                    epoch_offset = int(hist_existing["epoch"].max())
                    print(f"  Resuming from epoch {epoch_offset} (no checkpoint column found, keeping all history)")
        except Exception as e:
            print(f"  Warning: failed to parse existing history: {e}")

    precomputed_val_loss = None
    precomputed_val_acc = None

    if is_resuming or (from_checkpoint is not None):
        if val_loader is not None:
            print(f"  [Checkpoint Loading] Recomputing validation parameters before starting training...")
            val_loss, val_acc = _evaluate_epoch(model, val_loader, device, loss_hyperparam, accuracy_hyperparam)
            print(f"  [Checkpoint Loading] Recomputed validation: val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
            print(f"  [Checkpoint Loading] Updating best_accuracy from historical {best_accuracy:.4f} to recomputed {val_acc:.4f}")
            best_accuracy = val_acc
            precomputed_val_loss = val_loss
            precomputed_val_acc = val_acc

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    # Early stopping is disabled when save_last_epoch=True (saving the final epoch
    # only makes sense if all epochs run), or when patience/min_delta are None.
    effective_patience   = None if save_last_epoch else early_stopping_patience
    effective_min_delta  = None if save_last_epoch else early_stopping_min_delta
    if save_last_epoch and (early_stopping_patience is not None or early_stopping_min_delta is not None):
        print("  Early stopping disabled (save_last_epoch=True — all epochs will run).")
    elif effective_patience is None or effective_min_delta is None:
        print("  Early stopping disabled (patience or min_delta is None).")
    early_stop = EarlyStopping(
        patience=effective_patience,
        min_delta=effective_min_delta,
    )
    
    history: list[dict] = []

    # Epoch 0: Baseline Evaluation
    if epoch_offset == 0:
        if val_loader is not None:
            if precomputed_val_loss is not None and precomputed_val_acc is not None:
                val_loss = precomputed_val_loss
                val_acc = precomputed_val_acc
                print(f"  Using precomputed validation parameters from checkpoint load.")
            else:
                val_loss, val_acc = _evaluate_epoch(model, val_loader, device, loss_hyperparam, accuracy_hyperparam)
            
            # Score validation set unique images
            all_ids = list(
                set(effective_val_human_df["img_id_A"].astype(str).tolist())
                | set(effective_val_human_df["img_id_B"].astype(str).tolist())
            )
            val_scores = _score_images(model, all_ids, val_img_lookup, IMAGE_TRANSFORM, device)
            val_score_mean, val_score_std, val_entropy = _compute_unique_image_stats(val_scores)

            epoch0_row = {
                "epoch": 0,
                "train_loss": float("nan"),
                "train_acc": float("nan"),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_score_mean": round(val_score_mean, 4),
                "val_score_std": round(val_score_std, 4),
                "entropy": round(val_entropy, 4),
                "checkpoint": False,
            }
            # Set initial best accuracy
            best_accuracy = val_acc
        else:
            epoch0_row = {
                "epoch": 0,
                "train_loss": float("nan"),
                "train_acc": float("nan"),
                "val_loss": float("nan"),
                "val_acc": float("nan"),
                "val_score_mean": float("nan"),
                "val_score_std": float("nan"),
                "entropy": float("nan"),
                "checkpoint": False,
            }

        print(f"Epoch   0 (Baseline before training):")
        if val_loader is not None:
            print(f"  val_loss={epoch0_row['val_loss']:.4f} | val_acc={epoch0_row['val_acc']:.4f} | "
                  f"score_mean={epoch0_row['val_score_mean']:.2f} | "
                  f"score_std={epoch0_row['val_score_std']:.2f} | entropy={epoch0_row['entropy']:.4f}")
        else:
            print("  (No validation set provided)")
        
        history.append(epoch0_row)
        _save_history([epoch0_row], csv_path)
        _plot_curves(csv_path, jpg_path, metric, loss_hyperparam, accuracy_hyperparam)

    # Main Training Loop
    for epoch in range(1, epochs + 1):
        global_epoch = epoch_offset + epoch

        # --- Train Phase ---
        model.train()
        train_loss = 0.0
        correct_acc = 0
        total_samples = 0

        for images_A, images_B, labels, is_tie in tqdm(train_loader, desc=f"Epoch {global_epoch} [train]"):
            images_A = images_A.to(device)
            images_B = images_B.to(device)
            labels = labels.to(device)
            is_tie = is_tie.to(device)

            optimizer.zero_grad()

            # Single forward pass for both sets of images
            images = torch.cat([images_A, images_B], dim=0)
            logits = model(images)
            logits_A, logits_B = torch.chunk(logits, 2, dim=0)

            # Targets for cross entropy classification
            targets = torch.cat([labels, 1 - labels], dim=0).long()

            # Scores (scaled probabilities)
            scores_A = torch.softmax(logits_A, dim=1)[:, 1] * 10.0
            scores_B = torch.softmax(logits_B, dim=1)[:, 1] * 10.0

            # Select optimized loss
            if loss_hyperparam == "crossentropy":
                loss = crossentropy_loss(logits, targets)
            elif loss_hyperparam == "mixed":
                loss = mixed_loss(logits, scores_A, scores_B, labels, targets, is_tie=is_tie)
            else:
                loss = pair_loss(scores_A, scores_B, labels.float(), is_tie=is_tie)

            # Select monitored accuracy
            if accuracy_hyperparam in ("single", "crossentropy"):
                acc_val = f1_score(logits, targets)
                num_items = targets.size(0)
            else:
                acc_val = pair_accuracy(scores_A, scores_B, labels, is_tie=is_tie)
                num_items = labels.size(0)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            correct_acc += int(acc_val * num_items)
            total_samples += num_items

        train_loss /= len(train_loader.dataset)
        train_acc = correct_acc / total_samples

        # --- Validation Phase ---
        if val_loader is not None:
            val_loss, val_acc = _evaluate_epoch(model, val_loader, device, loss_hyperparam, accuracy_hyperparam)
            
            # Score unique images
            all_ids = list(
                set(effective_val_human_df["img_id_A"].astype(str).tolist())
                | set(effective_val_human_df["img_id_B"].astype(str).tolist())
            )
            val_scores = _score_images(model, all_ids, val_img_lookup, IMAGE_TRANSFORM, device)
            val_score_mean, val_score_std, val_entropy = _compute_unique_image_stats(val_scores)
        else:
            val_loss = float("nan")
            val_acc = float("nan")
            val_score_mean = float("nan")
            val_score_std = float("nan")
            val_entropy = float("nan")

        # Logging output
        print(
            f"Epoch {global_epoch:>3} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"score_mean={val_score_mean:.2f} | score_std={val_score_std:.2f} | "
            f"entropy={val_entropy:.4f}"
        )

        # Model Checkpoint Saving Logic (based on chosen Accuracy hyperparam)
        is_saved = False
        if val_loader is not None:
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                core = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(core, save_path)
                print(f"  ✓ Checkpoint saved (val_acc={val_acc:.4f} improved over best {best_accuracy:.4f})")
                is_saved = True
            elif save_last_epoch:
                core = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(core, save_path)
                print(f"  ✓ Last-epoch checkpoint saved (val_acc={val_acc:.4f} <= best {best_accuracy:.4f}, save_last_epoch=True)")
            else:
                print(f"  ✗ Not saving (val_acc={val_acc:.4f} <= best {best_accuracy:.4f})")
        else:
            # Without validation, always save latest model
            core = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(core, save_path)
            print("  ✓ Checkpoint saved (no validation set — saving every epoch)")
            is_saved = True

        row = {
            "epoch": global_epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_score_mean": round(val_score_mean, 4),
            "val_score_std": round(val_score_std, 4),
            "entropy": round(val_entropy, 4),
            "checkpoint": is_saved,
        }
        history.append(row)
        _save_history([row], csv_path)
        _plot_curves(csv_path, jpg_path, metric, loss_hyperparam, accuracy_hyperparam)

        # Early stopping check
        if val_loader is not None:
            if early_stop.step(val_acc):
                break

    print(f"\nTraining completed. Saved best model to: {save_path}")
    return os.path.abspath(save_path)