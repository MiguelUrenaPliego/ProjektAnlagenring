# inference.py
# coding: utf-8

"""Inference utilities for street perception scoring with uncertainty quantification."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm import tqdm

from model import (
    HF_REPO_ID,
    IMAGE_TRANSFORM,
    Net,
    get_model_filename,
)
from uncertainty import mc_dropout_passes, get_score_confidence_explanation


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def download_pretrained_models(model_dir: str) -> None:
    """Download the pretrained .pth files from Hugging Face into *model_dir*.

    Safe to call even when the files already exist (HF hub is idempotent).
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=["*.pth", "README.md"],
        local_dir=model_dir,
    )


def load_model(
    metric: str,
    model_dir: str,
    device: torch.device,
) -> nn.Module:
    """Load a trained perception model from *model_dir*.

    The file is expected at ``{model_dir}/{metric_lowercase}.pth``.

    Args:
        metric (str): Metric name (e.g. ``"safety"``).
        model_dir (str): Directory containing the .pth files.
        device (torch.device): Torch device to map the model to.

    Returns:
        nn.Module: Loaded model in eval mode.
    """
    model_path = os.path.join(model_dir, get_model_filename(metric))

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Run download_pretrained_models('{model_dir}') first, or check "
            "the model_dir / metric name."
        )

    import model as _model_module
    sys.modules["Model_01"] = _model_module

    with torch.serialization.safe_globals([Net]):
        loaded = torch.load(
            model_path,
            map_location=device,
            weights_only=False,
        )

    if torch.cuda.device_count() > 1:
        loaded = nn.DataParallel(loaded)

    loaded = loaded.to(device)
    loaded.eval()
    return loaded


# ---------------------------------------------------------------------------
# Single-image prediction with multiple uncertainty models
# ---------------------------------------------------------------------------

def predict_image_score(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    mc_passes: int = 0,
) -> tuple[float, float | None, float]:
    """Return a perception score and dual-uncertainty values for one image.

    Dual-Uncertainty Framework:
      1. Aleatoric/Entropy Uncertainty: Measures predicted distribution confidence.
         It maps the score (0-10) back to probability p = score / 10.
         Binary Shannon entropy = -p*log2(p) - (1-p)*log2(1-p).
         A score of 5.0 results in 1.0 (maximal uncertainty), while scores of 0.0 or 10.0
         yield 0.0 (maximal certainty).
      2. Epistemic/MC-Dropout Uncertainty: Measures model parameter consistency.
         When mc_passes > 1, the model runs multiple times with dropout active.
         The standard deviation of those scores measures epistemic uncertainty (on the 0-10 scale).

    Args:
        model (nn.Module): Loaded Net in eval mode.
        image_path (str): Absolute path to the image file.
        device (torch.device): Torch device.
        mc_passes (int, optional): Number of stochastic forward passes for MC-Dropout.
            0 or 1 disables MC-Dropout (returns None).

    Returns:
        tuple[float, float | None, float]: A tuple containing:
            - Predicted score (float, 0.0 to 10.0).
            - Epistemic MC-dropout uncertainty (float on the 0.0-10.0 scale, or None).
            - Aleatoric entropy uncertainty (float on the 0.0-1.0 scale).
    """
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    # 1. Score prediction & MC-dropout standard deviation
    if mc_passes <= 1:
        # Deterministic single forward pass
        model.eval()
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.softmax(logits, dim=1)[0][1].item()
        mean_score = prob * 10.0
        mc_std = None
    else:
        # MC-Dropout epistemic uncertainty estimation
        # Force dropout layers to be in training mode and have non-zero probability
        def enable_dropout(m):
            if m.__class__.__name__.startswith('Dropout'):
                if hasattr(m, 'p') and m.p == 0.0:
                    m.p = 0.1
                m.train()
        
        model.eval()
        model.apply(enable_dropout)
        
        mc_scores: list[float] = []
        with torch.no_grad():
            for _ in range(mc_passes):
                logits = model(tensor)
                prob = torch.softmax(logits, dim=1)[0][1].item()
                mc_scores.append(prob * 10.0)
                
        model.eval()  # restore pure eval mode
        
        mean_score = sum(mc_scores) / len(mc_scores)
        variance = sum((s - mean_score) ** 2 for s in mc_scores) / len(mc_scores)
        mc_std = variance ** 0.5

    # 2. Aleatoric / Entropy Uncertainty calculation
    # Answer: "if model outputs score 5, how sure is it?" -> Maximum entropy = 1.0!
    expl = get_score_confidence_explanation(mean_score)
    entropy_unc = expl["entropy_bits"]

    return round(mean_score, 2), (round(mc_std, 3) if mc_std is not None else None), round(entropy_unc, 4)


# ---------------------------------------------------------------------------
# Batch inference on a GeoDataFrame / DataFrame
# ---------------------------------------------------------------------------

def run(
    gdf: gpd.GeoDataFrame,
    metrics: str | Iterable[str],
    model_dir: str,
    image_column: str = "path",
    device: torch.device | None = None,
    download_missing_models: bool = False,
    mc_passes: int = 0,
) -> gpd.GeoDataFrame:
    """Score all images in *gdf* for one or more perception metrics.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame (or plain DataFrame) with absolute image paths.
        metrics (str | Iterable[str]): Single metric name or list of metrics.
        model_dir (str): Directory containing the trained .pth files.
        image_column (str, optional): Name of the column with paths. Defaults to "path".
        device (torch.device, optional): Torch device. Defaults to auto.
        download_missing_models (bool, optional): Auto-downloads standard checkpoints if missing.
        mc_passes (int, optional): Number of stochastic passes for MC-Dropout.

    Returns:
        gpd.GeoDataFrame: Copy of *gdf* with added score and uncertainty columns.
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Inference device: {device} | mc_passes={mc_passes}")

    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = list(metrics)

    if download_missing_models:
        from model import DEFAULT_METRICS
        non_default = [m for m in metrics if m not in DEFAULT_METRICS]
        if non_default:
            raise ValueError(
                f"download_missing_models=True only works for the built-in default "
                f"metrics {DEFAULT_METRICS}. Custom metrics must be present locally: {non_default}"
            )
        missing = [
            m for m in metrics
            if not os.path.isfile(os.path.join(model_dir, get_model_filename(m)))
        ]
        if missing:
            print(f"Downloading pretrained models for: {missing} …")
            download_pretrained_models(model_dir)
        else:
            print("All pretrained models already present locally — skipping download.")

    result = gdf.copy()

    for metric in metrics:
        print(f"\n######### Scoring Metric: {metric} #########")
        model = load_model(metric=metric, model_dir=model_dir, device=device)

        scores: list[float | None] = []
        mc_uncertainties: list[float | None] = []
        entropy_uncertainties: list[float | None] = []

        for img_path in tqdm(result[image_column], desc=metric):
            if (
                img_path is None
                or not isinstance(img_path, str)
                or not os.path.isfile(img_path)
            ):
                scores.append(None)
                mc_uncertainties.append(None)
                entropy_uncertainties.append(None)
                continue

            score, mc_unc, ent_unc = predict_image_score(
                model=model,
                image_path=img_path,
                device=device,
                mc_passes=mc_passes,
            )
            scores.append(score)
            mc_uncertainties.append(mc_unc)
            entropy_uncertainties.append(ent_unc)

        # Write columns to output dataframe
        result[metric] = scores
        result[f"entropy_{metric}"] = entropy_uncertainties

        if mc_passes > 1:
            result[f"uncertainty_mc_{metric}"] = mc_uncertainties

        print(f"Metric '{metric}' inference complete.")

    return result
