# model.py
# coding: utf-8

"""Model architecture and shared constants for street perception scoring."""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms import functional as F_t
from torchvision.models import ViT_B_16_Weights, vit_b_16

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DEFAULT_METRICS: list[str] = [
    "safety",
    "lively",
    "wealthy",
    "beautiful",
    "boring",
    "depressing",
]

# Maps every *known* metric name to its .pth filename.
# Custom / new metrics follow the same convention: {metric_lowercase}.pth
MODEL_FILENAMES: dict[str, str] = {
    "safety": "safety.pth",
    "lively": "lively.pth",
    "wealthy": "wealthy.pth",
    "beautiful": "beautiful.pth",
    "boring": "boring.pth",
    "depressing": "depressing.pth",
}

HF_REPO_ID: str = "Jiani11/human-perception-place-pulse"

class CenterCropToSquare:
    """Crops the input PIL Image to a square aspect ratio at the center, 
    preserving the original dimensions of the shorter side before resizing.
    """
    def __call__(self, img):
        w, h = img.size
        if w != h:
            min_dim = min(w, h)
            return F_t.center_crop(img, [min_dim, min_dim])
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


IMAGE_TRANSFORM = T.Compose(
    [
        CenterCropToSquare(),
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

TRAIN_IMAGE_TRANSFORM = T.Compose(
    [
        CenterCropToSquare(),
        T.RandomResizedCrop(size=(384, 384), scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def get_model_filename(metric: str) -> str:
    """Return the .pth filename for any metric (known or custom).

    Known metrics use the canonical name from MODEL_FILENAMES.
    Unknown metrics follow the convention {metric_lowercase}.pth.
    """
    filename = MODEL_FILENAMES.get(metric, f"{metric.lower()}")

    if not filename.endswith(".pth"):
        filename += ".pth"

    return filename

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Net(nn.Module):
    """Vision Transformer (ViT-B/16) with a 3-layer MLP head for binary perception classification.

    Args:
        num_classes:  Number of output logits (default 2 for binary).
        vit_weights:  When *True* (default) initialise the ViT backbone with
                      ``ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1`` pretrained
                      weights.  Pass *False* to start from a random backbone.
        freeze_vit:   When *True* the ViT backbone parameters are frozen so
                      only the MLP head is trained.  Default *False*.
    """

    def __init__(
        self,
        num_classes: int = 2,
        vit_weights: bool = True,
        freeze_vit: bool = False,
    ) -> None:
        super().__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if vit_weights else None
        self.model = vit_b_16(weights=weights)

        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

        nn.init.xavier_uniform_(self.model.heads.head[0].weight)
        nn.init.xavier_uniform_(self.model.heads.head[2].weight)
        nn.init.xavier_uniform_(self.model.heads.head[4].weight)

        if freeze_vit:
            self.freeze_backbone()

    # ------------------------------------------------------------------
    def freeze_backbone(self) -> None:
        """Freeze all ViT backbone parameters (leave MLP head trainable)."""
        for name, param in self.model.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Un-freeze the ViT backbone (make all parameters trainable)."""
        for param in self.model.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
