# losses.py
# coding: utf-8

"""Loss and accuracy functions for street perception model training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def crossentropy_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the standard cross-entropy loss for single-image binary classification.

    Args:
        outputs (torch.Tensor): Logits predicted by the model of shape (N, 2).
        targets (torch.Tensor): Ground truth binary labels of shape (N,), containing values in {0, 1}.

    Returns:
        torch.Tensor: Scalar tensor containing the computed cross-entropy loss.
    """
    return F.cross_entropy(outputs, targets)


def f1_score(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Computes the binary F1 score on single images.

    Args:
        outputs (torch.Tensor): Logits predicted by the model of shape (N, 2).
        targets (torch.Tensor): Ground truth binary labels of shape (N,), containing values in {0, 1}.

    Returns:
        float: F1 score in the range [0.0, 1.0].
    """
    preds = outputs.argmax(dim=1)
    tp = ((preds == 1) & (targets == 1)).float().sum()
    fp = ((preds == 1) & (targets == 0)).float().sum()
    fn = ((preds == 0) & (targets == 1)).float().sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()


def pair_loss(
    scores_A: torch.Tensor,
    scores_B: torch.Tensor,
    labels: torch.Tensor,
    is_tie: torch.Tensor | None = None,
    margin: float = 1.0,
) -> torch.Tensor:
    """Computes the pairwise ranking loss based on the score difference.

    For strict comparisons, loss is: scores_rejected - scores_preferred.
    For tie comparisons, loss is: abs(scores_A - scores_B).
    The contribution of all pairs is summed and normalized by dividing by the total
    number of images in the set (2 * number of pairs).
    """
    if is_tie is None:
        is_tie = torch.zeros_like(labels)

    scores_preferred = labels * scores_A + (1.0 - labels) * scores_B
    scores_rejected = (1.0 - labels) * scores_A + labels * scores_B
    
    std_loss = scores_rejected - scores_preferred
    tie_loss = torch.abs(scores_A - scores_B)
    
    loss_contribs = torch.where(is_tie == 1, tie_loss, std_loss)
    
    # Add contributions of all pairs and divide by the total number of images (2 * number of pairs)
    total_images = 2 * len(labels)
    return loss_contribs.sum() / total_images


def pair_accuracy(
    scores_A: torch.Tensor,
    scores_B: torch.Tensor,
    labels: torch.Tensor,
    is_tie: torch.Tensor | None = None,
) -> float:
    """Computes the pairwise ranking accuracy.

    For strict comparisons, contribution is: scores_preferred - scores_rejected.
    For tie comparisons, contribution is: -abs(scores_A - scores_B).
    The contributions of all pairs are added and divided by the number of pairs.
    """
    if is_tie is None:
        is_tie = torch.zeros_like(labels)

    scores_preferred = labels * scores_A + (1.0 - labels) * scores_B
    scores_rejected = (1.0 - labels) * scores_A + labels * scores_B
    
    std_contrib = scores_preferred - scores_rejected
    tie_contrib = -torch.abs(scores_A - scores_B)
    
    contributions = torch.where(is_tie == 1, tie_contrib, std_contrib)
    
    # Add contributions of all pairs and divide by number of pairs
    return (contributions.sum() / len(labels)).item()


def mixed_loss(
    logits: torch.Tensor,
    scores_A: torch.Tensor,
    scores_B: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
    is_tie: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes a mixed loss where each loss contributes exactly 50% to the total loss.

    To balance their influence, the pairwise ranking loss is scaled to match the
    detached absolute scale of the cross-entropy loss. We then apply a shift to prevent
    the loss values from canceling out (e.g. when pair_loss is negative), ensuring a
    meaningful, positive total loss value while preserving the balanced 50/50 gradient flow.
    """
    loss_ce = crossentropy_loss(logits, targets)
    loss_p = pair_loss(scores_A, scores_B, labels, is_tie=is_tie)
    
    # Detach scales to avoid flow of secondary gradients
    scale_ce = loss_ce.detach().abs().clamp(min=1e-8)
    scale_p = loss_p.detach().abs().clamp(min=1e-8)
    
    # Scale pair_loss so its gradient magnitude matches cross-entropy loss
    loss_p_scaled = loss_p * (scale_ce / scale_p)
    
    # By adding scale_ce when pair_loss is negative, we prevent cancellation of values
    # (i.e. loss_ce and -loss_ce canceling to 0.0) without altering the gradient flow.
    return 0.5 * loss_ce + 0.5 * loss_p_scaled + 0.5 * scale_ce

