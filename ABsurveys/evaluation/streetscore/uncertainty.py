# uncertainty.py
# coding: utf-8

"""Uncertainty quantification and explanation utilities for street perception models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def calculate_entropy(probs: torch.Tensor, base: float = 2.0) -> torch.Tensor:
    """Calculates the Shannon entropy for a given probability distribution.

    Entropy is a measure of aleatoric uncertainty (the inherent statistical variance
    or randomness of the prediction).
    H = 0 means absolute certainty (probabilities like [0.0, 1.0]).
    H = 1 (when base=2.0) means absolute uncertainty (uniform distribution, [0.5, 0.5]).

    Args:
        probs (torch.Tensor): Logits or probabilities. Expected shape (..., C) containing class probabilities.
        base (float, optional): Logarithmic base. Defaults to 2.0 (for bits).

    Returns:
        torch.Tensor: The Shannon entropy values.
    """
    eps = 1e-9
    probs = torch.clamp(probs, min=eps)
    entropy = -torch.sum(probs * (torch.log(probs) / torch.log(torch.tensor(base, dtype=probs.dtype, device=probs.device))), dim=-1)
    return entropy


def get_score_confidence_explanation(score: float) -> dict[str, any]:
    """Explains how sure the model is for a given 0-10 score.

    A score is mathematically defined as p_preferred * 10.0, where p_preferred is the
    probability that the image is preferred.
    - If score = 5.0, p_preferred = 0.5, representing maximum entropy/uncertainty
      (the model is completely unsure/neutral).
    - If score approaches 0.0 or 10.0, p_preferred is near 0.0 or 1.0, representing
      maximum certainty.

    Args:
        score (float): Predicted score between 0.0 and 10.0.

    Returns:
        dict[str, any]: A dictionary containing:
            - 'score': The original score.
            - 'probability': The implied probability p.
            - 'entropy_bits': Binary Shannon entropy in bits (range [0.0, 1.0]).
            - 'certainty_percentage': Confidence/certainty level as a percentage.
            - 'explanation': A human-readable description of how sure the model is.
    """
    p = max(0.0, min(1.0, score / 10.0))
    eps = 1e-9
    p_clamped = max(eps, min(1.0 - eps, p))
    # Binary Shannon entropy: -p*log2(p) - (1-p)*log2(1-p)
    entropy = -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))
    
    certainty = (1.0 - entropy) * 100.0
    
    if entropy > 0.9:
        desc = "Completely unsure / highly ambiguous (the model cannot distinguish preference)."
    elif entropy > 0.6:
        desc = "Somewhat unsure (the model has a weak preference inclination)."
    elif entropy > 0.3:
        desc = "Moderately confident (the model shows a clear preference)."
    else:
        desc = "Highly confident (the model is extremely sure of its prediction)."
        
    return {
        "score": round(score, 2),
        "probability": round(p, 4),
        "entropy_bits": round(entropy, 4),
        "certainty_percentage": round(certainty, 2),
        "explanation": desc,
    }


def mc_dropout_passes(
    model: nn.Module,
    x: torch.Tensor,
    num_passes: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes epistemic uncertainty using Monte Carlo (MC) Dropout.

    By keeping the dropout layers active during forward passes, we can measure the
    variance of the predictions. High variance/standard deviation suggests high
    epistemic uncertainty (model has not seen enough similar data during training).

    Args:
        model (nn.Module): The PyTorch neural network model.
        x (torch.Tensor): A batch of preprocessed images, shape (B, C, H, W).
        num_passes (int, optional): Number of stochastic forward passes. Defaults to 20.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Mean probability of the preferred class over all passes, shape (B,).
            - Standard deviation of the probabilities, representing the epistemic
              uncertainty (on the [0.0, 1.0] scale), shape (B,).
    """
    # Force dropout layers to be in training mode and have non-zero probability
    def enable_dropout(m):
        if m.__class__.__name__.startswith('Dropout'):
            if hasattr(m, 'p') and m.p == 0.0:
                m.p = 0.1
            m.train()

    model.eval()  # Put everything in eval mode first (e.g., BatchNorm)
    model.apply(enable_dropout)  # Re-enable dropout layers for stochastic prediction
    
    all_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(preferred)
            all_probs.append(probs)
            
    all_probs_tensor = torch.stack(all_probs, dim=0)  # Shape: (num_passes, B)
    mean_probs = all_probs_tensor.mean(dim=0)
    std_probs = all_probs_tensor.std(dim=0)
    
    return mean_probs, std_probs
