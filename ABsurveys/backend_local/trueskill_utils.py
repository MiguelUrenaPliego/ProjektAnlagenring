"""
=======================================================================
TRUESKILL STATE  (user_data/{scenario}_images.csv)
=======================================================================

Each image has the following columns per AB question_id:

    score_{question_id}        — TrueSkill mu (mean skill), scaled 0–10
                                 Default: 5.0

    uncertainty_{question_id}  — TrueSkill sigma, scaled proportionally
                                 Default: 0.5

    n_answers_{question_id}    — Number of AB comparisons this image has
                                 been involved in for this question.

    batch_id                   — Static batch assignment (1-based int).

    active_batch_{question_id} — Highest batch currently included in the
                                 study. Starts at 1.

=======================================================================
"""

from __future__ import annotations

import os
from typing import Dict, Set, Tuple, Callable

import pandas as pd
import trueskill

from . import utils


# ============================================================================
# TRUE SKILL ENVIRONMENT
# ============================================================================

ENV = trueskill.TrueSkill(
    mu=25.0,
    sigma=25.0 / 3.0,
    beta=25.0 / 6.0,
    tau=25.0 / 300.0,
    draw_probability=0.0,
)


# ============================================================================
# SCALING (TrueSkill <-> CSV 0–10 space)
# ============================================================================

MU_SCALE: float = 5.0
SIG_SCALE: float = MU_SCALE / (25.0 / 3.0)

DEFAULT_SCORE: float = 25.0 / MU_SCALE          # = 5.0
DEFAULT_UNCERTAINTY: float = (25.0 / 3.0) * SIG_SCALE / 10.0  # ≈ 0.5


# ============================================================================
# CONVERSIONS
# ============================================================================

def score_to_mu(score: float) -> float:
    """Convert 0–10 score → TrueSkill mu."""
    return score * MU_SCALE


def mu_to_score(mu: float) -> float:
    """Convert TrueSkill mu → 0–10 score."""
    return mu / MU_SCALE


def uncertainty_to_sigma(u: float) -> float:
    """Convert 0–10 uncertainty → TrueSkill sigma."""
    return u * 10.0 / SIG_SCALE


def sigma_to_uncertainty(sigma: float) -> float:
    """Convert TrueSkill sigma → 0–10 uncertainty."""
    return sigma * SIG_SCALE / 10.0


# ============================================================================
# CORE TRUE SKILL UPDATE
# ============================================================================

def trueskill_update(
    mu_a: float,
    sig_a: float,
    mu_b: float,
    sig_b: float,
    draw: bool = False,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Apply a single TrueSkill update.

    Args:
        mu_a: Player A mean skill.
        sig_a: Player A uncertainty.
        mu_b: Player B mean skill.
        sig_b: Player B uncertainty.
        draw: Whether match ended in draw.

    Returns:
        ((new_mu_a, new_sig_a), (new_mu_b, new_sig_b))

    NOTE: A is treated as the WINNER when draw=False.
    Pass players in winner-first order when calling.
    """
    a = ENV.create_rating(mu_a, sig_a)
    b = ENV.create_rating(mu_b, sig_b)

    new_a, new_b = trueskill.rate_1vs1(a, b, drawn=draw, env=ENV)

    return (new_a.mu, new_a.sigma), (new_b.mu, new_b.sigma)


# ============================================================================
# ENSURE SCENARIO STATE (thin wrapper kept here for server.py import compat)
# ============================================================================

def ensure_scenario_state(
    scenario: str,
    images_df: pd.DataFrame,
    save_qids: Set[str],
    config: dict,
    user_data_path: str,
    default_score: float,
    default_uncertainty: float,
    recompute_fn: Callable | None = None,
) -> pd.DataFrame:
    """Delegates to utils.ensure_scenario_state."""
    if recompute_fn is None:
        recompute_fn = recompute_from_history

    return utils.ensure_scenario_state(
        scenario=scenario,
        images_df=images_df,
        save_qids=save_qids,
        config=config,
        user_data_path=user_data_path,
        default_score=default_score,
        default_uncertainty=default_uncertainty,
        recompute_fn=recompute_fn,
    )


# ============================================================================
# ONLINE UPDATE (MAIN ENTRY POINT)
# ============================================================================

def update_image_state(
    scenario: str,
    question_id: str,
    img_id_A: str,
    img_id_B: str,
    winner: str,
    scenario_state: Dict[str, pd.DataFrame],
    user_data_path: str,
    config: dict | None = None,
) -> None:
    """
    ONLINE update after a single AB comparison.

    Args:
        scenario: Scenario name.
        question_id: Question ID.
        img_id_A: Left image.
        img_id_B: Right image.
        winner: "A", "B", or "=".
        scenario_state: In-memory state dict (mutated in place).
        user_data_path: Persistence path.
        config: App config dict; required for batch expansion after update.
    """
    qid = utils.to_str_id(question_id)
    df = scenario_state[scenario]

    s_col = f"score_{qid}"
    u_col = f"uncertainty_{qid}"
    n_col = f"n_answers_{qid}"

    for col, default in [(s_col, DEFAULT_SCORE), (u_col, DEFAULT_UNCERTAINTY)]:
        if col not in df.columns:
            df[col] = default

    aid = utils.to_str_id(img_id_A)
    bid = utils.to_str_id(img_id_B)

    mask_a = df["img_id"].map(utils.to_str_id) == aid
    mask_b = df["img_id"].map(utils.to_str_id) == bid

    if not mask_a.any() or not mask_b.any():
        return

    mu_a = score_to_mu(float(df.loc[mask_a, s_col].iloc[0]))
    mu_b = score_to_mu(float(df.loc[mask_b, s_col].iloc[0]))
    sig_a = uncertainty_to_sigma(float(df.loc[mask_a, u_col].iloc[0]))
    sig_b = uncertainty_to_sigma(float(df.loc[mask_b, u_col].iloc[0]))

    draw = winner == "="

    if winner == "A" or draw:
        # A wins (or draw): pass A first as winner
        (new_mu_a, new_sig_a), (new_mu_b, new_sig_b) = trueskill_update(
            mu_a, sig_a, mu_b, sig_b, draw=draw
        )
    else:
        # B wins: pass B first as winner, then swap back
        (new_mu_b, new_sig_b), (new_mu_a, new_sig_a) = trueskill_update(
            mu_b, sig_b, mu_a, sig_a, draw=False
        )

    df.loc[mask_a, s_col] = mu_to_score(new_mu_a)
    df.loc[mask_b, s_col] = mu_to_score(new_mu_b)
    df.loc[mask_a, u_col] = sigma_to_uncertainty(new_sig_a)
    df.loc[mask_b, u_col] = sigma_to_uncertainty(new_sig_b)

    if n_col not in df.columns:
        df[n_col] = 0

    df.loc[mask_a, n_col] += 1
    df.loc[mask_b, n_col] += 1

    # Check batch expansion on fresh uncertainty values, then persist atomically.
    # img_type is read from the anchor row so maybe_expand_batch gets the right group.
    if config is not None:
        anchor_rows = df[df["img_id"].map(utils.to_str_id) == aid]
        if not anchor_rows.empty and "img_type" in df.columns:
            img_type = str(anchor_rows.iloc[0]["img_type"])
            df = maybe_expand_batch(scenario, img_type, qid, df, config)
            scenario_state[scenario] = df

    # Persist updated state (includes any active_batch_* changes from expansion)
    path = os.path.join(user_data_path, f"{scenario}_images.csv")
    df.to_csv(path, index=False)


# ============================================================================
# BATCH EXPANSION (ACTIVE LEARNING CONTROLLER)
# ============================================================================

def maybe_expand_batch(
    scenario: str,
    img_type: str,
    qid: str,
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Expand active batch if ≥90% of the active pool is below the uncertainty
    threshold. Mutates df in place and returns it.
    """
    threshold = float(config.get("uncertainty_threshold", 0.25))

    ab_col = f"active_batch_{qid}"
    u_col = f"uncertainty_{qid}"

    if ab_col not in df.columns or u_col not in df.columns:
        return df

    grp = df[df["img_type"] == img_type]
    current = int(grp[ab_col].max())

    active = grp[grp["batch_id"] <= current]

    if active.empty:
        return df

    pct_ready = (active[u_col] <= threshold).mean()

    if pct_ready >= 0.90:
        max_batch = int(grp["batch_id"].max())
        if current < max_batch:
            df.loc[df["img_type"] == img_type, ab_col] = current + 1

    return df


# ============================================================================
# OPTIONAL: FULL ONLINE REBUILD (DEBUG ONLY)
# ============================================================================

def recompute_from_history(
    history_df: pd.DataFrame,
    df: pd.DataFrame,
    qids: Set[str],
) -> pd.DataFrame:
    """
    DEBUG TOOL ONLY. Replays full history to rebuild state.
    Not needed for normal online inference.
    """
    for qid in qids:
        df[f"score_{qid}"] = DEFAULT_SCORE
        df[f"uncertainty_{qid}"] = DEFAULT_UNCERTAINTY
        df[f"n_answers_{qid}"] = 0

    return df