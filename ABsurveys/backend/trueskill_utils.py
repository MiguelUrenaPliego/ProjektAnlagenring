"""
=======================================================================
trueskill_utils.py — TrueSkill state with MongoDB persistence
=======================================================================

TrueSkill state lives in two places simultaneously:
  1. In-memory scenario_state dict (pandas DataFrame) — fast reads for pairing
  2. MongoDB image_state collection — durable, survives restarts

An asyncio.Lock per scenario serialises concurrent TrueSkill updates so
two answers arriving simultaneously for the same image never overwrite
each other.
=======================================================================
"""

from __future__ import annotations

import asyncio
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
# SCALING
# ============================================================================

MU_SCALE: float = 5.0
SIG_SCALE: float = MU_SCALE / (25.0 / 3.0)

DEFAULT_SCORE: float = 25.0 / MU_SCALE           # = 5.0
DEFAULT_UNCERTAINTY: float = (25.0 / 3.0) * SIG_SCALE / 10.0  # ≈ 0.5

# One lock per scenario — serialises concurrent TrueSkill updates for the
# same scenario without blocking unrelated scenarios.
_scenario_locks: Dict[str, asyncio.Lock] = {}


def _get_lock(scenario: str) -> asyncio.Lock:
    if scenario not in _scenario_locks:
        _scenario_locks[scenario] = asyncio.Lock()
    return _scenario_locks[scenario]


# ============================================================================
# CONVERSIONS
# ============================================================================

def score_to_mu(score: float) -> float:
    return score * MU_SCALE


def mu_to_score(mu: float) -> float:
    return mu / MU_SCALE


def uncertainty_to_sigma(u: float) -> float:
    return u * 10.0 / SIG_SCALE


def sigma_to_uncertainty(sigma: float) -> float:
    return sigma * SIG_SCALE / 10.0


# ============================================================================
# CORE TRUESKILL UPDATE
# ============================================================================

def trueskill_update(
    mu_a: float, sig_a: float,
    mu_b: float, sig_b: float,
    draw: bool = False,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    a = ENV.create_rating(mu_a, sig_a)
    b = ENV.create_rating(mu_b, sig_b)
    new_a, new_b = trueskill.rate_1vs1(a, b, drawn=draw, env=ENV)
    return (new_a.mu, new_a.sigma), (new_b.mu, new_b.sigma)


# ============================================================================
# ENSURE SCENARIO STATE (async, MongoDB-backed)
# ============================================================================

async def ensure_scenario_state(
    scenario: str,
    images_df: pd.DataFrame,
    save_qids: Set[str],
    config: dict,
    default_score: float,
    default_uncertainty: float,
    col,  # motor AsyncIOMotorCollection for image_state
    recompute_fn: Callable | None = None,
) -> pd.DataFrame:
    """Delegates to utils.ensure_scenario_state (async, MongoDB version)."""
    return await utils.ensure_scenario_state(
        scenario=scenario,
        images_df=images_df,
        save_qids=save_qids,
        config=config,
        default_score=default_score,
        default_uncertainty=default_uncertainty,
        recompute_fn=recompute_fn or recompute_from_history,
        col=col,
    )


# ============================================================================
# BATCH EXPANSION
# ============================================================================

async def maybe_expand_batch(
    scenario: str,
    img_type: str,
    qid: str,
    df: pd.DataFrame,
    config: dict,
    col,  # motor collection
) -> pd.DataFrame:
    """
    Expand active batch if ≥90% of the active pool is below the uncertainty
    threshold. Updates in-memory df and persists to MongoDB.
    """
    threshold = float(config.get("uncertainty_threshold", 0.25))

    ab_col = f"active_batch_{qid}"
    u_col  = f"uncertainty_{qid}"

    if ab_col not in df.columns or u_col not in df.columns:
        return df

    grp     = df[df["img_type"] == img_type]
    current = int(grp[ab_col].max())
    active  = grp[grp["batch_id"] <= current]

    if active.empty:
        return df

    pct_ready = (active[u_col] <= threshold).mean()

    if pct_ready >= 0.90:
        max_batch = int(grp["batch_id"].max())
        if current < max_batch:
            new_batch = current + 1
            df.loc[df["img_type"] == img_type, ab_col] = new_batch
            # Persist to MongoDB
            await col.update_many(
                {"scenario": scenario, "img_type": img_type},
                {"$set": {ab_col: new_batch}},
            )

    return df


# ============================================================================
# ONLINE UPDATE
# ============================================================================

async def update_image_state(
    scenario: str,
    question_id: str,
    img_id_A: str,
    img_id_B: str,
    winner: str,
    scenario_state: Dict[str, pd.DataFrame],
    config: dict | None,
    col,  # motor collection for image_state
) -> None:
    """
    Async, lock-serialised TrueSkill update after a single AB comparison.

    The asyncio.Lock ensures two concurrent answers for the same scenario
    never interleave their read-modify-write sequences.
    """
    async with _get_lock(scenario):
        qid = utils.to_str_id(question_id)
        df  = scenario_state[scenario]

        s_col = f"score_{qid}"
        u_col = f"uncertainty_{qid}"
        n_col = f"n_answers_{qid}"

        for col_name, default in [(s_col, DEFAULT_SCORE), (u_col, DEFAULT_UNCERTAINTY)]:
            if col_name not in df.columns:
                df[col_name] = default

        aid = utils.to_str_id(img_id_A)
        bid = utils.to_str_id(img_id_B)

        mask_a = df["img_id"].map(utils.to_str_id) == aid
        mask_b = df["img_id"].map(utils.to_str_id) == bid

        if not mask_a.any() or not mask_b.any():
            return

        mu_a  = score_to_mu(float(df.loc[mask_a, s_col].iloc[0]))
        mu_b  = score_to_mu(float(df.loc[mask_b, s_col].iloc[0]))
        sig_a = uncertainty_to_sigma(float(df.loc[mask_a, u_col].iloc[0]))
        sig_b = uncertainty_to_sigma(float(df.loc[mask_b, u_col].iloc[0]))

        draw = winner == "="

        if winner == "A" or draw:
            (new_mu_a, new_sig_a), (new_mu_b, new_sig_b) = trueskill_update(
                mu_a, sig_a, mu_b, sig_b, draw=draw
            )
        else:
            (new_mu_b, new_sig_b), (new_mu_a, new_sig_a) = trueskill_update(
                mu_b, sig_b, mu_a, sig_a, draw=False
            )

        new_score_a = mu_to_score(new_mu_a)
        new_score_b = mu_to_score(new_mu_b)
        new_unc_a   = sigma_to_uncertainty(new_sig_a)
        new_unc_b   = sigma_to_uncertainty(new_sig_b)

        # Update in-memory DataFrame
        df.loc[mask_a, s_col] = new_score_a
        df.loc[mask_b, s_col] = new_score_b
        df.loc[mask_a, u_col] = new_unc_a
        df.loc[mask_b, u_col] = new_unc_b

        if n_col not in df.columns:
            df[n_col] = 0
        df.loc[mask_a, n_col] += 1
        df.loc[mask_b, n_col] += 1

        n_a = int(df.loc[mask_a, n_col].iloc[0])
        n_b = int(df.loc[mask_b, n_col].iloc[0])

        # Persist both images to MongoDB
        await col.update_one(
            {"scenario": scenario, "img_id": aid},
            {"$set": {s_col: new_score_a, u_col: new_unc_a, n_col: n_a}},
        )
        await col.update_one(
            {"scenario": scenario, "img_id": bid},
            {"$set": {s_col: new_score_b, u_col: new_unc_b, n_col: n_b}},
        )

        # Batch expansion check
        if config is not None:
            anchor_rows = df[df["img_id"].map(utils.to_str_id) == aid]
            if not anchor_rows.empty and "img_type" in df.columns:
                img_type = str(anchor_rows.iloc[0]["img_type"])
                df = await maybe_expand_batch(scenario, img_type, qid, df, config, col)
                scenario_state[scenario] = df


# ============================================================================
# DEBUG ONLY
# ============================================================================

def recompute_from_history(
    history_df: pd.DataFrame,
    df: pd.DataFrame,
    qids: Set[str],
) -> pd.DataFrame:
    for qid in qids:
        df[f"score_{qid}"]       = DEFAULT_SCORE
        df[f"uncertainty_{qid}"] = DEFAULT_UNCERTAINTY
        df[f"n_answers_{qid}"]   = 0
    return df