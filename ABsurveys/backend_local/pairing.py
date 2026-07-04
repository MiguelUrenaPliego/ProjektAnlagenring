"""
=======================================================================
ACTIVE LEARNING PAIRING ENGINE (ONLINE TRUE SKILL SYSTEM)
=======================================================================

Selects image pairs for AB testing using:

1. TrueSkill uncertainty
2. Active learning (information gain)
3. Batch expansion (curriculum learning)
4. Hard constraints (Rules 1–7)

KEY PRINCIPLE
-------------
Pairing always reflects CURRENT TrueSkill state.
Batch expansion only increases candidate pool.

=======================================================================
"""

from __future__ import annotations

import random
import math
from typing import Dict, FrozenSet, Tuple, Optional

import pandas as pd

from . import utils
from . import trueskill_utils


# ============================================================================
# RULE HELPERS
# ============================================================================

def _to_id(x: str) -> str:
    return utils.to_str_id(x)


def _build_incompat_set(pool: pd.DataFrame, anchor_id: str) -> set[str]:
    """
    Return the set of img_ids that are incompatible with *anchor_id*.
    Reads the 'incompatible_ids' column which stores comma/list-separated ids.
    """
    rows = pool[pool["img_id"].map(_to_id) == _to_id(anchor_id)]
    blocked: set[str] = set()
    for _, row in rows.iterrows():
        for iid in utils.parse_list(row.get("incompatible_ids", "")):
            if iid:
                blocked.add(_to_id(iid))
    return blocked


def _eligible_mask(
    pool: pd.DataFrame,
    used_types: Dict[str, str],
    img_type: str,
    incompat_blocked: set[str],
) -> pd.Series:
    """
    Rule 2 filter:
    - Prevents incompatible pairs.
    - Enforces per-user img_type locking (an image seen under one img_type
      cannot appear under a different one for the same user).
    """
    ids = pool["img_id"].map(_to_id)

    def locked(iid: str) -> bool:
        t = used_types.get(iid)
        return t is not None and t != img_type

    return ~ids.map(locked) & ~ids.isin(incompat_blocked)


def _prefer_unseen_by_user(
    pool: pd.DataFrame, seen_img_ids: set[str]
) -> Tuple[pd.DataFrame, bool]:
    """
    RULE 7 helper: prefer rows whose img_id has not yet been shown to this
    user. Returns (filtered_pool, relaxed) where relaxed=True means the
    constraint couldn't be satisfied and the original pool was returned
    unchanged.
    """
    if not seen_img_ids:
        return pool, False
    ids = pool["img_id"].map(_to_id)
    filtered = pool[~ids.isin(seen_img_ids)]
    if filtered.empty:
        return pool, True
    return filtered, False


def _exclude_repeat_pairs(
    pool: pd.DataFrame,
    anchor_id: str,
    seen_pairs: set[FrozenSet[str]],
) -> Tuple[pd.DataFrame, bool]:
    """
    RULE 6 helper: exclude candidates that would recreate a pair already
    shown to this user, regardless of left/right order. Returns
    (filtered_pool, relaxed).
    """
    if not seen_pairs:
        return pool, False

    anchor = _to_id(anchor_id)
    ids = pool["img_id"].map(_to_id)

    def is_repeat(iid: str) -> bool:
        return frozenset((anchor, iid)) in seen_pairs

    filtered = pool[~ids.map(is_repeat)]
    if filtered.empty:
        return pool, True
    return filtered, False


# ============================================================================
# CORE PAIR SELECTION HELPERS
# ============================================================================

def _make_pair(a: pd.Series, b: pd.Series) -> dict:
    """Build the pair dict, randomising left/right assignment."""
    if random.random() < 0.5:
        a, b = b, a

    # pd.Series membership uses .index, not `in series`
    col = "_serve_path" if "_serve_path" in a.index else "path"

    return {
        "A": str(a[col]),
        "B": str(b[col]),
        "img_id_A": str(a["img_id"]),
        "img_id_B": str(b["img_id"]),
    }


# ============================================================================
# ACTIVE LEARNING SCORING
# ============================================================================

def _info_gain(
    pool: pd.DataFrame,
    anchor: pd.Series,
    s_col: str,
    u_col: str,
    n_col: str,
) -> pd.Series:
    """
    Compute information gain for every row in *pool* relative to *anchor*.

    Higher is better:
        0.5 × uncertainty  +  0.3 × score similarity  +  0.2 × novelty
    """
    diff = (pool[s_col] - float(anchor[s_col])).abs()
    sim = 1.0 / (1.0 + diff)
    unc = pool[u_col]
    seen = pool[n_col].fillna(0)

    return 0.5 * unc + 0.3 * sim + 0.2 * (1.0 / (1.0 + seen))


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_random_pair(
    scenario: str,
    img_type: str,
    question_id: str,
    scenario_state: Dict[str, pd.DataFrame],
    images_df: pd.DataFrame,
    config: Dict,
    exclude: Optional[list[str]] = None,
    user_id: str = "",
    used_img_types: Optional[Dict[str, str]] = None,
    seen_images: Optional[list[str]] = None,
    seen_pairs: Optional[list[list[str]]] = None,
) -> Tuple[Optional[dict], float, str]:
    """
    ACTIVE LEARNING PAIR GENERATOR (ONLINE SYSTEM)

    Rules are applied in priority order — a lower-numbered rule is NEVER
    broken to satisfy a higher-numbered one. When a constraint must be
    relaxed, always relax the highest-numbered rule first.

        RULE 0: Batch expansion is handled by update_image_state (trueskill_utils.py)
                after each answer is saved, on fresh uncertainty values, and persisted
                atomically. create_random_pair reads the current active_batch as-is.
        RULE 1: Prefer pairing one seen image with one unseen (cold start).
        RULE 2: No incompatible pairs; respect per-user img_type locking.
        RULE 3: Unseen exploration first.
        RULE 4: Uncertainty-driven selection once all images are seen.
                Blended with uniform random sampling via config["active_learning"]
                (0.0 = fully random, 1.0 = fully uncertainty/info-gain driven).
        RULE 5: Avoid recreating a pair already shown to this user, in
                either left/right order (relaxed if no alternative exists).
                Driven by user_id / seen_pairs.
        RULE 6: Avoid showing this user an image they've already seen for
                this question (relaxed if no alternative exists).
                Driven by user_id / seen_images.

    Args:
        scenario: Scenario name.
        img_type: Image type to filter on.
        question_id: Question ID.
        scenario_state: Full in-memory state dict (read-only here).
        images_df: Global images DataFrame (used for metadata lookups).
        config: Application config dict.
        exclude: List of img_ids to exclude from both slots (e.g. the
            immediately preceding pair, for UX freshness).
        user_id: Current user identifier (unused for now, reserved).
        used_img_types: {img_id → img_type} mapping for this user session.
        seen_images: All img_ids already shown to this user for this
            question (RULE 6).
        seen_pairs: All [img_id_A, img_id_B] pairs already shown to this
            user for this question, order-independent (RULE 5).

    Returns:
        (pair_dict, information_gain, violation_string)
    """
    qid = _to_id(question_id)
    exclude_set: set[str] = {_to_id(x) for x in (exclude or [])}
    used_types: Dict[str, str] = used_img_types or {}

    seen_img_set: set[str] = {_to_id(x) for x in (seen_images or [])}
    seen_pair_set: set[FrozenSet[str]] = {
        frozenset((_to_id(p[0]), _to_id(p[1])))
        for p in (seen_pairs or [])
        if len(p) == 2
    }

    if scenario not in scenario_state:
        return None, 0.0, "unknown scenario"

    df = scenario_state[scenario]

    # Batch expansion is handled inside update_image_state (trueskill_utils.py)
    # immediately after each TrueSkill update, on fresh uncertainty values, and
    # persisted atomically. Here we just read whatever active_batch is current.

    s_col = f"score_{qid}"
    u_col = f"uncertainty_{qid}"
    n_col = f"n_answers_{qid}"
    ab_col = f"active_batch_{qid}"

    # Filter to this img_type
    grp = df[df["img_type"] == img_type].copy()

    if len(grp) < 2:
        return None, 0.0, "not enough images"

    # Ensure columns exist with defaults
    for col, default in [
        (s_col, trueskill_utils.DEFAULT_SCORE),
        (u_col, trueskill_utils.DEFAULT_UNCERTAINTY),
        (n_col, 0),
    ]:
        if col not in grp.columns:
            grp[col] = default

    grp[n_col] = grp[n_col].fillna(0).astype(int)

    # ── RULE 0: Active batch filter (batch expansion already ran above) ──
    active_batch = int(grp[ab_col].max()) if ab_col in grp.columns else 1
    imgs = grp[grp["batch_id"] <= active_batch].copy()

    if len(imgs) < 2:
        return None, 0.0, "active batch too small"

    # ── UX freshness: exclude the immediately preceding pair ─────────────
    imgs = imgs[~imgs["img_id"].map(_to_id).isin(exclude_set)]

    if len(imgs) < 2:
        # Relax exclude constraint rather than returning nothing
        imgs = grp[grp["batch_id"] <= active_batch].copy()

    # ── Split seen / unseen ──────────────────────────────────────────────
    unseen = imgs[imgs[n_col] == 0]
    seen = imgs[imgs[n_col] > 0]

    violation = ""

    # =========================================================
    # RULE 3: unseen first (cold-start exploration)
    # =========================================================
    if not unseen.empty and not seen.empty:
        # RULE 6: prefer an unseen-by-this-system image the user hasn't seen
        unseen_pref, relaxed_a = _prefer_unseen_by_user(unseen, seen_img_set)
        a = unseen_pref.sample(1).iloc[0]
        if relaxed_a:
            violation = f"rule6 relaxed: img {a['img_id']} already shown to user"

        # RULE 2: build incompat set for chosen anchor
        incompat = _build_incompat_set(imgs, str(a["img_id"]))
        eligible = seen[_eligible_mask(seen, used_types, img_type, incompat)]

        if eligible.empty:
            eligible = seen  # relax constraint, log violation
            violation = f"incompat/lock constraint relaxed for img {a['img_id']}"

        # RULE 6: prefer B not already shown to this user
        eligible, relaxed_b = _prefer_unseen_by_user(eligible, seen_img_set)
        if relaxed_b and not violation:
            violation = f"rule6 relaxed: no candidates for B unseen by user (anchor {a['img_id']})"

        # RULE 5: avoid recreating a pair already shown to this user
        eligible, relaxed_pair = _exclude_repeat_pairs(eligible, str(a["img_id"]), seen_pair_set)
        if relaxed_pair and not violation:
            violation = f"rule5 relaxed: all candidates would repeat a pair with img {a['img_id']}"

        b = eligible.sample(1).iloc[0]
        return _make_pair(a, b), float(a[u_col]), violation

    # =========================================================
    # RULE 1 fallback: all unseen — bootstrap with two unseen
    # =========================================================
    if not unseen.empty and seen.empty:
        if len(unseen) < 2:
            return None, 0.0, "not enough unseen images for bootstrap pair"

        # RULE 6: prefer images the user hasn't seen before
        unseen_pref, relaxed_a = _prefer_unseen_by_user(unseen, seen_img_set)
        pool_for_pair = unseen_pref if len(unseen_pref) >= 2 else unseen
        if pool_for_pair is unseen and relaxed_a:
            violation = "rule6 relaxed: not enough images unseen by user for bootstrap pair"

        sample = pool_for_pair.sample(min(2, len(pool_for_pair)))
        a = sample.iloc[0]
        b = sample.iloc[1]

        # RULE 5: avoid recreating a pair already shown to this user
        if frozenset((_to_id(a["img_id"]), _to_id(b["img_id"]))) in seen_pair_set:
            alt = pool_for_pair[pool_for_pair["img_id"] != a["img_id"]]
            alt, relaxed_pair = _exclude_repeat_pairs(alt, str(a["img_id"]), seen_pair_set)
            if not alt.empty:
                b = alt.sample(1).iloc[0]
            elif not violation:
                violation = f"rule5 relaxed: pair {a['img_id']}/{b['img_id']} repeats a prior pairing"

        return _make_pair(a, b), float(a[u_col]), violation

    # =========================================================
    # RULE 4: all seen — active learning (uncertainty-driven)
    # Blended with uniform random via config["active_learning"]:
    #   0.0 → fully random anchor + random B
    #   1.0 → fully uncertainty-weighted anchor + info-gain B
    # =========================================================
    al = float(config.get("active_learning", 1.0))
    al = max(0.0, min(1.0, al))  # clamp to [0, 1]

    # RULE 6: prefer an anchor the user hasn't seen yet
    seen_pref, relaxed_a = _prefer_unseen_by_user(seen, seen_img_set)
    anchor_pool = seen_pref if not seen_pref.empty else seen
    if anchor_pool is seen and relaxed_a:
        violation = "rule6 relaxed: no candidates for anchor unseen by user"

    # Anchor selection: blend uncertainty-weighted vs uniform random
    if al == 0.0:
        # Fully random
        A = anchor_pool.sample(1).iloc[0]
    elif al == 1.0:
        # Fully uncertainty-weighted
        weights = anchor_pool[u_col].clip(lower=0.0) + 0.01
        A = anchor_pool.sample(1, weights=weights).iloc[0]
    else:
        # Blended: mix uncertainty weights with uniform weights
        unc_weights = anchor_pool[u_col].clip(lower=0.0) + 0.01
        uniform_weights = pd.Series(1.0, index=anchor_pool.index)
        blended_weights = al * unc_weights + (1.0 - al) * uniform_weights
        A = anchor_pool.sample(1, weights=blended_weights).iloc[0]

    remaining = seen[seen["img_id"] != A["img_id"]].copy()

    if remaining.empty:
        return None, 0.0, "only one image available"

    # RULE 2: filter incompatible images
    incompat = _build_incompat_set(imgs, str(A["img_id"]))
    eligible_remaining = remaining[
        _eligible_mask(remaining, used_types, img_type, incompat)
    ]

    if eligible_remaining.empty:
        eligible_remaining = remaining
        violation = f"incompat/lock constraint relaxed for img {A['img_id']}"

    # RULE 6: prefer B not already shown to this user
    eligible_remaining, relaxed_b = _prefer_unseen_by_user(eligible_remaining, seen_img_set)
    if relaxed_b and not violation:
        violation = f"rule6 relaxed: no candidates for B unseen by user (anchor {A['img_id']})"

    # RULE 5: avoid recreating a pair already shown to this user
    eligible_remaining, relaxed_pair = _exclude_repeat_pairs(eligible_remaining, str(A["img_id"]), seen_pair_set)
    if relaxed_pair and not violation:
        violation = f"rule5 relaxed: all remaining candidates repeat a pair with img {A['img_id']}"

    eligible_remaining = eligible_remaining.copy()

    # B selection: blend info-gain vs uniform random
    if al == 0.0:
        # Fully random — skip info_gain computation entirely
        B = eligible_remaining.sample(1).iloc[0]
        ig = 0.0
    else:
        eligible_remaining["ig"] = _info_gain(eligible_remaining, A, s_col, u_col, n_col)
        top_n = max(1, round(al * 10))  # al=1 → top 10, al=0.5 → top 5, al=0.1 → top 1
        B = eligible_remaining.sort_values("ig", ascending=False).head(top_n).sample(1).iloc[0]
        ig = float(B["ig"]) if not math.isnan(float(B["ig"])) else 0.0

    return _make_pair(A, B), ig, violation