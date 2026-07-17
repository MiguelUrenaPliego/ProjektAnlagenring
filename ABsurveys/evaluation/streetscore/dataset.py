# dataset.py
# coding: utf-8

"""Dataset creation, image-level train/val/test splitting, and pair-to-label conversion.

Splitting contract
------------------
All splits are performed **at the image level** (by img_id), never at the pair level.
Each img_id is randomly assigned to exactly one of train / val / test.

Pair resolution then depends on the active loss/accuracy mode:

  pair / mixed mode
  ~~~~~~~~~~~~~~~~~
  A pair can only be used by a split if BOTH of its images belong to that split.
  Cross-split pairs (one image in train, one in val) are dropped from both sides.
  This prevents any gradient leakage through the pairwise ranking signal.

  crossentropy (single-image) mode
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  The model only sees one image at a time, so there is no leakage.
  A pair can be *shared* between splits: each image contributes to its own split.
  Concretely, img_id 1 (train) vs img_id 3 (val) in pair b where img_id 3 wins:
    • train gets  img_id 1 with label 0  (img_id 1 lost pair b)
    • val   gets  img_id 3 with label 1  (img_id 3 won  pair b)
  We call this the "image-level" resolution: build_single_image_df filtered to
  each split's img_id_set naturally produces the correct rows.

Single-image dataset
--------------------
The model is trained with pairs, but for logging / validation statistics we also
expose the single-image view:  every image that appears in at least one AB pair
receives a binary label for each appearance:

    label = 1  if this image was the *preferred* one in that pair
    label = 0  if this image was the *rejected* one in that pair

For tie (answer == "=") pairs the image appears once with label 1 and once with
label 0 (symmetric), just like the pairwise training convention.

This module is **stateless** — call the public functions directly; nothing is
stored globally.

Public API
----------
split_images_by_id(img_df, human_df, val_pct, test_pct, *, random_state)
    → dict[str, pd.DataFrame]          # keys: "train", "val", "test"

resolve_pairs_for_split(human_df, img_id_set, ...)
    → pd.DataFrame                     # pairs where BOTH images are in img_id_set

resolve_pairs_for_val_split(human_df, train_img_ids, val_img_ids, ...)
    → tuple[pd.DataFrame, pd.DataFrame]  # (train_pairs, val_pairs) — pair mode

resolve_pairs_image_level(human_df, img_id_set, ...)
    → pd.DataFrame                     # single-image rows for crossentropy mode

build_single_image_df(human_df, img_id_set)
    → pd.DataFrame                     # columns: img_id, label, pair_id, is_tie

get_split_label(img_id, split_dfs)
    → str | None                       # "train" | "val" | "test" | None
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Core splitting
# ---------------------------------------------------------------------------

def split_images_by_id(
    img_df: pd.DataFrame,
    human_df: pd.DataFrame,
    val_pct: int | None,
    test_pct: int | None,
    *,
    random_state: int = 42,
    img_types: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Split images into train / val / test **by img_id**, never by pair.

    Images that do not appear in any AB pair (orphans) are placed into the
    test set automatically — they carry no training signal but should be
    scored at inference time.

    Args:
        img_df: Full image registry DataFrame (must have an ``img_id`` column).
        human_df: Human survey DataFrame (must have ``img_id_A``, ``img_id_B``,
            ``type``, and ``answer`` columns).
        val_pct: Percentage (0–100) of *labeled* images to hold out for validation.
            ``None`` or ``0`` → no validation split.
        test_pct: Percentage (0–100) of *all* images to hold out for testing.
            ``100`` is special: test = all images, train is kept intact (useful
            when you want to score every image with an already-trained model).
            ``None`` or ``0`` → no test split beyond orphans.
        random_state: Seed for reproducible shuffling.
        img_types: Optional list of ``img_type`` values to restrict *labeled*
            image resolution (same filter as used in main.py training).
        scenarios: Optional list of ``scenario`` values to restrict *labeled*
            image resolution.

    Returns:
        dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to
        sub-DataFrames of *img_df* (possibly empty).
    """
    img_df = img_df.copy()
    img_df["img_id"] = img_df["img_id"].astype(str)

    # Filter human_df to AB rows with valid answers
    ab_df = human_df[
        (human_df["type"] == "AB") & human_df["answer"].isin(["A", "B", "="])
    ].copy()

    # Optional filters (mirror what main.py does per metric)
    if img_types:
        ab_df = ab_df[ab_df["img_type"].isin(img_types)]
    if scenarios:
        ab_df = ab_df[ab_df["scenario"].isin(scenarios)]

    # All unique img_ids that appear in at least one AB pair
    labeled_ids = (
        pd.concat([ab_df["img_id_A"], ab_df["img_id_B"]])
        .astype(str)
        .unique()
        .tolist()
    )
    labeled_ids_set = set(labeled_ids)

    # Separate labeled vs orphan images
    mask_labeled = img_df["img_id"].isin(labeled_ids_set)
    img_labeled = img_df[mask_labeled].reset_index(drop=True)
    img_orphaned = img_df[~mask_labeled].reset_index(drop=True)

    n_labeled = len(img_labeled)
    n_orphaned = len(img_orphaned)
    n_total = n_labeled + n_orphaned

    print(f"  [split_images_by_id] Total images: {n_total:,}  "
          f"(labeled: {n_labeled:,}, orphans→test: {n_orphaned:,})")

    # ── Special case: test_pct == 100 ────────────────────────────────────────
    # test=100 means "score every image at inference time" — it is completely
    # independent from the train/val split.  We still carve val out of the
    # labeled pool so training has proper validation.
    if isinstance(test_pct, int) and test_pct == 100:
        test_df = img_df.copy()  # all images go to test (for inference)
        if isinstance(val_pct, int) and 0 < val_pct < 100 and n_labeled > 0:
            img_train, img_val = train_test_split(
                img_labeled,
                test_size=val_pct / 100.0,
                random_state=random_state,
            )
            print(f"  [split_images_by_id] test_pct=100 → test=all {len(test_df):,} images, "
                  f"train={len(img_train):,} images, val={len(img_val):,} images")
        else:
            img_train = img_labeled.copy()
            img_val = pd.DataFrame(columns=img_df.columns)
            print(f"  [split_images_by_id] test_pct=100 → test=all {len(test_df):,} images, "
                  f"train={len(img_train):,} images, val=0")
        return {"train": img_train.reset_index(drop=True),
                "val": img_val.reset_index(drop=True),
                "test": test_df.reset_index(drop=True)}

    # ── Normal percentage split ───────────────────────────────────────────────
    # Step 1 — carve out test images from labeled pool
    test_labeled_ids: list[str] = []
    remaining_labeled = img_labeled.copy()

    if isinstance(test_pct, int) and test_pct > 0:
        target_test_n = round(n_total * test_pct / 100)
        still_needed = max(0, target_test_n - n_orphaned)

        if still_needed > 0 and len(remaining_labeled) > 0:
            take = min(still_needed, len(remaining_labeled))
            # Split **by img_id** (each row is already one unique image here)
            extra_test_df, remaining_labeled = train_test_split(
                remaining_labeled,
                test_size=take,
                random_state=random_state,
            )
            test_labeled_ids = extra_test_df["img_id"].tolist()

        n_test_from_labeled = len(test_labeled_ids)
        print(
            f"  [split_images_by_id] test_pct={test_pct}% → target {target_test_n:,} test images "
            f"(orphans: {n_orphaned:,} + from labeled: {n_test_from_labeled:,})"
        )

    img_test = pd.concat(
        [img_orphaned,
         img_labeled[img_labeled["img_id"].isin(test_labeled_ids)]],
        ignore_index=True,
    )

    # Step 2 — carve out val images from *remaining* labeled pool (image-level)
    img_val = pd.DataFrame(columns=img_df.columns)
    img_train = remaining_labeled.copy()

    if isinstance(val_pct, int) and 0 < val_pct < 100 and len(remaining_labeled) > 0:
        img_train, img_val = train_test_split(
            remaining_labeled,
            test_size=val_pct / 100.0,
            random_state=random_state,
        )
        print(f"  [split_images_by_id] val_pct={val_pct}% → "
              f"train {len(img_train):,} images, val {len(img_val):,} images")
    else:
        print(f"  [split_images_by_id] No val split → "
              f"train {len(img_train):,} images, val 0")

    return {
        "train": img_train.reset_index(drop=True),
        "val": img_val.reset_index(drop=True),
        "test": img_test.reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Pair resolution per split
# ---------------------------------------------------------------------------

def resolve_pairs_for_split(
    human_df: pd.DataFrame,
    img_id_set: set[str],
    *,
    question_ids: list[str] | None = None,
    img_types: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Return the subset of AB pairs where BOTH images belong to *img_id_set*.

    A pair can only belong to a split if both its images are in that split —
    this prevents any cross-split leakage.

    Args:
        human_df: Full human survey DataFrame.
        img_id_set: Set of img_ids that define the split.
        question_ids: Optional filter on ``question_id``.
        img_types: Optional filter on ``img_type``.
        scenarios: Optional filter on ``scenario``.

    Returns:
        Filtered DataFrame of AB pairs, reset index.
    """
    df = human_df[
        (human_df["type"] == "AB") & human_df["answer"].isin(["A", "B", "="])
    ].copy()

    if question_ids:
        df = df[df["question_id"].isin(question_ids)]
    if img_types:
        df = df[df["img_type"].isin(img_types)]
    if scenarios:
        df = df[df["scenario"].isin(scenarios)]

    img_id_set = {str(i) for i in img_id_set}
    mask = (
        df["img_id_A"].astype(str).isin(img_id_set)
        & df["img_id_B"].astype(str).isin(img_id_set)
    )
    result = df[mask].reset_index(drop=True)

    excluded = len(df) - len(result)
    if excluded > 0 and len(img_id_set) > 0:
        print(f"  [resolve_pairs_for_split] {excluded:,} pair(s) excluded "
              "(at least one image not in this split's img_id_set).")

    return result


def resolve_pairs_for_val_split(
    human_df: pd.DataFrame,
    train_img_ids: set[str],
    val_img_ids: set[str],
    *,
    question_ids: list[str] | None = None,
    img_types: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve AB pairs for train and val using pre-computed image-level id sets.

    A pair is assigned to train only when BOTH its images are in *train_img_ids*.
    A pair is assigned to val only when BOTH its images are in *val_img_ids*.
    Pairs that span the boundary (one image in train, one in val) are dropped
    from both splits — this is the correct behaviour to prevent leakage.

    The id sets come from :func:`split_images_by_id` which already performed
    the image-level split; this function just resolves the pairs accordingly.

    Args:
        human_df: Human survey DataFrame.
        train_img_ids: Set of img_ids assigned to the train split.
        val_img_ids: Set of img_ids assigned to the val split.
        question_ids: Optional filter on ``question_id``.
        img_types: Optional filter on ``img_type``.
        scenarios: Optional filter on ``scenario``.

    Returns:
        Tuple of (train_pairs_df, val_pairs_df), both reset-indexed.
    """
    train_img_ids = {str(i) for i in train_img_ids}
    val_img_ids = {str(i) for i in val_img_ids}

    train_pairs = resolve_pairs_for_split(
        human_df, train_img_ids,
        question_ids=question_ids, img_types=img_types, scenarios=scenarios,
    )
    val_pairs = resolve_pairs_for_split(
        human_df, val_img_ids,
        question_ids=question_ids, img_types=img_types, scenarios=scenarios,
    )

    print(f"  [resolve_pairs_for_val_split] train: {len(train_pairs):,} pairs "
          f"({len(train_img_ids):,} imgs) | val: {len(val_pairs):,} pairs "
          f"({len(val_img_ids):,} imgs)")

    return train_pairs.reset_index(drop=True), val_pairs.reset_index(drop=True)


def resolve_pairs_image_level(
    human_df: pd.DataFrame,
    img_id_set: set[str],
    *,
    question_ids: list[str] | None = None,
    img_types: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Return single-image label rows for *img_id_set* — for crossentropy mode.

    Unlike :func:`resolve_pairs_for_split` (which requires BOTH images of a pair
    to be in the same split), this function works at the **image level**: a pair
    is split between splits, and each image contributes its label row only to the
    split that owns that image.

    This is the correct strategy for crossentropy / single-image training because
    the loss is computed per image independently — there is no cross-split leakage.

    For example, given pair b: img_id 1 (train) vs img_id 3 (val), img_id 3 wins:
      • Calling with img_id_set = train_ids  →  img_id 1, label 0
      • Calling with img_id_set = val_ids    →  img_id 3, label 1

    The returned DataFrame has the same schema as :func:`build_single_image_df`:
    ``img_id, label, is_tie, pair_id, img_id_A, img_id_B, answer``.

    Args:
        human_df: Human survey DataFrame with AB pairs.
        img_id_set: Set of img_ids that define this split.
        question_ids: Optional filter on ``question_id``.
        img_types: Optional filter on ``img_type``.
        scenarios: Optional filter on ``scenario``.

    Returns:
        DataFrame of single-image rows restricted to images in *img_id_set*,
        including rows from cross-split pairs (where the other image is in a
        different split).
    """
    # build_single_image_df with img_id_set=None gets all rows, then we filter
    # to only the images owned by this split — including cross-split pair rows.
    all_rows = build_single_image_df(
        human_df,
        img_id_set=None,
        question_ids=question_ids,
        img_types=img_types,
        scenarios=scenarios,
    )
    img_id_set = {str(i) for i in img_id_set}
    result = all_rows[all_rows["img_id"].isin(img_id_set)].reset_index(drop=True)

    cross_split = result[
        ~(
            result["img_id_A"].astype(str).isin(img_id_set)
            & result["img_id_B"].astype(str).isin(img_id_set)
        )
    ]
    n_cross = len(cross_split["pair_id"].unique()) if len(cross_split) > 0 else 0
    print(f"  [resolve_pairs_image_level] {len(result):,} single-image rows "
          f"for {len(img_id_set):,} img_ids "
          f"(includes {n_cross:,} cross-split pair(s) contributing one image each).")
    return result


def build_single_image_df(
    human_df: pd.DataFrame,
    img_id_set: set[str] | None = None,
    *,
    question_ids: list[str] | None = None,
    img_types: list[str] | None = None,
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """Convert AB pairs into a single-image binary label DataFrame.

    Each row represents **one image in one comparison**:
    - label=1 means the image was preferred in that pair.
    - label=0 means the image was rejected in that pair.
    - For tie (answer="=") pairs: the image appears **twice** — once as
      preferred (label=1) and once as rejected (label=0), matching the
      symmetric tie convention used in ABPairDataset.

    This format is useful for:
    - Single-image classification loss (cross-entropy / F1).
    - Computing per-image win-rates for analysis.
    - Understanding how frequently a given image "wins".

    Args:
        human_df: Human survey DataFrame with AB pairs.
        img_id_set: If provided, only include rows where the image's img_id
            is in this set.  Useful to restrict to train/val/test images.
        question_ids: Optional filter on ``question_id``.
        img_types: Optional filter on ``img_type``.
        scenarios: Optional filter on ``scenario``.

    Returns:
        DataFrame with columns:
            img_id      — the individual image
            label       — 1 (preferred) or 0 (rejected)
            is_tie      — 1 if this row comes from a tie pair, else 0
            pair_id     — integer index of the source AB pair (for grouping)
            img_id_A    — original img_id_A from the pair (for reference)
            img_id_B    — original img_id_B from the pair (for reference)
            answer      — original answer field ("A", "B", or "=")
    """
    df = human_df[
        (human_df["type"] == "AB") & human_df["answer"].isin(["A", "B", "="])
    ].copy().reset_index(drop=True)

    if question_ids:
        df = df[df["question_id"].isin(question_ids)]
    if img_types:
        df = df[df["img_type"].isin(img_types)]
    if scenarios:
        df = df[df["scenario"].isin(scenarios)]

    df = df.reset_index(drop=True)
    df["_pair_id"] = df.index  # stable pair reference

    rows: list[dict] = []
    for _, row in df.iterrows():
        img_a = str(row["img_id_A"])
        img_b = str(row["img_id_B"])
        answer = row["answer"]
        pair_id = int(row["_pair_id"])
        base = dict(
            img_id_A=img_a,
            img_id_B=img_b,
            answer=answer,
            pair_id=pair_id,
        )

        if answer == "A":
            rows.append({**base, "img_id": img_a, "label": 1, "is_tie": 0})
            rows.append({**base, "img_id": img_b, "label": 0, "is_tie": 0})
        elif answer == "B":
            rows.append({**base, "img_id": img_a, "label": 0, "is_tie": 0})
            rows.append({**base, "img_id": img_b, "label": 1, "is_tie": 0})
        elif answer == "=":
            # Symmetric: each image appears once as preferred, once as rejected
            rows.append({**base, "img_id": img_a, "label": 1, "is_tie": 1})
            rows.append({**base, "img_id": img_b, "label": 0, "is_tie": 1})
            rows.append({**base, "img_id": img_a, "label": 0, "is_tie": 1})
            rows.append({**base, "img_id": img_b, "label": 1, "is_tie": 1})

    result = pd.DataFrame(rows, columns=[
        "img_id", "label", "is_tie", "pair_id", "img_id_A", "img_id_B", "answer"
    ])

    if img_id_set is not None:
        img_id_set = {str(i) for i in img_id_set}
        result = result[result["img_id"].isin(img_id_set)].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Win-rate summary (optional analysis helper)
# ---------------------------------------------------------------------------

def compute_win_rates(single_img_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-image win-rate statistics from a single-image label DataFrame.

    Args:
        single_img_df: Output of :func:`build_single_image_df`.

    Returns:
        DataFrame with columns: img_id, n_comparisons, n_wins, n_losses,
        n_ties_as_preferred, win_rate (wins / non-tie comparisons).
    """
    strict = single_img_df[single_img_df["is_tie"] == 0]
    tie = single_img_df[single_img_df["is_tie"] == 1]

    stats = (
        strict.groupby("img_id")["label"]
        .agg(n_comparisons="count", n_wins="sum")
        .reset_index()
    )
    stats["n_losses"] = stats["n_comparisons"] - stats["n_wins"]
    stats["win_rate"] = stats["n_wins"] / stats["n_comparisons"].clip(lower=1)

    tie_counts = (
        tie[tie["label"] == 1]
        .groupby("img_id")
        .size()
        .rename("n_ties_as_preferred")
        .reset_index()
    )

    stats = stats.merge(tie_counts, on="img_id", how="left")
    stats["n_ties_as_preferred"] = stats["n_ties_as_preferred"].fillna(0).astype(int)
    return stats.sort_values("win_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split label helper (for scores.csv stamping)
# ---------------------------------------------------------------------------

def get_split_label(img_id: str, split_dfs: dict[str, pd.DataFrame]) -> str | None:
    """Return the split name ("train", "val", or "test") for a given img_id.

    When test_pct=100, "test" means "score every image at inference time" and
    is not a genuine held-out set — every train/val image is also in test.
    Priority order is therefore train > val > test: an image only gets labelled
    "test" when it is not part of the actual train/val split (e.g. an orphan
    image with no training signal, or a real percentage-based test carve-out).

    Args:
        img_id: The image identifier to look up.
        split_dfs: Mapping returned by :func:`split_images_by_id`.

    Returns:
        Split label string, or ``None`` if the img_id is not found in any split.
    """
    img_id = str(img_id)
    for split_name in ("train", "val", "test"):
        df = split_dfs.get(split_name)
        if df is not None and not df.empty:
            if img_id in df["img_id"].astype(str).values:
                return split_name
    return None


def assign_split_column(
    scores_df: pd.DataFrame,
    split_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Add (or overwrite) a ``split`` column in *scores_df*.

    For each row, the value is "train", "val", "test", or None.

    When ``test_pct=100``, "test" means "score every image at inference time"
    and is not a genuine held-out set — every train/val image is also in test.
    Stamping it naively would make every row read "test" and hide the actual
    train/val assignment used for training, so train/val always takes priority:
    an image is only labelled "test" when it isn't part of the real train/val
    split (orphans with no training signal, or an actual percentage-based test
    carve-out when ``test_pct`` isn't the "score everything" special case).

    Args:
        scores_df: DataFrame that includes an ``img_id`` column.
        split_dfs: Output of :func:`split_images_by_id`.

    Returns:
        Copy of *scores_df* with a ``split`` column inserted right after
        ``img_id``.
    """
    result = scores_df.copy()
    result["img_id"] = result["img_id"].astype(str)

    # Build fast lookup: img_id → split_name.
    # Iterate in reverse priority order — later writes win, so "train" (highest
    # priority) is written last and overrides "val", which overrides "test".
    label_map: dict[str, str] = {}
    for split_name in ("test", "val", "train"):
        df = split_dfs.get(split_name)
        if df is not None and not df.empty:
            for iid in df["img_id"].astype(str):
                label_map[iid] = split_name

    result["split"] = result["img_id"].map(label_map)

    # Reorder: img_id, split, ...rest
    cols = result.columns.tolist()
    cols.remove("split")
    insert_at = cols.index("img_id") + 1
    cols.insert(insert_at, "split")
    result = result[cols]

    counts = result["split"].value_counts(dropna=False)
    print(f"  [assign_split_column] Split distribution in scores.csv:")
    for label in ("train", "val", "test", None):
        key = label if label is not None else float("nan")
        n = counts.get(label, 0)
        print(f"    {str(label):6s}: {n:,}")

    return result