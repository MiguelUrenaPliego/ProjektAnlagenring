import numpy as np
import math
import pandas as pd
import os
import ast
import random
import json
from typing import Any, Callable, Set

# =======================================================================
# CONFIG AND ROOT PATHS
# =======================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

try:
    with open(os.path.join(ROOT, "config.json"), "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    CONFIG = {}

USER_DATA_PATH = os.path.join(ROOT, CONFIG.get("user_data_path", "user_data"))
os.makedirs(USER_DATA_PATH, exist_ok=True)

# =======================================================================
# SAVE_QIDS
# =======================================================================

def _load_save_qids(cfg: dict) -> Set[str]:
    raw = cfg.get("save_qids", [])
    if isinstance(raw, list):
        return {str(q).strip() for q in raw if str(q).strip()}
    return {q.strip() for q in str(raw).split(",") if q.strip()}


SAVE_QIDS: Set[str] = _load_save_qids(CONFIG)


# =======================================================================
# HELPERS
# =======================================================================

def parse_list(value: Any) -> list[str]:
    """Parses a string value into a list of stripped strings."""
    if value is None:
        return []
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            return [x.strip().strip('"').strip("'") for x in ast.literal_eval(s) if x.strip()]
        except (ValueError, SyntaxError):
            return [x.strip().strip('"').strip("'") for x in s[1:-1].split(",") if x.strip()]
    return [s]


def safe_json(v: Any) -> float | None | Any:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer, np.floating)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return v


def to_str_id(v: Any) -> str:
    return str(v).strip()


# =======================================================================
# CSV LOADERS (used for static source files — images.csv, questions, etc.)
# =======================================================================

def load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, quotechar='"', skipinitialspace=True)
    return pd.DataFrame()


def load_text_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(
            path,
            quotechar='"',
            skipinitialspace=True,
            keep_default_na=False,
        )
    return pd.DataFrame()


def _resolve_paths(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        entries = value
    else:
        s = str(value).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                entries = ast.literal_eval(s)
            except Exception:
                entries = [x.strip().strip('"').strip("'") for x in s[1:-1].split(",") if x.strip()]
        else:
            entries = [s]
    resolved = []
    for p in entries:
        p = p.strip()
        if os.path.isabs(p):
            resolved.append(p)
        else:
            resolved.append(os.path.join(ROOT, p))
    return resolved


def load_multi_csv(paths_value: str | list[str], text: bool = False) -> pd.DataFrame:
    paths = _resolve_paths(paths_value)
    frames = []
    for p in paths:
        df = load_text_csv(p) if text else load_csv(p)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates()
    return combined


def explode_list_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    df = df.copy()
    df[col] = df[col].apply(parse_list)
    df = df.explode(col).reset_index(drop=True)
    df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else "")
    return df


# =======================================================================
# BATCH ASSIGNMENT
# =======================================================================

def assign_batch_ids(base: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Assigns a batch_id to every image in *base* (already filtered to one scenario).

    If the source CSV has a batch_id/batch column, those values are used directly.
    Otherwise images are randomly distributed into batches of batch_size per
    (scenario, img_type) group.

    In all cases, batch IDs are remapped to be contiguous starting at 1
    per group — so gaps like (1, 3, 5) become (1, 2, 3) and batch 1 always exists.
    """
    batch_size = int(config.get("batch_size", 10))

    src_col = None
    for cname in ("batch_id", "batch"):
        if cname in base.columns:
            src_col = cname
            break

    df = base.copy()

    if src_col is not None:
        df["batch_id"] = pd.to_numeric(df[src_col], errors="coerce").fillna(1).astype(int)
    else:
        df["batch_id"] = 0
        for (sc, it), grp_idx in df.groupby(["scenario", "img_type"]).groups.items():
            shuffled = list(grp_idx)
            random.shuffle(shuffled)
            for i, idx in enumerate(shuffled):
                df.at[idx, "batch_id"] = i // batch_size + 1

    # Always remap per group so IDs are contiguous from 1 (no gaps).
    for (sc, it), grp_idx in df.groupby(["scenario", "img_type"]).groups.items():
        raw_ids = df.loc[grp_idx, "batch_id"].values
        unique_sorted = sorted(set(raw_ids))
        remap = {old: new for new, old in enumerate(unique_sorted, start=1)}
        df.loc[grp_idx, "batch_id"] = [remap[v] for v in raw_ids]

    return df


# =======================================================================
# SCENARIO STATE — MongoDB-backed
# =======================================================================

async def ensure_scenario_state(
    scenario: str,
    images_df: pd.DataFrame,
    save_qids: Set[str],
    config: dict,
    default_score: float,
    default_uncertainty: float,
    recompute_fn: Callable,
    col,  # motor collection: image_state
) -> pd.DataFrame:
    """
    Guarantees MongoDB image_state has documents for every image in *scenario*.

    Strategy:
    - Build the canonical 'base' DataFrame from images_df (same as before).
    - Fetch existing documents from MongoDB for this scenario.
    - For images already in Mongo: preserve mutable TrueSkill columns,
      overwrite static/path-derived columns from base (fresh source of truth).
    - For new images (not yet in Mongo): insert with default TrueSkill values.
    - Return the full in-memory DataFrame for pairing.py to use.
    """
    required_cols = ["img_id", "path", "img_type", "scenario", "incompatible_ids", "_serve_path"]

    base = images_df[images_df["scenario"].apply(to_str_id) == to_str_id(scenario)].copy()
    for _rc in required_cols:
        if _rc not in base.columns:
            base[_rc] = ""

    if base.empty:
        raise RuntimeError(f"No images found for scenario '{scenario}' in images.csv.")

    base["img_id"] = base["img_id"].apply(to_str_id)
    base = assign_batch_ids(base, config)

    # ── Load existing documents from MongoDB ────────────────────────────
    cursor = col.find({"scenario": scenario}, {"_id": 0})
    existing_docs = await cursor.to_list(length=None)
    existing: dict[str, dict] = {d["img_id"]: d for d in existing_docs}

    rows = []
    to_insert = []
    to_update = []  # list of (img_id, update_doc)

    for _, base_row in base.iterrows():
        img_id = to_str_id(base_row["img_id"])
        doc = existing.get(img_id)

        # Static/path columns always come from base (fresh source of truth)
        static = {
            "img_id":           img_id,
            "path":             str(base_row.get("path", "")),
            "img_type":         str(base_row.get("img_type", "")),
            "scenario":         scenario,
            "incompatible_ids": str(base_row.get("incompatible_ids", "")),
            "_serve_path":      str(base_row.get("_serve_path", "")),
            "_abs_path":        str(base_row.get("_abs_path", "")),
            "_img_source_idx":  int(base_row.get("_img_source_idx", 0)),
            "batch_id":         int(base_row["batch_id"]),
        }

        if doc is None:
            # New image — build full document with TrueSkill defaults
            new_doc = dict(static)
            for qid in save_qids:
                new_doc[f"score_{qid}"]        = default_score
                new_doc[f"uncertainty_{qid}"]  = default_uncertainty
                new_doc[f"n_answers_{qid}"]    = 0
                new_doc[f"active_batch_{qid}"] = 1
            to_insert.append(new_doc)
            rows.append(new_doc)
        else:
            # Existing image — merge: static from base, TrueSkill from Mongo
            merged = dict(doc)
            merged.update(static)  # overwrite static/path columns

            # Ensure all qid columns exist (new questions added after initial seed)
            changed = dict(static)
            for qid in save_qids:
                for col_name, default in [
                    (f"score_{qid}",        default_score),
                    (f"uncertainty_{qid}",  default_uncertainty),
                    (f"n_answers_{qid}",    0),
                    (f"active_batch_{qid}", 1),
                ]:
                    if col_name not in merged:
                        merged[col_name] = default
                        changed[col_name] = default

            # Clamp active_batch to valid range
            for qid in save_qids:
                ab_col = f"active_batch_{qid}"
                max_b = int(base_row["batch_id"])  # max in group computed later; use 1 as floor
                merged[ab_col] = max(1, int(merged.get(ab_col, 1)))

            to_update.append((img_id, {"$set": changed}))
            rows.append(merged)

    # ── Persist changes to MongoDB ───────────────────────────────────────
    if to_insert:
        await col.insert_many(to_insert)

    for img_id, update_doc in to_update:
        await col.update_one({"scenario": scenario, "img_id": img_id}, update_doc)

    # ── Ensure unique index exists ───────────────────────────────────────
    await col.create_index([("scenario", 1), ("img_id", 1)], unique=True, background=True)

    # ── Build and return in-memory DataFrame ─────────────────────────────
    df = pd.DataFrame(rows)

    # Clamp active_batch per (scenario, img_type) group to [1, max_batch_id]
    for qid in save_qids:
        ab_col = f"active_batch_{qid}"
        if ab_col in df.columns and "batch_id" in df.columns:
            for (sc, it), grp_idx in df.groupby(["scenario", "img_type"]).groups.items():
                max_b = int(df.loc[grp_idx, "batch_id"].max())
                df.loc[grp_idx, ab_col] = (
                    pd.to_numeric(df.loc[grp_idx, ab_col], errors="coerce")
                    .fillna(1).astype(int)
                    .clip(lower=1, upper=max_b)
                )

    for c in required_cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    return df