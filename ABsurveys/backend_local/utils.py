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
# SAVE_QIDS — question IDs for which TrueSkill state is persisted.
# Populated from config.json key "save_qids" (comma-separated string or list).
# Falls back to an empty set; server.py may override after loading questions_df.
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
    """Safely converts numpy types and NaNs/Infs for JSON serialization."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer, np.floating)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return v


def to_str_id(v: Any) -> str:
    """Normalises any id (int or str) to a stripped string for safe comparison."""
    return str(v).strip()


# =======================================================================
# CSV LOADERS
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
# SCENARIO STATE MANAGEMENT HELPERS
# =======================================================================

def assign_batch_ids(base: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assigns a batch_id to every image in *base* (already filtered to one scenario)."""
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

    # Always remap batch_ids per (scenario, img_type) group to be
    # contiguous starting at 1. This ensures batch 1 always exists
    # regardless of source data gaps (e.g. 1,3,5 → 1,2,3).
    for (sc, it), grp_idx in df.groupby(["scenario", "img_type"]).groups.items():
        raw_ids = df.loc[grp_idx, "batch_id"].values
        unique_sorted = sorted(set(raw_ids))
        remap = {old: new for new, old in enumerate(unique_sorted, start=1)}
        df.loc[grp_idx, "batch_id"] = [remap[v] for v in raw_ids]

    return df


def ensure_scenario_state(
    scenario: str,
    images_df: pd.DataFrame,
    save_qids: Set[str],
    config: dict,
    user_data_path: str,
    default_score: float,
    default_uncertainty: float,
    recompute_fn: Callable,
) -> pd.DataFrame:
    """Guarantees user_data/{scenario}_images.csv exists with correct schema."""
    path = os.path.join(user_data_path, f"{scenario}_images.csv")
    required_cols = ["img_id", "path", "img_type", "scenario", "incompatible_ids", "_serve_path"]

    base = images_df[images_df["scenario"].apply(to_str_id) == to_str_id(scenario)].copy()
    for _rc in required_cols:
        if _rc not in base.columns:
            base[_rc] = ""

    if base.empty:
        raise RuntimeError(f"No images found for scenario '{scenario}' in images.csv.")

    base["img_id"] = base["img_id"].apply(to_str_id)
    base = assign_batch_ids(base, config)

    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"img_id": str})
        df["img_id"] = df["img_id"].apply(to_str_id)
        merge_on = ["img_id", "path", "img_type", "scenario"]
        df = df.merge(base[required_cols + ["batch_id"]], on=merge_on, how="right", suffixes=("", "_new"))
        # Always overwrite path-derived columns from base (fresh from images_df).
        # Never keep a stale _serve_path that was saved to disk on a previous run.
        for _rc in ["incompatible_ids", "_serve_path", "_abs_path"]:
            col_new = f"{_rc}_new"
            if col_new in df.columns:
                df[_rc] = df[col_new]          # take fresh value unconditionally
                df.drop(columns=[col_new], inplace=True)
            elif _rc not in df.columns and _rc in base.columns:
                fresh = base.set_index("img_id")[_rc]
                df[_rc] = df["img_id"].map(fresh)
        if "batch_id_new" in df.columns:
            df["batch_id"] = df["batch_id_new"].combine_first(df.get("batch_id"))
            df.drop(columns=["batch_id_new"], inplace=True)
    else:
        df = base.copy()

    for qid in save_qids:
        s_col = f"score_{qid}"
        u_col = f"uncertainty_{qid}"
        n_col = f"n_answers_{qid}"
        ab_col = f"active_batch_{qid}"

        if s_col not in df.columns:
            df[s_col] = default_score
        else:
            df[s_col] = pd.to_numeric(df[s_col], errors="coerce").fillna(default_score)

        if u_col not in df.columns:
            df[u_col] = default_uncertainty
        else:
            df[u_col] = pd.to_numeric(df[u_col], errors="coerce").fillna(default_uncertainty)

        if n_col not in df.columns:
            df[n_col] = 0
        else:
            df[n_col] = pd.to_numeric(df[n_col], errors="coerce").fillna(0).astype(int)

        if ab_col not in df.columns:
            df[ab_col] = 1
        else:
            df[ab_col] = pd.to_numeric(df[ab_col], errors="coerce").fillna(1).astype(int)
            for (sc, it), grp_idx in df.groupby(["scenario", "img_type"]).groups.items():
                max_b = int(df.loc[grp_idx, "batch_id"].max()) if "batch_id" in df.columns else 1
                df.loc[grp_idx, ab_col] = df.loc[grp_idx, ab_col].clip(lower=1, upper=max_b)

    for c in required_cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    df.to_csv(path, index=False)
    return df