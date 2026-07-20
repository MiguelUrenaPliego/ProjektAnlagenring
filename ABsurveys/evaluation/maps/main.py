# coding: utf-8
"""
main.py

End-to-end pipeline to configure, load, clean, normalize, and render multi-model 
street perception data on an interactive Leaflet map. Handles both real-world CSV 
inputs and robust fallback simulation for testing without data files.
"""

from __future__ import annotations
import os
import re
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

# Import our customized map generation engine
from map import generate_custom_html_map
from utils import (
    _all_files_exist,
    normalize_and_align_distributions,
    generate_simulation_data,
    load_bild_label_coordinates
)

# ============================================================================
# CONFIGURATION — Edit freely to point to your survey and model datasets
# ============================================================================

# Define the absolute root output path for generated artifacts (map.html, etc.)
# Fallback is current directory
ROOT_PATH: str = "/home/miguel/Documents/UNI/Master/2/ProjektVerkehr/ABsurveys"

# List of image CSV file(s) that match images with img_ids.
# Can be a single str or a list of files.
IMG_PATHS: Union[str, list[str]] = "images/Anlagenring/images.csv"

# Human survey data (CSV file mapping user clicks/choices)
HUMAN_DF_PATHS: Union[str, list[str]] = "user_data/Anlagenring_user_data_merged.csv"

# TrueSkill score outcomes from human surveys (CSV contains score, uncertainty, n_answers columns)
TRUESKILL_DF_PATHS: Union[str, list[str], None] = "user_data/Anlagenring_user_images_merged.csv"

# StreetScore predictions from ML models (CSV contains walk/bike/stay score, entropy, MC dropout columns)
STREETSCORE_DF_PATHS: Union[str, list[str], None] = "evaluation/streetscore/models/FrankfurtAnlagenring/scores.csv"

# Optional path to SWM2 database containing photo coordinates and bearing metadata
SWM2_DATABASE_PATH: Union[str, list[str], None] = "images/Anlagenring/database.swm2"

# --- Manually geo-located image groups (images absent from the SWM2 database) ---
# Each entry maps images whose path starts with "path_prefix" to coordinates read
# from a small xlsx sheet (label "Bild N" -> "lat, lon") and tags them for display.
GRUPPE_LABEL_OVERRIDES = [
    {
        "path_prefix": "images/Anlagenring/GruppeSafety/",
        "filename_pattern": r"Bild_Sicherheit_(\d+)\.",
        "label_template": "Bild {n}",
        "coords_xlsx": "images/Anlagenring/GruppeSafety/Koordinaten.xlsx",
        "tag": "Gruppe2",
    }
]

# --- Metrics mapping settings ------------------------------------------------
# Define the active metrics you want to load and compare
METRICS_MAP = [
    {
        "streetscore_metric": "walk",
        "question_id": "walk-preference",
        "img_type": "walk",
        "scenario": "Anlagenring"
    },
    {
        "streetscore_metric": "bike",
        "question_id": "bike-preference",
        "img_type": "bike",
        "scenario": "Anlagenring"
    },
    {
        "streetscore_metric": "stay",
        "question_id": "stay-preference",
        "img_type": "stay",
        "scenario": "Anlagenring"
    }
]

# SCORE NORMALIZATION METHODOLOGY has been moved to utils.py

# Helper function to resolve configuration paths relative to ROOT_PATH
def get_abs_path(p: Union[str, list[str], None]) -> Union[str, list[str], None]:
    if p is None:
        return None
    if isinstance(p, list):
        return [get_abs_path(item) for item in p]
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(ROOT_PATH, p))

# ============================================================================
# MAIN DATA PIPELINE
# ============================================================================
def load_and_compile_perceptions() -> tuple[list[dict], int, int, bool, bool]:
    """
    Loads raw tables, merges coordinates, matches image assets, resolves and scales
    uncertainty parameters, executes dual-distribution normalization, and outputs 
    compiled data ready for Leaflet dashboard rendering.
    """
    print("[Pipeline] Starting data ingestion and cleaning...")

    resolved_img_paths = get_abs_path(IMG_PATHS)
    resolved_human_paths = get_abs_path(HUMAN_DF_PATHS)
    resolved_trueskill_paths = get_abs_path(TRUESKILL_DF_PATHS)
    resolved_streetscore_paths = get_abs_path(STREETSCORE_DF_PATHS)
    resolved_swm2_paths = get_abs_path(SWM2_DATABASE_PATH)

    # Verification: If local files don't exist, we fallback to a beautiful, realistic 
    # mock simulation so that the code is fully testable and runnable out-of-the-box!
    files_exist = (
        _all_files_exist(resolved_img_paths) and
        _all_files_exist(resolved_human_paths) and 
        (resolved_trueskill_paths is None or _all_files_exist(resolved_trueskill_paths)) and 
        (resolved_streetscore_paths is None or _all_files_exist(resolved_streetscore_paths))
    )

    if not files_exist:
        print("[Pipeline] Data files not fully found. Activating Realistic Simulation Mode!")
        pts, users, clicks = generate_simulation_data()
        return pts, users, clicks, True, True

    # --- IMAGE PATHS & SWM2 COORDINATES RESOLUTION ---
    # Normalize resolved_img_paths to a list of strings
    if isinstance(resolved_img_paths, str):
        img_paths_list = [resolved_img_paths]
    else:
        img_paths_list = resolved_img_paths

    # Normalize resolved_swm2_paths to a list of corresponding length
    if resolved_swm2_paths is None:
        swm2_paths_list = [None] * len(img_paths_list)
    elif isinstance(resolved_swm2_paths, str):
        swm2_paths_list = [resolved_swm2_paths] * len(img_paths_list)
    else:
        swm2_paths_list = resolved_swm2_paths

    # Pad swm2_paths_list with None to match img_paths_list if shorter
    if len(swm2_paths_list) < len(img_paths_list):
        swm2_paths_list = swm2_paths_list + [None] * (len(img_paths_list) - len(swm2_paths_list))

    # For each pair, load the images CSV and merge SWM2 database metadata if exists
    img_dfs = []
    for img_csv, swm2_db in zip(img_paths_list, swm2_paths_list):
        print(f"[Pipeline] Loading images index from: {img_csv}")
        df = pd.read_csv(img_csv)
        
        # Resolve paths to be relative to ROOT_PATH
        csv_dir = os.path.dirname(os.path.abspath(img_csv))
        def resolve_rel_path(row_path):
            if pd.isna(row_path):
                return row_path
            full_img_path = os.path.join(csv_dir, str(row_path))
            try:
                return os.path.relpath(full_img_path, os.path.abspath(ROOT_PATH))
            except Exception:
                return str(row_path)
        
        if "path" in df.columns:
            df["path"] = df["path"].apply(resolve_rel_path)
        
        # Merge photo locations from SWM2 if it exists
        if swm2_db is not None and os.path.exists(swm2_db):
            print(f"[Pipeline] Merging photo locations from SWM2 database: {swm2_db}")
            try:
                conn = sqlite3.connect(swm2_db)
                db_meta = pd.read_sql_query("""
                    SELECT photo_path, lon AS x, lat AS y, bearing
                    FROM photos p JOIN points pt ON p.uuid = pt.fid
                """, conn)
                conn.close()
                
                db_meta["filename"] = db_meta["photo_path"].apply(os.path.basename)
                df["filename"] = df["path"].apply(os.path.basename)
                
                df = df.merge(db_meta[["filename", "x", "y", "bearing"]], on="filename", how="left", suffixes=("", "_db"))
                
                # Resolve coordinates and bearing
                for col in ["x", "y", "bearing"]:
                    db_col = f"{col}_db"
                    if db_col in df.columns:
                        df[col] = df[db_col].combine_first(df[col]) if col in df.columns else df[db_col]
                        df.drop(columns=[db_col], inplace=True)
            except Exception as e:
                print(f"[Pipeline] SWM2 Join failed for {swm2_db}: {e}")
        
        img_dfs.append(df)
        
    img_df = pd.concat(img_dfs, ignore_index=True)

    # Load human clicks (human_df)
    if resolved_human_paths is not None:
        print(f"[Pipeline] Loading human clicks from: {resolved_human_paths}")
        human_df = pd.read_csv(resolved_human_paths) if isinstance(resolved_human_paths, str) else pd.concat([pd.read_csv(p) for p in resolved_human_paths])
    else:
        human_df = pd.DataFrame()

    # Load TrueSkill outcomes
    if resolved_trueskill_paths is not None:
        print(f"[Pipeline] Loading TrueSkill outcomes from: {resolved_trueskill_paths}")
        trueskill_df = pd.read_csv(resolved_trueskill_paths) if isinstance(resolved_trueskill_paths, str) else pd.concat([pd.read_csv(p) for p in resolved_trueskill_paths])
    else:
        print("[Pipeline] TrueSkill outcomes omitted (None).")
        trueskill_df = pd.DataFrame()

    # Load StreetScore ML outcomes
    if resolved_streetscore_paths is not None:
        print(f"[Pipeline] Loading StreetScore outcomes from: {resolved_streetscore_paths}")
        streetscore_df = pd.read_csv(resolved_streetscore_paths) if isinstance(resolved_streetscore_paths, str) else pd.concat([pd.read_csv(p) for p in resolved_streetscore_paths])
    else:
        print("[Pipeline] StreetScore outcomes omitted (None).")
        streetscore_df = pd.DataFrame()

    # Strip any trailing/leading whitespaces from all loaded columns
    img_df.columns = img_df.columns.str.strip()
    if not trueskill_df.empty:
        trueskill_df.columns = trueskill_df.columns.str.strip()
    if not streetscore_df.empty:
        streetscore_df.columns = streetscore_df.columns.str.strip()

    # Normalize lat/lon column names
    if "lon" in img_df.columns: img_df.rename(columns={"lon": "x"}, inplace=True)
    if "lat" in img_df.columns: img_df.rename(columns={"lat": "y"}, inplace=True)
    
    # Fill in default mock bearings/coords if missing
    if "x" not in img_df.columns: img_df["x"] = 8.6821 + np.random.normal(0, 0.002, len(img_df))
    if "y" not in img_df.columns: img_df["y"] = 50.1109 + np.random.normal(0, 0.002, len(img_df))
    if "bearing" not in img_df.columns: img_df["bearing"] = None

    # Define a robust clean function for img_id keys to bypass type coercion issues (.0 ending)
    def clean_img_id(val):
        if pd.isna(val):
            return ""
        s = str(val).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    # Apply robust cleaning to all img_id values across all datasets
    img_df["img_id"] = img_df["img_id"].apply(clean_img_id)
    if not trueskill_df.empty:
        trueskill_df["img_id"] = trueskill_df["img_id"].apply(clean_img_id)
    if not streetscore_df.empty:
        streetscore_df["img_id"] = streetscore_df["img_id"].apply(clean_img_id)

    # --- Apply manually geo-located overrides & tags for image groups missing from SWM2 ---
    img_id_tags: dict[str, str] = {}
    for override in GRUPPE_LABEL_OVERRIDES:
        prefix = override["path_prefix"]
        matches = img_df["path"].astype(str).str.replace("\\", "/", regex=False).str.startswith(prefix)
        if not matches.any():
            continue

        coords_path = get_abs_path(override["coords_xlsx"])
        label_coords = load_bild_label_coordinates(coords_path)
        pattern = re.compile(override["filename_pattern"])

        for idx in img_df[matches].index:
            filename = os.path.basename(str(img_df.at[idx, "path"]))
            m = pattern.search(filename)
            if not m:
                continue
            label = override["label_template"].format(n=m.group(1))
            coord = label_coords.get(label)
            if coord is not None:
                lat, lon = coord
                img_df.at[idx, "x"] = lon
                img_df.at[idx, "y"] = lat
            img_id_tags[str(img_df.at[idx, "img_id"])] = override["tag"]

    # Normalize source-of-truth metadata columns to lowercase & stripped for case-insensitive joins/filters
    if "img_type" in img_df.columns:
        img_df["img_type"] = img_df["img_type"].astype(str).str.strip().str.lower()
    if "scenario" in img_df.columns:
        img_df["scenario"] = img_df["scenario"].astype(str).str.strip().str.lower()

    # Re-merge img_type and scenario from img_df as source of truth if they exist in img_df
    cols_to_merge = ["img_id"]
    if "img_type" in img_df.columns:
        cols_to_merge.append("img_type")
    if "scenario" in img_df.columns:
        cols_to_merge.append("scenario")

    # For TrueSkill metadata alignment
    if not trueskill_df.empty:
        # Drop paths or other redundant metadata columns that might be stale, inaccurate, or wrong
        for col in ["path", "_base_dir", "abs_path", "img_type", "scenario"]:
            if col in trueskill_df.columns:
                trueskill_df.drop(columns=[col], inplace=True)
        if len(cols_to_merge) > 1:
            trueskill_df = trueskill_df.merge(img_df[cols_to_merge], on="img_id", how="left")

    # For StreetScore metadata alignment
    if not streetscore_df.empty:
        # Drop paths or other redundant metadata columns that might be stale, inaccurate, or wrong
        for col in ["path", "_base_dir", "abs_path", "img_type", "scenario"]:
            if col in streetscore_df.columns:
                streetscore_df.drop(columns=[col], inplace=True)
        if len(cols_to_merge) > 1:
            streetscore_df = streetscore_df.merge(img_df[cols_to_merge], on="img_id", how="left")

    # Extract split values if available
    split_map = {}
    if "split" in img_df.columns:
        split_map.update(img_df.set_index("img_id")["split"].dropna().to_dict())
    if not streetscore_df.empty and "split" in streetscore_df.columns:
        split_map.update(streetscore_df.set_index("img_id")["split"].dropna().to_dict())
    if not trueskill_df.empty and "split" in trueskill_df.columns:
        split_map.update(trueskill_df.set_index("img_id")["split"].dropna().to_dict())
    split_map = {clean_img_id(k): str(v).strip() for k, v in split_map.items()}

    # StreetScore's "test" split is scored against 100% of images (IMG_TEST_PATHS=100
    # in evaluation/streetscore/main.py), so it does not denote a genuine held-out
    # set and is meaningless as a filter category. Only the real train/val split
    # (carved out before that whole-dataset test pass) is informative.
    split_map = {k: v for k, v in split_map.items() if v.lower() != "test"}

    # Calculate survey stats
    unique_users = int(human_df["user_id"].nunique()) if "user_id" in human_df.columns else 0
    if "type" in human_df.columns:
        total_clicks = int((human_df["type"] == "AB").sum())
    else:
        total_clicks = len(human_df)

    # Compile the final mapping of points
    points_dict: dict[str, dict] = {}

    # Initialize master index for each image
    for _, img_row in img_df.iterrows():
        img_id = str(img_row["img_id"])
        # Format path to be relative to ROOT_PATH for portability
        rel_path = str(img_row["path"])
        
        points_dict[img_id] = {
            "id": img_id,
            "x": float(img_row["x"]) if pd.notna(img_row["x"]) else None,
            "y": float(img_row["y"]) if pd.notna(img_row["y"]) else None,
            "bearing": float(img_row["bearing"]) if pd.notna(img_row["bearing"]) else None,
            "img_path": rel_path,
            "split": img_id_tags.get(img_id, split_map.get(img_id, None)),
            "tag": img_id_tags.get(img_id, None),
            "metrics": {}
        }

    # Loop through configured metrics and perform alignment/normalization
    for mconfig in METRICS_MAP:
        metric_val = mconfig["streetscore_metric"]
        qid_val = mconfig["question_id"]
        img_type_val = mconfig["img_type"]
        scenario_val = mconfig.get("scenario")

        # Determine target metric key
        metric_key = "-".join(metric_val) if isinstance(metric_val, list) else metric_val
        
        print(f"[Pipeline] Processing metric: '{metric_key}'...")

        # Prepare lists of columns
        qids = qid_val if isinstance(qid_val, list) else [qid_val]
        ts_cols = [f"score_{q}" for q in qids]
        ts_unc_cols = [f"uncertainty_{q}" for q in qids]
        ts_n_answers_cols = [f"n_answers_{q}" for q in qids]

        metrics = metric_val if isinstance(metric_val, list) else [metric_val]
        ss_cols = [m for m in metrics]
        ss_unc_cols = [f"uncertainty_mc_{m}" for m in metrics]
        ss_entropy_cols = [f"entropy_{m}" for m in metrics]

        # Filter matching records
        ts_sub = trueskill_df.copy()
        ss_sub = streetscore_df.copy()

        # Execute case-insensitive filters for robust joins and types comparison
        if not ts_sub.empty and "img_type" in ts_sub.columns and img_type_val is not None:
            if isinstance(img_type_val, list):
                val_list_lower = [str(v).lower() for v in img_type_val]
                filtered = ts_sub[ts_sub["img_type"].astype(str).str.lower().isin(val_list_lower)]
                if not filtered.empty:
                    ts_sub = filtered
            else:
                filtered = ts_sub[ts_sub["img_type"].astype(str).str.lower() == str(img_type_val).lower()]
                if not filtered.empty:
                    ts_sub = filtered

        if not ss_sub.empty and "img_type" in ss_sub.columns and img_type_val is not None:
            if isinstance(img_type_val, list):
                val_list_lower = [str(v).lower() for v in img_type_val]
                filtered = ss_sub[ss_sub["img_type"].astype(str).str.lower().isin(val_list_lower)]
                if not filtered.empty:
                    ss_sub = filtered
            else:
                filtered = ss_sub[ss_sub["img_type"].astype(str).str.lower() == str(img_type_val).lower()]
                if not filtered.empty:
                    ss_sub = filtered

        if scenario_val is not None:
            if not ts_sub.empty and "scenario" in ts_sub.columns:
                if isinstance(scenario_val, list):
                    sc_list_lower = [str(s).lower() for s in scenario_val]
                    filtered = ts_sub[ts_sub["scenario"].astype(str).str.lower().isin(sc_list_lower)]
                    if not filtered.empty:
                        ts_sub = filtered
                else:
                    filtered = ts_sub[ts_sub["scenario"].astype(str).str.lower() == str(scenario_val).lower()]
                    if not filtered.empty:
                        ts_sub = filtered
            if not ss_sub.empty and "scenario" in ss_sub.columns:
                if isinstance(scenario_val, list):
                    sc_list_lower = [str(s).lower() for s in scenario_val]
                    filtered = ss_sub[ss_sub["scenario"].astype(str).str.lower().isin(sc_list_lower)]
                    if not filtered.empty:
                        ss_sub = filtered
                else:
                    filtered = ss_sub[ss_sub["scenario"].astype(str).str.lower() == str(scenario_val).lower()]
                    if not filtered.empty:
                        ss_sub = filtered

        # Extract score sub-slices
        existing_ts_cols = [c for c in ts_cols if c in ts_sub.columns] if not ts_sub.empty else []
        if existing_ts_cols:
            ts_scores = ts_sub.set_index("img_id")[existing_ts_cols].mean(axis=1)
        else:
            ts_scores = pd.Series(dtype=float)

        existing_ss_cols = [c for c in ss_cols if c in ss_sub.columns] if not ss_sub.empty else []
        if existing_ss_cols:
            ss_scores = ss_sub.set_index("img_id")[existing_ss_cols].mean(axis=1)
        else:
            ss_scores = pd.Series(dtype=float)

        ts_aligned_vals, ss_aligned_vals, ts_unc_mult, ss_unc_mult = normalize_and_align_distributions(
            ts_scores, ss_scores
        )

        # Store aligned scores and scaled uncertainties in master points index
        for img_id in points_dict:
            ts_row = ts_sub[ts_sub["img_id"] == img_id] if not ts_sub.empty and "img_id" in ts_sub.columns else pd.DataFrame()
            ss_row = ss_sub[ss_sub["img_id"] == img_id] if not ss_sub.empty and "img_id" in ss_sub.columns else pd.DataFrame()
            
            ts_score = float(ts_aligned_vals.get(img_id, np.nan)) if not ts_aligned_vals.empty else np.nan
            ss_score = float(ss_aligned_vals.get(img_id, np.nan)) if not ss_aligned_vals.empty else np.nan

            # Retrieve and scale TrueSkill uncertainty
            ts_unc = np.nan
            existing_ts_unc_cols = [c for c in ts_unc_cols if c in ts_row.columns] if not ts_row.empty else []
            if not ts_row.empty and existing_ts_unc_cols:
                ts_unc = float(ts_row.iloc[0][existing_ts_unc_cols].mean()) * ts_unc_mult
                if pd.isna(ts_unc): ts_unc = 0.5 * ts_unc_mult

            # Retrieve and scale StreetScore uncertainty
            ss_unc = np.nan
            existing_ss_unc_cols = [c for c in ss_unc_cols if c in ss_row.columns] if not ss_row.empty else []
            existing_ss_entropy_cols = [c for c in ss_entropy_cols if c in ss_row.columns] if not ss_row.empty else []
            
            if not ss_row.empty and existing_ss_unc_cols:
                ss_unc = float(ss_row.iloc[0][existing_ss_unc_cols].mean()) * ss_unc_mult
            elif not ss_row.empty and existing_ss_entropy_cols:
                ss_unc = float(ss_row.iloc[0][existing_ss_entropy_cols].mean()) * 1.5 * ss_unc_mult
                
            # Grab answers count
            n_answers = None
            existing_ts_n_ans_cols = [c for c in ts_n_answers_cols if c in ts_row.columns] if not ts_row.empty else []
            if not ts_row.empty and existing_ts_n_ans_cols:
                n_answers = int(ts_row.iloc[0][existing_ts_n_ans_cols].sum())

            # Populate metrics tree
            points_dict[img_id]["metrics"][metric_key] = {
                "trueskill": {
                    "score": None if pd.isna(ts_score) else ts_score,
                    "uncertainty": None if pd.isna(ts_unc) else ts_unc,
                    "n_answers": n_answers
                },
                "streetscore": {
                    "score": None if pd.isna(ss_score) else ss_score,
                    "uncertainty": None if pd.isna(ss_unc) else ss_unc,
                    "n_answers": None
                }
            }

    # Convert dictionary to list of records
    compiled_points = list(points_dict.values())
    has_trueskill = TRUESKILL_DF_PATHS is not None
    has_streetscore = STREETSCORE_DF_PATHS is not None
    return compiled_points, unique_users, total_clicks, has_trueskill, has_streetscore

# REALISTIC SIMULATION GENERATOR has been moved to utils.py

# ============================================================================
# PIPELINE EXECUTION ENTRYPOINT
# ============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("      FRANKFURT ANLAGENRING PERCEPTION DATA MAPPING SYSTEM      ")
    print("=" * 65)
    
    # 1. Load and compile dataset
    points, users, clicks, has_ts, has_ss = load_and_compile_perceptions()
    
    # 2. Extract metrics list
    metrics_list = list(points[0]["metrics"].keys()) if points else ["walk", "bike", "stay"]
    default_metric = "walk" if "walk" in metrics_list else metrics_list[0]
    
    # 3. Output path
    output_html_file = os.path.join(ROOT_PATH, "map.html")
    
    # 4. Trigger Folium builder
    generate_custom_html_map(
        points_data=points,
        unique_users=users,
        total_clicks=clicks,
        metrics_list=metrics_list,
        default_metric=default_metric,
        output_path=output_html_file,
        has_trueskill=has_ts,
        has_streetscore=has_ss
    )
    
    print("-" * 65)
    print(f"Success! Perceptual map successfully compiled!")
    print(f"Interactive File: {output_html_file}")
    print("=" * 65)
