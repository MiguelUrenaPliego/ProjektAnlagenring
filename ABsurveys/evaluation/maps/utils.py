# coding: utf-8
"""
utils.py

Helper functions for the street perception mapping and data normalization pipeline.
"""

from __future__ import annotations
import os
import re
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from typing import Union

_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

def load_bild_label_coordinates(xlsx_path: str) -> dict[str, tuple[float, float] | None]:
    """
    Parses a simple two/three-column "Bild N | lat, lon | note" xlsx sheet
    (as exported for manually geo-located images that have no SWM2 database
    entry) without requiring openpyxl, using only the stdlib zipfile/ElementTree.

    Returns a dict mapping "Bild N" -> (lat, lon), or None if the cell holds
    a non-coordinate placeholder (e.g. "X" for unknown location).
    """
    if not os.path.exists(xlsx_path):
        return {}

    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{_XLSX_NS}si"):
                text = "".join(t.text or "" for t in si.iter(f"{_XLSX_NS}t"))
                shared_strings.append(text)

        sheet_name = next(n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n))
        sheet_root = ET.fromstring(zf.read(sheet_name))

        rows_cells: dict[str, dict[str, str]] = {}
        for row in sheet_root.iter(f"{_XLSX_NS}row"):
            for cell in row.findall(f"{_XLSX_NS}c"):
                ref = cell.get("r")
                col = re.match(r"[A-Z]+", ref).group()
                row_num = re.match(r"[A-Z]+(\d+)", ref).group(1)
                v = cell.find(f"{_XLSX_NS}v")
                if v is None:
                    continue
                value = shared_strings[int(v.text)] if cell.get("t") == "s" else v.text
                rows_cells.setdefault(row_num, {})[col] = value

    result: dict[str, tuple[float, float] | None] = {}
    for cells in rows_cells.values():
        label = cells.get("A")
        coord_str = cells.get("B")
        if not label or coord_str is None:
            continue
        match = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", coord_str)
        result[label] = (float(match.group(1)), float(match.group(2))) if match else None

    return result

def _all_files_exist(paths: Union[str, list[str], None]) -> bool:
    """
    Checks if all given paths exist safely.
    """
    if paths is None:
        return True
    if isinstance(paths, str):
        return os.path.exists(paths)
    if isinstance(paths, list):
        return all(os.path.exists(p) for p in paths)
    return False

def normalize_and_align_distributions(
    ts_scores: pd.Series, 
    ss_scores: pd.Series
) -> tuple[pd.Series, pd.Series, float, float]:
    """
    Standardizes and normalizes two score series to have exactly:
      1. An aligned mean of 5.0
      2. Exactly the same standard deviation (s)
      3. Global absolute bounds of [0.0, 10.0] (the extreme value reaches 0 or 10)
    
    This preserves relative rankings and guarantees perfect side-by-side comparability.
    It also returns the scale multipliers to scale their corresponding uncertainties.
    """
    ts_is_empty = ts_scores.empty or ts_scores.isna().all()
    ss_is_empty = ss_scores.empty or ss_scores.isna().all()

    if ts_is_empty and ss_is_empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 1.0, 1.0

    if ts_is_empty:
        ss_mean = ss_scores.mean()
        ss_zero = ss_scores - ss_mean
        ss_std = ss_zero.std()
        if ss_std == 0 or np.isnan(ss_std): ss_std = 1.0
        ss_norm = ss_zero / ss_std
        max_deviation = ss_norm.abs().max()
        if max_deviation == 0 or np.isnan(max_deviation): max_deviation = 1.0
        target_std = 5.0 / max_deviation
        ss_aligned = 5.0 + (ss_norm * target_std)
        return pd.Series(dtype=float), ss_aligned, 1.0, target_std / ss_std

    if ss_is_empty:
        ts_mean = ts_scores.mean()
        ts_zero = ts_scores - ts_mean
        ts_std = ts_zero.std()
        if ts_std == 0 or np.isnan(ts_std): ts_std = 1.0
        ts_norm = ts_zero / ts_std
        max_deviation = ts_norm.abs().max()
        if max_deviation == 0 or np.isnan(max_deviation): max_deviation = 1.0
        target_std = 5.0 / max_deviation
        ts_aligned = 5.0 + (ts_norm * target_std)
        return ts_aligned, pd.Series(dtype=float), target_std / ts_std, 1.0

    # Fill NaNs with means or defaults to avoid mathematical breaks
    ts_clean = ts_scores.fillna(ts_scores.mean() if not ts_scores.isna().all() else 5.0)
    ss_clean = ss_scores.fillna(ss_scores.mean() if not ss_scores.isna().all() else 5.0)

    # 1. Zero-center both distributions
    ts_mean = ts_clean.mean()
    ss_mean = ss_clean.mean()
    
    ts_zero = ts_clean - ts_mean
    ss_zero = ss_clean - ss_mean
    
    # 2. Extract standard deviations
    ts_std = ts_zero.std()
    ss_std = ss_zero.std()
    
    if ts_std == 0 or np.isnan(ts_std): ts_std = 1.0
    if ss_std == 0 or np.isnan(ss_std): ss_std = 1.0
    
    # 3. Standardize to unit variance
    ts_norm = ts_zero / ts_std
    ss_norm = ss_zero / ss_std
    
    # 4. Find the global maximum absolute deviation from the mean across both
    combined_max_deviation = max(ts_norm.abs().max(), ss_norm.abs().max())
    if combined_max_deviation == 0 or np.isnan(combined_max_deviation):
        combined_max_deviation = 1.0
        
    # Scale multiplier to make the extreme deviation exactly 5.0 (bounds [0, 10] centered at 5)
    target_std = 5.0 / combined_max_deviation
    
    # 5. Project back to target standard deviation and shift center to 5.0
    ts_aligned = 5.0 + (ts_norm * target_std)
    ss_aligned = 5.0 + (ss_norm * target_std)
    
    # 6. Compute uncertainty scale factors
    # Since scores were multiplied by (target_std / original_std), 
    # the standard deviations (uncertainties) must scale by the exact same factors.
    ts_unc_multiplier = target_std / ts_std
    ss_unc_multiplier = target_std / ss_std
    
    return ts_aligned, ss_aligned, ts_unc_multiplier, ss_unc_multiplier

def generate_simulation_data() -> tuple[list[dict], int, int]:
    """
    Generates rich, fully populated mock data matching the exact geographic boundary
    of Frankfurt's Anlagenring ring road to ensure a gorgeous and ready-to-run demo.
    """
    print("[Pipeline] Injected realistic mock dataset for Frankfurt Anlagenring...")
    
    # Simulation parameters
    num_points = 180
    unique_users = 412
    total_clicks = 8945
    metrics_list = ["walk", "bike", "stay"]
    
    # Centered at Frankfurt Anlagenring
    center_y, center_x = 50.1158, 8.6881
    
    # Generate points in a circular ring representing the ring road (Anlagenring)
    angles = np.linspace(0, 2 * np.pi, num_points)
    radius = 0.006 # Ring radius in degrees coordinates
    
    compiled_points = []
    
    for i, angle in enumerate(angles):
        img_id = f"w{i+1}" if i % 3 == 0 else (f"b{i+1}" if i % 3 == 1 else f"s{i+1}")
        
        # Add slight jitter to simulate real camera captures
        jitter_x = np.random.normal(0, 0.0003)
        jitter_y = np.random.normal(0, 0.0003)
        
        x = center_x + radius * np.cos(angle) + jitter_x
        y = center_y + radius * np.sin(angle) + jitter_y
        
        # Calculate tangent angle for photo bearing (pointing along the ring road)
        bearing = int((np.degrees(-angle) + 270) % 360)
        
        # Structure metric predictions and human logs
        metrics = {}
        for m in metrics_list:
            # Simulate real TrueSkill vs StreetScore differences with correlated structures
            base_score = 4.0 + 3.5 * np.sin(angle * 2) + np.random.normal(0, 0.8)
            
            ts_score = max(0.5, min(9.5, base_score + np.random.normal(0, 0.6)))
            ss_score = max(0.5, min(9.5, base_score + np.random.normal(0, 1.2)))
            
            # MC Dropout uncertainty represents model confidence: 
            # higher when scores are neutral (around 5.0) or randomly noisy
            ss_unc = max(0.2, min(2.8, 1.8 - 0.12 * abs(ss_score - 5.0) + np.random.normal(0, 0.3)))
            
            # TrueSkill uncertainty decreases when there are more survey answers
            n_ans = int(np.random.poisson(lam=12))
            ts_unc = max(0.1, min(1.2, 1.5 / np.sqrt(n_ans + 1) + np.random.normal(0, 0.05)))
            
            metrics[m] = {
                "trueskill": {
                    "score": round(ts_score, 2),
                    "uncertainty": round(ts_unc, 2),
                    "n_answers": n_ans
                },
                "streetscore": {
                    "score": round(ss_score, 2),
                    "uncertainty": round(ss_unc, 2),
                    "n_answers": None
                }
            }
            
        compiled_points.append({
            "id": img_id,
            "x": round(x, 6),
            "y": round(y, 6),
            "bearing": bearing,
            "img_path": f"images/Anlagenring/{'Walk' if i%3==0 else ('Bike' if i%3==1 else 'Stay')}/image_{i+1}.jpg",
            "metrics": metrics
        })

    return compiled_points, unique_users, total_clicks
