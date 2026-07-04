"""
=======================================================================
server.py — Survey Backend
=======================================================================

Image folder mounting
---------------------
Each entry in config["images_path"] is an images.csv whose "path" column
contains paths relative to that CSV's parent directory.

    images_path = "/home/miguel/.../images/images.csv"
    parent      = "/home/miguel/.../images/"
    row path    = "Anlagenring/1.jpg"
    served at   = "/images/0/Anlagenring/1.jpg"

When images_path is a list each entry gets its own mount index:
    /images/0  →  parent of images_path[0]
    /images/1  →  parent of images_path[1]

IMPORTANT — mount order
-----------------------
FastAPI/Starlette matches routes in registration order.
StaticFiles mounts are greedy, so image mounts (/images/N) and the
frontend mount (/static) must be registered AFTER all @app route
decorators, otherwise API calls could be swallowed.
=======================================================================
"""

from __future__ import annotations

import os
import json
import random
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import trueskill_utils
from . import pairing
from . import utils

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------

app = FastAPI(title="Survey App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

with open(os.path.join(ROOT, "config.json"), "r") as f:
    CONFIG: dict = json.load(f)

USER_DATA_PATH = os.path.join(ROOT, CONFIG["user_data_path"])
os.makedirs(USER_DATA_PATH, exist_ok=True)


# =====================================================================
# HELPERS
# =====================================================================

def _to_path_list(value: str | list) -> list[str]:
    """Normalise a config value to a list of absolute paths."""
    entries: list[str] = value if isinstance(value, list) else [value]
    result: list[str] = []
    for p in entries:
        p = str(p).strip()
        result.append(p if os.path.isabs(p) else os.path.join(ROOT, p))
    return result


def _load_csvs(value: str | list, text: bool = False, drop_duplicates: bool = True) -> pd.DataFrame:
    """Load one or more CSVs into a single concatenated DataFrame."""
    frames: list[pd.DataFrame] = []
    for path in _to_path_list(value):
        df = utils.load_text_csv(path) if text else utils.load_csv(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if drop_duplicates:
        combined = combined.drop_duplicates()
    return combined


# =====================================================================
# IMAGE LOADING
# =====================================================================

def _load_images_df(images_path_value: str | list) -> pd.DataFrame:
    """
    Load all images CSVs.

    For each source CSV, the "path" column contains paths relative to
    that CSV's parent directory.  We compute:

        _abs_path   — full path on disk (for debugging)
        _serve_path — URL the browser will request, e.g. /images/0/Anlagenring/1.jpg
    """
    csv_paths = _to_path_list(images_path_value)
    frames: list[pd.DataFrame] = []

    for idx, csv_path in enumerate(csv_paths):
        df = utils.load_csv(csv_path)
        if df.empty:
            continue

        parent_dir = os.path.dirname(os.path.abspath(csv_path))

        def _abs(p: str, parent: str = parent_dir) -> str:
            rel = str(p).strip().lstrip("/\\")
            return os.path.join(parent, rel)

        def _url(p: str, i: int = idx) -> str:
            # Normalise: strip leading slashes, use forward slashes
            rel = str(p).strip().lstrip("/\\").replace("\\", "/")
            return f"/images/{i}/{rel}"

        df["_abs_path"]      = df["path"].apply(_abs)
        df["_serve_path"]    = df["path"].apply(_url)
        df["_img_source_idx"] = idx

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    key_cols = [c for c in ["img_id", "scenario", "img_type"] if c in frames[0].columns]
    combined = pd.concat(frames, ignore_index=True)
    if key_cols:
        combined = combined.drop_duplicates(subset=key_cols)
    return combined


# =====================================================================
# LOAD STATIC DATASETS
# =====================================================================

scenarios_df = _load_csvs(CONFIG["scenarios_path"], drop_duplicates=False)
languages_df = _load_csvs(CONFIG["languages_path"])
questions_df = _load_csvs(CONFIG["questions_path"])
images_df    = _load_images_df(CONFIG["images_path"])

for _df in [scenarios_df, languages_df]:
    if not _df.empty:
        obj_cols = _df.select_dtypes(include="object").columns
        _df[obj_cols] = _df[obj_cols].fillna("")

if not images_df.empty:
    images_df["img_id"] = images_df["img_id"].astype(str)

# =====================================================================
# SAVE_QIDS
# =====================================================================

SAVE_QIDS = utils.SAVE_QIDS

if not SAVE_QIDS and not questions_df.empty:
    if "question_id" in questions_df.columns and "question_type" in questions_df.columns:
        ab_rows = questions_df[questions_df["question_type"].str.upper() == "AB"]
        SAVE_QIDS = {utils.to_str_id(q) for q in ab_rows["question_id"]}
        utils.SAVE_QIDS = SAVE_QIDS

# =====================================================================
# SCENARIO LIST & STATE CACHE
# =====================================================================

scenario_list: List[str] = (
    CONFIG["scenario"]
    if isinstance(CONFIG["scenario"], list)
    else [CONFIG["scenario"]]
)

scenario_state: Dict[str, pd.DataFrame] = {}

for sc in scenario_list:
    scenario_state[sc] = trueskill_utils.ensure_scenario_state(
        scenario=sc,
        images_df=images_df,
        save_qids=SAVE_QIDS,
        config=CONFIG,
        user_data_path=USER_DATA_PATH,
        default_score=trueskill_utils.DEFAULT_SCORE,
        default_uncertainty=trueskill_utils.DEFAULT_UNCERTAINTY,
        recompute_fn=trueskill_utils.recompute_from_history,
    )
    # Startup sanity check
    _sp = scenario_state[sc].get("_serve_path", pd.Series(dtype=str))
    print(f"[server] scenario={sc!r}  img_types={images_df["img_type"].unique().tolist()} images={len(_sp)}  sample _serve_path={_sp.iloc[0] if not _sp.empty else 'EMPTY'}")

# =====================================================================
# API ENDPOINTS  (must be registered BEFORE static mounts)
# =====================================================================

@app.get("/")
def index() -> HTMLResponse:
    """Serve frontend entry point."""
    with open(os.path.join(ROOT, "frontend", "index.html"), "r") as f:
        return HTMLResponse(f.read())


@app.get("/api/images-debug")
def images_debug() -> list[dict]:
    """
    Debug endpoint — returns the first 20 rows of images_df showing
    path, _abs_path, _serve_path and whether the file exists on disk.
    Remove or restrict this endpoint in production.
    """
    if images_df.empty:
        return [{"error": "images_df is empty"}]
    cols = ["img_id", "path", "_abs_path", "_serve_path"]
    cols = [c for c in cols if c in images_df.columns]
    rows = images_df[cols].head(20).copy()
    rows["_file_exists"] = rows["_abs_path"].apply(os.path.exists)
    return rows.to_dict(orient="records")


@app.get("/api/languages")
def get_languages() -> list[dict]:
    return languages_df.to_dict(orient="records")


@app.get("/api/survey")
def get_survey(language: str = "english") -> list[dict]:
    matches = languages_df[languages_df["language"] == language]
    if matches.empty:
        return []

    lang_row  = matches.iloc[0]
    lang_file = os.path.join(ROOT, "questions", lang_row["file"])
    lang_df   = utils.load_text_csv(lang_file)

    if lang_df.empty:
        return []

    lang_df["question_id"] = lang_df["question_id"].astype(str)
    lang_df = lang_df.set_index("question_id")

    # Metadata (type, autocontinue) lives in questions.csv, not in the
    # per-language file, so build a lookup to merge in.
    meta_df = questions_df.copy()
    meta_lookup: dict[str, dict] = {}
    if not meta_df.empty and "question_id" in meta_df.columns:
        meta_df["question_id"] = meta_df["question_id"].astype(str)
        meta_lookup = meta_df.set_index("question_id").to_dict(orient="index")

    survey: list[dict] = []

    for scenario in scenario_state:
        rows = scenarios_df[scenarios_df["scenario"].astype(str) == scenario]

        for _, row in rows.iterrows():
            qids      = utils.parse_list(row.get("question_id", ""))
            img_types = utils.parse_list(row.get("img_type", ""))
            img_type  = random.choice(img_types) if img_types else ""

            questions: list[dict] = []
            for qid in qids:
                if qid not in lang_df.index:
                    continue
                q = lang_df.loc[qid].to_dict()
                q["question_id"] = qid
                q["scenario"]    = scenario
                q["img_type"]    = img_type
                q["pair"]        = None
                q["info_gain"]   = 0.0

                meta = meta_lookup.get(qid, {})
                q["type"]         = meta.get("question_type", "")
                q["autocontinue"] = meta.get("autocontinue", "")

                q["options"]      = utils.parse_list(q.get("options", ""))
                q["user_options"] = utils.parse_list(q.get("user_options", ""))

                questions.append(q)

            survey.append({
                "scenario":  scenario,
                "img_type":  img_type,
                "questions": questions,
            })

    return survey


@app.post("/api/new-user")
def new_user() -> dict:
    return {"user_id": random.randint(1, 10_000_000)}


@app.post("/api/next-pair")
async def next_pair(request: Request) -> dict:
    data = await request.json()

    scenario       = str(data.get("scenario", ""))
    img_type       = str(data.get("img_type", ""))
    question_id    = str(data.get("question_id", ""))
    prev_pair_ids  = data.get("prev_pair_ids", [])
    user_id        = str(data.get("user_id", ""))
    used_img_types = data.get("used_img_types", {})
    seen_images    = data.get("seen_images", [])
    seen_pairs     = data.get("seen_pairs", [])

    if scenario not in scenario_state:
        return {"pair": None, "info_gain": 0.0, "violation_info": "unknown scenario"}

    pair, info_gain, violation = pairing.create_random_pair(
        scenario=scenario,
        img_type=img_type,
        question_id=question_id,
        scenario_state=scenario_state,
        images_df=images_df,
        config=CONFIG,
        exclude=prev_pair_ids,
        user_id=user_id,
        used_img_types=used_img_types,
        seen_images=seen_images,
        seen_pairs=seen_pairs,
    )
    print(f"[debug] scenario={scenario!r} img_type={img_type!r} question_id={question_id!r}")
    df = scenario_state.get(scenario)
    if df is not None:
        print(f"[debug] img_types in state: {df['img_type'].unique().tolist()}")
        print(f"[debug] grp size: {len(df[df['img_type'] == img_type])}")

    print(f"[debug] pair={pair} violation={violation!r}")  # TEMP DEBUG

    return {"pair": pair, "info_gain": info_gain, "violation_info": violation}


@app.post("/api/save-answer")
async def save_answer(request: Request) -> dict:
    data = await request.json()

    scenario = str(data.get("scenario", ""))
    qid      = str(data.get("question_id", ""))
    qtype    = str(data.get("type", "")).upper()

    row = {
        "user_id":     data.get("user_id", ""),
        "scenario":    scenario,
        "language":    data.get("language", ""),
        "question_id": qid,
        "type":        qtype,
        "answer":      data.get("answer", ""),
        "img_id_A":    data.get("img_id_A", ""),
        "img_id_B":    data.get("img_id_B", ""),
        "img_type":    data.get("img_type", ""),
        "info":        data.get("violation_info", ""),
    }

    csv_path = os.path.join(USER_DATA_PATH, f"{scenario}_user_data.csv")
    pd.DataFrame([row]).to_csv(
        csv_path, mode="a", header=not os.path.exists(csv_path), index=False
    )

    if qtype == "AB" and qid in SAVE_QIDS and scenario in scenario_state:
        trueskill_utils.update_image_state(
            scenario=scenario,
            question_id=qid,
            img_id_A=str(data.get("img_id_A", "")),
            img_id_B=str(data.get("img_id_B", "")),
            winner=str(data.get("answer", "")),
            scenario_state=scenario_state,
            user_data_path=USER_DATA_PATH,
            config=CONFIG,
        )

    return {"success": True}

@app.get("/api/debug-pair")
def debug_pair():
    scenario = scenario_list[0]
    df = scenario_state[scenario]
    return {
        "scenario": scenario,
        "total_images": len(df),
        "img_types": df["img_type"].unique().tolist() if "img_type" in df.columns else [],
        "columns": df.columns.tolist(),
        "sample": df.head(3).to_dict(orient="records"),
    }

# =====================================================================
# STATIC MOUNTS  — registered LAST so API routes take priority
# =====================================================================

# Image folders: /images/<idx> → parent directory of images_path[idx]
for _idx, _csv_path in enumerate(_to_path_list(CONFIG["images_path"])):
    _parent = os.path.dirname(os.path.abspath(_csv_path))
    _route  = f"/images/{_idx}"
    app.mount(_route, StaticFiles(directory=_parent), name=f"images_{_idx}")
    print(f"[server] Mounted {_parent!r}  →  {_route}")

# Frontend static assets
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(ROOT, "frontend")),
    name="static",
)