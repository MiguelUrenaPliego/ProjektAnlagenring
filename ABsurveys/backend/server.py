"""
=======================================================================
server.py — Survey Backend (MongoDB edition)
=======================================================================

Image folder mounting
---------------------
Each entry in config["images_path"] is an images.csv whose "path" column
contains paths relative to that CSV's parent directory.

    images_path = "/home/miguel/.../images/images.csv"
    parent      = "/home/miguel/.../images/"
    row path    = "Anlagenring/1.jpg"
    served at   = "/images/0/Anlagenring/1.jpg"

MongoDB
-------
Set MONGODB_URI in the environment before starting (defaults to localhost).
Set MONGODB_DB to override the database name (default: "survey_app").

    export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net/"
    export MONGODB_DB="survey_app"

Static mounts must be registered AFTER all @app route decorators.
=======================================================================
"""

from __future__ import annotations

import os
import json
import random
from contextlib import asynccontextmanager
from typing import Dict, List

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import trueskill_utils
from . import pairing
from . import utils
from . import db as _db

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with open(os.path.join(ROOT, "config.json"), "r") as f:
    CONFIG: dict = json.load(f)

USER_DATA_PATH = os.path.join(ROOT, CONFIG["user_data_path"])
os.makedirs(USER_DATA_PATH, exist_ok=True)


# =====================================================================
# HELPERS
# =====================================================================

def _to_path_list(value: str | list) -> list[str]:
    entries: list[str] = value if isinstance(value, list) else [value]
    result: list[str] = []
    for p in entries:
        p = str(p).strip()
        result.append(p if os.path.isabs(p) else os.path.join(ROOT, p))
    return result


def _load_csvs(value: str | list, text: bool = False, drop_duplicates: bool = True) -> pd.DataFrame:
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


def _load_images_df(images_path_value: str | list) -> pd.DataFrame:
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
            rel = str(p).strip().lstrip("/\\").replace("\\", "/")
            return f"/images/{i}/{rel}"

        df["_abs_path"]       = df["path"].apply(_abs)
        df["_serve_path"]     = df["path"].apply(_url)
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
# LOAD STATIC DATASETS (at import time — these are read-only CSVs)
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
# SCENARIO LIST
# =====================================================================

scenario_list: List[str] = (
    CONFIG["scenario"]
    if isinstance(CONFIG["scenario"], list)
    else [CONFIG["scenario"]]
)

# In-memory state (populated during lifespan startup)
scenario_state: Dict[str, pd.DataFrame] = {}


# =====================================================================
# LIFESPAN — async startup/shutdown (replaces @app.on_event)
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Seed MongoDB state and load in-memory scenario_state on startup."""
    col = _db.image_state_col()

    for sc in scenario_list:
        df = await trueskill_utils.ensure_scenario_state(
            scenario=sc,
            images_df=images_df,
            save_qids=SAVE_QIDS,
            config=CONFIG,
            default_score=trueskill_utils.DEFAULT_SCORE,
            default_uncertainty=trueskill_utils.DEFAULT_UNCERTAINTY,
            col=col,
        )
        scenario_state[sc] = df

        img_types = df["img_type"].unique().tolist() if "img_type" in df.columns else []
        _sp = df.get("_serve_path", pd.Series(dtype=str))
        print(f"[server] scenario={sc!r}  img_types={img_types}  images={len(df)}"
              f"  sample _serve_path={_sp.iloc[0] if not _sp.empty else 'EMPTY'}")

    yield  # server runs here

    # Shutdown: nothing to clean up (motor handles connection pool)


# =====================================================================
# APP
# =====================================================================

app = FastAPI(title="Survey App", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# API ENDPOINTS
# =====================================================================

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

    scenario      = str(data.get("scenario", ""))
    img_type      = str(data.get("img_type", ""))
    question_id   = str(data.get("question_id", ""))
    prev_pair_ids = data.get("prev_pair_ids", [])
    user_id       = str(data.get("user_id", ""))
    seen_images   = data.get("seen_images", [])
    seen_pairs    = data.get("seen_pairs", [])

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
        used_img_types={},
        seen_images=seen_images,
        seen_pairs=seen_pairs,
    )

    return {"pair": pair, "info_gain": info_gain, "violation_info": violation}


@app.post("/api/save-answer")
async def save_answer(request: Request) -> dict:
    data = await request.json()

    scenario = str(data.get("scenario", ""))
    qid      = str(data.get("question_id", ""))
    qtype    = str(data.get("type", "")).upper()
    save_ans = str(data.get("save_answer", "true")).lower() != "false"

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

    if save_ans:
        await _db.user_answers_col().insert_one(row)

    if qtype == "AB" and qid in SAVE_QIDS and scenario in scenario_state and save_ans:
        await trueskill_utils.update_image_state(
            scenario=scenario,
            question_id=qid,
            img_id_A=str(data.get("img_id_A", "")),
            img_id_B=str(data.get("img_id_B", "")),
            winner=str(data.get("answer", "")),
            scenario_state=scenario_state,
            config=CONFIG,
            col=_db.image_state_col(),
        )

    return {"success": True}


@app.get("/api/images-debug")
def images_debug() -> list[dict]:
    if images_df.empty:
        return [{"error": "images_df is empty"}]
    cols = ["img_id", "path", "_abs_path", "_serve_path"]
    cols = [c for c in cols if c in images_df.columns]
    rows = images_df[cols].head(20).copy()
    rows["_file_exists"] = rows["_abs_path"].apply(os.path.exists)
    return rows.to_dict(orient="records")


# =====================================================================
# STATIC MOUNTS — registered LAST
# =====================================================================

for _idx, _csv_path in enumerate(_to_path_list(CONFIG["images_path"])):
    _parent = os.path.dirname(os.path.abspath(_csv_path))
    _route  = f"/images/{_idx}"
    app.mount(_route, StaticFiles(directory=_parent), name=f"images_{_idx}")
    print(f"[server] Mounted {_parent!r}  →  {_route}")

FRONTEND_DIST = os.path.join(ROOT, "frontend", "dist")

app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

from fastapi.responses import FileResponse

@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))