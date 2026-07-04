# main.py
# coding: utf-8

"""End-to-end pipeline to fine-tune and score street perception models with custom losses and dual uncertainty."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import torch
from huggingface_hub import login

import tokens

# ============================================================================
# CONFIG — edit freely, nothing else needs to change
# ============================================================================

# --- Paths: Human survey data -----------------------------------------------
HUMAN_DF_PATHS: Union[str, list[str]] = (
    "/home/miguel/Documents/UNI/Master/2/ProjektVerkehr/"
    "ABsurveys/user_data/Anlagenring_user_data_merged.csv"
)

# --- Paths: Image data (train/val/test splits) ----------------------------
IMG_TRAIN_PATHS: Union[str, list[str]] = (
    "/home/miguel/Documents/UNI/Master/2/ProjektVerkehr/"
    "ABsurveys/images/Anlagenring/images.csv"
)

IMG_VAL_PATHS: Union[str, list[str], int] = 30   # 30 % of labeled train images → val
IMG_TEST_PATHS: Union[str, list[str], int] = 100  # 100 = all, keep train intact

# --- Metrics to train -------------------------------------------------------
METRICS: list[str] = [
    "walk","bike","stay"
]

# question_id filter(s) applied to human_df
QUESTION_IDS: list | str | None = [
    "walk-preference","bike-preference","stay-preference"
]

# img_type filter(s) applied to human_df (train) and img_df (inference)
IMG_TYPES: list | str | None = [
    "walk","bike","stay"
]

# scenario filter(s) applied to human_df (train) and img_df (inference)
SCENARIOS: list | str | None = [
    "Anlagenring","Anlagenring","Anlagenring"
]

# --- Starting checkpoints ---------------------------------------------------
FROM_CHECKPOINTS: list[str | None] | str | None = [
    "safety","safety","safety"
]

# --- Resume / skip behaviour ------------------------------------------------
RESUME_TRAINING: bool = False

# --- Baseline comparison (original_{metric} column) -------------------------
SCORE_ORIGINAL_CHECKPOINT: bool = False

# --- Output directories -----------------------------------------------------
MODEL_FOLDER: str = "models/FrankfurtAnlagenring"
PRETRAINED_MODEL_DIR: str = "models/default_models"

# --- Model initialisation ---------------------------------------------------
VIT_WEIGHTS: bool = True
FREEZE_VIT: bool = True

# --- Training hyperparameters -----------------------------------------------
EPOCHS: int = 8
BATCH_SIZE: int = 16
LEARNING_RATE: float = 5e-5

# --- Custom Loss and Accuracy optimization functions ------------------------
# LOSS can be:
#   • "pair"          → Optimize model with pairwise ranking loss directly
#   • "crossentropy"  → Optimize model with standard single-image classification loss
#   • "mixed"         → Optimize model with mixed loss (50% cross-entropy, 50% pair loss)
# ACCURACY can be:
#   • "pair"          → Track pairwise ranking accuracy and save checkpoint on improvement
#   • "single"        → Track single-image classification accuracy and save on improvement
LOSS: str = "crossentropy"
ACCURACY: str = "single"

NUM_WORKERS: int = 4

# --- Early stopping ---------------------------------------------------------
# Set either to None to disable early stopping entirely.
# Early stopping is also automatically disabled when SAVE_LAST_EPOCH = True.
EARLY_STOPPING_PATIENCE: int | None = None
EARLY_STOPPING_MIN_DELTA: float | None = None

# --- Checkpoint saving ------------------------------------------------------
# When True, the .pth file is always overwritten with the model from the most
# recent epoch, regardless of whether validation accuracy improved.
# The history CSV still marks checkpoint=True only on genuine accuracy gains;
# this flag only controls which weights end up on disk at the end of training.
# Set to True when you want the freshest weights (e.g. fixed-epoch fine-tuning).
# Set to False (default) to keep the best-accuracy checkpoint on disk.
SAVE_LAST_EPOCH: bool = True

# --- MC-Dropout uncertainty — INFERENCE --------------------------------------
INFERENCE_MC_PASSES: int = 20

# --- Device -----------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Inference column -------------------------------------------------------
IMAGE_COLUMN: str = "abs_path"

# ============================================================================
# Authenticate with Hugging Face
# ============================================================================
try:
    login(token=tokens.token)
except Exception as e:
    print(f"Hugging Face login skipped or failed: {e}")


# ============================================================================
# Internal helpers
# ============================================================================

def _as_list_or_none(value) -> list | None:
    """Wrap a scalar in a list; pass lists and None through unchanged."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _broadcast_to_metrics(value, metrics: list[str]) -> list:
    """Broadcast a scalar or list to match the length of metrics."""
    if value is None:
        return [None] * len(metrics)

    if not isinstance(value, list):
        return [value] * len(metrics)

    if len(value) == len(metrics):
        return [
            [v] if not isinstance(v, list) else v
            for v in value
        ]

    return [[v] if not isinstance(v, list) else v for v in value] * len(metrics)


def _filter_df(df: pd.DataFrame, column: str, values: list | None) -> pd.DataFrame:
    """Keep only rows where df[column] is in values."""
    if values is None or not values:
        return df
    flat = []
    for v in values:
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)
    if not flat:
        return df
    return df[df[column].isin(flat)]


def _load_image_dataframes(paths: Union[str, list[str], int]) -> pd.DataFrame:
    """Load image files from one or more paths and concatenate them."""
    if isinstance(paths, (int, float)):
        return None

    if isinstance(paths, str):
        paths = [paths]

    dfs = []
    for path in paths:
        if isinstance(path, float):
            continue
        path = str(path)
        base_dir = str(Path(path).parent)
        print(f"  Loading {path} …")
        try:
            if any(path.endswith(ext) for ext in ['.geojson', '.shp', '.gpkg', '.geoparquet']):
                df = gpd.read_file(path)
            elif path.endswith('.json'):
                df = pd.read_json(path)
            else:
                df = pd.read_csv(path)
            df["_base_dir"] = base_dir
            dfs.append(df)
        except Exception as e:
            print(f"    Warning: Failed to load {path}: {e}")

    if not dfs:
        raise ValueError(f"No valid image dataframes loaded from {paths}")

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Total images loaded: {len(result):,}")
    return result


def _load_human_dataframes(paths: Union[str, list[str]]) -> pd.DataFrame:
    """Load human survey data from one or more CSV/JSON files and concatenate."""
    if isinstance(paths, str):
        paths = [paths]

    dfs = []
    for path in paths:
        path = str(path)
        print(f"  Loading {path} …")
        try:
            if path.endswith('.json'):
                dfs.append(pd.read_json(path))
            else:
                dfs.append(pd.read_csv(path))
        except Exception as e:
            print(f"    Warning: Failed to load {path}: {e}")

    if not dfs:
        raise ValueError(f"No valid human dataframes loaded from {paths}")

    result = pd.concat(dfs, ignore_index=True)
    print(f"  Total AB rows loaded: {len(result):,}")
    return result



def build_image_gdf(img_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert image DataFrame to GeoDataFrame with absolute paths."""
    if "path" not in img_df.columns:
        raise ValueError("img_df must have a 'path' column")

    result = img_df.copy()
    result["img_id"] = result["img_id"].astype(str)

    def _resolve(row) -> str:
        p = str(row["path"])
        if os.path.isabs(p):
            return p
        base = str(row["_base_dir"]) if "_base_dir" in row.index else ""
        return os.path.join(base, p) if base else p

    result[IMAGE_COLUMN] = result.apply(_resolve, axis=1)

    if "geometry" in result.columns and not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result, geometry="geometry")
    elif not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result)

    return result


# ============================================================================
# Main pipeline
# ============================================================================

def main() -> None:
    """Fine-tune street perception models with custom parameters and dual uncertainty."""
    import dataset as _ds
    from inference import run
    from train import train

    print(f"Device: {DEVICE}  |  Precision: float32")
    print(f"{'='*80}")

    # ====================================================================
    # 0. Load and validate data
    # ====================================================================
    print("[0/3] Loading data …")

    print("\n  Loading human survey responses …")
    human_df_raw = _load_human_dataframes(HUMAN_DF_PATHS)

    print("\n  Loading image registry …")
    img_train_paths = IMG_TRAIN_PATHS if IMG_TRAIN_PATHS is not None else []
    img_val_pct  = IMG_VAL_PATHS  if isinstance(IMG_VAL_PATHS,  (int, float)) else None
    img_test_pct = IMG_TEST_PATHS if isinstance(IMG_TEST_PATHS, (int, float)) else None

    if img_val_pct is not None:
        img_val_pct = int(img_val_pct)
        if not (0 <= img_val_pct <= 100):
            raise ValueError(f"IMG_VAL_PATHS must be 0–100, got {img_val_pct}")
    if img_test_pct is not None:
        img_test_pct = int(img_test_pct)
        if not (0 <= img_test_pct <= 100):
            raise ValueError(f"IMG_TEST_PATHS must be 0–100, got {img_test_pct}")

    img_train_df = _load_image_dataframes(img_train_paths)
    img_val_df   = _load_image_dataframes(IMG_VAL_PATHS)   if not isinstance(IMG_VAL_PATHS,  (int, float)) else None
    img_test_df  = _load_image_dataframes(IMG_TEST_PATHS)  if not isinstance(IMG_TEST_PATHS, (int, float)) else None

    img_dfs_to_concat = [df for df in [img_train_df, img_val_df, img_test_df] if df is not None]
    if not img_dfs_to_concat:
        raise ValueError("No image dataframes loaded")

    img_df_combined = pd.concat(img_dfs_to_concat, ignore_index=True)
    img_df_combined["img_id"] = img_df_combined["img_id"].astype(str)
    print(f"  Total images across all sources: {len(img_df_combined):,}")

    # ── Image-level splitting via dataset.py ─────────────────────────────────
    # All splits are performed on unique img_ids, never on pairs, to prevent
    # data leakage.  See dataset.py for the full contract.
    print("\n  Splitting train / val / test by img_id (image-level) …")
    split_dfs = _ds.split_images_by_id(
        img_df=img_df_combined,
        human_df=human_df_raw,
        val_pct=img_val_pct,
        test_pct=img_test_pct,
        random_state=42,
    )
    img_train_final = split_dfs["train"]
    img_val_final   = split_dfs["val"]
    img_test_final  = split_dfs["test"]

    # Merge in any explicitly-provided test CSV paths
    if img_test_df is not None and len(img_test_df) > 0:
        img_test_final = pd.concat([img_test_df, img_test_final], ignore_index=True)
        img_test_final = img_test_final.drop_duplicates(subset="img_id").reset_index(drop=True)
        split_dfs["test"] = img_test_final

    print(f"  Image split: train {len(img_train_final):,}, "
          f"val {len(img_val_final):,}, test {len(img_test_final):,}")

    train_img_id_set = set(img_train_final["img_id"].astype(str).tolist())
    img_df_raw = img_test_final if img_test_pct == 100 else img_train_final

    # ====================================================================
    # 1. Training
    # ====================================================================
    print(f"\n[1/3] Training models …")

    question_ids_list = _broadcast_to_metrics(QUESTION_IDS, METRICS)
    img_types_list = _broadcast_to_metrics(IMG_TYPES, METRICS)
    scenarios_list = _broadcast_to_metrics(SCENARIOS, METRICS)
    checkpoints = _broadcast_to_metrics(FROM_CHECKPOINTS, METRICS)
    checkpoints = [c[0] for c in checkpoints]
    skipped_metrics = set()

    for metric, q_ids, img_types, from_ckpt_for_train in zip(
        METRICS, question_ids_list, img_types_list, checkpoints
    ):
        print(f"\n  Metric '{metric}':")

        model_path = os.path.join(MODEL_FOLDER, f"{metric}.pth")
        if os.path.isfile(model_path) and not RESUME_TRAINING:
            print(f"    Model exists and RESUME_TRAINING=False → skipping training")
            skipped_metrics.add(metric)
            continue

        human_df_all = human_df_raw[
            (human_df_raw["type"] == "AB")
            & (human_df_raw["answer"].isin(["A", "B", "="]))
        ].copy()

        human_df_all = _filter_df(human_df_all, "question_id", q_ids)
        human_df_all = _filter_df(human_df_all, "img_type", img_types)
        human_df_all = _filter_df(human_df_all, "scenario", scenarios_list[METRICS.index(metric)])
        human_df_all = human_df_all.reset_index(drop=True)

        print(f"    AB rows after filtering: {len(human_df_all):,}")

        if len(human_df_all) == 0:
            raise ValueError(
                f"No AB rows found for metric '{metric}' after filtering."
            )

        human_df_train, human_df_val = _ds.resolve_pairs_for_val_split(
            human_df=human_df_all,
            train_img_ids=train_img_id_set,
            val_img_ids=set(img_val_final["img_id"].astype(str)),
        )

        # ── Single-image resolution for crossentropy mode ─────────────────────
        # For crossentropy loss the split is at the image level: a cross-split
        # pair (one image in train, one in val) contributes its train-image row
        # to train and its val-image row to val.  We resolve this via
        # resolve_pairs_image_level which calls build_single_image_df and filters
        # by img_id_set, capturing cross-split pairs correctly.
        single_img_train_df = None
        single_img_val_df = None
        if LOSS == "crossentropy":
            single_img_train_df = _ds.resolve_pairs_image_level(
                human_df=human_df_all,
                img_id_set=train_img_id_set,
            )
            val_img_id_set = set(img_val_final["img_id"].astype(str))
            single_img_val_df = _ds.resolve_pairs_image_level(
                human_df=human_df_all,
                img_id_set=val_img_id_set,
            ) if val_img_id_set else None

        print(f"    Train AB pairs: {len(human_df_train):,}  |  Val AB pairs: {len(human_df_val):,}")
        if LOSS == "crossentropy":
            print(f"    Train single-img rows: {len(single_img_train_df) if single_img_train_df is not None else 0:,}  |  "
                  f"Val single-img rows: {len(single_img_val_df) if single_img_val_df is not None else 0:,}")

        if len(human_df_train) == 0 and (single_img_train_df is None or len(single_img_train_df) == 0):
            raise ValueError(
                f"No AB training pairs remain for metric '{metric}' after val split."
            )

        # Determine which image rows to include in training/val img_df.
        # For crossentropy mode: all images that appear in the single-image train/val df.
        # For pair/mixed mode: only images that appear in both-sides-in-split pairs.
        if LOSS == "crossentropy" and single_img_train_df is not None and len(single_img_train_df) > 0:
            train_valid_ids = single_img_train_df["img_id"].astype(str).unique().tolist()
        else:
            train_valid_ids = (
                pd.concat([human_df_train["img_id_A"], human_df_train["img_id_B"]])
                .astype(str)
                .unique()
                .tolist()
            )
        img_df_train = img_train_final[
            img_train_final["img_id"].isin(train_valid_ids)
        ].reset_index(drop=True)
        print(f"    Images used for training: {len(img_df_train):,}")

        val_img_df_metric = None
        if LOSS == "crossentropy" and single_img_val_df is not None and len(single_img_val_df) > 0:
            val_valid_ids = single_img_val_df["img_id"].astype(str).unique().tolist()
            val_img_df_metric = img_val_final[
                img_val_final["img_id"].isin(val_valid_ids)
            ].reset_index(drop=True)
            if val_img_df_metric.empty:
                val_img_df_metric = img_train_final[
                    img_train_final["img_id"].isin(val_valid_ids)
                ].reset_index(drop=True)
        elif len(human_df_val) > 0:
            val_valid_ids = (
                pd.concat([human_df_val["img_id_A"], human_df_val["img_id_B"]])
                .astype(str)
                .unique()
                .tolist()
            )
            # Use val split images; fall back to train if val_pct was 0
            val_img_df_metric = img_val_final[
                img_val_final["img_id"].isin(val_valid_ids)
            ].reset_index(drop=True)
            if val_img_df_metric.empty:
                val_img_df_metric = img_train_final[
                    img_train_final["img_id"].isin(val_valid_ids)
                ].reset_index(drop=True)

        saved_path = train(
            human_df=human_df_train,
            img_df=img_df_train,
            metric=metric,
            model_folder=MODEL_FOLDER,
            val_human_df=human_df_val if len(human_df_val) > 0 else None,
            val_img_df=val_img_df_metric,
            single_img_train_df=single_img_train_df,
            single_img_val_df=single_img_val_df,
            from_checkpoint=from_ckpt_for_train,
            vit_weights=VIT_WEIGHTS,
            freeze_vit=FREEZE_VIT,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            num_workers=NUM_WORKERS,
            device=DEVICE,
            pretrained_model_dir=PRETRAINED_MODEL_DIR,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
            # Custom Loss/Accuracy Hyperparams passed to train
            loss_hyperparam=LOSS,
            accuracy_hyperparam=ACCURACY,
            save_last_epoch=SAVE_LAST_EPOCH,
        )
        print(f"    ✓ Saved → {saved_path}")

        # ── Score original baseline checkpoint if requested ──
        if SCORE_ORIGINAL_CHECKPOINT and from_ckpt_for_train is not None:
            orig_sidecar = os.path.join(MODEL_FOLDER, f"{metric}_original_scores.csv")
            infer_gdf_for_orig = build_image_gdf(img_df_raw).copy()
            infer_gdf_for_orig = _filter_df(infer_gdf_for_orig, "img_type",
                                            img_types_list[METRICS.index(metric)])
            infer_gdf_for_orig = _filter_df(infer_gdf_for_orig, "scenario",
                                            scenarios_list[METRICS.index(metric)])

            already_orig_ids: set[str] = set()
            if os.path.isfile(orig_sidecar):
                _existing_orig = pd.read_csv(orig_sidecar)
                orig_col_name = f"original_{metric}"
                if orig_col_name in _existing_orig.columns:
                    already_orig_ids = set(
                        _existing_orig.loc[
                            pd.to_numeric(_existing_orig[orig_col_name],
                                          errors="coerce").notna(),
                            "img_id",
                        ].astype(str)
                    )

            need_orig_gdf = infer_gdf_for_orig[
                ~infer_gdf_for_orig["img_id"].astype(str).isin(already_orig_ids)
            ]
            print(
                f"\n  Scoring pretrained '{from_ckpt_for_train}' checkpoint "
                f"(original_{metric}) from {PRETRAINED_MODEL_DIR} — "
                f"({len(need_orig_gdf):,} images) → {orig_sidecar}"
            )

            if len(need_orig_gdf) > 0:
                _orig_scored = run(
                    gdf=need_orig_gdf,
                    metrics=[from_ckpt_for_train],
                    model_dir=PRETRAINED_MODEL_DIR,
                    image_column=IMAGE_COLUMN,
                    device=DEVICE,
                    download_missing_models=True,
                    mc_passes=INFERENCE_MC_PASSES,
                )
                orig_col_name = f"original_{metric}"
                orig_ent_col  = f"entropy_original_{metric}"
                orig_mc_col   = f"uncertainty_original_mc_{metric}"
                
                src_ent_col   = f"entropy_{from_ckpt_for_train}"
                src_mc_col    = f"uncertainty_mc_{from_ckpt_for_train}"

                new_orig_rows = _orig_scored[["img_id"]].copy()
                new_orig_rows[orig_col_name] = _orig_scored[from_ckpt_for_train].values
                new_orig_rows[orig_ent_col] = _orig_scored[src_ent_col].values
                if INFERENCE_MC_PASSES > 1 and src_mc_col in _orig_scored.columns:
                    new_orig_rows[orig_mc_col] = _orig_scored[src_mc_col].values

                if already_orig_ids and os.path.isfile(orig_sidecar):
                    _old = pd.read_csv(orig_sidecar)
                    new_orig_rows = pd.concat([_old, new_orig_rows], ignore_index=True)

                Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
                new_orig_rows.to_csv(orig_sidecar, index=False)
                print(f"  Original checkpoint scores saved → {orig_sidecar}")
            else:
                print(f"  All images already have original scores — skipping.")

    # ====================================================================
    # 2. Inference
    # ====================================================================
    print(f"\n[2/3] Running inference …")

    full_image_gdf = build_image_gdf(img_df_raw)

    out_csv = os.path.join(MODEL_FOLDER, "scores.csv")
    existing_scores = None
    if os.path.isfile(out_csv):
        existing_scores = pd.read_csv(out_csv)
        print(f"  Found existing scores ({len(existing_scores):,} rows) — upserting …")

    run_result: gpd.GeoDataFrame | None = None

    for metric, img_types, scenarios, from_ckpt in zip(
        METRICS, img_types_list, scenarios_list, checkpoints
    ):
        was_skipped = metric in skipped_metrics

        print(f"\n  Metric '{metric}' (skipped_training={was_skipped}): filtering inference images …")

        infer_gdf = full_image_gdf.copy()
        infer_gdf = _filter_df(infer_gdf, "img_type", img_types)
        infer_gdf = _filter_df(infer_gdf, "scenario", scenarios)
        infer_gdf["img_id"] = infer_gdf["img_id"].astype(str)
        print(f"    Total images in scope: {len(infer_gdf):,}")

        # Determine which images still need new inference scores
        if existing_scores is not None and metric in existing_scores.columns:
            already_scored_ids = set(
                existing_scores.loc[
                    existing_scores[metric].notna(), "img_id"
                ].astype(str)
            )
            new_infer_gdf = infer_gdf[~infer_gdf["img_id"].isin(already_scored_ids)]
            print(f"    Already scored (metric): {len(already_scored_ids):,}  →  need to score: {len(new_infer_gdf):,}")
        else:
            new_infer_gdf = infer_gdf
            already_scored_ids = set()

        # Score new images with the trained model
        if len(new_infer_gdf) > 0:
            scored = run(
                gdf=new_infer_gdf,
                metrics=[metric],
                model_dir=MODEL_FOLDER,
                image_column=IMAGE_COLUMN,
                device=DEVICE,
                download_missing_models=False,
                mc_passes=INFERENCE_MC_PASSES,
            )
        else:
            print(f"    All images already scored for '{metric}' — skipping model inference.")
            scored = new_infer_gdf.copy()
            scored[metric] = pd.Series(dtype=float)

        # Merge original_{metric} sidecar values
        if SCORE_ORIGINAL_CHECKPOINT and not was_skipped and from_ckpt is not None:
            orig_sidecar = os.path.join(MODEL_FOLDER, f"{metric}_original_scores.csv")
            orig_col = f"original_{metric}"
            orig_ent_col = f"entropy_original_{metric}"
            orig_mc_col = f"uncertainty_original_mc_{metric}"

            if os.path.isfile(orig_sidecar):
                sidecar_df = pd.read_csv(orig_sidecar)
                sidecar_df["img_id"] = sidecar_df["img_id"].astype(str)

                keep_cols = ["img_id", orig_col, orig_ent_col]
                if orig_mc_col in sidecar_df.columns:
                    keep_cols.append(orig_mc_col)
                sidecar_df = sidecar_df[[c for c in keep_cols if c in sidecar_df.columns]]

                scored = scored.merge(sidecar_df, on="img_id", how="left")
                n_filled = scored[orig_col].notna().sum() if orig_col in scored.columns else 0
                print(f"    Merged {orig_col} from sidecar: {n_filled:,}/{len(scored):,} images.")
            else:
                print(f"    ⚠️  Sidecar not found: {orig_sidecar}")

        # Accumulate scores
        if run_result is None:
            run_result = scored
        else:
            new_cols = [c for c in scored.columns if c not in run_result.columns]
            run_result = run_result.merge(
                scored[["img_id"] + new_cols],
                on="img_id",
                how="outer",
            )

    if run_result is None:
        print("No inference was run — nothing to save.")
        return

    # ====================================================================
    # 3. Save scores
    # ====================================================================
    all_score_cols = METRICS + [
        f"original_{m}"
        for m, ckpt in zip(METRICS, checkpoints)
        if ckpt is not None and m not in skipped_metrics
    ]
    
    # Add aleatoric entropy uncertainties (always calculated)
    all_score_cols += [f"entropy_{m}" for m in METRICS]
    all_score_cols += [
        f"entropy_original_{m}"
        for m, ckpt in zip(METRICS, checkpoints)
        if ckpt is not None and m not in skipped_metrics
    ]

    # Add epistemic MC uncertainties if enabled
    if INFERENCE_MC_PASSES > 1:
        all_score_cols += [f"uncertainty_mc_{m}" for m in METRICS]
        all_score_cols += [
            f"uncertainty_original_mc_{m}"
            for m, ckpt in zip(METRICS, checkpoints)
            if ckpt is not None and m not in skipped_metrics
        ]

    all_score_cols = [c for c in all_score_cols if c in run_result.columns]

    final_scores = run_result.copy()
    final_scores = final_scores[["img_id"] + [c for c in final_scores.columns if c != "img_id"]]

    # Upsert into existing scores if needed
    if existing_scores is not None:
        existing_ids = set(existing_scores["img_id"].astype(str))
        new_ids = set(final_scores["img_id"].astype(str))
        to_keep = existing_scores[~existing_scores["img_id"].astype(str).isin(new_ids)]
        final_scores = pd.concat([to_keep, final_scores], ignore_index=True)

    # ── Stamp the 'split' column ─────────────────────────────────────────────
    # For every image we record which dataset split it came from.
    # When test_pct=100, images from train are still labelled "test" in this
    # column so the reader of scores.csv knows what "all images" means.
    final_scores = _ds.assign_split_column(final_scores, split_dfs)

    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    final_scores.to_csv(out_csv, index=False)

    display_cols = ["img_id", "split", IMAGE_COLUMN] + [
        c for c in all_score_cols if c in final_scores.columns
    ]
    display_cols = [c for c in display_cols if c in final_scores.columns]

    print(f"\nScores saved → {out_csv}  ({len(final_scores):,} total rows)")
    print(final_scores[display_cols].head(10).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()