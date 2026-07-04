import asyncio
import pandas as pd
from collections import defaultdict

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import trueskill_utils

# =========================================================
# CONFIG
# =========================================================
SCENARIO = "Anlagenring"

USER_CSVS = [
    "/home/miguel/Documents/UNI/Master/2/ProjektVerkehr/ABsurveys/user_data/Anlagenring_user_data_local.csv",
    "/home/miguel/Documents/UNI/Master/2/ProjektVerkehr/ABsurveys/user_data/user_answers.csv",
]

IMAGE_JSON = "/home/miguel/Documents/UNI/Master/2/ProjektVerkehr/ABsurveys/user_data/image_state.json"


# =========================================================
# Load image universe
# =========================================================
def load_images(json_path):
    import json
    with open(json_path, "r") as f:
        data = json.load(f)

    return pd.DataFrame(data)


# =========================================================
# Build initial scenario state (IMPORTANT)
# =========================================================
def init_scenario_state(images_df: pd.DataFrame):
    df = images_df.copy()

    qids = [
        "start",
        "walk-preference",
        "bike-preference",
        "general-preference",
        "stay-preference",
    ]

    for qid in qids:
        df[f"score_{qid}"] = trueskill_utils.DEFAULT_SCORE
        df[f"uncertainty_{qid}"] = trueskill_utils.DEFAULT_UNCERTAINTY
        df[f"n_answers_{qid}"] = 0
        df[f"active_batch_{qid}"] = 1

    return {SCENARIO: df}


# =========================================================
# Parse AB interactions from user logs
# =========================================================
def extract_interactions(csv_paths):
    frames = []

    for path in csv_paths:
        df = pd.read_csv(path)
        df = df[df["type"] == "AB"]
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    return merged


# =========================================================
# Dummy Mongo collection (no persistence)
# =========================================================
class DummyCollection:
    async def update_one(self, *args, **kwargs):
        return None

    async def update_many(self, *args, **kwargs):
        return None


# =========================================================
# MAIN RECOMPUTE LOOP
# =========================================================
async def recompute():
    images_df = load_images(IMAGE_JSON)

    scenario_state = init_scenario_state(images_df)

    interactions_df = extract_interactions(USER_CSVS)

    dummy_col = DummyCollection()

    config = {
        "uncertainty_threshold": 0.25
    }

    for i, row in interactions_df.iterrows():

        await trueskill_utils.update_image_state(
            scenario=SCENARIO,
            question_id=row["question_id"],
            img_id_A=row["img_id_A"],
            img_id_B=row["img_id_B"],
            winner=row["answer"],
            scenario_state=scenario_state,
            config=config,
            col=dummy_col,
        )

        if i % 20 == 0:
            print(f"Processed {i}/{len(interactions_df)} comparisons")

    return scenario_state[SCENARIO]


# =========================================================
# EXPORT IMAGE STATE
# =========================================================
def export(df: pd.DataFrame):
    df.to_csv(
        "/home/miguel/Documents/UNI/Master/2/ProjektVerkehr/ABsurveys/user_data/Anlagenring_user_images_merged.csv",
        index=False
    )


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    result_df = asyncio.run(recompute())

    pd.concat([pd.read_csv(p) for p in USER_CSVS], ignore_index=True)\
        .to_csv(
            "/home/miguel/Documents/UNI/Master/2/ProjektVerkehr/ABsurveys/user_data/Anlagenring_user_data_merged.csv",
            index=False
        )

    print("Saved merged user data CSV")

    export(result_df)