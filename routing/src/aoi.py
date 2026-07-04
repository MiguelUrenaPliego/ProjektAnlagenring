from pathlib import Path
import sys
import os

# Add this file's directory (src/) to Python path so sibling modules are
# importable. Data paths are derived from the project root (this file's
# parent's parent), since data/ lives alongside main.py, not inside src/.
parent_dir = Path(__file__).resolve().parent
project_root = parent_dir.parent
sys.path.append(str(parent_dir))

import geopandas as gpd
import numpy as np
import pandas as pd

import population
import raster_utils

# ============================================================
# PATHS  (all absolute, derived from the project root)
# ============================================================

data_dir       = project_root / "data/region_boundaries"
aoi_dir        = project_root / "data"          # aoi saved alongside other data
population_dir = project_root / "population"
aoi_path       = aoi_dir / "aoi.gpkg"
map_path       = project_root / "region_map.html"

os.makedirs(aoi_dir, exist_ok=True)
os.makedirs(population_dir, exist_ok=True)

# ============================================================
# LOAD SOURCE LAYERS
# ============================================================

landkreise   = gpd.read_file(data_dir / "Landkreise.gpkg")
gemeinden    = gpd.read_file(data_dir / "Gemeinden.gpkg")
metroregionen = gpd.read_file(data_dir / "Metropolregionen.geojson")
stadtteile   = gpd.read_file(data_dir / "Gemarkungen.gpkg")

# ============================================================
# PROJECT TO UTM  (use Frankfurt metro as reference)
# ============================================================

f_metro = (
    metroregionen[metroregionen["KMR"].str.contains("Frankfurt", na=False)]
    .reset_index(drop=True)
)
f_metro = f_metro.to_crs(f_metro.estimate_utm_crs())
utm_crs = f_metro.crs

def _clean(gdf):
    gdf = gdf.to_crs(utm_crs)
    gdf["geometry"] = gdf["geometry"].make_valid().buffer(0)
    return gdf

stadtteile = _clean(stadtteile)
gemeinden  = _clean(gemeinden)
landkreise = _clean(landkreise)

metro_union = f_metro.union_all()

stadtteile = stadtteile[stadtteile.centroid.intersects(metro_union)].reset_index(drop=True)
gemeinden  = gemeinden[gemeinden.centroid.intersects(metro_union)].reset_index(drop=True)
landkreise = landkreise[landkreise.centroid.intersects(metro_union)].reset_index(drop=True)

# ============================================================
# DEFINE CITY AND SUBURBAN AREA
# ============================================================

city = (
    landkreise[landkreise["KREIS_BZ"].str.contains("Frankfurt", na=False)]
    .reset_index(drop=True)
)
city_union = city.union_all()

suburban_area = (
    landkreise[landkreise.centroid.intersects(city_union.buffer(10_000))]
    .reset_index(drop=True)
)
suburban_union = suburban_area.union_all()

# ============================================================
# BUILD AOI PARTS
# ============================================================

# --- Stadtteile inside Frankfurt city ---
stadtteile_part = (
    stadtteile
    .loc[stadtteile.centroid.intersects(city_union), ["geometry", "GMK_BZ", "GMK_NR"]]
    .rename(columns={"GMK_BZ": "Name"})
    .copy()
)
stadtteile_part["type"] = "Stadtteil"
# "source" records which raw layer the row's geometry came from (as
# opposed to "type", which is the semantic city/suburb/outer grouping) —
# used later to know whether a row's geometry is smaller or larger than
# the workplaces data (Gemeinde-level), to decide split vs. sum when
# joining. This layer is loaded from Gemarkungen.gpkg.
stadtteile_part["source"] = "Gemarkung"

# --- Gemeinden in suburban ring (outside city) ---
gemeinden_part = (
    gemeinden
    .loc[
        gemeinden.centroid.intersects(suburban_union.difference(city_union)),
        ["geometry", "GMDE_BZ"],
    ]
    .rename(columns={"GMDE_BZ": "Name"})
    .copy()
)
gemeinden_part["type"] = "Gemeinde"
gemeinden_part["source"] = "Gemeinde"

# --- Landkreise outside suburban ring ---
landkreise_part = (
    landkreise
    .loc[~landkreise.centroid.intersects(suburban_union), ["geometry", "KREIS_BZ"]]
    .rename(columns={"KREIS_BZ": "Name"})
    .copy()
)
landkreise_part["type"] = "Landkreis"
landkreise_part["source"] = "Landkreis"

aoi = pd.concat(
    [stadtteile_part, gemeinden_part, landkreise_part],
    ignore_index=True,
)

# ============================================================
# ADD BUNDESLAND REMAINDER (Rheinland-Pfalz / Bayern)
# ============================================================

geom_raw = f_metro.difference(landkreise.union_all()).union_all()
parts = [g for g in geom_raw.geoms if g.area > 1e9]

if parts:
    metro_parts = gpd.GeoDataFrame(
        {"geometry": parts, "Name": ["Rheinland-Pfalz", "Bayern"][: len(parts)]},
        crs=utm_crs,
    )
    metro_parts["type"] = "Bundesland"
    metro_parts["source"] = "Metropolregion"
    aoi = pd.concat([aoi, metro_parts], ignore_index=True)

# ============================================================
# FINALISE AOI
# ============================================================

aoi = gpd.GeoDataFrame(aoi, geometry="geometry", crs=utm_crs)
aoi = aoi.reset_index(drop=True)

# Integer id column – stable, 0-based
aoi["id"] = aoi.index.astype(int)

# Gemarkung 500 ("Main") is the river running through Frankfurt – it must
# stay in the AOI (dropping it would leave a gap in the AOI union that
# streets.py clips the street graph to, cutting off the bridges crossing
# it) but has no population/workplaces of its own. Its computed values are
# forced to 0 and redistributed proportionally onto the other rows so
# totals are preserved (see _zero_and_redistribute below).
ZERO_MASK = (aoi["source"] == "Gemarkung") & (aoi["GMK_NR"] == "500")


def _zero_and_redistribute(values: pd.Series, zero_mask: pd.Series) -> pd.Series:
    """Force ``values`` to 0 wherever ``zero_mask`` is True, redistributing
    what was there proportionally over the remaining rows so the overall
    total is unchanged from before zeroing."""
    values = values.astype(float).copy()
    if not zero_mask.any():
        return values
    original_total = values.sum()
    values.loc[zero_mask] = 0.0
    remaining_total = values.loc[~zero_mask].sum()
    if remaining_total > 0:
        values.loc[~zero_mask] *= original_total / remaining_total
    return values


# First save (without population) so streets.py can use it if needed early
aoi.to_file(str(aoi_path), driver="GPKG")
print(f"AOI saved ({len(aoi)} polygons) → {aoi_path}")

# ============================================================
# POPULATION
# ============================================================

population_file = population.download_worldpop_population(
    aoi,
    2025,
    folder=str(population_dir),
    resolution="100m",
)

if population_file is not None:
    for i in range(len(aoi)):
        aoi_i = aoi.iloc[i : i + 1]
        pop, transform, crs = raster_utils.read_raster(
            population_file, aoi=aoi_i.to_crs(4326)
        )
        pop_gdf = raster_utils.vectorize(pop, transform, crs)
        pop_gdf = pop_gdf.to_crs(aoi.crs)
        pop_gdf = pop_gdf[
            pop_gdf.centroid.intersects(aoi_i.union_all())
        ].reset_index(drop=True)
        aoi.loc[i, "population"] = pop_gdf["value"].sum()

    aoi["population"] = _zero_and_redistribute(aoi["population"], ZERO_MASK)

    aoi.to_file(str(aoi_path), driver="GPKG")
    print(f"AOI with population saved → {aoi_path}")
else:
    print("Warning: population download returned None – population column not added.")

# ============================================================
# WORKPLACES ("Arbeitsort" jobs) from data/labor/labor.csv
#
# labor.csv is a Regionalstatistik-style export (semicolon-separated,
# multi-row header, one blank footnote row after every data row) keyed by
# an AGS-like "Schl. Nr." code: 8 digits for a Gemeinde, 5 for a
# Landkreis, both starting with the 2-digit Bundesland code ("06" =
# Hessen). Stripping that land prefix gives exactly Gemeinden.gpkg's
# GMDE_NR (6 digits) / Landkreise.gpkg's KREIS_NR (3 digits).
#
# AOI rows can be smaller (Stadtteil/Gemarkung), equal (Gemeinde), or
# larger (Landkreis/Metropolregion) than that Gemeinde-level source
# geometry, so the workplaces value has to be joined directionally:
#   - AOI row <= Gemeinde tier: its centroid falls inside exactly one
#     Gemeinde polygon; that Gemeinde's job count is split among every
#     AOI row landing in the same Gemeinde, weighted by AOI population
#     (this is also how a Gemeinde-tier AOI row gets its own 1:1 value —
#     it's the only row whose centroid falls in that Gemeinde).
#   - AOI row > Gemeinde tier: sum the job counts of every Gemeinde
#     centroid that falls inside the AOI row's own polygon.
# ============================================================

labor_path = project_root / "data/labor/labor.csv"

SOURCE_TIER = {
    "Gemarkung": 0,
    "Stadtteil": 0,
    "Flure": 0,
    "Gemeinde": 1,
    "Landkreis": 2,
    "Metropolregion": 3,
}


def _read_labor_csv(path) -> dict:
    """Parse labor.csv into {AGS-like code (digits-only str): jobs at
    workplace value ("Arbeitsort", column 11 / index 10)}."""
    import csv

    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter=";"))

    jobs_by_code = {}
    for row in rows:
        if not row:
            continue
        code = row[0].strip().strip('"')
        if not code.isdigit() or len(row) <= 10:
            continue
        raw_value = row[10].strip().strip('"')
        if not raw_value or raw_value in ("-", "*"):
            continue
        try:
            jobs_by_code[code] = float(raw_value)
        except ValueError:
            continue

    return jobs_by_code


def _split_or_sum_by_tier(
    aoi_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
    source_value_col: str,
    weight_col: str = "population",
) -> pd.Series:
    """Assign ``source_value_col`` from ``source_gdf`` (Gemeinde- or
    Landkreis-level polygons) onto every row of ``aoi_gdf``, per the
    smaller-vs-larger rule described above. Returns a Series aligned with
    ``aoi_gdf.index``; NaN where an AOI row matched no source polygon."""
    aoi_tier = aoi_gdf["source"].map(SOURCE_TIER)
    gemeinde_tier = SOURCE_TIER["Gemeinde"]

    source_polys = source_gdf[["geometry", source_value_col]].copy()
    source_polys["_source_id"] = range(len(source_polys))
    source_centroids = gpd.GeoDataFrame(
        {source_value_col: source_gdf[source_value_col].to_numpy()},
        geometry=source_gdf.geometry.centroid.to_numpy(),
        crs=source_gdf.crs,
    )

    result = pd.Series(np.nan, index=aoi_gdf.index, dtype=float)

    # --- AOI row <= Gemeinde tier: split the containing source polygon's
    # value among every AOI row sharing it, weighted by population ---
    small_mask = aoi_tier <= gemeinde_tier
    if small_mask.any():
        aoi_small = gpd.GeoDataFrame(
            {
                "_aoi_idx": aoi_gdf.index[small_mask].to_numpy(),
                weight_col: aoi_gdf.loc[small_mask, weight_col].to_numpy(),
            },
            geometry=aoi_gdf.loc[small_mask].geometry.centroid.to_numpy(),
            crs=aoi_gdf.crs,
        )
        joined = gpd.sjoin(aoi_small, source_polys, predicate="within", how="inner")
        if len(joined) > 0:
            group_weight = joined.groupby("_source_id")[weight_col].transform("sum")
            group_count = joined.groupby("_source_id")["_source_id"].transform("count")
            share = np.where(group_weight > 0, joined[weight_col] / group_weight, 1.0 / group_count)
            allocated = share * joined[source_value_col].to_numpy()
            per_aoi = pd.Series(allocated, index=joined["_aoi_idx"].to_numpy()).groupby(level=0).sum()
            result.loc[per_aoi.index] = per_aoi.to_numpy()

    # --- AOI row > Gemeinde tier: sum every source centroid inside it ---
    big_mask = aoi_tier > gemeinde_tier
    if big_mask.any():
        aoi_big = gpd.GeoDataFrame(
            {"_aoi_idx": aoi_gdf.index[big_mask].to_numpy()},
            geometry=aoi_gdf.loc[big_mask].geometry.to_numpy(),
            crs=aoi_gdf.crs,
        )
        joined = gpd.sjoin(source_centroids, aoi_big, predicate="within", how="inner")
        if len(joined) > 0:
            per_aoi = joined.groupby("_aoi_idx")[source_value_col].sum()
            result.loc[per_aoi.index] = per_aoi.to_numpy()

    return result


if labor_path.is_file():
    jobs_by_code = _read_labor_csv(labor_path)

    gem_source = gemeinden.copy()
    gem_source["workplaces"] = gem_source["GMDE_NR"].astype(str).str.zfill(6).map(
        lambda code: jobs_by_code.get("06" + code)
    )
    gem_source = gem_source[gem_source["workplaces"].notna()].reset_index(drop=True)

    aoi["workplaces"] = _split_or_sum_by_tier(aoi, gem_source, "workplaces", weight_col="population")

    # Fallback to Landkreis-level totals for AOI rows that matched no
    # Gemeinde data at all (e.g. a Gemarkung whose own Gemeinde has no
    # labor.csv row). Metropolregion (Bundesland-remainder) rows are left
    # out of this fallback — a whole-Bundesland total would grossly
    # overstate the tiny sliver of it these rows actually cover.
    missing = aoi["workplaces"].isna() & (aoi["source"] != "Metropolregion")
    if missing.any():
        kreis_source = landkreise.copy()
        kreis_source["workplaces"] = kreis_source["KREIS_NR"].astype(str).str.zfill(3).map(
            lambda code: jobs_by_code.get("06" + code)
        )
        kreis_source = kreis_source[kreis_source["workplaces"].notna()].reset_index(drop=True)

        fallback = _split_or_sum_by_tier(aoi.loc[missing], kreis_source, "workplaces", weight_col="population")
        aoi.loc[missing, "workplaces"] = fallback

    aoi["workplaces"] = aoi["workplaces"].fillna(0.0)
    aoi["workplaces"] = _zero_and_redistribute(aoi["workplaces"], ZERO_MASK)

    aoi.to_file(str(aoi_path), driver="GPKG")
    print(f"AOI with workplaces saved → {aoi_path}")
else:
    print(f"Warning: {labor_path} not found – workplaces column not added.")

# ============================================================
# REGION / CENSUS MAP (population, workplaces, and their densities)
# ============================================================

import map as map_module

region_city_union = aoi.loc[aoi["type"] == "Stadtteil"].union_all()
region_suburban_union = aoi.loc[aoi["type"] == "Gemeinde"].union_all()

map_module.build_region_map(
    aoi,
    map_path,
    city_union=region_city_union,
    suburban_union=region_suburban_union,
)