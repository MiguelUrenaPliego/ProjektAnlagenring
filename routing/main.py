from pathlib import Path
import sys
import os

# Add parent directory and src/ to Python path so sibling modules (now
# living in src/) are importable regardless of the working directory.
parent_dir = Path(__file__).resolve().parent
src_dir = parent_dir / "src"
sys.path.append(str(parent_dir))
sys.path.append(str(src_dir))

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd

import scenario
import graph as graph_utils

# ============================================================
# CONFIG
# ============================================================

node_penalty = 3            # seconds added per edge (intersection delay)
acceleration = 1.5          # m/s²
min_cruising_time = 5       # seconds
min_cruising_speed = 10     # km/h
max_stop_and_go_speed = 50  # km/h
nogo_speed = 1               # km/h applied to maxspeed_car on closed edges
affected_time_threshold_min = 5.0  # routes slower by more than this count as "affected"

maxspeeds = {
    "living_street":  30,
    "motorway":       100,
    "motorway_link":  60,
    "primary":        50,
    "primary_link":   50,
    "residential":    30,
    "secondary":      40,
    "secondary_link": 40,
    "service":        20,
    "tertiary":       40,
    "tertiary_link":  40,
    "trunk":          80,
    "trunk_link":     60,
    "unclassified":   40,
}

SCORE_TRAVEL_TIME_REDUCTION = 0.25
score = {
    "living_street":  1,
    "motorway":       10,
    "motorway_link":  9.5,
    "primary":        6,
    "primary_link":   5.5,
    "residential":    1,
    "secondary":      1,
    "secondary_link": 1,
    "service":        1,
    "tertiary":       3,
    "tertiary_link":  2.5,
    "trunk":          8,
    "trunk_link":     7.5,
    "unclassified":   1,
}

# Closure scenarios: name -> boundary gpkg path
SCENARIOS = {
    "center":       "region_boundaries/frankfurt_center.gpkg",
    "anlagenring":  "region_boundaries/frankfurt_anlagenring.gpkg",
    "bahnhof":      "region_boundaries/frankfurt_bahnhof.gpkg",
}

# Paths
data_dir            = os.path.join(parent_dir, "data")
output_dir          = os.path.join(data_dir, "routing_analysis")
aoi_path            = os.path.join(data_dir, "aoi.gpkg")
streets_graph_path  = os.path.join(data_dir, "streets/streets.graphml")

# Where the final interactive maps are written — change these to move
# them elsewhere; both default to the project root.
map_output_path     = os.path.join(parent_dir, "carfree_zones_map.html")
region_map_path     = os.path.join(parent_dir, "region_map.html")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# AOI
# ============================================================

if not os.path.isfile(aoi_path):
    print("AOI file not found – running aoi.py …")
    import aoi as aoi_module   # noqa: F401  (side-effects create the file)

aoi = gpd.read_file(aoi_path)
aoi = aoi.to_crs(aoi.estimate_utm_crs())
crs = aoi.crs

if "id" not in aoi.columns:
    aoi["id"] = aoi.index.astype(int)
else:
    aoi["id"] = aoi["id"].astype(int)

if "population" not in aoi.columns:
    aoi["population"] = 0.0
aoi["population"] = pd.to_numeric(aoi["population"], errors="coerce").fillna(0.0)

if "workplaces" not in aoi.columns:
    aoi["workplaces"] = 0.0
aoi["workplaces"] = pd.to_numeric(aoi["workplaces"], errors="coerce").fillna(0.0)

pop_lookup = dict(zip(aoi["id"], aoi["population"]))
workplace_lookup = dict(zip(aoi["id"], aoi["workplaces"]))
aoi_ids = aoi["id"].tolist()

# Weight used for every *summary*-level weighted mean (the "Overall
# results" box + scenario-comparison CSV) — population + workplaces
# combined on both ends of the pair, as a single measure of an AOI row's
# overall importance. This is deliberately different from the routes'
# weighting (population at the origin, workplaces at the destination,
# used for edge_traffic/per_origin_weighted/the route explorer) — the
# summary is asking "how important is this row overall," not "how much of
# this row's traffic goes to that destination."
summary_weight_lookup = {i: pop_lookup.get(i, 0.0) + workplace_lookup.get(i, 0.0) for i in aoi_ids}

# AOI rows built from the fine-grained source layers (Flure/Gemarkung/
# Stadtteil/Gemeinde) sit in the high-resolution "suburban ring + city"
# area; Landkreis/Metropolregion rows are the coarse outer fringe, whose
# geometry is far too big for a per-row time/distance/speed/co2 mean or
# the general summary to say anything meaningful. Every stats layer and
# the summary — except the traffic-increase layer/raster, which needs the
# whole network's real traffic — only considers routes *to* a destination
# in this fine-grained set.
FINE_SOURCES = {"Gemarkung", "Stadtteil", "Gemeinde", "Flure"}
fine_ids = (
    set(aoi.loc[aoi["source"].isin(FINE_SOURCES), "id"]) if "source" in aoi.columns else set(aoi_ids)
)


def _to_fine_destinations(od):
    return od[od["destination_id"].isin(fine_ids)]

# ============================================================
# STREET GRAPH
# ============================================================

if not os.path.isfile(streets_graph_path):
    print("Street graph not found – running streets.run() …")
    import streets as streets_module

    streets_module.run({
        "aoi_path": aoi_path,
        "streets_dir": os.path.join(data_dir, "streets"),
        "streets_graph_path": streets_graph_path,
    })

    aoi = gpd.read_file(aoi_path)
    aoi = aoi.to_crs(crs)
    if "id" not in aoi.columns:
        aoi["id"] = aoi.index.astype(int)
    else:
        aoi["id"] = aoi["id"].astype(int)

G = ox.load_graphml(streets_graph_path)
nodes, edges = ox.graph_to_gdfs(G)

if "osmid" not in aoi.columns:
    print("Snapping AOI centroids to nearest graph nodes …")
    aoi_centroids = aoi.copy()
    aoi_centroids.geometry = aoi_centroids.geometry.centroid
    aoi["osmid"] = graph_utils.nearest_nodes(aoi_centroids, nodes)

# ============================================================
# BASE EDGE SPEEDS (shared by every scenario)
# ============================================================

print("Inferring base maxspeeds …")
edges["maxspeed_car"] = graph_utils.infer_maxspeed(
    edges, maxspeeds, enforce=False, maxspeed_col="maxspeed"
)

scenario_kwargs = dict(
    acceleration=acceleration,
    min_cruising_time=min_cruising_time,
    min_cruising_speed=min_cruising_speed,
    max_stop_and_go_speed=max_stop_and_go_speed,
    node_penalty=node_penalty,
    road_score=score,
    score_travel_time_reduction=SCORE_TRAVEL_TIME_REDUCTION,
)

# Edge row order shared by every scenario's igraph (topology/geometry never
# changes between scenarios, only speed/time/co2 do), used to align
# per-scenario traffic arrays onto a single edges GeoDataFrame.
edges_reset_base = edges.reset_index()

# ============================================================
# BASELINE ("current") SCENARIO
# ============================================================

print("Computing baseline ('current') network …")
edges_current = scenario.build_scenario_edges(edges, nogo_union=None, **scenario_kwargs)
ig_current = scenario.build_igraph(nodes, edges_current)

od_current = scenario.od_matrix(ig_current, aoi, crs, desc="OD matrix: current")
traffic_current = scenario.edge_traffic(
    ig_current, aoi, pop_lookup, workplace_lookup, blocked_ids=None, desc="Traffic: current"
)

od_matrices = {"current": od_current}
traffic_by_scenario = {"current": traffic_current}
blocked_by_scenario = {"current": None}

# ============================================================
# CLOSURE SCENARIOS
# ============================================================

closure_boundaries_wgs84 = {}
closed_by_scenario = {}

for name, rel_path in SCENARIOS.items():
    print(f"Computing '{name}' closure network …")
    boundary_path = os.path.join(data_dir, rel_path)
    boundary = gpd.read_file(boundary_path).to_crs(crs)
    nogo_union = boundary.union_all()

    edges_s = scenario.build_scenario_edges(
        edges, nogo_union=nogo_union, nogo_speed=nogo_speed, **scenario_kwargs
    )
    ig_s = scenario.build_igraph(nodes, edges_s)

    blocked = scenario.blocked_ids(aoi, nodes, nogo_union)
    print(f"  {len(blocked)} AOI rows blocked by '{name}' closure.")

    # Pairs with an origin or destination inside the closure are routed
    # without the closure-avoidance penalty (see od_matrix/_weight_for_pair
    # in scenario.py) — otherwise a trip that genuinely has to start/end
    # inside a "prohibited" area would get reported as absurdly slow just
    # because the router refuses to route through its own destination.
    od_s = scenario.od_matrix(ig_s, aoi, crs, desc=f"OD matrix: {name}", blocked_ids=blocked)

    # No AOI row/road is excluded from traffic assignment either — every
    # row sends/receives traffic, and _weight_for_pair (via blocked_ids)
    # handles routing through vs. around the closure per pair.
    traffic_s = scenario.edge_traffic(
        ig_s, aoi, pop_lookup, workplace_lookup, blocked_ids=blocked, desc=f"Traffic: {name}"
    )

    od_matrices[name] = od_s
    traffic_by_scenario[name] = traffic_s
    blocked_by_scenario[name] = blocked
    # edges_s is built from the same (unreset, positionally-ordered) `edges`
    # GeoDataFrame every time, so this aligns 1:1 with edges_reset_base /
    # traffic_gdf below — used to exclude closed (blocked-off) roads from
    # the traffic-increase %Δ entirely, since their trivial "-100%, now
    # unreachable" isn't a real traffic-increase signal.
    closed_by_scenario[name] = edges_s["closed"].to_numpy()
    # The closure boundary itself (blocked-road area), reprojected once
    # here for the map to draw as a plain overlay polygon.
    closure_boundaries_wgs84[name] = boundary.to_crs(4326).union_all()

scenario_names = ["current", *SCENARIOS.keys()]

# Fine-destination-only views of the OD matrices, used for every stats
# layer + summary computation below (od_matrices itself stays unfiltered
# for the GPKG export and the map's route explorer, which should be able
# to show any route, not just the "stats" subset).
od_matrices_fine = {name: _to_fine_destinations(od) for name, od in od_matrices.items()}
od_current_fine = od_matrices_fine["current"]

# ============================================================
# TRAFFIC EDGES GPKG
# (gravity-allocated traffic per edge, one column per scenario)
# ============================================================

print("Building traffic edges GeoDataFrame …")
traffic_gdf = edges_reset_base[["u", "v", "key", "geometry", "highway", "length"]].copy()
for name, arr in traffic_by_scenario.items():
    traffic_gdf[f"traffic_{name}"] = arr

traffic_edges_path = os.path.join(output_dir, "traffic_edges.gpkg")
traffic_gdf.to_file(traffic_edges_path, driver="GPKG")
print(f"  {traffic_edges_path} ({len(traffic_gdf)} edges)")

# ============================================================
# TRAFFIC-INCREASE RASTER (per road edge)
# (rasterized instead of a vector H3 grid — embedding thousands of vector
#  cells in the map HTML was what made it slow to load; a small raster
#  PNG overlay is far cheaper to ship and render.)
# ============================================================

print("Rasterizing traffic increase per edge …")
import rasterio
from rasterio.transform import from_bounds as rio_transform_from_bounds
from rasterio.features import rasterize as rio_rasterize

import raster_utils

RASTER_MAX_DIM = 2000  # pixels along the longer side

# Edges with a near-zero baseline blow up the %Δ formula (dividing by
# almost nothing turns a tiny absolute change into a five-digit percentage)
# without representing any real congestion change, so %Δ is only computed
# for edges whose "current" traffic is above a floor derived from the
# actual distribution (5th percentile of edges that carry any traffic at
# all, with a sane absolute minimum). This feeds the raster; the map
# summary's "Max/Avg traffic increase" instead reads from busiest_road_pct
# (per-AOI-row values, computed further below).
positive_traffic_current = traffic_gdf.loc[traffic_gdf["traffic_current"] > 0, "traffic_current"]
MIN_BASELINE_TRAFFIC = (
    max(50.0, float(np.percentile(positive_traffic_current, 5)))
    if len(positive_traffic_current) > 0 else 50.0
)

for name in SCENARIOS.keys():
    pct_increase = np.where(
        traffic_gdf["traffic_current"] >= MIN_BASELINE_TRAFFIC,
        (traffic_gdf[f"traffic_{name}"] - traffic_gdf["traffic_current"])
        / traffic_gdf["traffic_current"] * 100.0,
        np.nan,
    )
    # Roads inside the closure are no longer excluded — they now carry
    # real (if reduced) traffic from pairs that genuinely start/end there
    # (see scenario._weight_for_pair), so their %Δ is a real signal too.
    # Exact -100% is still dropped (an edge losing literally all its
    # traffic, for any reason, isn't informative on this scale).
    pct_increase = np.where(pct_increase <= -100.0, np.nan, pct_increase)
    traffic_gdf[f"pct_increase_{name}"] = pct_increase

minx, miny, maxx, maxy = traffic_gdf.total_bounds
extent_w, extent_h = maxx - minx, maxy - miny
if extent_w >= extent_h:
    raster_width = RASTER_MAX_DIM
    pixel_size = extent_w / raster_width
    raster_height = max(1, round(extent_h / pixel_size))
else:
    raster_height = RASTER_MAX_DIM
    pixel_size = extent_h / raster_height
    raster_width = max(1, round(extent_w / pixel_size))

raster_transform = rio_transform_from_bounds(minx, miny, maxx, maxy, raster_width, raster_height)
raster_shape = (raster_height, raster_width)

traffic_increase_path = os.path.join(output_dir, "traffic_increase.tif")
band_names = list(SCENARIOS.keys())
bands_utm = []

for name in band_names:
    valid = traffic_gdf[traffic_gdf[f"pct_increase_{name}"].notna()]
    band = rio_rasterize(
        ((geom, val) for geom, val in zip(valid.geometry, valid[f"pct_increase_{name}"])),
        out_shape=raster_shape,
        transform=raster_transform,
        fill=np.nan,
        all_touched=True,
        dtype="float32",
    )
    bands_utm.append(band)

# Reprojected to EPSG:3857 (Web Mercator) *before* writing to disk, once,
# here — not EPSG:4326. Leaflet always renders internally in Web Mercator;
# an ImageOverlay is placed by linearly stretching the source pixel grid
# between two lat/lng corners *in that projected space*, so a pixel grid
# that is linear in plain lat/lng (EPSG:4326) instead of Web Mercator
# still comes out warped against the basemap/roads, worse the further
# from the equator. Building the raster's own grid to be linear in
# EPSG:3857 makes that stretch exact.
bands_merc = []
dst_transform_merc = None
for band in bands_utm:
    band_merc, dst_transform_merc, dst_crs_merc = raster_utils.reproject(
        band.astype(np.float64), raster_transform, crs,
        src_nodata=np.nan, dst_nodata=np.nan, dst_crs=3857,
    )
    bands_merc.append(band_merc.astype("float32"))

merc_height, merc_width = bands_merc[0].shape

# Always overwrite: remove any stale file first (rather than relying on
# "w" mode alone) so a run never leaves pixels from a previous run's grid
# mixed in if e.g. the process was interrupted mid-write before.
if os.path.exists(traffic_increase_path):
    os.remove(traffic_increase_path)

with rasterio.open(
    traffic_increase_path,
    "w",
    driver="GTiff",
    height=merc_height,
    width=merc_width,
    count=len(bands_merc),
    dtype="float32",
    crs=dst_crs_merc,
    transform=dst_transform_merc,
    nodata=np.nan,
    compress="lzw",
) as dst:
    for i, (name, band) in enumerate(zip(band_names, bands_merc), start=1):
        dst.write(band, i)
        dst.set_band_description(i, name)

print(
    f"  {os.path.relpath(traffic_increase_path, parent_dir)} "
    f"({merc_width}x{merc_height}px, EPSG:3857, bands={band_names})"
)

# ============================================================
# BUSIEST-ROAD %Δ PER AOI ROW
# (for each AOI polygon, the road with the most traffic under 'current'
#  that intersects it, and that same road's %Δ traffic in each scenario —
#  not necessarily the biggest %Δ road in the polygon. This is a cheap
#  vector computation, added as an extra AOI choropleth layer.)
# ============================================================

print("Computing busiest-road %Δ per AOI row …")
for name in SCENARIOS.keys():
    traffic_gdf[f"closed_{name}"] = closed_by_scenario[name]

edges_aoi_join = gpd.sjoin(
    traffic_gdf[
        ["geometry", "traffic_current"]
        + [f"traffic_{n}" for n in SCENARIOS.keys()]
        + [f"closed_{n}" for n in SCENARIOS.keys()]
    ],
    aoi[["id", "geometry"]],
    predicate="intersects",
    how="inner",
).reset_index(drop=True)
# sjoin repeats the same edge's (non-unique) index across every AOI polygon
# it intersects; reset_index above gives each output row a unique label so
# idxmax()+.loc[] below select exactly one row per AOI id, not a many-to-
# many expansion (which produced duplicate "id" values -> reindex failure).

busiest_road_pct = {"current": pd.Series(0.0, index=aoi_ids)}
# Each AOI row's own busiest road's baseline traffic — the weight used
# below for the summary's "Avg traffic increase" (weighted by how much
# traffic that road actually carries, not by population/workplaces).
busiest_road_traffic_current = pd.Series(0.0, index=aoi_ids)
if len(edges_aoi_join) > 0:
    busiest_idx = edges_aoi_join.groupby("id")["traffic_current"].idxmax()
    aoi_busiest = edges_aoi_join.loc[busiest_idx].set_index("id")
    busiest_road_traffic_current = aoi_busiest["traffic_current"].reindex(aoi_ids).fillna(0.0)

    for name in SCENARIOS.keys():
        pct = np.where(
            aoi_busiest["traffic_current"] > 0,
            (aoi_busiest[f"traffic_{name}"] - aoi_busiest["traffic_current"])
            / aoi_busiest["traffic_current"] * 100.0,
            np.nan,
        )
        # Same as the raster: a busiest road inside the closure is no
        # longer excluded (it can carry real traffic from pairs that
        # start/end there); exact -100% is still dropped.
        pct = np.where(pct <= -100.0, np.nan, pct)
        busiest_road_pct[name] = pd.Series(pct, index=aoi_busiest.index).reindex(aoi_ids)
else:
    for name in SCENARIOS.keys():
        busiest_road_pct[name] = pd.Series(np.nan, index=aoi_ids)

# "City" = Stadtteil rows (inside Frankfurt proper), "suburban" = Gemeinde
# rows (the ring around it) — the two finest-resolution AOI tiers, used to
# report the traffic-increase average separately for each.
aoi_type_by_id = aoi.set_index("id")["type"].reindex(aoi_ids)
city_row_mask = (aoi_type_by_id == "Stadtteil").to_numpy()
suburban_row_mask = (aoi_type_by_id == "Gemeinde").to_numpy()


def _traffic_weighted_mean(pct_series: pd.Series, row_mask: np.ndarray) -> float:
    """Weighted mean of an AOI-row %Δ series, restricted to ``row_mask``,
    weighted by each row's own busiest road's baseline traffic."""
    valid = pct_series.notna().to_numpy() & row_mask
    if not valid.any():
        return float("nan")
    weights = busiest_road_traffic_current.to_numpy()[valid]
    values = pct_series.to_numpy()[valid]
    if weights.sum() <= 0:
        return float(np.mean(values))
    return float(np.average(values, weights=weights))

# ============================================================
# SAVE OD MATRIX GPKGS
# ============================================================

print("Saving OD matrix GPKGs …")
for name, od in od_matrices.items():
    out_path = os.path.join(output_dir, f"od_{name}.gpkg")
    od_wgs84 = od.to_crs(4326)
    od_wgs84.to_file(out_path, driver="GPKG")
    print(f"  {out_path} ({len(od_wgs84)} rows)")

# ============================================================
# SCENARIO COMPARISON CSV
# (population-weighted MEAN; weight = each origin's population split among
#  destinations by destination-population share, i.e. scenario.pair_weight;
#  restricted to AOI pairs not blocked by that scenario's closure)
# ============================================================

print("Computing scenario comparison CSV …")
comparison_rows = []
per_row_affected = {"current": pd.Series(0.0, index=aoi_ids)}

# unfiltered baseline (except the fine-destination-only restriction that
# applies to every stats/summary figure), used for the "current" row of
# the map summary box
overall_stats = {
    "current": {
        **scenario.route_weighted_mean(od_current_fine, summary_weight_lookup),
        "diff_time_min": 0.0,
        "diff_distance_km": 0.0,
        "diff_avg_speed_kmh": 0.0,
        "diff_co2_kg": 0.0,
        "n_od_pairs": len(od_current_fine),
        "n_routes_affected": 0,
        "pct_routes_affected": 0.0,
        "people_affected": 0.0,
        "pct_people_affected": 0.0,
        "max_traffic_increase_pct": 0.0,
        "avg_traffic_increase_pct_city": 0.0,
        "avg_traffic_increase_pct_suburban": 0.0,
        "max_time_increase_min": 0.0,
        "max_distance_increase_km": 0.0,
        "max_speed_decrease_kmh": 0.0,
    }
}

for name in SCENARIOS.keys():
    # Routes to/from AOI rows inside this scenario's closure are no longer
    # excluded from stats — od_matrix already routes them sensibly (see
    # scenario._weight_for_pair), so their real travel time is meaningful
    # and should be included like any other pair.
    current_sub = od_current_fine
    scenario_sub = od_matrices_fine[name]

    # Population-weighted (population at origin, workplaces at destination)
    # — feeds per_row_affected, which the AOI choropleth's "% affected" layer
    # reads from. This keeps that layer's meaning as "% of this row's own
    # population affected," not a population+workplaces blend.
    impact = scenario.scenario_impact(
        current_sub, scenario_sub, pop_lookup, aoi_ids,
        dest_lookup=workplace_lookup, threshold_min=affected_time_threshold_min,
    )
    per_row_affected[name] = impact["people_affected_by_origin"]
    n_routes_affected = impact["n_routes_affected"]

    # Summary-only figures: population+workplaces combined weight (see
    # summary_weight_lookup above) for the "Overall results" box and the
    # scenario-comparison CSV, kept separate from the per-row layer above.
    current_avg = scenario.route_weighted_mean(current_sub, summary_weight_lookup)
    scenario_avg = scenario.route_weighted_mean(scenario_sub, summary_weight_lookup)
    summary_impact = scenario.scenario_impact(
        current_sub, scenario_sub, summary_weight_lookup, aoi_ids,
        threshold_min=affected_time_threshold_min,
    )
    people_affected = summary_impact["people_affected"]
    total_population = summary_impact["total_population"]
    pct_routes_affected = (
        n_routes_affected / len(scenario_sub) * 100.0 if len(scenario_sub) > 0 else float("nan")
    )
    pct_people_affected = (
        people_affected / total_population * 100.0 if total_population > 0 else float("nan")
    )

    row = {
        "scenario": name,
        "n_aoi_in_closure": len(blocked_by_scenario[name]),
        "n_od_pairs_current": len(current_sub),
        "n_od_pairs_scenario": len(scenario_sub),
        "n_routes_affected": n_routes_affected,
        "pct_routes_affected": pct_routes_affected,
        "people_affected": people_affected,
        "pct_people_affected": pct_people_affected,
    }

    for metric in scenario.METRIC_COLS:
        cur_v = current_avg[metric]
        scn_v = scenario_avg[metric]
        diff = scn_v - cur_v
        diff_pct = (diff / cur_v * 100.0) if cur_v not in (0, None) and cur_v == cur_v else float("nan")

        row[f"current_{metric}"] = cur_v
        row[f"scenario_{metric}"] = scn_v
        row[f"diff_{metric}"] = diff
        row[f"diff_pct_{metric}"] = diff_pct

    comparison_rows.append(row)

    # Both "Max" and "Avg" traffic increase in the summary are computed
    # over AOI rows' own busiest-road %Δ (busiest_road_pct, already
    # computed above) — i.e. numbers actually attributable to, and
    # look-up-able on, a specific AOI row — not over all ~90k edges (most
    # of which aren't any AOI row's busiest road).
    busiest_pct_this_scenario = busiest_road_pct[name]
    valid_busiest = busiest_pct_this_scenario.notna()
    if valid_busiest.any():
        # 99th percentile (not the true max) over AOI rows' busiest-road
        # %Δ — a single row's outlier shouldn't stand in as "the" max.
        max_traffic_increase_pct = float(np.percentile(busiest_pct_this_scenario[valid_busiest], 99))
    else:
        max_traffic_increase_pct = float("nan")

    # Two separate means — city (Stadtteil) vs. suburban (Gemeinde) rows —
    # each weighted by that row's own busiest road's baseline traffic (see
    # _traffic_weighted_mean above), not by population/workplaces.
    avg_traffic_increase_pct_city = _traffic_weighted_mean(busiest_pct_this_scenario, city_row_mask)
    avg_traffic_increase_pct_suburban = _traffic_weighted_mean(busiest_pct_this_scenario, suburban_row_mask)

    overall_stats[name] = {
        **{m: scenario_avg[m] for m in scenario.METRIC_COLS},
        **{f"diff_{m}": row[f"diff_{m}"] for m in scenario.METRIC_COLS},
        "n_od_pairs": row["n_od_pairs_scenario"],
        "n_routes_affected": n_routes_affected,
        "pct_routes_affected": pct_routes_affected,
        "people_affected": people_affected,
        "pct_people_affected": pct_people_affected,
        "max_traffic_increase_pct": max_traffic_increase_pct,
        "avg_traffic_increase_pct_city": avg_traffic_increase_pct_city,
        "avg_traffic_increase_pct_suburban": avg_traffic_increase_pct_suburban,
    }

comparison_df = pd.DataFrame(comparison_rows)
comparison_csv_path = os.path.join(output_dir, "scenario_comparison.csv")
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"  {comparison_csv_path}")

# ============================================================
# CHOROPLETH DATA FOR MAP
# (per-AOI-row, per-scenario, population-weighted MEAN of the 4 metrics +
#  their diff vs. that scenario's filtered current, plus the % of that
#  row's population affected by a >5min-slower route, and static
#  population/population-density layers)
# ============================================================

print("Computing per-AOI choropleth data …")
popup_cols = [c for c in ["id", "Name", "type", "source", "population", "workplaces"] if c in aoi.columns]
choropleth = aoi[[*popup_cols, "geometry"]].copy().set_index("id", drop=False)

# Population + population density (pop/km²) are static per AOI row (not
# scenario-dependent), but replicated under every scenario name so they
# slot into the same {scenario}_{layer} choropleth column scheme as the
# other map layers.
row_population_static = pd.Series(pop_lookup).reindex(aoi_ids).fillna(0.0)
row_workplaces_static = pd.Series(workplace_lookup).reindex(aoi_ids).fillna(0.0)
area_km2 = (aoi.set_index("id").geometry.area / 1e6).reindex(aoi_ids)
row_pop_density = (row_population_static / area_km2.replace(0, np.nan)).fillna(0.0)
row_workplace_density = (row_workplaces_static / area_km2.replace(0, np.nan)).fillna(0.0)

worst_time_dest_lookup: dict = {}

for name in scenario_names:
    # No valid_ids filtering — AOI rows inside a closure are routed to/from
    # sensibly by od_matrix (see scenario._weight_for_pair) and their stats
    # are as meaningful as any other row's.
    scenario_avg = scenario.per_origin_weighted(
        od_matrices_fine[name], aoi_ids, pop_lookup, workplace_lookup, None
    )

    if name == "current":
        current_avg_for_diff = scenario_avg
    else:
        current_avg_for_diff = scenario.per_origin_weighted(
            od_current_fine, aoi_ids, pop_lookup, workplace_lookup, None
        )

    for metric in scenario.METRIC_COLS:
        choropleth[f"{name}_{metric}"] = scenario_avg[metric].reindex(aoi_ids).values
        choropleth[f"{name}_diff_{metric}"] = (
            scenario_avg[metric] - current_avg_for_diff[metric]
        ).reindex(aoi_ids).values

    # "Worst" layers: the single worst route out of each AOI row (biggest
    # time/distance increase, biggest speed drop) vs. current — not a mean
    # across routes like the layers above.
    current_sub_fine = od_current_fine[od_current_fine["destination_id"].isin(fine_ids)]
    scenario_sub_fine = od_matrices_fine[name][od_matrices_fine[name]["destination_id"].isin(fine_ids)]
    worst = scenario.per_origin_worst(current_sub_fine, scenario_sub_fine, aoi_ids)
    choropleth[f"{name}_worst_time_increase"] = worst["worst_time_increase"].reindex(aoi_ids).values
    choropleth[f"{name}_worst_distance_increase"] = worst["worst_distance_increase"].reindex(aoi_ids).values
    choropleth[f"{name}_worst_speed_decrease"] = worst["worst_speed_decrease"].reindex(aoi_ids).values
    # Destination id of each row's worst-time-increase route, for the
    # map's "draw the worst route" click feature (see map.py).
    worst_time_dest_lookup[name] = {
        int(k): int(v) for k, v in worst["worst_time_dest_id"].dropna().items()
    }

    # Summary-box "Max" figures: 99th percentile (not the true max) across
    # AOI rows' own worst-route values — same "a single outlier row
    # shouldn't stand in as the max" reasoning as max_traffic_increase_pct
    # above, but for the per-route worst-time/distance/speed layers.
    worst_time_valid = worst["worst_time_increase"].dropna()
    worst_distance_valid = worst["worst_distance_increase"].dropna()
    worst_speed_valid = worst["worst_speed_decrease"].dropna()
    overall_stats[name]["max_time_increase_min"] = (
        float(np.percentile(worst_time_valid, 99)) if len(worst_time_valid) else float("nan")
    )
    overall_stats[name]["max_distance_increase_km"] = (
        float(np.percentile(worst_distance_valid, 99)) if len(worst_distance_valid) else float("nan")
    )
    overall_stats[name]["max_speed_decrease_kmh"] = (
        float(np.percentile(worst_speed_valid, 1)) if len(worst_speed_valid) else float("nan")
    )

    row_people_affected = per_row_affected[name].reindex(aoi_ids)
    row_population = pd.Series(pop_lookup).reindex(aoi_ids)
    pct_people_affected = (row_people_affected / row_population.replace(0, np.nan) * 100.0).fillna(0.0)

    # Only the % version is exposed as a choropleth layer (the raw
    # head-count layer was redundant with it and harder to read across
    # AOI rows of very different population).
    choropleth[f"{name}_pct_people_affected_5min"] = pct_people_affected.values
    choropleth[f"{name}_busiest_road_pct_change"] = busiest_road_pct[name].reindex(aoi_ids).values
    choropleth[f"{name}_population"] = row_population_static.values
    choropleth[f"{name}_population_density"] = row_pop_density.values
    choropleth[f"{name}_workplaces"] = row_workplaces_static.values
    choropleth[f"{name}_workplace_density"] = row_workplace_density.values

# Snapped routing-node position per AOI row, in WGS84 — used by the map to
# drop an origin marker at the exact point routes start from (not just the
# polygon centroid).
nodes_wgs84 = nodes.to_crs(4326)
osmid_series = aoi.set_index("id")["osmid"].reindex(aoi_ids)
node_points = nodes_wgs84.geometry.reindex(osmid_series.values)
choropleth["node_lon"] = [p.x if p is not None else None for p in node_points.values]
choropleth["node_lat"] = [p.y if p is not None else None for p in node_points.values]

choropleth = choropleth.reset_index(drop=True)
choropleth = gpd.GeoDataFrame(choropleth, geometry="geometry", crs=crs)

# ============================================================
# CITY / SUBURBAN AREA OUTLINES
# (the overall outline of all Stadtteil rows = "city", all Gemeinde rows =
# "suburban" — drawn as a plain, slightly thicker border on the map)
# ============================================================

area_boundaries_wgs84 = {}
_city_union = aoi.loc[aoi["type"] == "Stadtteil"].union_all()
if _city_union is not None and not _city_union.is_empty:
    area_boundaries_wgs84["city"] = gpd.GeoSeries([_city_union], crs=crs).to_crs(4326).iloc[0]
_suburban_union = aoi.loc[aoi["type"] == "Gemeinde"].union_all()
if _suburban_union is not None and not _suburban_union.is_empty:
    area_boundaries_wgs84["suburban"] = gpd.GeoSeries([_suburban_union], crs=crs).to_crs(4326).iloc[0]

# Centroid of each scenario's closure area, for the "no cars here" emoji
# marker (already have the WGS84 boundary geometry from the closure loop).
closure_centroids_wgs84 = {
    name: geom.centroid for name, geom in closure_boundaries_wgs84.items() if geom is not None
}

# ============================================================
# BUILD INTERACTIVE MAP
# ============================================================

print("Building interactive map …")
import map as map_module  # noqa: E402  (local module, avoid shadowing stdlib name earlier)

# traffic_increase_path stays under data/routing_analysis/; map_output_path
# (configured near the top of this file, default: project-root
# carfree_zones_map.html) can live anywhere, so the raster is located
# relative to the *data*
# folder, not relative to the map file. Note the raster is still embedded
# directly into the HTML as inline PNG data (see map.py) — there is no
# on-disk reference to this .tif from the saved HTML itself, only at
# build time here.
print(f"  Traffic raster source: {os.path.relpath(traffic_increase_path, parent_dir)}")
map_module.build_map(
    choropleth,
    od_matrices,
    scenario_names=scenario_names,
    overall_stats=overall_stats,
    affected_time_threshold_min=affected_time_threshold_min,
    traffic_raster_path=traffic_increase_path,
    output_path=map_output_path,
    pop_lookup=pop_lookup,
    workplace_lookup=workplace_lookup,
    closure_boundaries=closure_boundaries_wgs84,
    closure_centroids=closure_centroids_wgs84,
    worst_dest_lookup=worst_time_dest_lookup,
    area_boundaries=area_boundaries_wgs84,
)

print("Building region/census map …")
map_module.build_region_map(
    aoi, region_map_path, city_union=_city_union, suburban_union=_suburban_union,
)

print("Done.")
