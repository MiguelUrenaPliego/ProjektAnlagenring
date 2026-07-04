"""Helpers to compute a single scenario's road network, igraph, and full
AOI-to-AOI OD matrix (time, distance, speed, CO2, route geometry).

A "scenario" is one road network variant: either the unmodified baseline
("current") or a variant where edges intersecting a closure polygon have
their speed reduced to ``nogo_speed`` (mimicking closing that area to cars).
"""
from __future__ import annotations

from typing import Optional

import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
from tqdm import tqdm
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

import co2 as co2_module
import graph as graph_utils


def _orient_edges_u_to_v(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reverse each edge's LineString coordinates if they run v->u instead
    of u->v.

    Two-way streets are stored as two directed rows (u,v) and (v,u), but
    OSM/osmnx don't guarantee the LineString itself is reversed for the
    "backwards" row — both rows can end up pointing at the *same*
    coordinate order. Concatenating per-edge geometries along a shortest
    path (see ``_merge_path_geometry``) assumes each segment already runs
    in the direction it's traversed; an edge stored backwards produces a
    visible kink/zigzag where it's spliced in — most noticeable on a
    single distinctive span like a river bridge, where there's no
    alternate edge to mask the glitch.
    """
    edges = edges.copy()
    u_pts = nodes.geometry.reindex(edges["u"])
    v_pts = nodes.geometry.reindex(edges["v"])

    new_geoms = []
    for geom, u_pt, v_pt in zip(edges["geometry"], u_pts, v_pts):
        if geom is None or u_pt is None or v_pt is None or len(geom.coords) < 2:
            new_geoms.append(geom)
            continue
        start = Point(geom.coords[0])
        end = Point(geom.coords[-1])
        # Reverse only when the end points are clearly swapped (start
        # nearer v than u, end nearer u than v) — anything ambiguous
        # (e.g. a loop edge where u == v) is left as-is.
        if start.distance(v_pt) < start.distance(u_pt) and end.distance(u_pt) < end.distance(v_pt):
            new_geoms.append(LineString(list(geom.coords)[::-1]))
        else:
            new_geoms.append(geom)

    edges["geometry"] = new_geoms
    return edges


def build_scenario_edges(
    edges_base: gpd.GeoDataFrame,
    nogo_union: Optional[BaseGeometry] = None,
    nogo_speed: float = 1,
    acceleration: float = 1.5,
    min_cruising_time: float = 5,
    min_cruising_speed: float = 10,
    max_stop_and_go_speed: float = 50,
    node_penalty: float = 5,
    vehicle_type: str = "gasoline_pc",
    road_score: Optional[dict] = None,
    score_travel_time_reduction: float = 0.0,
) -> gpd.GeoDataFrame:
    """Return edges with maxspeed/travel_time/co2 recomputed for a scenario.

    ``edges_base`` must already contain a ``maxspeed_car`` column (the
    unmodified inferred speed limits).

    ``nogo_union`` (a closure/"prohibited area" polygon) only ever affects
    route *selection*, never the reported real travel time: ``closed``
    just flags which edges intersect it, and ``car_travel_time`` (real) is
    always computed from the unmodified ``maxspeed_car`` — a route that
    genuinely has to cross the closure (its origin or destination is
    inside it) should not be reported as absurdly slow just because it
    couldn't be routed around. See ``car_perceived_travel_time_avoiding_closure``
    below for where the closure penalty actually lives.

    If ``road_score`` is given (a highway-type -> score in [1, 10] mapping,
    10 = most attractive to drive on e.g. a motorway, 1 = least e.g. a
    residential street), two perceived-time columns are computed:

      - ``car_perceived_travel_time``: equal to ``car_travel_time`` at
        score 10, inflated by up to a factor of
        ``1 / (1 - score_travel_time_reduction)`` at score 1 (linearly
        interpolated in between) — modeling a driver's reluctance to route
        through low-score roads even when they're nominally faster.
      - ``car_perceived_travel_time_avoiding_closure``: the same, but
        ``closed`` edges get an extra large multiplicative penalty (scaled
        by how much slower ``nogo_speed`` would make that edge look, so a
        fast road is deterred more strongly than an already-slow one) —
        for routing calls where neither endpoint is inside the closure, so
        the route actually detours around it instead of cutting through.

    In every case, route *selection* should weight by one of the two
    perceived-time columns (see build_igraph); the route's reported
    time/distance/CO2 should always be summed from the real
    ``car_travel_time``/length/co2 columns.
    """
    edges = edges_base.copy()
    edges["closed"] = False

    if nogo_union is not None:
        edges["closed"] = edges.geometry.intersects(nogo_union)

    edges["travel_time_car"], edges["avg_speed_car"] = graph_utils.travel_time(
        edges=edges,
        acceleration=acceleration,
        min_cruising_speed=min_cruising_speed,
        min_cruising_time=min_cruising_time,
        max_stop_and_go_speed=max_stop_and_go_speed,
        node_penalty=node_penalty,
        maxspeed_col="maxspeed_car",
        return_speed=True,
    )

    edges["co2_car"] = co2_module.route_hbefa(
        edges,
        avg_speed_col="avg_speed_car",
        vehicle_type=vehicle_type,
        maxspeed_col="maxspeed_car",
        return_total=False,
    )

    if road_score is not None:
        road_score_score = edges["highway"].apply(
            lambda h: graph_utils.infer_maxspeed_row(h, road_score)
        )
        # score 1 -> avoidance = 1 (full reduction factor applied), score
        # 10 -> avoidance = 0 (no penalty at all), linear in between.
        avoidance = (10.0 - road_score_score) / 9.0
        denom = 1.0 - score_travel_time_reduction * avoidance
        edges["car_perceived_travel_time"] = edges["travel_time_car"] / denom

        # Same relative deterrent as clamping maxspeed to nogo_speed would
        # have produced on the real time (a motorway penalized far more
        # than an already-slow residential street), but applied only to
        # this avoid-closure perceived column, never to the real time.
        closure_penalty = (edges["maxspeed_car"] / nogo_speed).clip(lower=1.0)
        edges["car_perceived_travel_time_avoiding_closure"] = np.where(
            edges["closed"],
            edges["car_perceived_travel_time"] * closure_penalty,
            edges["car_perceived_travel_time"],
        )

    return edges


def build_igraph(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame) -> ig.Graph:
    """Build an igraph from nodes/edges with time, length, co2 and geometry
    attached to edges (positionally aligned with ``edges.reset_index()``,
    matching the ordering assumption ``build_igraph_from_gdfs`` relies on).
    """
    edges_reset = edges.reset_index() if "u" not in edges.columns else edges
    edges_reset = _orient_edges_u_to_v(edges_reset, nodes)

    g = graph_utils.build_igraph_from_gdfs(
        nodes,
        edges_reset,
        weight="travel_time_car",
    )

    n = g.ecount()
    if n != len(edges_reset):
        raise ValueError(
            f"igraph edge count ({n}) does not match edges_reset rows "
            f"({len(edges_reset)}) - cannot align attributes positionally."
        )

    g.es["co2"] = edges_reset["co2_car"].astype(float).tolist()
    g.es["length"] = edges_reset["length"].astype(float).tolist()
    g.es["geometry"] = edges_reset["geometry"].tolist()

    # Route *selection* weights: perceived travel time (penalizes
    # low-score roads), with a second variant that also strongly
    # deters cutting through a closure — used when neither the origin nor
    # destination is inside it (see build_scenario_edges and the
    # blocked_ids-aware routing in od_matrix/edge_traffic). Both fall back
    # to the real travel time in g.es["time"] if road_score wasn't given.
    # Reported route metrics (time/length/co2) always sum the real
    # "time"/"length"/"co2" attributes above, never these.
    if "car_perceived_travel_time" in edges_reset.columns:
        g.es["perceived_time"] = edges_reset["car_perceived_travel_time"].astype(float).tolist()
    else:
        g.es["perceived_time"] = g.es["time"]

    if "car_perceived_travel_time_avoiding_closure" in edges_reset.columns:
        g.es["perceived_time_avoiding_closure"] = (
            edges_reset["car_perceived_travel_time_avoiding_closure"].astype(float).tolist()
        )
    else:
        g.es["perceived_time_avoiding_closure"] = g.es["perceived_time"]

    return g


def blocked_ids(aoi: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, polygon_union: BaseGeometry) -> set:
    """AOI ids whose snapped osmid node falls inside the closure polygon."""
    blocked = set()
    for _, row in aoi.iterrows():
        osmid = row["osmid"]
        if osmid is None or osmid not in nodes.index:
            continue
        if nodes.geometry.loc[osmid].within(polygon_union):
            blocked.add(int(row["id"]))
    return blocked


def _merge_path_geometry(geometries: list) -> Optional[LineString]:
    """Concatenate ordered, connected edge LineStrings into a single route
    LineString, dropping duplicate junction vertices."""
    if not geometries:
        return None

    coords: list = []
    for i, geom in enumerate(geometries):
        seg = list(geom.coords)
        if i == 0:
            coords.extend(seg)
        else:
            coords.extend(seg[1:])

    if len(coords) < 2:
        return None

    return LineString(coords)


def _weight_for_pair(src_id, dest_id, blocked_ids: Optional[set]) -> str:
    """Which perceived-time edge attribute to route a given pair with: if
    either end of the trip is actually inside the closed/prohibited area,
    that closure isn't a detour-able obstacle for this trip — it's the
    trip's own start or end — so route selection shouldn't penalize
    cutting through it. Only pairs entirely outside the closure get the
    closure-avoiding weights."""
    if blocked_ids and (src_id in blocked_ids or dest_id in blocked_ids):
        return "perceived_time"
    return "perceived_time_avoiding_closure"


def od_matrix(
    ig_graph: ig.Graph,
    aoi: gpd.GeoDataFrame,
    crs,
    desc: str = "OD matrix",
    blocked_ids: Optional[set] = None,
) -> gpd.GeoDataFrame:
    """Compute the full AOI-to-AOI OD matrix (excluding self-pairs) for one
    scenario's igraph. One igraph call per origin computes all destinations
    at once (mirrors the existing all-pairs loop in main.py).

    ``blocked_ids``: AOI ids inside this scenario's closure (see
    ``blocked_ids()``/``_weight_for_pair``) — pairs with an origin or
    destination in this set are routed by the plain ``perceived_time``
    (no closure penalty); every other pair is routed by
    ``perceived_time_avoiding_closure``, so the closure is treated as
    something to route around unless the trip itself starts or ends there.
    """
    osmid_to_vidx = {osmid: i for i, osmid in enumerate(ig_graph.vs["osmid"])}

    aoi_ids = aoi["id"].tolist()
    aoi_names = aoi["Name"].tolist() if "Name" in aoi.columns else [None] * len(aoi)
    aoi_osmids = aoi["osmid"].tolist()
    aoi_vidx = [osmid_to_vidx.get(osmid) for osmid in aoi_osmids]

    rows = []

    origins = tqdm(
        list(zip(aoi_vidx, aoi_ids, aoi_names)),
        desc=desc,
        unit="origin",
    )

    for src_vidx, src_id, src_name in origins:
        if src_vidx is None:
            continue

        dest_entries = [
            (j, dest_id, dest_name, dest_vidx)
            for j, (dest_id, dest_name, dest_vidx) in enumerate(zip(aoi_ids, aoi_names, aoi_vidx))
            if (dest_vidx is not None) and (dest_id != src_id)
        ]
        if not dest_entries:
            continue

        # Split this origin's destinations by which perceived-time weight
        # they should be routed with, and issue one get_shortest_paths
        # call per weight actually needed (usually just one, since most
        # origins aren't themselves inside a closure).
        by_weight: dict = {}
        for entry in dest_entries:
            _, dest_id, _, _ = entry
            by_weight.setdefault(_weight_for_pair(src_id, dest_id, blocked_ids), []).append(entry)

        entry_epath_pairs = []
        for weight_name, entries in by_weight.items():
            epaths = ig_graph.get_shortest_paths(
                src_vidx,
                to=[e[3] for e in entries],
                weights=weight_name,
                output="epath",
            )
            entry_epath_pairs.extend(zip(entries, epaths))

        for (j, dest_id, dest_name, dest_vidx), epath in entry_epath_pairs:
            if not epath:
                continue

            total_time = sum(ig_graph.es[eid]["time"] for eid in epath)
            total_length = sum(ig_graph.es[eid]["length"] for eid in epath)
            total_co2 = sum(ig_graph.es[eid]["co2"] for eid in epath)

            if total_time <= 0 or total_length <= 0:
                continue

            geom = _merge_path_geometry([ig_graph.es[eid]["geometry"] for eid in epath])
            if geom is None:
                continue

            rows.append(
                {
                    "origin_id": int(src_id),
                    "destination_id": int(dest_id),
                    "origin_name": src_name,
                    "destination_name": dest_name,
                    "time_min": round(total_time / 60.0, 4),
                    "distance_km": round(total_length / 1000.0, 4),
                    "avg_speed_kmh": round((total_length / 1000.0) / (total_time / 3600.0), 4),
                    "co2_kg": round(total_co2, 4),
                    "geometry": geom,
                }
            )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return gdf


METRIC_COLS = ["time_min", "distance_km", "avg_speed_kmh", "co2_kg"]


def filter_valid(od: gpd.GeoDataFrame, valid_ids: Optional[set]) -> gpd.GeoDataFrame:
    """Restrict an OD matrix to pairs whose origin AND destination are not
    blocked (excluded) for a given scenario. ``valid_ids=None`` means no
    filtering (baseline/current with no closure)."""
    if valid_ids is None:
        return od
    mask = od["origin_id"].isin(valid_ids) & od["destination_id"].isin(valid_ids)
    return od[mask]


def pair_weight(od: gpd.GeoDataFrame, origin_lookup: dict, dest_lookup: Optional[dict] = None) -> pd.Series:
    """Gravity-style importance weight for each OD pair, consistent with
    ``scenario_impact``/``edge_traffic``: each origin's total traffic/
    importance (proportional to its own population, ``origin_lookup``) is
    split among its destinations in proportion to each destination's
    workplaces share within ``od`` (``dest_lookup``) — i.e.
    ``origin_value * dest_value / sum(dest_value over the same origin)``.
    ``dest_lookup`` defaults to ``origin_lookup`` (population on both
    ends) if not given. This is the single weighting definition used for
    every weighted mean/sum in this module, so all of them represent "how
    much of the origin's traffic goes to this destination," not a raw,
    unnormalized product."""
    dest_lookup = origin_lookup if dest_lookup is None else dest_lookup
    origin_val = od["origin_id"].map(origin_lookup).fillna(0.0)
    dest_val = od["destination_id"].map(dest_lookup).fillna(0.0)
    total_dest_val = dest_val.groupby(od["origin_id"]).transform("sum")
    return pd.Series(
        np.where(total_dest_val > 0, origin_val * dest_val / total_dest_val, 0.0),
        index=od.index,
    )


def route_weighted_mean(od: gpd.GeoDataFrame, origin_lookup: dict, dest_lookup: Optional[dict] = None) -> dict:
    """Weighted MEAN of the metric columns across all OD pairs in ``od``,
    weight = ``pair_weight`` (each origin's traffic — by population —
    split among its destinations by workplaces share) — the travel
    time/distance/speed/co2 an average unit of system-wide traffic
    experiences. Used for the scenario-comparison CSV."""
    if len(od) == 0:
        return {m: np.nan for m in METRIC_COLS}

    w = pair_weight(od, origin_lookup, dest_lookup).to_numpy(dtype=float)

    total_w = w.sum()
    if total_w <= 0:
        return {m: np.nan for m in METRIC_COLS}

    return {m: float(np.sum(od[m].to_numpy(dtype=float) * w) / total_w) for m in METRIC_COLS}


def per_origin_weighted(
    od: gpd.GeoDataFrame,
    aoi_ids: list,
    origin_lookup: dict,
    dest_lookup: Optional[dict] = None,
    valid_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Per-AOI-row (as origin) weighted MEAN (weight = ``pair_weight``,
    origin population split across destinations by workplaces share) of
    the metric columns, restricted to non-blocked destinations. Rows whose
    own id is blocked (not in ``valid_ids``) get NaN. Used for the map
    choropleth layers.

    Note: since the origin's own population is a constant factor within
    each row's group, it cancels out of this particular weighted mean
    (weighting by dest_lookup alone would give the identical result) — but
    pair_weight is used anyway for a single, consistent weighting
    definition across the module."""
    od_valid_dest = od if valid_ids is None else od[od["destination_id"].isin(valid_ids)]

    weights = pair_weight(od_valid_dest, origin_lookup, dest_lookup)
    weighted_vals = od_valid_dest[METRIC_COLS].multiply(weights, axis=0)

    grouped_w = weights.groupby(od_valid_dest["origin_id"]).sum()
    grouped_vals = weighted_vals.groupby(od_valid_dest["origin_id"]).sum()

    result = grouped_vals.div(grouped_w, axis=0)
    result = result.reindex(aoi_ids)

    if valid_ids is not None:
        blocked_mask = ~pd.Series(aoi_ids, index=aoi_ids).isin(valid_ids)
        result.loc[blocked_mask] = np.nan

    return result


WORST_PERCENTILE = 99  # vs. the true max/min single outlier route


def _group_percentile_with_dest(
    df: pd.DataFrame, value_col: str, pct: float, aoi_ids: list,
) -> tuple:
    """Per ``origin_id`` group: the row whose ``value_col`` is nearest to
    that group's ``pct``-th percentile (not the true max/min single
    route), so one freak outlier route doesn't dominate the whole AOI
    row's "worst" value. Returns (value Series, destination_id Series),
    both reindexed to ``aoi_ids`` — a real destination id is attached
    (unlike a plain ``.quantile()``) for drawing that route on the map."""
    def _pick(g):
        qval = np.percentile(g[value_col], pct)
        idx = (g[value_col] - qval).abs().idxmin()
        return pd.Series(
            {value_col: g.loc[idx, value_col], "destination_id": g.loc[idx, "destination_id"]}
        )

    picked = df.groupby("origin_id").apply(_pick)
    return (
        picked[value_col].reindex(aoi_ids),
        picked["destination_id"].reindex(aoi_ids),
    )


def per_origin_worst(
    current_sub: gpd.GeoDataFrame,
    scenario_sub: gpd.GeoDataFrame,
    aoi_ids: list,
) -> dict:
    """Per-AOI-row (as origin) *worst single route* vs. the current
    baseline — not a mean across the row's routes, but the row's
    ``WORST_PERCENTILE``-th percentile route (99th for the two "increase"
    metrics, 1st/most-negative-equivalent for the speed decrease). Matches
    OD pairs present in both ``current_sub`` and ``scenario_sub`` (same
    origin/destination).

    Returns a dict of pd.Series (index=aoi_ids):
      - worst_time_increase / worst_time_dest_id
      - worst_distance_increase / worst_distance_dest_id
      - worst_speed_decrease / worst_speed_dest_id
    """
    cols = ["origin_id", "destination_id", "time_min", "distance_km", "avg_speed_kmh"]
    merged = current_sub[cols].merge(
        scenario_sub[cols], on=["origin_id", "destination_id"], suffixes=("_current", "_scenario"),
    )

    if len(merged) == 0:
        nan_series = pd.Series(np.nan, index=aoi_ids)
        return {
            "worst_time_increase": nan_series,
            "worst_time_dest_id": nan_series,
            "worst_distance_increase": nan_series,
            "worst_distance_dest_id": nan_series,
            "worst_speed_decrease": nan_series,
            "worst_speed_dest_id": nan_series,
        }

    merged["diff_time_min"] = merged["time_min_scenario"] - merged["time_min_current"]
    merged["diff_distance_km"] = merged["distance_km_scenario"] - merged["distance_km_current"]
    merged["diff_avg_speed_kmh"] = merged["avg_speed_kmh_scenario"] - merged["avg_speed_kmh_current"]

    time_val, time_dest = _group_percentile_with_dest(merged, "diff_time_min", WORST_PERCENTILE, aoi_ids)
    dist_val, dist_dest = _group_percentile_with_dest(merged, "diff_distance_km", WORST_PERCENTILE, aoi_ids)
    speed_val, speed_dest = _group_percentile_with_dest(
        merged, "diff_avg_speed_kmh", 100 - WORST_PERCENTILE, aoi_ids
    )
    return {
        "worst_time_increase": time_val,
        "worst_time_dest_id": time_dest,
        "worst_distance_increase": dist_val,
        "worst_distance_dest_id": dist_dest,
        "worst_speed_decrease": speed_val,
        "worst_speed_dest_id": speed_dest,
    }


def scenario_impact(
    current_sub: gpd.GeoDataFrame,
    scenario_sub: gpd.GeoDataFrame,
    origin_lookup: dict,
    aoi_ids: list,
    dest_lookup: Optional[dict] = None,
    threshold_min: float = 5.0,
) -> dict:
    """How many people are affected by routes that got more than
    ``threshold_min`` minutes slower.

    Each AOI row's own population is split across its outgoing routes in
    proportion to each destination's workplaces (a gravity-style trip
    distribution: an origin sends more of its population toward bigger
    job centers). The "people affected" for a row is the portion of its
    population allocated to routes that got slower — this is bounded by
    the row's own population (no double counting across origin AND
    destination like a plain pop_o+pop_d sum would produce).

    Returns a dict with:
      - n_routes_affected: number of OD pairs slower by > threshold_min
      - people_affected_by_origin: pd.Series (index=aoi_ids), allocated
        affected-people count per AOI row (as origin only)
      - people_affected: total affected people (sum of the above)
      - total_population: sum of population over AOI rows that appear as
        an origin in ``current_sub`` (denominator for a system-wide %)
    """
    dest_lookup = origin_lookup if dest_lookup is None else dest_lookup

    merged = current_sub[["origin_id", "destination_id", "time_min"]].merge(
        scenario_sub[["origin_id", "destination_id", "time_min"]],
        on=["origin_id", "destination_id"],
        suffixes=("_current", "_scenario"),
    )
    merged["delta_time_min"] = merged["time_min_scenario"] - merged["time_min_current"]

    merged["origin_pop"] = merged["origin_id"].map(origin_lookup).fillna(0.0)
    merged["dest_val"] = merged["destination_id"].map(dest_lookup).fillna(0.0)

    total_dest_val = merged.groupby("origin_id")["dest_val"].transform("sum")
    merged["allocated_people"] = np.where(
        total_dest_val > 0,
        merged["origin_pop"] * merged["dest_val"] / total_dest_val,
        0.0,
    )

    affected = merged[merged["delta_time_min"] > threshold_min]

    people_affected_by_origin = (
        affected.groupby("origin_id")["allocated_people"].sum().reindex(aoi_ids).fillna(0.0)
    )

    origin_ids_present = merged["origin_id"].unique()
    total_population = float(sum(origin_lookup.get(i, 0.0) for i in origin_ids_present))

    return {
        "n_routes_affected": int(len(affected)),
        "people_affected_by_origin": people_affected_by_origin,
        "people_affected": float(people_affected_by_origin.sum()),
        "total_population": total_population,
    }


def edge_traffic(
    ig_graph: ig.Graph,
    aoi: gpd.GeoDataFrame,
    origin_lookup: dict,
    dest_lookup: Optional[dict] = None,
    blocked_ids: Optional[set] = None,
    desc: str = "Traffic assignment",
) -> np.ndarray:
    """Estimate traffic load per edge for one scenario's igraph.

    Every AOI row's own population is split across all of its destinations
    in proportion to each destination's workplaces (a gravity trip
    distribution — e.g. a row with population 200 and destinations with
    20/30/50 workplaces sends 40/60/100 "people" to each). Each of those
    person-flows is then assigned onto every edge of its shortest-path
    route, and edge loads are summed across all AOI rows.

    No AOI row is excluded here — every row, including those inside a
    scenario's closure, sends/receives traffic like any other; only the
    *routing weight* for a pair changes when it touches the closure (see
    ``_weight_for_pair``/``blocked_ids``), never whether it's counted.

    Returns a numpy array of accumulated traffic (people), aligned
    positionally with ``ig_graph``'s edges (same order as the edges
    GeoDataFrame used to build the graph — see ``build_igraph``).
    """
    dest_lookup = origin_lookup if dest_lookup is None else dest_lookup
    osmid_to_vidx = {osmid: i for i, osmid in enumerate(ig_graph.vs["osmid"])}

    aoi_ids = aoi["id"].tolist()
    aoi_osmids = aoi["osmid"].tolist()
    aoi_vidx = [osmid_to_vidx.get(osmid) for osmid in aoi_osmids]

    edge_flow = np.zeros(ig_graph.ecount(), dtype=float)

    origins = tqdm(list(zip(aoi_vidx, aoi_ids)), desc=desc, unit="origin")

    for src_vidx, src_id in origins:
        if src_vidx is None:
            continue

        dest_entries = [
            (dest_id, dest_vidx)
            for dest_id, dest_vidx in zip(aoi_ids, aoi_vidx)
            if dest_vidx is not None and dest_id != src_id
        ]
        if not dest_entries:
            continue

        dest_vals = np.array([dest_lookup.get(d, 0.0) for d, _ in dest_entries], dtype=float)
        total_dest_val = dest_vals.sum()
        if total_dest_val <= 0:
            continue

        origin_pop = origin_lookup.get(src_id, 0.0)
        flows = origin_pop * dest_vals / total_dest_val

        by_weight: dict = {}
        for (dest_id, dest_vidx), flow in zip(dest_entries, flows):
            by_weight.setdefault(_weight_for_pair(src_id, dest_id, blocked_ids), []).append((dest_vidx, flow))

        for weight_name, entries in by_weight.items():
            epaths = ig_graph.get_shortest_paths(
                src_vidx, to=[e[0] for e in entries], weights=weight_name, output="epath"
            )
            for (_, flow), epath in zip(entries, epaths):
                if flow <= 0 or not epath:
                    continue
                for eid in epath:
                    edge_flow[eid] += flow

    return edge_flow
