from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import geopandas as gpd
import igraph as ig
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import re

from sklearn.cluster import DBSCAN

# ============================================================
# TYPE ALIASES
# ============================================================
GraphInput = Union[
    ig.Graph,
    nx.MultiDiGraph,
    nx.Graph,
    nx.DiGraph,
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[object, pd.DataFrame, pd.DataFrame],
    Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
    Tuple[object, gpd.GeoDataFrame, gpd.GeoDataFrame],
]

MAXSPEEDS = {
    "living_street": 30,
    "motorway": 100,
    "motorway_link": 60,
    "primary": 50,
    "primary_link": 50,
    "residential": 30,
    "secondary": 40,
    "secondary_link": 40,
    "service": 20,
    "tertiary": 40,
    "tertiary_link": 40,
    "trunk": 80,
    "trunk_link": 60,
    "unclassified": 40,
}

ROUTE_TYPE_PRIORITY = [
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "service",
    "living_street",
    "residential",
    "unclassified",
]

# ============================================================
# BUILD IGARPH FROM GDFS
# ============================================================
def build_igraph_from_gdfs(
    nodes_gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
    edges_gdf: Union[gpd.GeoDataFrame, pd.DataFrame],
    weight: str = "length",
) -> ig.Graph:
    """
    Build igraph from node/edge GeoDataFrames while preserving:

    - original node IDs
    - multi-edges
    - edge keys
    - edge metadata
    """

    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy().reset_index()

    # --------------------------------------------------------
    # NODE INDEX MAPPING
    # --------------------------------------------------------
    node_ids = list(nodes.index)

    node_to_idx = {
        node_id: i
        for i, node_id in enumerate(node_ids)
    }

    # --------------------------------------------------------
    # VALIDATE EDGES
    # --------------------------------------------------------
    if "u" not in edges.columns or "v" not in edges.columns:
        raise ValueError("Edges must contain 'u' and 'v' columns")

    # --------------------------------------------------------
    # MAP EDGES TO IGRAPH INDICES
    # --------------------------------------------------------
    edges["u_idx"] = edges["u"].map(node_to_idx)
    edges["v_idx"] = edges["v"].map(node_to_idx)

    edges = edges.dropna(subset=["u_idx", "v_idx"])

    edges["u_idx"] = edges["u_idx"].astype(int)
    edges["v_idx"] = edges["v_idx"].astype(int)

    # --------------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------------
    g = ig.Graph(
        n=len(node_ids),
        edges=list(zip(edges["u_idx"], edges["v_idx"])),
        directed=True,
    )

    # --------------------------------------------------------
    # VERTEX ATTRIBUTES
    # --------------------------------------------------------
    g.vs["osmid"] = node_ids

    # --------------------------------------------------------
    # EDGE ATTRIBUTES
    # --------------------------------------------------------
    g.es["u"] = edges["u"].tolist()
    g.es["v"] = edges["v"].tolist()

    if "key" in edges.columns:
        g.es["key"] = edges["key"].tolist()

    # travel time / weight
    if weight in edges.columns:
        g.es["time"] = (
            edges[weight]
            .astype(float)
            .tolist()
        )
    else:
        g.es["time"] = [1.0] * g.ecount()

    # preserve length if available
    if "length" in edges.columns:
        g.es["length"] = (
            edges["length"]
            .astype(float)
            .tolist()
        )

    return g


# ============================================================
# GRAPH NORMALIZATION
# ============================================================
def _to_igraph(
    G: GraphInput,
    weight: str = "length",
) -> ig.Graph:
    """
    Convert supported graph formats into igraph.
    """

    # --------------------------------------------------------
    # CASE 1: ALREADY IGRAPH
    # --------------------------------------------------------
    if isinstance(G, ig.Graph):
        return G

    # --------------------------------------------------------
    # CASE 2: NETWORKX / OSMNX
    # --------------------------------------------------------
    if isinstance(
        G,
        (
            nx.MultiDiGraph,
            nx.DiGraph,
            nx.Graph,
        ),
    ):
        nodes, edges = ox.graph_to_gdfs(G)

        return build_igraph_from_gdfs(
            nodes,
            edges,
            weight=weight,
        )

    # --------------------------------------------------------
    # CASE 3: TUPLES
    # --------------------------------------------------------
    if isinstance(G, tuple):

        if len(G) == 2:
            nodes, edges = G

        elif len(G) == 3:
            _, nodes, edges = G

        else:
            raise ValueError(
                "Unsupported tuple format"
            )

        return build_igraph_from_gdfs(
            nodes,
            edges,
            weight=weight,
        )

    raise ValueError("Unsupported graph type")



def normalize_route_type(route_type,route_type_priority=ROUTE_TYPE_PRIORITY):
    if isinstance(route_type,(pd.DataFrame,gpd.GeoDataFrame)):
        route_type = route_type["highway"] 

    if isinstance(route_type, pd.Series):
        route_type = route_type.map(
            lambda x: normalize_route_type(x, route_type_priority)
        ).astype(str)
        
    if isinstance(route_type, (list, tuple)):
        return next(
            (rt for rt in route_type_priority if rt in route_type),
            None
        )
    return route_type  # already a single value


def normalize_maxspeed(x) -> float:
    """
    Normalize OSM maxspeed values into numeric km/h values.

    Handles:
    - numeric values (int, float)
    - numeric strings ("50", " 50 ")
    - unit strings ("50 km/h", "50kmh", "30 mph")
    - list of values
    - invalid values ("none", "signals", etc.)

    Parameters
    ----------
    x : Any
        Raw OSM maxspeed attribute.

    Returns
    -------
    float
        Parsed speed in km/h or np.nan if invalid.
    """

    # ============================================================
    # 1. HANDLE LIST INPUTS
    # ============================================================

    if isinstance(x, list):

        vals = [normalize_maxspeed(v) for v in x]
        vals = [v for v in vals if np.isfinite(v)]

        return float(min(vals)) if vals else np.nan

    # ============================================================
    # 2. HANDLE NULLS
    # ============================================================

    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    # ============================================================
    # 3. HANDLE STRINGS (CLEAN + PARSE)
    # ============================================================

    if isinstance(x, str):

        x = x.lower().strip()

        # remove ALL spaces (fixes "km / h", "50 kmh", etc.)
        x = x.replace(" ", "")

        # normalize unit variants
        x = re.sub(r"km/?h|kmh|kph", "", x)
        x = re.sub(r"mph", "", x)

        # split multiple values
        if ";" in x:
            vals = [normalize_maxspeed(v) for v in x.split(";")]
            vals = [v for v in vals if np.isfinite(v)]
            return float(min(vals)) if vals else np.nan

        # extract first numeric value
        match = re.search(r"(\d+(\.\d+)?)", x)
        if match:
            return float(match.group(1))

        return np.nan

    # ============================================================
    # 4. NUMERIC INPUTS
    # ============================================================

    if isinstance(x, (int, float, np.number)):
        return float(x)

    return np.nan


def infer_maxspeed_row(
    highway,
    maxspeeds: dict = MAXSPEEDS,
) -> float:
    """Infer default maxspeed from OSM highway classification.

    Supports both scalar and list-based highway values commonly
    produced by OSMnx.

    Args:
        highway: OSM highway classification.
        maxspeeds: Mapping between highway type and inferred speed.

    Returns:
        float: Inferred speed limit in km/h.
    """

    def get_speed(h):

        if pd.isna(h):
            return maxspeeds["unclassified"]

        return maxspeeds.get(
            h,
            maxspeeds["unclassified"],
        )

    if isinstance(highway, list):

        vals = [get_speed(h) for h in highway]
        vals = [v for v in vals if not pd.isna(v)]

        if len(vals) == 0:
            return maxspeeds["unclassified"]

        return float(max(vals))

    return float(get_speed(highway))


def infer_maxspeed(
    edges: gpd.GeoDataFrame,
    maxspeeds: Union[Dict[str, float], float] = MAXSPEEDS,
    enforce: bool = False,
    maxspeed_col: str = "maxspeed",
) -> List[float]:
    """
    Infer maxspeed values from OSM edges and return as a list of floats.

    Pipeline:
        1. Normalize existing maxspeed values using `normalize_maxspeed`
        2. Optionally overwrite using highway-based inference (`enforce=True`)
        3. Fill missing values using `infer_maxspeed_row`
        4. Validate numeric correctness
        5. Return values as Python list (row-aligned)

    Parameters
    ----------
    edges : gpd.GeoDataFrame
        Edge GeoDataFrame containing a "highway" column.

    maxspeeds : dict[str, float] or float
        Mapping from highway type to default speed in km/h.

    enforce : bool, default=False
        If True, ignore existing maxspeed values and recompute all
        values purely from highway classification.

    maxspeed_col : str, default="maxspeed"
        Column name containing maxspeed values.

    Returns
    -------
    List[float]
        Clean maxspeed values aligned with input edges order.

    Raises
    ------
    ValueError
        If invalid or non-finite values remain.
    """

    # ============================================================
    # WORK ON COPY OF SERIES ONLY (FAST PATH)
    # ============================================================

    if maxspeed_col not in edges.columns:
        values = np.full(len(edges), np.nan, dtype=float)
    else:
        values = edges[maxspeed_col].to_numpy(dtype=object)

    # ============================================================
    # 1. NORMALIZE OR RECOMPUTE
    # ============================================================

    if enforce:
        values = np.array([
            infer_maxspeed_row(h, maxspeeds=maxspeeds)
            for h in edges["highway"]
        ], dtype=float)

    else:
        values = np.array([
            normalize_maxspeed(v)
            for v in values
        ], dtype=float)

        # fill missing via highway inference
        mask = ~np.isfinite(values)

        if mask.any():
            values[mask] = np.array([
                infer_maxspeed_row(h, maxspeeds=maxspeeds)
                for h in edges.loc[mask, "highway"]
            ], dtype=float)

    # ============================================================
    # 2. VALIDATION
    # ============================================================

    if not np.all(np.isfinite(values)):
        bad = values[~np.isfinite(values)]
        raise ValueError(f"Invalid maxspeed values detected: {bad[:10].tolist()}")

    # ============================================================
    # 3. RETURN LIST
    # ============================================================

    return values.tolist()

def travel_time_row(
    L: float,
    vmax_kmh: float,
    a: float = 1,
    min_speed_kmh: float = 10,
    min_time: float = 5,
    stop_speed: float = 50,
    v0: Optional[float] = None,
) -> float:
    """Estimate edge travel time using asymmetric kinematics.

    The model includes:

    - acceleration,
    - deceleration,
    - cruise phases,
    - minimum cruise duration constraints,
    - intersection stopping behavior.

    Args:
        L: Segment length in meters.
        vmax_kmh: Maximum allowed speed in km/h.
        a: Constant acceleration/deceleration in m/s².
        min_speed_kmh: Minimum cruise speed threshold.
        min_time: Minimum cruise duration in seconds.
        stop_speed: Threshold speed for stop behavior.
        v0: Initial incoming speed in km/h.
            If ``None``, inferred automatically.

    Returns:
        float: Estimated travel time in seconds.
    """

    vmax = vmax_kmh / 3.6
    vmin = min_speed_kmh / 3.6
    v_stop = stop_speed / 3.6

    if v0 is None:
        v0 = v_stop if vmax_kmh > stop_speed else 0.0
    else:
        v0 = v0 / 3.6

    if vmax_kmh <= stop_speed:
        v1 = 0.0
    else:
        v1 = v_stop

    if vmax <= max(v0, v1):
        return L / vmax

    d_acc = (vmax**2 - v0**2) / (2 * a)
    d_dec = (vmax**2 - v1**2) / (2 * a)

    d_total = d_acc + d_dec

    if d_total >= L:

        v_peak = np.sqrt(
            (2 * a * L + v0**2 + v1**2) / 2
        )

        return (
            (v_peak - v0) / a
            + (v_peak - v1) / a
        )

    d_cruise = L - d_total
    t_cruise = d_cruise / vmax

    t_acc = (vmax - v0) / a
    t_dec = (vmax - v1) / a

    if vmax >= vmin and t_cruise < min_time:

        A = 1 / a
        B = min_time

        C = -(
            L
            + (v0**2 + v1**2) / (2 * a)
        )

        v_allowed = (
            -B
            + np.sqrt(B**2 - 4 * A * C)
        ) / (2 * A)

        vmax_eff = min(v_allowed, vmax)

        d_acc = (
            vmax_eff**2 - v0**2
        ) / (2 * a)

        d_dec = (
            vmax_eff**2 - v1**2
        ) / (2 * a)

        if d_acc + d_dec >= L:

            v_peak = np.sqrt(
                (
                    2 * a * L
                    + v0**2
                    + v1**2
                ) / 2
            )

            return (
                (v_peak - v0) / a
                + (v_peak - v1) / a
            )

        d_cruise = L - d_acc - d_dec

        t_acc = (vmax_eff - v0) / a
        t_dec = (vmax_eff - v1) / a
        t_cruise = d_cruise / vmax_eff

        return t_acc + t_cruise + t_dec

    return t_acc + t_cruise + t_dec


def signalized_node_ids(nodes: gpd.GeoDataFrame, highway_col: str = "highway") -> set:
    """OSM node ids tagged ``highway=traffic_signals`` (signalized
    intersections), used to pick the node delay penalty in `travel_time`."""
    if highway_col not in nodes.columns:
        return set()

    def _is_signalized(h):
        if isinstance(h, list):
            return "traffic_signals" in h
        return h == "traffic_signals"

    return set(nodes.index[nodes[highway_col].map(_is_signalized)])


def travel_time(
    edges: gpd.GeoDataFrame,
    acceleration: float = 1,
    min_cruising_speed: float = 10,
    min_cruising_time: float = 5,
    max_stop_and_go_speed: float = 50,
    node_penalty: float = 5,
    signalized_node_penalty: Optional[float] = None,
    signalized_nodes: Optional[set] = None,
    node_col: str = "v",
    maxspeed_col: str = "maxspeed",
    length_col: str = "length",
    return_speed: bool = False,
) -> Union[List[float], Tuple[List[float], List[float]]]:
    """
    Compute travel time and optionally average speed for graph edges.

    The function applies a row-wise physics-based travel time model
    using a vectorized pandas `.apply(axis=1)` operation.

    Args:
        edges:
            Edge GeoDataFrame containing at least:
            - length column (meters)
            - maxspeed column (km/h)

        acceleration:
            Vehicle acceleration in m/s².

        min_cruising_speed:
            Minimum cruising speed in km/h.

        min_cruising_time:
            Minimum time spent in cruising phase (seconds).

        max_stop_and_go_speed:
            Speed threshold above which stop-and-go effects are ignored.

        node_penalty:
            Fixed time penalty per edge (seconds), applied at the edge's
            end node (``node_col``) when it is not signalized (or when
            ``signalized_nodes`` is not given, in which case every node
            uses this value).

        signalized_node_penalty:
            Fixed time penalty per edge (seconds) applied instead of
            ``node_penalty`` when the edge's end node is in
            ``signalized_nodes`` (an OSM ``highway=traffic_signals``
            node). Ignored if ``signalized_nodes`` is not given.

        signalized_nodes:
            Set of node ids (matching ``edges[node_col]``) that are
            signalized intersections — see ``signalized_node_ids``.

        node_col:
            Name of the column identifying each edge's end node, used to
            look up ``signalized_nodes``.

        maxspeed_col:
            Name of column containing maximum speed (km/h).

        length_col:
            Name of column containing edge length (meters).

        return_speed:
            If True, also return average speeds (km/h).

    Returns:
        Union[List[float], Tuple[List[float], List[float]]]:
            - travel_time list in seconds
            - optionally (travel_time, avg_speed)
    """

    edges = edges.copy()

    # ============================================================
    # ENSURE NUMERIC INPUTS
    # ============================================================

    edges[maxspeed_col] = edges[maxspeed_col].astype(float)
    edges[length_col] = edges[length_col].astype(float)

    # ============================================================
    # PER-EDGE NODE PENALTY (SIGNALIZED VS. UNSIGNALIZED)
    # ============================================================

    if signalized_nodes is not None and node_col in edges.columns:
        penalties = np.where(
            edges[node_col].isin(signalized_nodes),
            signalized_node_penalty,
            node_penalty,
        )
    else:
        penalties = np.full(len(edges), node_penalty, dtype=float)

    # ============================================================
    # TRAVEL TIME COMPUTATION (VECTORIZED APPLY)
    # ============================================================

    edges["travel_time"] = edges.apply(
        lambda row: travel_time_row(
            row[length_col],
            row[maxspeed_col],
            acceleration,
            min_cruising_speed,
            min_cruising_time,
            max_stop_and_go_speed,
        ),
        axis=1,
    ) + penalties

    # ============================================================
    # OPTIONAL SPEED COMPUTATION
    # ============================================================

    if return_speed:
        edges["avg_speed"] = (
            (edges[length_col] / 1000)
            / (edges["travel_time"] / 3600)
        )

        return (
            edges["travel_time"].tolist(),
            edges["avg_speed"].tolist(),
        )

    return edges["travel_time"].tolist()

def crop_graph(nodes,edges,aoi,fix_border=True,exclusive=False,connected=True):
    if exclusive:
        edges_filtered = edges[~edges.intersects(aoi.to_crs(edges.crs).union_all())]
    else:
        edges_filtered = edges[edges.intersects(aoi.to_crs(edges.crs).union_all())]

    edges_filtered = edges_filtered[~edges_filtered.index.duplicated(keep='first')]
    connected_nodes = set(edges_filtered.index.get_level_values("u")).union(set(edges_filtered.index.get_level_values("v")))
    nodes_filtered = nodes[nodes.index.isin(connected_nodes)]
    nodes_filtered.index = nodes_filtered.index.astype(int)
    idx = edges_filtered.index

    edges_filtered.index = pd.MultiIndex.from_arrays(
        [
            idx.get_level_values("u").astype(int),
            idx.get_level_values("v").astype(int),
            idx.get_level_values("key").astype(int),
        ],
        names=idx.names
    )

    if fix_border:
        nodes_filtered, edges_filtered = fix_graph_border(nodes_filtered,edges_filtered,aoi)
    
    if connected:
        G = ox.graph_from_gdfs(nodes_filtered,edges_filtered)
        G = ox.truncate.largest_component(G,strongly=True)
        nodes_filtered, edges_filtered = ox.graph_to_gdfs(G)

    return nodes_filtered, edges_filtered

def fix_graph_border(nodes,edges,aoi):
    border_nodes = nodes[~nodes.intersects(aoi.to_crs(nodes.crs).union_all())]
    idx = edges.index

    reversed_pairs = pd.MultiIndex.from_arrays(
        [
            idx.get_level_values("v"),
            idx.get_level_values("u"),
        ],
        names=["u", "v"]
    )

    existing_pairs = pd.MultiIndex.from_arrays(
        [
            idx.get_level_values("u"),
            idx.get_level_values("v"),
        ],
        names=["u", "v"]
    )

    mask = (
        (
            idx.get_level_values("u").isin(border_nodes.index) |
            idx.get_level_values("v").isin(border_nodes.index)
        ) &
        (
            ~reversed_pairs.isin(existing_pairs)
        )
    )
    border_edges = edges[mask]

    border_nodes = border_nodes[
        border_nodes.index.isin(border_edges.index.get_level_values("u")) |
        border_nodes.index.isin(border_edges.index.get_level_values("v"))
    ]


    coords = np.array([
        (geom.x, geom.y)
        for geom in border_nodes.geometry
    ])

    border_nodes["cluster_id"] = DBSCAN(
        eps=5000,
        min_samples=1
    ).fit_predict(coords)

    border_nodes["cluster_id"] = (
        border_nodes.groupby("cluster_id")["cluster_id"]
        .transform(lambda s: s.index[0])
    )

    connecting_edges_u = border_edges.copy()
    idx = connecting_edges_u.index
    u_mapping = border_nodes["cluster_id"].to_dict()
    connecting_edges_u.index = pd.MultiIndex.from_arrays(
        [
            idx.get_level_values("u").map(u_mapping),
            idx.get_level_values("v"),
            idx.get_level_values("key"),
        ],
        names=["u", "v", "key"]
    )
    connecting_edges_u = connecting_edges_u[
        connecting_edges_u.index.get_level_values("u").notna()
    ]

    connecting_edges_v = border_edges.copy()
    idx = connecting_edges_v.index
    v_mapping = border_nodes["cluster_id"].to_dict()
    connecting_edges_v.index = pd.MultiIndex.from_arrays(
        [
            idx.get_level_values("u"),
            idx.get_level_values("v").map(v_mapping),
            idx.get_level_values("key"),
        ],
        names=["u", "v", "key"]
    )
    connecting_edges_v = connecting_edges_v[
        connecting_edges_v.index.get_level_values("v").notna()
    ]

    edges_filtered = pd.concat([
        edges,
        connecting_edges_u,
        connecting_edges_v
    ])
    edges_filtered = edges_filtered[~edges_filtered.index.duplicated(keep='first')]
    connected_nodes = set(edges_filtered.index.get_level_values("u")).union(set(edges_filtered.index.get_level_values("v")))
    nodes_filtered = nodes[nodes.index.isin(connected_nodes)]
    nodes_filtered.index = nodes_filtered.index.astype(int)
    idx = edges_filtered.index

    edges_filtered.index = pd.MultiIndex.from_arrays(
        [
            idx.get_level_values("u").astype(int),
            idx.get_level_values("v").astype(int),
            idx.get_level_values("key").astype(int),
        ],
        names=idx.names
    )

    return nodes_filtered, edges_filtered

def nearest_nodes(
    geometries: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    nodes,
    max_dist: Optional[float] = None,
) -> list:
    """Find nearest graph nodes to geometries.

    Args:
        geometries: Input geometries.
        G: GraphInput.
        max_dist: Optional maximum search distance in meters.

    Returns:
        list:
            List of nearest node IDs.
            Returns ``None`` for unmatched geometries.
    """
    if nodes.crs.is_geographic:
        nodes = nodes.to_crs(nodes.estimate_utm_crs())

    geom = geometries.to_crs(nodes.crs).copy()

    geom["node_id"] = None

    idx_geom, idx_nodes = nodes.sindex.nearest(
        geom.geometry,
        max_distance=max_dist,
        return_all=False,
    )

    geom.iloc[
        idx_geom,
        geom.columns.get_loc("node_id"),
    ] = list(nodes.index[idx_nodes])

    return list(geom["node_id"])


def nearest_edges(
    geometries: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    edges,
    max_dist: Optional[float] = None,
) -> list:
    """Find nearest graph edges to geometries.

    Args:
        geometries: Input geometries.
        edges: graph edges.
        max_dist: Optional maximum search distance in meters.

    Returns:
        list:
            List of nearest edge IDs.
    """
    if edges.crs.is_geographic:
        edges = edges.to_crs(edges.estimate_utm_crs())
    
    geom = geometries.to_crs(edges.crs).copy()

    geom["edge_id"] = None

    idx_geom, idx_edges = edges.sindex.nearest(
        geom.geometry,
        max_distance=max_dist,
        return_all=False,
    )

    geom.iloc[
        idx_geom,
        geom.columns.get_loc("edge_id"),
    ] = list(edges.index[idx_edges])

    return list(geom["edge_id"])