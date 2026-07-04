"""Builds/filters the drivable street graph for the AOI and snaps AOI
centroids to it.

Callable programmatically (e.g. from main.py) via ``run(config)``, where
``config`` is a dict, a path to a JSON file, or a JSON string — so callers
can pass whatever paths/parameters apply to their own layout without this
module hardcoding them relative to its own file location. Running the file
directly (``python streets.py [config.json]``) uses the same entry point,
falling back to paths relative to the project root if no config is given.
"""
from pathlib import Path
import sys
import os
import json

# Add this file's directory (src/) to Python path so sibling modules are
# importable regardless of the working directory or caller's own path.
parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))

import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import DBSCAN
import osmnx as ox

import osm
import graph as graph_utils

# ============================================================
# DEFAULT CONFIG
# (used for any key not supplied in the config passed to run())
# ============================================================

DEFAULT_CONFIG = {
    # Highway types kept everywhere in the region
    "highway_filter": [
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
    ],
    # Additional types kept close to the city core
    "highway_filter_city_extra": [
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
    ],
    "city_buffer_m": 10_000,          # buffer around the city core (metres)
    "border_cluster_eps_m": 5_000,    # DBSCAN eps for border-node dedup (metres)
    "city_name_filter": "Frankfurt",  # substring matched against aoi["Name"]
    # AOI polygons often leave small internal gaps that aren't a real
    # "outside the study area" boundary — rivers/canals in particular are
    # usually excluded from municipal polygon coverage, so a node sitting
    # on a bridge can fail `intersects(aoi_union)` despite being deep in
    # the interior. Buffering aoi_union outward by this margin before
    # testing keeps those nodes out of the border-node clustering below
    # (which reassigns edge endpoints to a distant "representative" node
    # without updating geometry — fine for nodes genuinely far outside the
    # modeled region, but it turns a river-bridge edge into a bogus
    # multi-km straight-line teleport if wrongly applied there).
    "internal_gap_buffer_m": 300,



    # Paths (default to <project_root>/data/..., project_root = this
    # file's parent's parent, since this module lives in src/).
    "aoi_path": None,
    "streets_dir": None,
    "streets_graph_path": None,
    "osm_xml_file": None,
    "edges_debug_path": None,
}


def _load_config(config) -> dict:
    """Merge ``config`` (dict, JSON file path, or JSON string) over
    ``DEFAULT_CONFIG``. ``None`` uses the defaults untouched."""
    resolved = dict(DEFAULT_CONFIG)

    if config is None:
        user_config = {}
    elif isinstance(config, dict):
        user_config = config
    elif isinstance(config, (str, Path)):
        config_str = str(config)
        path = Path(config_str)
        if path.is_file():
            user_config = json.loads(path.read_text())
        else:
            user_config = json.loads(config_str)
    else:
        raise TypeError(f"config must be a dict, JSON path/string, or None, got {type(config)}")

    resolved.update(user_config)

    project_root = parent_dir.parent
    default_data_dir = project_root / "data"
    default_streets_dir = default_data_dir / "streets"

    if resolved["aoi_path"] is None:
        resolved["aoi_path"] = str(default_data_dir / "aoi.gpkg")
    if resolved["streets_dir"] is None:
        resolved["streets_dir"] = str(default_streets_dir)
    if resolved["streets_graph_path"] is None:
        resolved["streets_graph_path"] = str(Path(resolved["streets_dir"]) / "streets.graphml")
    if resolved["osm_xml_file"] is None:
        resolved["osm_xml_file"] = str(Path(resolved["streets_dir"]) / "streets.osm")
    if resolved["edges_debug_path"] is None:
        resolved["edges_debug_path"] = str(Path(resolved["streets_dir"]) / "edges.gpkg")

    return resolved


def run(config=None) -> None:
    """Build (or reuse) the filtered drivable street graph and snap AOI
    centroids to it, saving the updated AOI + street graph back to disk.

    ``config`` accepts the keys in ``DEFAULT_CONFIG`` — pass a dict, a
    path to a JSON file, or a JSON string. Any key not supplied falls back
    to its default (paths default to ``<project_root>/data/...``).
    """
    cfg = _load_config(config)

    highway_filter = cfg["highway_filter"]
    highway_filter_city = highway_filter + cfg["highway_filter_city_extra"]
    city_buffer_m = cfg["city_buffer_m"]
    border_cluster_eps_m = cfg["border_cluster_eps_m"]
    internal_gap_buffer_m = cfg["internal_gap_buffer_m"]

    aoi_path = Path(cfg["aoi_path"])
    streets_dir = Path(cfg["streets_dir"])
    streets_graph_path = Path(cfg["streets_graph_path"])
    osm_xml_file = Path(cfg["osm_xml_file"])
    edges_debug_path = Path(cfg["edges_debug_path"])

    os.makedirs(str(streets_dir), exist_ok=True)

    # ============================================================
    # LOAD AOI
    # ============================================================

    aoi = gpd.read_file(str(aoi_path))
    aoi = aoi.to_crs(aoi.estimate_utm_crs())

    aoi_city = (
        aoi[aoi["Name"].str.contains(cfg["city_name_filter"], na=False)]
        .reset_index(drop=True)
    )

    # ============================================================
    # DOWNLOAD AND CONVERT OSM DATA
    # ============================================================

    if not os.path.isfile(streets_graph_path):
        network_filter = osm.osmium_network_filter("drive")

        osm.geofabrik_to_osm(
            str(osm_xml_file),
            input_file=str(streets_dir),
            aoi=aoi,
            osmium_filter_args=network_filter,
            overwrite=False,
        )

        # ============================================================
        # BUILD INITIAL GRAPH
        # ============================================================

        G = ox.graph_from_xml(str(osm_xml_file))
        G = ox.project_graph(G, to_crs=aoi.estimate_utm_crs())

        # Save full raw graph before filtering (fast reload fallback)
        ox.save_graphml(G, str(streets_graph_path))

    else:
        G = ox.load_graphml(streets_graph_path)
        G = ox.project_graph(G, to_crs=aoi.estimate_utm_crs())

    nodes, edges = ox.graph_to_gdfs(G)

    # ============================================================
    # NORMALISE HIGHWAY TAGS
    # ============================================================

    # Reuse the version already defined in graph_utils to avoid duplication.
    edges["highway"] = graph_utils.normalize_route_type(edges["highway"])

    # ============================================================
    # EDGE FILTERING
    # ============================================================

    city_union = aoi_city.union_all()
    city_union_buffered = city_union.buffer(city_buffer_m)
    aoi_union = aoi.union_all()

    edges_filtered = edges[
        edges.intersects(city_union)
        | (
            edges.intersects(city_union_buffered)
            & edges["highway"].isin(highway_filter_city)
        )
        | edges["highway"].isin(highway_filter)
    ].copy()

    # ============================================================
    # ADD REVERSE CONNECTOR EDGES FOR ONE-WAY BOUNDARY NODES
    # ============================================================

    # Nodes that appear in filtered edges on exactly one side need a
    # reverse edge so they stay reachable inside the largest component.
    u_set = set(edges_filtered.index.get_level_values("u"))
    v_set = set(edges_filtered.index.get_level_values("v"))

    dangling_mask = (
        (
            edges.index.get_level_values("u").isin(u_set)
            | edges.index.get_level_values("v").isin(v_set)
            | edges.index.get_level_values("u").isin(v_set)
            | edges.index.get_level_values("v").isin(u_set)
        )
        & ~edges.index.isin(edges_filtered.index)
    )

    oneway_edges = edges[dangling_mask].copy()
    oneway_edges["oneway"] = False

    oneway_swapped = oneway_edges.copy()
    idx = oneway_swapped.index
    oneway_swapped.index = pd.MultiIndex.from_arrays(
        [idx.get_level_values("v"), idx.get_level_values("u"), idx.get_level_values("key")],
        names=["u", "v", "key"],
    )

    edges_filtered = pd.concat([oneway_edges, oneway_swapped, edges_filtered])
    edges_filtered = edges_filtered[~edges_filtered.index.duplicated(keep="first")]

    # ============================================================
    # CLUSTER BORDER NODES (outside AOI) TO AVOID DISCONNECTED STUBS
    # ============================================================

    # Buffered outward so small internal gaps in AOI coverage (rivers,
    # slivers between adjacent polygons) don't get misclassified as
    # "outside the study area" — see internal_gap_buffer_m above.
    aoi_union_buffered_inward_gaps = aoi_union.buffer(internal_gap_buffer_m)
    border_nodes = nodes[~nodes.intersects(aoi_union_buffered_inward_gaps)].copy()

    idx = edges.index
    reversed_pairs = pd.MultiIndex.from_arrays(
        [idx.get_level_values("v"), idx.get_level_values("u")], names=["u", "v"]
    )
    existing_pairs = pd.MultiIndex.from_arrays(
        [idx.get_level_values("u"), idx.get_level_values("v")], names=["u", "v"]
    )

    border_mask = (
        idx.get_level_values("u").isin(border_nodes.index)
        | idx.get_level_values("v").isin(border_nodes.index)
    ) & ~reversed_pairs.isin(existing_pairs)

    border_edges = edges[border_mask].copy()

    # Keep only border nodes that actually appear in border edges
    border_nodes = border_nodes[
        border_nodes.index.isin(border_edges.index.get_level_values("u"))
        | border_nodes.index.isin(border_edges.index.get_level_values("v"))
    ].copy()

    if len(border_nodes) > 0:
        coords = np.array([(g.x, g.y) for g in border_nodes.geometry])
        labels = DBSCAN(eps=border_cluster_eps_m, min_samples=1).fit_predict(coords)
        border_nodes["cluster_id"] = labels
        # Representative node per cluster = first node index in that cluster
        border_nodes["cluster_id"] = (
            border_nodes.groupby("cluster_id")["cluster_id"]
            .transform(lambda s: s.index[0])
        )

        u_mapping = border_nodes["cluster_id"].to_dict()
        v_mapping = u_mapping  # same mapping

        # Remap u-side
        conn_u = border_edges.copy()
        idx_u = conn_u.index
        conn_u.index = pd.MultiIndex.from_arrays(
            [
                idx_u.get_level_values("u").map(u_mapping),
                idx_u.get_level_values("v"),
                idx_u.get_level_values("key"),
            ],
            names=["u", "v", "key"],
        )
        conn_u = conn_u[conn_u.index.get_level_values("u").notna()]

        # Remap v-side
        conn_v = border_edges.copy()
        idx_v = conn_v.index
        conn_v.index = pd.MultiIndex.from_arrays(
            [
                idx_v.get_level_values("u"),
                idx_v.get_level_values("v").map(v_mapping),
                idx_v.get_level_values("key"),
            ],
            names=["u", "v", "key"],
        )
        conn_v = conn_v[conn_v.index.get_level_values("v").notna()]

        edges_filtered = pd.concat([edges_filtered, conn_u, conn_v])
        edges_filtered = edges_filtered[~edges_filtered.index.duplicated(keep="first")]

    # ============================================================
    # REBUILD CLEAN NODE SET AND CAST INDEX TYPES
    # ============================================================

    connected_nodes = set(edges_filtered.index.get_level_values("u")).union(
        set(edges_filtered.index.get_level_values("v"))
    )
    nodes_filtered = nodes[nodes.index.isin(connected_nodes)].copy()
    nodes_filtered.index = nodes_filtered.index.astype(int)

    idx = edges_filtered.index
    edges_filtered.index = pd.MultiIndex.from_arrays(
        [
            idx.get_level_values("u").astype(int),
            idx.get_level_values("v").astype(int),
            idx.get_level_values("key").astype(int),
        ],
        names=idx.names,
    )

    # ============================================================
    # TRIM TO LARGEST STRONGLY CONNECTED COMPONENT
    # ============================================================

    H = ox.graph_from_gdfs(nodes_filtered, edges_filtered)
    H = ox.truncate.largest_component(H, strongly=True)
    ox.save_graphml(H, str(streets_graph_path))
    print(f"Street graph saved → {streets_graph_path}")

    nodes, edges = ox.graph_to_gdfs(H)

    # Debug export so the filtered edge set can be inspected in QGIS etc.
    edges.to_file(str(edges_debug_path), driver="GPKG")
    print(f"Edge debug layer saved → {edges_debug_path}")

    # ============================================================
    # SNAP AOI CENTROIDS TO NEAREST GRAPH NODES
    # ============================================================

    aoi_centroids = aoi.copy()
    aoi_centroids.geometry = aoi_centroids.geometry.centroid

    # graph_utils.nearest_nodes expects a GeoDataFrame and a nodes GDF
    aoi["osmid"] = graph_utils.nearest_nodes(aoi_centroids, nodes)

    node_geom = nodes.geometry.loc[aoi["osmid"]].values
    aoi["distance_to_node"] = aoi.geometry.centroid.distance(
        gpd.GeoSeries(node_geom, index=aoi.index, crs=aoi.crs)
    )

    # Ensure id column is preserved
    if "id" not in aoi.columns:
        aoi["id"] = aoi.index.astype(int)

    aoi.to_file(str(aoi_path), driver="GPKG")
    print(f"AOI updated with osmid and distance_to_node → {aoi_path}")


if __name__ == "__main__":
    cli_config = sys.argv[1] if len(sys.argv) > 1 else None
    run(cli_config)
