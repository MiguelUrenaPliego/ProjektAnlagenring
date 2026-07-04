from __future__ import annotations

from typing import (
    Any,
    Dict,
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

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

import graph


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


def prepare_point(
    point: Union[
        tuple[float, float],
        Point,
        BaseGeometry,
        gpd.GeoSeries,
        gpd.GeoDataFrame,
        int,
    ],
) -> Union[gpd.GeoDataFrame, int]:
    """
    Normalize heterogeneous point inputs.
    """
    if isinstance(point, int):
        return point

    if (
        isinstance(point, tuple)
        and len(point) == 2
        and all(isinstance(x, (float, int)) for x in point)
    ):
        return gpd.GeoDataFrame(geometry=[Point(point)], crs="EPSG:4326")

    if isinstance(point, gpd.GeoDataFrame):
        centroid = point.union_all().centroid
        return gpd.GeoDataFrame(geometry=[centroid], crs=point.crs)

    if isinstance(point, gpd.GeoSeries):
        centroid = point.union_all().centroid
        return gpd.GeoDataFrame(geometry=[centroid], crs=point.crs)

    if isinstance(point, BaseGeometry):
        geom = point if isinstance(point, Point) else point.centroid
        return gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

    raise TypeError(f"Unsupported point type: {type(point)}")


# Re-export helpers that live in graph.py for backwards compatibility
nearest_nodes = graph.nearest_nodes
nearest_edges  = graph.nearest_edges
infer_maxspeed = graph.infer_maxspeed
travel_time    = graph.travel_time


def route_nx(
    G: Union[
        nx.MultiDiGraph,
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
        tuple[nx.MultiDiGraph, gpd.GeoDataFrame, gpd.GeoDataFrame],
    ],
    origin: Union[gpd.GeoDataFrame, gpd.GeoSeries, Point, tuple[float, float], int],
    destination: Union[gpd.GeoDataFrame, gpd.GeoSeries, Point, tuple[float, float], int],
    max_dist: float = 1000,
    travel_time_col: str = "travel_time",
    length_col: str = "length",
):
    """Compute shortest-path route statistics using an OSMnx / NetworkX graph."""

    # --- graph normalisation ---
    if isinstance(G, tuple):
        if len(G) == 3:
            G, nodes, edges = G
        elif len(G) == 2:
            nodes, edges = G
            G = ox.graph_from_gdfs(nodes, edges)
        else:
            raise ValueError("Unsupported tuple format for G.")
    else:
        nodes, edges = ox.graph_to_gdfs(G)

    # --- snap to nodes ---
    if isinstance(origin, int):
        origin_osmid = origin
    else:
        origin_osmid = graph.nearest_nodes(prepare_point(origin), nodes, max_dist=max_dist)

    if isinstance(destination, int):
        destination_osmid = destination
    else:
        destination_osmid = graph.nearest_nodes(prepare_point(destination), nodes, max_dist=max_dist)

    # --- shortest path ---
    route_nodes = ox.shortest_path(G, origin_osmid, destination_osmid, weight=travel_time_col)
    if route_nodes is None:
        raise ValueError("No route found between origin and destination.")

    pairs = pd.DataFrame({"u": route_nodes[:-1], "v": route_nodes[1:]})

    edges_df = edges.reset_index()
    best_edges = (
        edges_df
        .sort_values(travel_time_col)
        .drop_duplicates(subset=["u", "v"], keep="first")
    )

    route_edges = pairs.merge(best_edges, on=["u", "v"], how="left")
    route_edges_gdf = gpd.GeoDataFrame(route_edges, geometry="geometry", crs=edges.crs)
    route_nodes_gdf = nodes.loc[route_nodes].copy()

    total_length      = route_edges[length_col].sum()
    total_travel_time = route_edges[travel_time_col].sum()
    avg_speed         = (total_length / 1000) / (total_travel_time / 3600)

    return (
        total_length / 1000,      # km
        total_travel_time / 60,   # minutes
        avg_speed,                # km/h
        route_nodes_gdf,
        route_edges_gdf,
    )


def route(
    G: GraphInput,
    origin: Union[gpd.GeoDataFrame, gpd.GeoSeries, Point, tuple[float, float], int],
    destination: Union[gpd.GeoDataFrame, gpd.GeoSeries, Point, tuple[float, float], int],
    travel_time_col: str = "travel_time",
    length_col: str = "length",
    max_dist: float = 1000,
) -> Dict[str, Any]:
    """
    Compute shortest path using igraph while preserving original node IDs,
    multi-edge keys, and edge identities.

    Returns
    -------
    dict with keys: node_indices, edge_indices, time (min), length (km), speed (km/h)
    """

    # --- build igraph ---
    g = graph._to_igraph(G, weight=travel_time_col)

    # --- recover nodes GDF to allow geometry-based snapping ---
    if isinstance(G, tuple):
        nodes = G[1] if len(G) == 2 else G[1]
    elif isinstance(G, (nx.MultiDiGraph, nx.DiGraph, nx.Graph)):
        nodes, _ = ox.graph_to_gdfs(G)
    else:
        # pure igraph – build a minimal node series from vertex osmids
        nodes = pd.DataFrame(
            {"osmid": g.vs["osmid"]},
            index=g.vs["osmid"],
        )

    # --- snap to nodes ---
    if isinstance(origin, int):
        origin_osmid = origin
    else:
        origin_osmid = graph.nearest_nodes(prepare_point(origin), nodes, max_dist=max_dist)

    if isinstance(destination, int):
        destination_osmid = destination
    else:
        destination_osmid = graph.nearest_nodes(prepare_point(destination), nodes, max_dist=max_dist)

    # --- vertex index lookup ---
    osmid_index = pd.Index(g.vs["osmid"], name="osmid")
    s = osmid_index.get_loc(origin_osmid)
    t = osmid_index.get_loc(destination_osmid)

    # --- shortest paths ---
    vpath = g.get_shortest_paths(s, to=t, weights="time", output="vpath")[0]
    epath = g.get_shortest_paths(s, to=t, weights="time", output="epath")[0]

    if not vpath:
        raise ValueError("No path found between origin and destination.")

    # --- metrics ---
    total_time   = sum(g.es[e]["time"] for e in epath)
    total_length = (
        sum(g.es[e][length_col] for e in epath)
        if length_col in g.es.attributes()
        else 0.0
    )
    speed = (
        (total_length / 1000) / (total_time / 3600)
        if total_length > 0 and total_time > 0
        else None
    )

    # --- edge index ---
    has_key = "key" in g.es.attributes()
    if has_key:
        edge_index = pd.MultiIndex.from_arrays(
            [
                [g.es[e]["u"] for e in epath],
                [g.es[e]["v"] for e in epath],
                [g.es[e]["key"] for e in epath],
            ],
            names=["u", "v", "key"],
        )
    else:
        edge_index = pd.MultiIndex.from_arrays(
            [
                [g.es[e]["u"] for e in epath],
                [g.es[e]["v"] for e in epath],
            ],
            names=["u", "v"],
        )

    node_index = pd.Index(osmid_index[vpath], name="osmid")

    return {
        "node_indices": node_index,
        "edge_indices": edge_index,
        "time":   total_time / 60,
        "length": total_length / 1000 if total_length else None,
        "speed":  speed,
    }