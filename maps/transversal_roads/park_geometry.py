"""Helper for the actual park polygons (OSM leisure=park) covering the
Anlagenring, used by the transversal-roads map generators.

Geometries live in anlagenring.gpkg (next to this file), meant to be
hand-edited afterwards (e.g. in QGIS) to fix any missing/wrong park
segments. Run this module directly to (re-)download it from OSM:

    python3 maps/transversal_roads/park_geometry.py

load_park_polygons() then just reads that file; it does not hit OSM.
"""

import sys
from pathlib import Path

import geopandas as gpd

MAPS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MAPS_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "routing" / "src"))

import osm  # noqa: E402

PARK_GEOMETRY_PATH = Path(__file__).resolve().parent / "anlagenring.gpkg"
AOI_PATH = MAPS_ROOT / "points_of_interest" / "aoi.gpkg"

PARK_OVERPASS_QUERY = """[out:xml][timeout:120];
    (
        way["leisure"="park"]({{bbox}});
        relation["leisure"="park"]({{bbox}});
    );
    (._;>;);
    out body;
"""


def download_park_polygons(bounds: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Download the individual leisure=park polygons within bounds from OSM."""
    gdf = osm.overpass_api_query(PARK_OVERPASS_QUERY, bounds)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise RuntimeError("No leisure=park polygons found for this AOI.")
    return gdf[["name", "geometry"]] if "name" in gdf.columns else gdf[["geometry"]]


def load_park_polygons() -> gpd.GeoDataFrame:
    """Load the (possibly hand-edited) park geometry from anlagenring.gpkg."""
    if not PARK_GEOMETRY_PATH.exists():
        raise FileNotFoundError(
            f"{PARK_GEOMETRY_PATH} not found. Run `python3 {Path(__file__).name}` "
            "in maps/transversal_roads to generate it (and optionally hand-edit it afterwards)."
        )
    return gpd.read_file(PARK_GEOMETRY_PATH)


def main():
    aoi = gpd.read_file(AOI_PATH)
    utm_crs = aoi.estimate_utm_crs()
    bounds = gpd.GeoDataFrame(
        geometry=aoi.to_crs(utm_crs).geometry.buffer(300).to_crs(4326), crs=4326
    )

    print("Downloading OSM park polygons (leisure=park) for the Anlagenring...")
    parks = download_park_polygons(bounds)
    parks.to_file(PARK_GEOMETRY_PATH, driver="GPKG")
    print(f"Saved {len(parks)} park polygon(s) to {PARK_GEOMETRY_PATH}")


if __name__ == "__main__":
    main()
