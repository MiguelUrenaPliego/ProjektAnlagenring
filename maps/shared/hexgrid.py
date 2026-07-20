"""Shared H3 hexagon-grid helpers for folium maps.

Import with a sys.path.insert(0, "<repo_root>/maps/shared") from any map
generator script (matches the convention used by map_i18n.py).
"""

import folium
import geopandas as gpd
import h3
from shapely.geometry import Polygon

# H3 resolution 10 has an average edge length (~circumradius) of about
# 0.066 km (~66 m). Chosen so the AOI (a park roughly 2.5 km long) is
# covered by a genuinely fine-grained grid — resolution 7 (~1.22 km
# edge) only produced a handful of hexagons across the whole park, and
# resolution 9 (~170 m) was still bumped one step finer on request.
DEFAULT_RESOLUTION = 10


def literal_hex_grid(aoi: gpd.GeoDataFrame, resolution: int = DEFAULT_RESOLUTION) -> gpd.GeoDataFrame:
    """Build the literal, uncut H3 hexagon grid for every cell touching the AOI.

    Cells are kept as their full hexagon shape (not clipped to the AOI
    boundary) so the grid reads as an actual hexagon tiling. Use
    "overlap" containment so small AOIs (smaller than a single hexagon at
    this resolution) still return at least the cells touching them,
    rather than zero cells.
    """
    aoi_union = aoi.to_crs(4326).geometry.union_all()
    cells = h3.h3shape_to_cells_experimental(
        h3.geo_to_h3shape(aoi_union), res=resolution, contain="overlap"
    )
    return gpd.GeoDataFrame(
        {"h3_cell": list(cells)},
        geometry=[
            Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in cells
        ],
        crs=4326,
    )


def filter_hex_grid_by_intersection(hex_gdf: gpd.GeoDataFrame, geoms) -> gpd.GeoDataFrame:
    """Keep only the hexagons that actually intersect the given geometry/geometries.

    ``geoms`` can be a single shapely geometry or anything
    ``GeoSeries.union_all()``-able (a GeoDataFrame/GeoSeries). Useful to
    drop hexagons that cover empty parts of the AOI with no underlying
    features (e.g. no street geometry running through them).
    """
    if hasattr(geoms, "union_all"):
        geoms = geoms.union_all()
    mask = hex_gdf.geometry.intersects(geoms)
    return hex_gdf[mask].reset_index(drop=True)


def add_aoi_background(
    m: folium.Map,
    aoi: gpd.GeoDataFrame,
    color: str = "#a8e6a3",
    opacity: float = 0.35,
    name: str = "AOI",
) -> folium.Map:
    """Draw the AOI polygon as a plain light-green background fill."""
    folium.GeoJson(
        aoi,
        name=name,
        style_function=lambda f: {
            "fillColor": color,
            "color": color,
            "weight": 0,
            "fillOpacity": opacity,
        },
    ).add_to(m)
    return m