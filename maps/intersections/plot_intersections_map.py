"""Build a folium map of the exact road geometries colored by distance to the
nearest crosswalk, with markers at the crosswalk points.

Reads data/intersections/crosswalks.gpkg, which contains 4 rows:
  - rows 0-1 ("Innen", "Aussen"): lines marking the crosswalks
  - rows 2-3 ("Innen_Strasse", "Aussen_Strasse"): the exact road geometries

Each road line is densified (extra points interpolated along it) so the
distance-to-nearest-crosswalk gradient is visible at high resolution, instead
of only at the original, sparser vertices. Markers are placed at every
vertex of the two crosswalk linestrings.

An H3 hexagon grid over aoi.gpkg additionally shows, per hexagon, how
"isolated" that part of the street network is: the average distance to the
nearest crosswalk of every road sub-segment and crosswalk point falling in
the hexagon, weighted by how much length (in meters) each observation
represents.
"""

import json
import sys
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import h3
import pandas as pd
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point

HERE = Path(__file__).resolve().parent
MAPS_ROOT = HERE.parent
sys.path.insert(0, str(MAPS_ROOT / "shared"))
from hexgrid import literal_hex_grid  # noqa: E402
from map_i18n import toggle_html  # noqa: E402

I18N = json.loads((MAPS_ROOT / "shared" / "i18n_content.json").read_text(encoding="utf-8"))["intersections"]

DEFAULT_LANG = "de"  # server-baked language for the colormap caption/tooltips below

GPKG_PATH = HERE / "crosswalks.gpkg"
AOI_PATH = HERE / "aoi.gpkg"
OUTPUT_PATH = HERE / "intersections_map.html"
DENSIFY_STEP_M = 1.0  # distance in meters between interpolated points along each road

gdf = gpd.read_file(GPKG_PATH)

crosswalks = gdf.iloc[[0, 1]]
roads = gdf.iloc[[2, 3]]

# Reproject to a metric CRS for accurate distances and densification.
gdf_m = gdf.to_crs(epsg=25832)  # UTM zone 32N, metric CRS for the area
roads_m = gdf_m.iloc[[2, 3]]
crosswalks_m = gdf_m.iloc[[0, 1]]

# Match each road to its own crosswalk by the "Innen"/"Aussen" name prefix.
# Each crosswalk's vertices are individual crossing points, not a continuous
# path, so distances must be measured to the nearest *vertex*, not to the
# nearest point on the line itself (which would cut across straight segments
# between crossings and understate the true distance).
road_to_crosswalk_m = {}
for _, road_m in roads_m.iterrows():
    prefix = road_m["Name"].split("_")[0]
    match = crosswalks_m[crosswalks_m["Name"] == prefix]
    crosswalk_line = match.iloc[0]["geometry"]
    road_to_crosswalk_m[road_m["Name"]] = MultiPoint(list(crosswalk_line.coords))

# Transformer back to WGS84 for plotting on the folium map, applied to the
# same densified metric coordinates used for the distance calculation so
# both stay in exact correspondence.
to_wgs84 = Transformer.from_crs(gdf_m.crs, gdf.crs, always_xy=True)

all_distances = []
road_segments = {}
# Per-observation records used for the isolation hex grid below: each
# road sub-segment and each crosswalk point contributes one observation
# of (location, distance-to-crosswalk value, weight in meters).
weighted_observations = []
for _, road_m in roads_m.iterrows():
    name = road_m["Name"]
    line_m = road_m["geometry"].segmentize(DENSIFY_STEP_M)
    crosswalk_m = road_to_crosswalk_m[name]

    coords_m = list(line_m.coords)
    coords = [to_wgs84.transform(x, y) for x, y in coords_m]
    distances = [crosswalk_m.distance(Point(pt)) for pt in coords_m]
    all_distances.extend(distances)
    road_segments[name] = (coords, distances)

    for i in range(len(coords) - 1):
        seg_len_m = Point(coords_m[i]).distance(Point(coords_m[i + 1]))
        seg_dist = (distances[i] + distances[i + 1]) / 2
        mid_lon = (coords[i][0] + coords[i + 1][0]) / 2
        mid_lat = (coords[i][1] + coords[i + 1][1]) / 2
        weighted_observations.append((mid_lat, mid_lon, seg_dist, seg_len_m))

# Crosswalk points are themselves the crossing (distance 0), weighted with
# a nominal length so they count on the same scale as the ~1 m road
# sub-segments instead of being swamped or dominating the average.
for _, crosswalk in crosswalks.iterrows():
    for lon, lat in crosswalk["geometry"].coords:
        weighted_observations.append((lat, lon, 0.0, DENSIFY_STEP_M))

colormap = cm.LinearColormap(
    colors=["#2c7bb6", "#ffffbf", "#d7191c"],
    vmin=0,
    vmax=100,
    caption=I18N[DEFAULT_LANG]["colormap_caption"],
)

# ------------------------------------------------------------------
# Isolation hex grid: weighted-mean distance-to-crosswalk per H3 cell.
# ------------------------------------------------------------------
aoi = gpd.read_file(AOI_PATH)
# Shrink the AOI by 100 m so the hex grid doesn't reach the very edge of the
# park, then build the grid over that buffered area.
aoi_buffered = aoi.to_crs(25832).assign(geometry=lambda d: d.geometry.buffer(-100.0)).to_crs(4326)
hex_gdf = literal_hex_grid(aoi_buffered)
resolution = h3.get_resolution(hex_gdf["h3_cell"].iloc[0])
obs_cells = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon, _, _ in weighted_observations]

weighted_sum = {}
weight_sum = {}
for cell, (_, _, value, weight) in zip(obs_cells, weighted_observations):
    weighted_sum[cell] = weighted_sum.get(cell, 0.0) + value * weight
    weight_sum[cell] = weight_sum.get(cell, 0.0) + weight

hex_gdf["weight_m"] = hex_gdf["h3_cell"].map(weight_sum).fillna(0.0)
hex_gdf["isolation_observed"] = hex_gdf["h3_cell"].map(
    lambda c: weighted_sum[c] / weight_sum[c] if c in weight_sum and weight_sum[c] > 0 else None
)

# Every hexagon in the buffered AOI must show a value, even ones with no
# road/crosswalk observations falling inside them. Fill those in by
# inverse-distance-weighted interpolation from the hexagons that do have an
# observed value, using centroid distance in a metric CRS.
centroids_m = hex_gdf.to_crs(25832).geometry.centroid
observed_mask = hex_gdf["isolation_observed"].notna()
known_xy = list(zip(centroids_m[observed_mask].x, centroids_m[observed_mask].y))
known_vals = hex_gdf.loc[observed_mask, "isolation_observed"].tolist()


def _idw(x, y, power=2.0):
    num = 0.0
    den = 0.0
    for (kx, ky), val in zip(known_xy, known_vals):
        d2 = (x - kx) ** 2 + (y - ky) ** 2
        w = 1.0 / d2 if d2 > 1e-6 else 1e6
        num += w * val
        den += w
    return num / den if den > 0 else None


hex_gdf["isolation"] = [
    val if pd.notna(val) else _idw(x, y)
    for val, x, y in zip(hex_gdf["isolation_observed"], centroids_m.x, centroids_m.y)
]

center = [gdf.geometry.union_all().centroid.y, gdf.geometry.union_all().centroid.x]
fmap = folium.Map(location=center, zoom_start=17, tiles="cartodbpositron")

NO_DATA_COLOR = "#cccccc"


def _hex_style(feature):
    value = feature["properties"]["isolation"]
    color = colormap(min(value, 100)) if value is not None else NO_DATA_COLOR
    return {"fillColor": color, "color": color, "weight": 0, "fillOpacity": 0.45}


hex_gdf["isolation_label"] = hex_gdf["isolation"].apply(
    lambda v: f"{v:.0f} m" if pd.notna(v) else "–"
)

folium.GeoJson(
    hex_gdf[["h3_cell", "isolation", "isolation_label", "geometry"]],
    name="Isoliertheit (Hexagon)",
    style_function=_hex_style,
    tooltip=folium.GeoJsonTooltip(
        fields=["isolation_label"],
        aliases=[I18N[DEFAULT_LANG]["hex_tooltip_isolation_label"] + ":"],
        localize=True,
    ),
).add_to(fmap)

for name, (coords, distances) in road_segments.items():
    for i in range(len(coords) - 1):
        seg_dist = (distances[i] + distances[i + 1]) / 2
        folium.PolyLine(
            locations=[(coords[i][1], coords[i][0]), (coords[i + 1][1], coords[i + 1][0])],
            color=colormap(min(seg_dist, 100)),
            weight=5,
            opacity=0.9,
            tooltip=I18N[DEFAULT_LANG]["tooltip_crosswalk_dist"].format(name=name, dist=f"{seg_dist:.1f}"),
        ).add_to(fmap)

for _, crosswalk in crosswalks.iterrows():
    name = crosswalk["Name"]
    for lon, lat in crosswalk["geometry"].coords:
        folium.CircleMarker(
            location=(lat, lon),
            radius=5,
            color="black",
            fill=True,
            fill_color="orange",
            fill_opacity=1.0,
            tooltip=I18N[DEFAULT_LANG]["tooltip_intersection_point"].format(name=name),
        ).add_to(fmap)

colormap.add_to(fmap)
folium.LayerControl().add_to(fmap)

# branca hardcodes the colormap legend to the top-right leaflet corner with no
# position option, so relocate it to the bottom-right after it renders.
legend_position_js = """
<script>
document.addEventListener("DOMContentLoaded", function () {
    var legend = document.querySelector(".legend.leaflet-control");
    if (legend) {
        legend.style.position = "fixed";
        legend.style.top = "auto";
        legend.style.bottom = "40px";
        legend.style.right = "40px";
        legend.style.left = "auto";
        legend.style.background = "rgba(255,255,255,0.92)";
        legend.style.padding = "4px 8px";
        legend.style.borderRadius = "6px";
        legend.style.boxShadow = "2px 2px 6px rgba(0,0,0,0.25)";
    }
});
</script>
"""
fmap.get_root().html.add_child(folium.Element(legend_position_js))

title_html = """
<div id="mapTitle" data-i18n="title" style="
    position:fixed;top:16px;left:16px;z-index:9999;
    background:rgba(8,25,55,0.75);color:#fff;padding:8px 14px;
    border-radius:8px;font:600 15px 'Segoe UI',system-ui,sans-serif;
    border:1px solid rgba(255,255,255,0.35);"></div>
"""
fmap.get_root().html.add_child(folium.Element(title_html))
fmap.get_root().html.add_child(folium.Element(toggle_html(I18N, default=DEFAULT_LANG)))

fmap.save(str(OUTPUT_PATH))
print(f"Saved map to {OUTPUT_PATH}")
