"""Build a folium map of the exact road geometries colored by distance to the
nearest crosswalk, with markers at the crosswalk points.

Reads data/intersections/crosswalks.gpkg, which contains 4 rows:
  - rows 0-1 ("Innen", "Aussen"): lines marking the crosswalks
  - rows 2-3 ("Innen_Strasse", "Aussen_Strasse"): the exact road geometries

Each road line is densified (extra points interpolated along it) so the
distance-to-nearest-crosswalk gradient is visible at high resolution, instead
of only at the original, sparser vertices. Markers are placed at every
vertex of the two crosswalk linestrings.
"""

import json
import sys
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point

HERE = Path(__file__).resolve().parent
MAPS_ROOT = HERE.parent
sys.path.insert(0, str(MAPS_ROOT / "shared"))
from map_i18n import toggle_html  # noqa: E402

I18N = json.loads((MAPS_ROOT / "shared" / "i18n_content.json").read_text(encoding="utf-8"))["intersections"]

GPKG_PATH = HERE / "crosswalks.gpkg"
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
for _, road_m in roads_m.iterrows():
    name = road_m["Name"]
    line_m = road_m["geometry"].segmentize(DENSIFY_STEP_M)
    crosswalk_m = road_to_crosswalk_m[name]

    coords_m = list(line_m.coords)
    coords = [to_wgs84.transform(x, y) for x, y in coords_m]
    distances = [crosswalk_m.distance(Point(pt)) for pt in coords_m]
    all_distances.extend(distances)
    road_segments[name] = (coords, distances)

colormap = cm.LinearColormap(
    colors=["#2c7bb6", "#ffffbf", "#d7191c"],
    vmin=0,
    vmax=100,
    caption="Distance from road to matching crosswalk (m, clipped to 100)",
)

center = [gdf.geometry.union_all().centroid.y, gdf.geometry.union_all().centroid.x]
fmap = folium.Map(location=center, zoom_start=17, tiles="cartodbpositron")

for name, (coords, distances) in road_segments.items():
    for i in range(len(coords) - 1):
        seg_dist = (distances[i] + distances[i + 1]) / 2
        folium.PolyLine(
            locations=[(coords[i][1], coords[i][0]), (coords[i + 1][1], coords[i + 1][0])],
            color=colormap(min(seg_dist, 100)),
            weight=5,
            opacity=0.9,
            tooltip=f"{name}: {seg_dist:.1f} m to crosswalk",
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
            tooltip=f"Intersection point ({name})",
        ).add_to(fmap)

colormap.add_to(fmap)
folium.LayerControl().add_to(fmap)

title_html = """
<div id="mapTitle" data-i18n="title" style="
    position:fixed;top:16px;left:16px;z-index:9999;
    background:rgba(8,25,55,0.75);color:#fff;padding:8px 14px;
    border-radius:8px;font:600 15px 'Segoe UI',system-ui,sans-serif;
    border:1px solid rgba(255,255,255,0.35);"></div>
"""
fmap.get_root().html.add_child(folium.Element(title_html))
fmap.get_root().html.add_child(folium.Element(toggle_html(I18N)))

fmap.save(str(OUTPUT_PATH))
print(f"Saved map to {OUTPUT_PATH}")
