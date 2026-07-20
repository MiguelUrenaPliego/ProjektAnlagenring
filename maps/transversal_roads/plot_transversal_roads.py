"""Build a folium map of the main transversal (park-crossing) roads along the
Anlagenring, as opposed to the roads running parallel to the park treated in
the intersections map (maps/intersections).

While the intersections map shows how far apart the crossing points are
along the roads that run *parallel* to the park (the "exit" problem), this
map instead shows the roads that run *across* the park, coming in from the
city centre and cutting it into disconnected pieces (the "interruption"
problem).

Reads transversal_roads.gpkg (label, severity, geometry columns), produced
by download_transversal_roads.py and meant to be hand-edited afterwards to
fix any misclassified or missing segments -- run that script first if the
gpkg doesn't exist yet.

The park polygon is the same AOI geometry already downloaded from OSM for
the points-of-interest map (maps/points_of_interest/aoi.gpkg).
"""

import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
from folium.plugins import MeasureControl

HERE = Path(__file__).resolve().parent
MAPS_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MAPS_ROOT / "shared"))

from map_i18n import toggle_html  # noqa: E402
from park_geometry import load_park_polygons  # noqa: E402

I18N = json.loads((MAPS_ROOT / "shared" / "i18n_content.json").read_text(encoding="utf-8"))["transversal"]

DEFAULT_LANG = "de"

ROADS_PATH = HERE / "transversal_roads.gpkg"
OUTPUT_PATH = HERE / "transversal_roads.html"

SEVERITY_COLOR = {"major": "#d7191c", "minor": "#fd8d3c"}
SEVERITY_WEIGHT = {"major": 7, "minor": 5}


def main():
    if not ROADS_PATH.exists():
        raise FileNotFoundError(
            f"{ROADS_PATH} not found. Run download_transversal_roads.py first "
            "to generate it (and optionally hand-edit it afterwards)."
        )

    roads = gpd.read_file(ROADS_PATH)
    park = load_park_polygons()

    center = [park.geometry.union_all().centroid.y, park.geometry.union_all().centroid.x]
    m = folium.Map(location=center, zoom_start=15, tiles="cartodbpositron")

    folium.GeoJson(
        park[["geometry"]],
        name="Anlagenring",
        style_function=lambda f: {
            "fillColor": "#1a9850",
            "color": "#1a9850",
            "weight": 2,
            "fillOpacity": 0.25,
        },
    ).add_to(m)

    severity_labels = I18N[DEFAULT_LANG]["severity_labels"]
    for severity, group_name in (("major", "major_group"), ("minor", "minor_group")):
        fg = folium.FeatureGroup(name=I18N[DEFAULT_LANG][group_name], show=True)
        subset = roads[roads["severity"] == severity]
        for _, row in subset.iterrows():
            geoms = row.geometry.geoms if row.geometry.geom_type == "MultiLineString" else [row.geometry]
            for geom in geoms:
                folium.PolyLine(
                    locations=[(lat, lon) for lon, lat in geom.coords],
                    color=SEVERITY_COLOR[severity],
                    weight=SEVERITY_WEIGHT[severity],
                    opacity=0.9,
                    tooltip=f"{row['label']} ({severity_labels[severity]})",
                ).add_to(fg)
        fg.add_to(m)

    m.add_child(MeasureControl())
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
    background:white;padding:10px 12px;border-radius:6px;
    box-shadow:2px 2px 6px rgba(0,0,0,0.3);
    font:13px 'Segoe UI',system-ui,sans-serif;">
    <b>{I18N[DEFAULT_LANG]['legend_title']}</b>
    <div><i style="background:{SEVERITY_COLOR['major']};width:14px;height:14px;
    display:inline-block;margin-right:6px;border:1px solid #999;"></i>{severity_labels['major']}</div>
    <div><i style="background:{SEVERITY_COLOR['minor']};width:14px;height:14px;
    display:inline-block;margin-right:6px;border:1px solid #999;"></i>{severity_labels['minor']}</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    title_html = """
    <div id="mapTitle" data-i18n="title" style="
        position:fixed;top:16px;left:25px;z-index:9999;
        background:rgba(8,25,55,0.75);color:#fff;padding:8px 14px;
        border-radius:8px;font:600 15px 'Segoe UI',system-ui,sans-serif;
        border:1px solid rgba(255,255,255,0.35);"></div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(toggle_html(I18N, default=DEFAULT_LANG)))

    m.save(str(OUTPUT_PATH))
    print(f"Saved map to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
