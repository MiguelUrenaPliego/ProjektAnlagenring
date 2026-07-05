import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import MeasureControl

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
MAPS_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "routing" / "src"))
sys.path.insert(0, str(MAPS_ROOT / "shared"))

import plot_helpers  # noqa: E402
from map_i18n import toggle_html  # noqa: E402

I18N = json.loads((MAPS_ROOT / "shared" / "i18n_content.json").read_text(encoding="utf-8"))["poi"]

DEFAULT_LANG = "de"  # server-baked language for content plot_helpers builds (e.g. the POI legend)


def main():
    pois = pd.read_csv(HERE / "Bestandsanalyse.csv")
    pois = pois.rename(columns={"Breitengrad": "lat", "Längengrad": "lon"})

    pois = gpd.GeoDataFrame(
        pois,
        geometry=gpd.points_from_xy(pois.lon, pois.lat),
        crs="EPSG:4326",
    )

    m = folium.Map(
        location=[pois.lat.mean(), pois.lon.mean()],
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Hybrid",
        overlay=False,
        control=True,
        opacity=0.5,
    ).add_to(m)

    m = plot_helpers.general_map(
        m=m,
        pois=pois,
        poi_column="Typ",
        poi_icon_column="emoji",
        legend_title=I18N[DEFAULT_LANG]["legend_title"],
    )

    m.add_child(MeasureControl())

    title_html = """
    <div id="mapTitle" data-i18n="title" style="
        position:fixed;top:16px;left:25px;z-index:9999;
        background:rgba(8,25,55,0.75);color:#fff;padding:8px 14px;
        border-radius:8px;font:600 15px 'Segoe UI',system-ui,sans-serif;
        border:1px solid rgba(255,255,255,0.35);"></div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(toggle_html(I18N, default=DEFAULT_LANG)))

    m.save(str(HERE / "points_of_interest.html"))


if __name__ == "__main__":
    main()
