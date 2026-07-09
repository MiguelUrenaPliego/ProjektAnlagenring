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

import osm  # noqa: E402
import plot_helpers  # noqa: E402
from map_i18n import toggle_html  # noqa: E402

I18N = json.loads((MAPS_ROOT / "shared" / "i18n_content.json").read_text(encoding="utf-8"))["poi"]

DEFAULT_LANG = "de"  # server-baked language for content plot_helpers builds (e.g. the POI legend)

# Kategorie / Typ / emoji to use for the OSM-downloaded POIs, keeping the
# same schema as Bestandsanalyse.csv (Kategorie, Typ, lat, lon, emoji, geometry).
# Classified after a single combined Overpass query by (tag_key, tag_value).
OSM_CATEGORIES = {
    ("amenity", "cafe"): ("Café", "☕"),
    ("amenity", "restaurant"): ("Restaurant", "🍽️"),
    ("amenity", "arts_centre"): ("Kultur-/Bildungseinrichtung", "🎭"),
    ("amenity", "community_centre"): ("Kultur-/Bildungseinrichtung", "🎭"),
    ("amenity", "theatre"): ("Kultur-/Bildungseinrichtung", "🎭"),
    ("amenity", "library"): ("Kultur-/Bildungseinrichtung", "🎭"),
    ("amenity", "public_bookcase"): ("Kultur-/Bildungseinrichtung", "🎭"),
    ("amenity", "conference_centre"): ("Kultur-/Bildungseinrichtung", "🎭"),
    ("amenity", "exhibition_centre"): ("Kultur-/Bildungseinrichtung", "🎭"),
    ("tourism", "museum"): ("Kultur-/Bildungseinrichtung", "🎭"),
    ("tourism", "gallery"): ("Kultur-/Bildungseinrichtung", "🎭"),
}

OSM_OVERPASS_QUERY = """[out:xml][timeout:120];
    (
        node["amenity"="cafe"]({{bbox}});
        way["amenity"="cafe"]({{bbox}});
        relation["amenity"="cafe"]({{bbox}});

        node["amenity"="restaurant"]({{bbox}});
        way["amenity"="restaurant"]({{bbox}});
        relation["amenity"="restaurant"]({{bbox}});

        node["amenity"="arts_centre"]({{bbox}});
        way["amenity"="arts_centre"]({{bbox}});
        relation["amenity"="arts_centre"]({{bbox}});

        node["amenity"="community_centre"]({{bbox}});
        way["amenity"="community_centre"]({{bbox}});
        relation["amenity"="community_centre"]({{bbox}});

        node["amenity"="theatre"]({{bbox}});
        way["amenity"="theatre"]({{bbox}});
        relation["amenity"="theatre"]({{bbox}});

        node["amenity"="library"]({{bbox}});
        way["amenity"="library"]({{bbox}});
        relation["amenity"="library"]({{bbox}});

        node["amenity"="public_bookcase"]({{bbox}});

        node["amenity"="conference_centre"]({{bbox}});
        way["amenity"="conference_centre"]({{bbox}});
        relation["amenity"="conference_centre"]({{bbox}});

        node["amenity"="exhibition_centre"]({{bbox}});
        way["amenity"="exhibition_centre"]({{bbox}});
        relation["amenity"="exhibition_centre"]({{bbox}});

        node["tourism"="museum"]({{bbox}});
        way["tourism"="museum"]({{bbox}});
        relation["tourism"="museum"]({{bbox}});

        node["tourism"="gallery"]({{bbox}});
        way["tourism"="gallery"]({{bbox}});
        relation["tourism"="gallery"]({{bbox}});
    );
    (._;>;);
    out body;
"""


def download_osm_pois(bounds: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Download cafes, restaurants and cultural/educational venues (museums, theatres, opera, libraries, etc.) from OSM for the AOI."""
    print("Downloading OSM POIs (cafes, restaurants, cultural/educational venues)...")
    gdf = osm.overpass_api_query(OSM_OVERPASS_QUERY, bounds)

    empty = gpd.GeoDataFrame(
        columns=["Kategorie", "Typ", "emoji", "name", "lat", "lon", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )
    if gdf.empty:
        return empty

    typ = pd.Series(None, index=gdf.index, dtype="object")
    emoji = pd.Series(None, index=gdf.index, dtype="object")
    for (key, value), (typ_label, emoji_label) in OSM_CATEGORIES.items():
        if key not in gdf.columns:
            continue
        mask = gdf[key] == value
        typ[mask] = typ_label
        emoji[mask] = emoji_label

    gdf = gdf[typ.notna()]
    typ = typ[typ.notna()]
    emoji = emoji.loc[typ.index]

    if gdf.empty:
        return empty

    # Use point location for nodes, centroid for way/relation polygons.
    # Centroid is computed in a projected CRS for accuracy, then reprojected back.
    utm_crs = gdf.estimate_utm_crs()
    points = gdf.to_crs(utm_crs).geometry.centroid.to_crs(gdf.crs)

    osm_pois = gpd.GeoDataFrame(
        {
            "Kategorie": "OSM",
            "Typ": typ,
            "emoji": emoji,
            "name": gdf["name"] if "name" in gdf.columns else None,
            "lat": points.y,
            "lon": points.x,
        },
        geometry=points,
        crs=gdf.crs,
    )
    return osm_pois.to_crs(4326)


def main():
    pois = pd.read_csv(HERE / "Bestandsanalyse.csv")
    pois = pois.rename(columns={"Breitengrad": "lat", "Längengrad": "lon"})

    pois = gpd.GeoDataFrame(
        pois,
        geometry=gpd.points_from_xy(pois.lon, pois.lat),
        crs="EPSG:4326",
    )

    aoi = gpd.read_file(HERE / "aoi.gpkg")
    osm_pois = download_osm_pois(aoi)

    pois = pd.concat([pois, osm_pois], ignore_index=True)
    pois = gpd.GeoDataFrame(pois, geometry="geometry", crs="EPSG:4326")

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
