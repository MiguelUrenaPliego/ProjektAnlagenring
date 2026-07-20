import json
import sys
from pathlib import Path

import folium
import geopandas as gpd
import h3
import pandas as pd
from folium.plugins import MeasureControl

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
MAPS_ROOT = HERE.parent

sys.path.insert(0, str(REPO_ROOT / "routing" / "src"))
sys.path.insert(0, str(MAPS_ROOT / "shared"))

import osm  # noqa: E402
import plot_helpers  # noqa: E402
from hexgrid import (  # noqa: E402
    DEFAULT_RESOLUTION,
    filter_hex_grid_by_intersection,
    literal_hex_grid,
)
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
    ("leisure", "park"): ("Wiese", "🌳"),
    ("leisure", "playground"): ("Spielplatz", "🛝"),
    ("leisure", "pitch"): ("Sportbereich", "⚽"),
    ("leisure", "sports_centre"): ("Sportbereich", "⚽"),
    ("leisure", "fitness_station"): ("Sportbereich", "⚽"),
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

        way["leisure"="park"]({{bbox}});
        relation["leisure"="park"]({{bbox}});

        node["leisure"="playground"]({{bbox}});
        way["leisure"="playground"]({{bbox}});
        relation["leisure"="playground"]({{bbox}});

        node["leisure"="pitch"]({{bbox}});
        way["leisure"="pitch"]({{bbox}});
        relation["leisure"="pitch"]({{bbox}});

        node["leisure"="sports_centre"]({{bbox}});
        way["leisure"="sports_centre"]({{bbox}});
        relation["leisure"="sports_centre"]({{bbox}});

        node["leisure"="fitness_station"]({{bbox}});
    );
    (._;>;);
    out body;
"""


# ------------------------------------------------------------------
# Diversity of daily usage patterns (Jane Jacobs: a park stays safe and
# alive only if it draws different kinds of users at different times of
# day/week, not a single group at a single hour). Each POI type is
# classified into one of four groups by the daypart/weekday it primarily
# attracts. A hexagon where several of these groups coexist has visitors
# spread across the whole day rather than a single peak-hour crowd.
# ------------------------------------------------------------------
USER_GROUPS = [
    "Sport: früh & abends",
    "Familien: nachmittags & Wochenende",
    "Ruhe & Erholung: ganztags",
    "Gastronomie & Kultur: abends & Wochenende",
]


def classify_user_group(kategorie: str, typ: str) -> str:
    """Map a POI's Kategorie/Typ to the daypart/weekday group it mainly attracts."""
    typ = typ or ""
    if "Sportbereich" in typ:
        return USER_GROUPS[0]  # joggers/sport before work and after work
    if "Spielplatz" in typ:
        return USER_GROUPS[1]  # children after school, families on weekends
    if "Springbrunnen" in typ:
        return USER_GROUPS[2]  # passive recreation, used throughout the day
    if typ == "Wiese":
        return USER_GROUPS[2]  # meadow/lawn, passive recreation throughout the day
    if kategorie == "OSM":
        return USER_GROUPS[3]  # cafés/restaurants/culture, evenings and weekends
    return USER_GROUPS[2]


DIVERSITY_COLORS = {
    0: "#e8e8e8",
    1: "#c7e9c0",
    2: "#74c476",
    3: "#238b45",
    4: "#00441b",
}


def build_diversity_hexgrid(
    pois: gpd.GeoDataFrame,
    aoi: gpd.GeoDataFrame,
    resolution: int = DEFAULT_RESOLUTION,
) -> gpd.GeoDataFrame:
    """Aggregate POIs into an H3 hexagon grid covering the AOI.

    Hexagons are drawn as their literal, uncut H3 shape (every cell that
    touches the AOI is kept whole, not clipped to the park boundary), so
    the grid reads as an actual hexagon tiling rather than an irregular
    patchwork.

    Each cell is colored by how many distinct daypart/weekday user groups
    (see USER_GROUPS) are present, i.e. how evenly activity in that part
    of the park is spread across the day and week rather than
    concentrated in a single use.
    """
    hex_gdf = literal_hex_grid(aoi, resolution=resolution)

    points = pois[pois.geometry.type.isin(["Point", "MultiPoint"])]
    polys = pois[pois.geometry.type.isin(["Polygon", "MultiPolygon"])]

    # Point POIs are assigned to the single hex cell their coordinate falls in.
    point_cells = points.assign(
        h3_cell=[
            h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(points.lat, points.lon)
        ],
        user_group=[
            classify_user_group(k, t) for k, t in zip(points.Kategorie, points.Typ)
        ],
    )
    groups_per_cell = point_cells.groupby("h3_cell")["user_group"].agg(set)
    counts_per_cell = point_cells.groupby("h3_cell").size()

    # Polygon POIs (e.g. Wiese/park areas) can span several hexagons, so
    # they're counted in every cell they intersect, not only the cell
    # containing their centroid.
    if not polys.empty:
        polys = polys.assign(
            user_group=[classify_user_group(k, t) for k, t in zip(polys.Kategorie, polys.Typ)]
        )
        joined = gpd.sjoin(
            hex_gdf[["h3_cell", "geometry"]],
            polys[["user_group", "geometry"]],
            predicate="intersects",
        )
        poly_groups_per_cell = joined.groupby("h3_cell")["user_group"].agg(set)
        poly_counts_per_cell = joined.groupby("h3_cell").size()
    else:
        poly_groups_per_cell = pd.Series(dtype="object")
        poly_counts_per_cell = pd.Series(dtype="int64")

    def combined_groups(cell):
        groups = set(groups_per_cell.get(cell, set())) | set(poly_groups_per_cell.get(cell, set()))
        return sorted(groups)

    hex_gdf["groups"] = hex_gdf["h3_cell"].apply(combined_groups)
    hex_gdf["diversity"] = hex_gdf["groups"].apply(len)
    hex_gdf["poi_count"] = hex_gdf["h3_cell"].apply(
        lambda c: int(counts_per_cell.get(c, 0)) + int(poly_counts_per_cell.get(c, 0))
    )
    hex_gdf["groups_label"] = hex_gdf["groups"].apply(lambda g: ", ".join(g) if g else "keine")

    return hex_gdf


def add_diversity_hexgrid(m: folium.Map, hex_gdf: gpd.GeoDataFrame) -> folium.Map:
    """Draw the diversity hex grid and its legend on the map."""
    folium.GeoJson(
        hex_gdf[["h3_cell", "diversity", "poi_count", "groups_label", "geometry"]],
        name="Anzahl Nutzungsarten",
        show=True,
        style_function=lambda f: {
            "fillColor": DIVERSITY_COLORS.get(f["properties"]["diversity"], DIVERSITY_COLORS[0]),
            "color": DIVERSITY_COLORS.get(f["properties"]["diversity"], DIVERSITY_COLORS[0]),
            "weight": 0,
            "fillOpacity": 0.55,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["diversity", "poi_count", "groups_label"],
            aliases=["Anzahl Nutzungsarten (0–4):", "Anzahl POIs:", "Vertretene Gruppen:"],
            localize=True,
        ),
    ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
    background:white;padding:10px 12px;border-radius:6px;
    box-shadow:2px 2px 6px rgba(0,0,0,0.3);
    font:13px 'Segoe UI',system-ui,sans-serif;">
    <b>Anzahl Nutzungsarten</b>
    """
    for i in range(5):
        legend_html += (
            f'<div><i style="background:{DIVERSITY_COLORS[i]};width:14px;height:14px;'
            f'display:inline-block;margin-right:6px;border:1px solid #999;"></i>'
            f"{i}</div>"
        )
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def add_poi_legend_by_group(m: folium.Map, pois: gpd.GeoDataFrame, legend_title: str) -> folium.Map:
    """Draw the POI legend grouped by daypart/weekday user group instead of a flat Typ list."""
    typ_info = pois.dropna(subset=["Typ"]).drop_duplicates(subset=["Typ"])[["Kategorie", "Typ", "emoji"]]

    entries_by_group = {group: [] for group in USER_GROUPS}
    for _, row in typ_info.iterrows():
        group = classify_user_group(row["Kategorie"], row["Typ"])
        entries_by_group[group].append((row["Typ"], row["emoji"]))

    legend_html = f"""
    <div style="position:fixed;bottom:40px;right:40px;z-index:9999;
    background:white;padding:10px 12px;border-radius:6px;
    box-shadow:2px 2px 6px rgba(0,0,0,0.3);
    font:13px 'Segoe UI',system-ui,sans-serif;max-width:270px;">
    <b>{legend_title}</b>
    """
    for group in USER_GROUPS:
        entries = entries_by_group.get(group, [])
        if not entries:
            continue
        legend_html += (
            f"<div style='margin-top:8px;font-weight:600;font-size:12px;color:#333;'>{group}</div>"
        )
        for typ, emoji in entries:
            legend_html += f"<div style='padding-left:4px'>{emoji} {typ}</div>"
    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def download_osm_pois(bounds: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Download cafes, restaurants, cultural/educational venues (museums, theatres, opera, libraries, etc.), park/meadow polygons, playgrounds and sports areas (pitches, sports centres, fitness stations) from OSM for the AOI."""
    print(
        "Downloading OSM POIs (cafes, restaurants, cultural/educational venues, "
        "parks/meadows, playgrounds, sports areas)..."
    )
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
    centroids = gdf.to_crs(utm_crs).geometry.centroid.to_crs(gdf.crs)

    # Wiese (leisure=park) keeps its full polygon geometry instead of being
    # collapsed to a point, so it can span and be counted in every hexagon
    # it overlaps, not just the one hexagon its centroid falls in.
    is_wiese = typ == "Wiese"
    geometry = gdf.geometry.where(is_wiese, centroids)

    osm_pois = gpd.GeoDataFrame(
        {
            "Kategorie": "OSM",
            "Typ": typ,
            "emoji": emoji,
            "name": gdf["name"] if "name" in gdf.columns else None,
            "lat": centroids.y,
            "lon": centroids.x,
        },
        geometry=geometry,
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

    # Shrink the AOI by 100m so both the displayed boundary and the hex grid
    # exclude the park's edge, which mostly covers surrounding streets.
    utm_crs = aoi.estimate_utm_crs()
    aoi_shrunk = gpd.GeoDataFrame(
        geometry=aoi.to_crs(utm_crs).geometry.buffer(-100).to_crs(4326),
        crs=4326,
    )

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

    hex_gdf = build_diversity_hexgrid(pois, aoi)
    hex_gdf = filter_hex_grid_by_intersection(hex_gdf, aoi_shrunk)

    m = add_diversity_hexgrid(m, hex_gdf)

    m = plot_helpers.general_map(
        m=m,
        aoi=aoi_shrunk,
        pois=pois,
        poi_column="Typ",
        poi_icon_column="emoji",
        legend_title=I18N[DEFAULT_LANG]["legend_title"],
        show_poi_legend=False,
        poi_group_name="POIs",
        show_aoi_outline=False,
    )
    m = add_poi_legend_by_group(m, pois, I18N[DEFAULT_LANG]["legend_title"])

    m.add_child(MeasureControl())
    folium.LayerControl(collapsed=False).add_to(m)

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
