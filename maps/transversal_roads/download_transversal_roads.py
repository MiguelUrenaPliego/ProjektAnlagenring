"""Download the main transversal (park-crossing) roads along the Anlagenring
from OSM and export them to transversal_roads.gpkg for manual review/editing.

Run this once to (re-)generate transversal_roads.gpkg, then hand-edit that
file (e.g. in QGIS) to fix misclassified or missing segments -- add/remove
rows, or change their "label"/"severity" attribute. plot_transversal_roads.py
reads the (possibly edited) gpkg rather than querying OSM itself, so manual
corrections persist across re-plots.

Two severity levels are distinguished:
  - "major": the two worst interruptions, at Friedberger Tor (Friedberger
    Landstraße / Konrad-Adenauer-Straße / Friedberger Tor itself) and at
    Eschenheimer Tor (Eschenheimer Tor / Große Eschenheimer Straße).
  - "minor": smaller streets that still carry significant traffic and cut
    across the park with long traffic-light phases, car noise and few good
    crossing options (Kaiserstraße, Taunustor, Junghofstraße, Peterstraße,
    Zeil, Allerheiligentor).

Roads running *parallel* to the park (e.g. "Eschenheimer Anlage",
"Friedberger Anlage") are deliberately excluded here even though their name
matches "Eschenheimer"/"Friedberger": they run alongside the park rather
than crossing it, and are already covered by the intersections map
(maps/intersections).
"""

import re
import sys
from pathlib import Path

import geopandas as gpd

HERE = Path(__file__).resolve().parent
MAPS_ROOT = HERE.parent
REPO_ROOT = MAPS_ROOT.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "routing" / "src"))

import osm  # noqa: E402
from park_geometry import load_park_polygons  # noqa: E402

AOI_PATH = MAPS_ROOT / "points_of_interest" / "aoi.gpkg"
OUTPUT_PATH = HERE / "transversal_roads.gpkg"

# (name regex for the OSM "name" tag, display label, severity). Order
# matters: the first matching pattern wins.
TRANSVERSAL_ROADS = [
    (r"^Friedberger Landstraße$|^Konrad-Adenauer-Straße$|^Friedberger Tor$", "Friedberger Tor", "major"),
    (r"^Eschenheimer Tor$|^Große Eschenheimer Straße$", "Eschenheimer Tor", "major"),
    (r"^Kaiserstraße$", "Kaiserstraße", "minor"),
    (r"^Taunustor$", "Taunustor", "minor"),
    (r"^Junghofstraße$", "Junghofstraße", "minor"),
    (r"^Peterstraße$", "Peterstraße", "minor"),
    (r"^Zeil$", "Zeil", "minor"),
    (r"^Allerheiligentor$", "Allerheiligentor", "minor"),
]

# Roads running alongside the park rather than across it -- excluded even
# though their name would otherwise match one of the patterns above (e.g.
# "Eschenheimer Anlage" contains "Eschenheimer").
EXCLUDE_NAME_PATTERN = r"Anlage"

NAME_PATTERN = "|".join(f"({pattern})" for pattern, _, _ in TRANSVERSAL_ROADS)

OSM_OVERPASS_QUERY = f"""[out:xml][timeout:120];
    (
        way["highway"]["name"~"{NAME_PATTERN}"]({{{{bbox}}}});
    );
    (._;>;);
    out body;
"""


def label_and_severity(name: str) -> tuple[str, str] | None:
    for pattern, label, severity in TRANSVERSAL_ROADS:
        if re.search(pattern, name or ""):
            return label, severity
    return None


def main():
    aoi = gpd.read_file(AOI_PATH)
    utm_crs = aoi.estimate_utm_crs()
    query_bounds = gpd.GeoDataFrame(
        geometry=aoi.to_crs(utm_crs).geometry.buffer(300).to_crs(4326), crs=4326
    )

    park = load_park_polygons()
    keep_bounds = gpd.GeoDataFrame(
        geometry=park.to_crs(utm_crs).geometry.buffer(60).to_crs(4326), crs=4326
    )

    print("Downloading OSM transversal roads (streets crossing the Anlagenring)...")
    roads = osm.overpass_api_query(OSM_OVERPASS_QUERY, query_bounds)

    if roads.empty:
        raise RuntimeError("No transversal roads found — check road names / AOI.")

    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])]
    roads = roads[~roads["name"].fillna("").str.contains(EXCLUDE_NAME_PATTERN)]
    roads = gpd.sjoin(roads, keep_bounds[["geometry"]], predicate="intersects").drop(columns="index_right")

    labels_severities = roads["name"].apply(label_and_severity)
    roads = roads[labels_severities.notna()]
    labels_severities = labels_severities[labels_severities.notna()]
    roads["label"] = [ls[0] for ls in labels_severities]
    roads["severity"] = [ls[1] for ls in labels_severities]
    roads = roads.dissolve(by=["label", "severity"], as_index=False)[["label", "severity", "geometry"]]

    print(roads[["label", "severity"]].to_string(index=False))

    found_labels = set(roads["label"])
    missing = {label for _, label, _ in TRANSVERSAL_ROADS} - found_labels
    if missing:
        print(f"Warning: no OSM segments found for: {', '.join(sorted(missing))}. "
              f"Add them manually to {OUTPUT_PATH.name} if needed.")

    roads.to_file(OUTPUT_PATH, driver="GPKG")
    print(f"Saved {len(roads)} road(s) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
