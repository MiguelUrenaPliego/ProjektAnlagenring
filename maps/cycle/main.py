"""Build a folium map of a GPX bike ride colored by speed, with red circles
marking the points where the bike was essentially stopped (< 1 km/h),
sized by how long the stop lasted.

Reads activity_23022180489.gpx (no gpxpy dependency, parsed via ElementTree)
and writes cycle_map.html next to it.
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import branca.colormap as cm
import folium

HERE = Path(__file__).resolve().parent
MAPS_ROOT = HERE.parent
sys.path.insert(0, str(MAPS_ROOT / "shared"))
from map_i18n import toggle_html  # noqa: E402

I18N = json.loads((MAPS_ROOT / "shared" / "i18n_content.json").read_text(encoding="utf-8"))["cycle"]
DEFAULT_LANG = "de"  # server-baked language for the colormap caption/tooltips below

GPX_PATH = HERE / "activity_23022180489.gpx"
OUTPUT_PATH = HERE / "cycle_map.html"

SLOW_SPEED_KMH = 3.0  # threshold below which the bike counts as "stopped"
GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def parse_trackpoints(path: Path):
    tree = ET.parse(path)
    root = tree.getroot()
    points = []
    for trkpt in root.findall(".//gpx:trkpt", GPX_NS):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])
        time_el = trkpt.find("gpx:time", GPX_NS)
        time = datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
        points.append((lat, lon, time))
    return points


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}" if m else f"{s}s"


def main():
    points = parse_trackpoints(GPX_PATH)

    # Per-segment (between consecutive points) distance, time and speed.
    segments = []
    for (lat1, lon1, t1), (lat2, lon2, t2) in zip(points, points[1:]):
        dt = (t2 - t1).total_seconds()
        dist_m = haversine_m(lat1, lon1, lat2, lon2)
        speed_kmh = (dist_m / dt) * 3.6 if dt > 0 else 0.0
        segments.append(
            {
                "start": (lat1, lon1),
                "end": (lat2, lon2),
                "dt": dt,
                "speed_kmh": speed_kmh,
            }
        )

    speeds = [seg["speed_kmh"] for seg in segments]
    speeds_sorted = sorted(speeds)
    p5_idx = min(len(speeds_sorted) - 1, int(round(0.05 * (len(speeds_sorted) - 1))))
    p90_idx = min(len(speeds_sorted) - 1, int(round(0.9 * (len(speeds_sorted) - 1))))
    vmin, vmax = speeds_sorted[p5_idx], speeds_sorted[p90_idx]

    colormap = cm.LinearColormap(
        colors=["#d7191c", "#ffffbf", "#1a9641", "#2c7bb6"],
        vmin=vmin,
        vmax=vmax,
        caption=I18N[DEFAULT_LANG]["legend_title"],
    )

    center = [
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    ]
    fmap = folium.Map(location=center, zoom_start=15, tiles="cartodbpositron")

    for seg in segments:
        folium.PolyLine(
            locations=[seg["start"], seg["end"]],
            color=colormap(min(max(seg["speed_kmh"], vmin), vmax)),
            weight=4,
            opacity=0.9,
            tooltip=f"{seg['speed_kmh']:.1f} km/h",
        ).add_to(fmap)

    # Group consecutive slow segments into stop events, each summarized by
    # its total duration and the mean location of its points.
    stops = []
    current = []
    for seg in segments:
        if seg["speed_kmh"] < SLOW_SPEED_KMH:
            current.append(seg)
        elif current:
            stops.append(current)
            current = []
    if current:
        stops.append(current)

    if stops:
        max_duration = max(sum(seg["dt"] for seg in stop) for stop in stops)
    else:
        max_duration = 0

    MIN_RADIUS_M, MAX_RADIUS_M = 4, 25
    for stop in stops:
        total_dt = sum(seg["dt"] for seg in stop)
        lats = [seg["start"][0] for seg in stop] + [stop[-1]["end"][0]]
        lons = [seg["start"][1] for seg in stop] + [stop[-1]["end"][1]]
        loc = (sum(lats) / len(lats), sum(lons) / len(lons))

        radius = MIN_RADIUS_M
        if max_duration > 0:
            radius += (MAX_RADIUS_M - MIN_RADIUS_M) * (total_dt / max_duration)

        duration_text = format_duration(total_dt)

        folium.Circle(
            location=loc,
            radius=radius,
            color="#d7191c",
            weight=1,
            fill=True,
            fill_color="#d7191c",
            fill_opacity=0.55,
            tooltip=I18N[DEFAULT_LANG]["slow_tooltip"].format(time=duration_text),
            popup=I18N[DEFAULT_LANG]["slow_popup"].format(time=duration_text),
        ).add_to(fmap)

        folium.Marker(
            location=loc,
            icon=folium.DivIcon(
                html=f"""<div style="
                    color:#fff; font:700 11px 'Segoe UI',system-ui,sans-serif;
                    text-shadow:0 0 3px #000,0 0 3px #000;
                    white-space:nowrap; transform:translate(-50%,-50%);
                    pointer-events:none;">{duration_text}</div>"""
            ),
        ).add_to(fmap)

    colormap.add_to(fmap)

    # branca hardcodes the colormap legend to the top-right leaflet corner
    # with no position option, so relocate it to the bottom-right after render.
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
        }
        function updateCaption() {
            var caption = document.querySelector(".legend.leaflet-control .caption");
            if (caption && window.t) caption.textContent = window.t("legend_title");
        }
        window.addEventListener("maplangchange", updateCaption);
        // The branca colormap builds its SVG (incl. the caption text) via a
        // script that runs after this one, so poll briefly until it exists.
        var tries = 0;
        var poll = setInterval(function () {
            var caption = document.querySelector(".legend.leaflet-control .caption");
            if (caption || ++tries > 50) {
                clearInterval(poll);
                updateCaption();
            }
        }, 100);
    });
    </script>
    """
    fmap.get_root().html.add_child(folium.Element(legend_position_js))

    title_html = """
    <div id="mapTitle" data-i18n="title" style="
        position:fixed;top:16px;left:70px;z-index:9999;
        background:rgba(8,25,55,0.75);color:#fff;padding:8px 14px;
        border-radius:8px;font:600 15px 'Segoe UI',system-ui,sans-serif;
        border:1px solid rgba(255,255,255,0.35);"></div>
    """
    fmap.get_root().html.add_child(folium.Element(title_html))
    fmap.get_root().html.add_child(folium.Element(toggle_html(I18N, default=DEFAULT_LANG)))

    fmap.save(str(OUTPUT_PATH))
    print(f"Saved map to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
