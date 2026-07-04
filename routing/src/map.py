"""Builds the interactive scenario-comparison Folium map.

Coordinated controls:
  1. AOI choropleth: a scenario dropdown (current / center / anlagenring /
     bahnhof) x a layer dropdown (time, distance, speed, co2, each metric's
     difference vs. that scenario's own filtered "current", people affected
     by a route that got >5min slower, and the busiest road's %Δ traffic
     for that AOI row) recolors a single GeoJson layer via injected Leaflet
     JS. A colorbar legend matching the actual colormap is always shown.
  2. Route explorer: click an AOI polygon to set the origin, click another
     to set the destination — draws the 4 scenario routes between them in
     different colors and shows a metrics summary table. Clicking also
     opens an info popup for that AOI.
  3. An "overall results" box (population-weighted means across all AOI
     pairs) for the selected scenario, always visible at the bottom.
  4. A "Show traffic increase (roads)" checkbox that overlays a raster PNG
     (per-road-edge %Δ traffic vs. current, for the selected scenario)
     underneath the AOI polygons (which stay underneath the route
     linestrings) — a raster overlay is far cheaper to ship in the page
     than the thousands of vector polygons a road-level or H3 layer would
     need, which is what made earlier versions slow to load. Z-order
     bottom-to-top: raster < AOI polygons < route linestrings/markers.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import rasterio.warp
from PIL import Image

import raster_utils
import scenario as scenario_module

_MAPS_ROOT = Path(__file__).resolve().parents[2] / "maps"
sys.path.insert(0, str(_MAPS_ROOT / "shared"))
from map_i18n import toggle_html  # noqa: E402

_I18N_CONTENT = json.loads((_MAPS_ROOT / "shared" / "i18n_content.json").read_text())

METRICS = ["time_min", "distance_km", "avg_speed_kmh", "co2_kg"]
# Base per-route units — every mean/diff layer's label is built from these,
# so it's explicit that the AOI-row value is a mean *across that row's
# routes*, not a single-route or per-edge figure.
METRIC_UNITS = {
    "time_min": "time per route (min)",
    "distance_km": "distance per route (km)",
    "avg_speed_kmh": "speed per route (km/h)",
    "co2_kg": "CO2 per route (kg)",
}
PCT_PEOPLE_AFFECTED_KEY = "pct_people_affected_5min"
BUSIEST_ROAD_KEY = "busiest_road_pct_change"
POPULATION_KEY = "population"
POPULATION_DENSITY_KEY = "population_density"
WORKPLACES_KEY = "workplaces"
WORKPLACE_DENSITY_KEY = "workplace_density"
# "Worst" layers: not a mean across a row's routes, but the single worst
# route out of that row (biggest time/distance increase, biggest speed
# drop) vs. the current baseline for the same origin/destination pairs.
WORST_TIME_KEY = "worst_time_increase"
WORST_DISTANCE_KEY = "worst_distance_increase"
WORST_SPEED_KEY = "worst_speed_decrease"
WORST_KEYS = [WORST_TIME_KEY, WORST_DISTANCE_KEY, WORST_SPEED_KEY]
DIVERGING_EXTRA_KEYS = {BUSIEST_ROAD_KEY, *WORST_KEYS}
# These layers use wider percentile bounds (vs. the 10th/90th every other
# sequential/diverging layer uses) — a % layer like "people affected",
# "traffic increase", or a "worst single route" layer needs to keep its
# real extremes visible, just with the single most extreme outlier
# edge/row clipped. Not forced symmetric around 0 either: each side is
# normalized against its own percentile bound (see _div_normalize).
WIDE_PERCENTILE_KEYS = {PCT_PEOPLE_AFFECTED_KEY, BUSIEST_ROAD_KEY, *WORST_KEYS}
WIDE_CMAP_PERCENTILE_LOW = 1
WIDE_CMAP_PERCENTILE_HIGH = 95
# The traffic-increase raster (per-road-edge, not per-AOI-row) uses its own,
# tighter percentile bounds — it has vastly more data points than an AOI
# choropleth layer, so a wider (99.5th) bound is needed to keep it from
# being washed out by a handful of extreme edges.
RASTER_CMAP_PERCENTILE_LOW = 0.5
RASTER_CMAP_PERCENTILE_HIGH = 99.5
NO_LAYER_KEY = "none"
LAYER_KEYS = (
    [POPULATION_KEY, POPULATION_DENSITY_KEY, WORKPLACES_KEY, WORKPLACE_DENSITY_KEY]
    + METRICS
    + [f"diff_{m}" for m in METRICS]
    + [PCT_PEOPLE_AFFECTED_KEY, BUSIEST_ROAD_KEY, *WORST_KEYS, NO_LAYER_KEY]
)
LAYER_LABELS = {
    NO_LAYER_KEY: "None",
    POPULATION_KEY: "Population",
    POPULATION_DENSITY_KEY: "Population density (/km²)",
    WORKPLACES_KEY: "Workplaces",
    WORKPLACE_DENSITY_KEY: "Workplace density (/km²)",
    **{m: f"{METRIC_UNITS[m]}" for m in METRICS},
    **{f"diff_{m}": f"Δ {METRIC_UNITS[m]}" for m in METRICS},
    PCT_PEOPLE_AFFECTED_KEY: "Pop. affected >5min (%)",
    BUSIEST_ROAD_KEY: "Traffic Δ, busiest road (%)",
    WORST_TIME_KEY: "Worst Δ time (min)",
    WORST_DISTANCE_KEY: "Worst Δ distance (km)",
    WORST_SPEED_KEY: "Worst Δ speed (km/h)",
}

# Percentile bounds used for every colormap's min/max instead of the raw
# data min/max, so a handful of outliers don't wash out the whole scale.
CMAP_PERCENTILE_LOW = 10
CMAP_PERCENTILE_HIGH = 90

SCENARIO_LABELS = {
    "current": "Current",
    "center": "Center closure",
    "anlagenring": "Anlagenring closure",
    "bahnhof": "Bahnhof closure",
}

# Fixed color stops shared by _seq_color/_div_color and the CSS legend
# gradients below, so the legend always matches the actual feature colors.
SEQ_STOPS = [
    (1.0, 1.0, 0.898),
    (0.996, 0.851, 0.463),
    (0.996, 0.549, 0.235),
    (0.902, 0.192, 0.161),
    (0.502, 0.0, 0.149),
]
DIV_STOPS = [
    (0.020, 0.188, 0.380),
    (0.969, 0.969, 0.969),
    (0.404, 0.0, 0.121),
]


def _rgb_to_hex(c) -> str:
    return f"#{int(round(c[0]*255)):02x}{int(round(c[1]*255)):02x}{int(round(c[2]*255)):02x}"


def _gradient_css(stops) -> str:
    hex_stops = [_rgb_to_hex(c) for c in stops]
    return "linear-gradient(to right, " + ", ".join(hex_stops) + ")"


SEQ_GRADIENT_CSS = _gradient_css(SEQ_STOPS)
DIV_GRADIENT_CSS = _gradient_css(DIV_STOPS)
SCENARIO_ROUTE_COLORS = {
    "current": "#555555",
    "center": "#e6194B",
    "anlagenring": "#4363d8",
    "bahnhof": "#3cb44b",
}

# Shared by every map this module builds: two mutually-exclusive
# background layers (the plain default basemap vs. a semi-transparent
# Google Hybrid satellite+labels overlay), toggled by a pair of checkboxes
# rather than folium's own radio-style base-layer control, so the look
# matches each map's other custom controls.
GOOGLE_HYBRID_TILES = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
GOOGLE_HYBRID_ATTR = "Google"


def _add_background_layers(m: folium.Map) -> tuple:
    """Add the two background TileLayers to ``m`` (map must be created
    with ``tiles=None``). Returns (current_tile_js_var, hybrid_tile_js_var)."""
    current_tile = folium.TileLayer(
        tiles="CartoDB positron", name="CartoDB positron", control=False, show=True,
    )
    current_tile.add_to(m)
    hybrid_tile = folium.TileLayer(
        tiles=GOOGLE_HYBRID_TILES, attr=GOOGLE_HYBRID_ATTR, name="Google Hybrid",
        control=False, show=False, opacity=0.6,
    )
    hybrid_tile.add_to(m)
    return current_tile.get_name(), hybrid_tile.get_name()


def _background_controls_html() -> str:
    """Inner HTML only (no wrapping box) — embedded inside each map's own
    existing control box ("part of the layer box"), not a separate one."""
    return """
      <hr style="margin:8px 0;">
      <b><span data-i18n="background_layers">Background</span></b><br>
      <label><input type="checkbox" id="bg-current-toggle" checked> CartoDB positron</label><br>
      <label><input type="checkbox" id="bg-hybrid-toggle"> Google Hybrid</label>
    """


def _scale_control_js(map_var: str) -> str:
    """Leaflet scale control placed top-left, next to (not stacked under)
    the zoom +/- buttons — relies on TOPLEFT_CONTROLS_ROW_CSS, which lays
    same-corner controls out horizontally instead of Leaflet's default
    vertical stack."""
    return f"""
    <script>
    window.addEventListener('load', function () {{
        L.control.scale({{position: 'topleft'}}).addTo({map_var});
    }});
    </script>
    """


# Leaflet stacks multiple controls in the same corner vertically by
# default, which would put a topleft scale bar directly under (hiding) the
# zoom buttons — this lays same-corner controls out in a row instead, so
# the scale bar lands to the right of the zoom buttons.
TOPLEFT_CONTROLS_ROW_CSS = """
<style>
.leaflet-top.leaflet-left { display: flex; flex-direction: row; align-items: flex-start; }
.leaflet-top.leaflet-left .leaflet-control { margin-top: 10px; margin-left: 10px; }
</style>
"""


def _background_controls_js(current_var: str, hybrid_var: str, map_var: str) -> str:
    """Standalone deferred script (not dependent on any other map-specific
    JS) wiring the two checkboxes above to ``map_var`` — checking one
    always unchecks (and hides) the other, so exactly one background is
    shown at a time."""
    return f"""
    <script>
    window.addEventListener('load', function () {{
        var currentTileLayer = {current_var};
        var hybridTileLayer = {hybrid_var};
        var mapObj = {map_var};

        function setBackground(which) {{
            if (which === 'hybrid') {{
                if (!mapObj.hasLayer(hybridTileLayer)) hybridTileLayer.addTo(mapObj);
                if (mapObj.hasLayer(currentTileLayer)) mapObj.removeLayer(currentTileLayer);
            }} else {{
                if (!mapObj.hasLayer(currentTileLayer)) currentTileLayer.addTo(mapObj);
                if (mapObj.hasLayer(hybridTileLayer)) mapObj.removeLayer(hybridTileLayer);
            }}
            document.getElementById('bg-current-toggle').checked = (which === 'current');
            document.getElementById('bg-hybrid-toggle').checked = (which === 'hybrid');
        }}

        document.getElementById('bg-current-toggle').addEventListener('change', function () {{
            setBackground(this.checked ? 'current' : 'hybrid');
        }});
        document.getElementById('bg-hybrid-toggle').addEventListener('change', function () {{
            setBackground(this.checked ? 'hybrid' : 'current');
        }});
    }});
    </script>
    """


def _seq_color(t: float) -> str:
    """Sequential light-yellow -> dark-red colormap for t in [0, 1]."""
    t = 0.0 if math.isnan(t) else min(max(t, 0.0), 1.0)
    stops = SEQ_STOPS
    n = len(stops) - 1
    pos = t * n
    i = min(int(pos), n - 1)
    frac = pos - i
    r = stops[i][0] + (stops[i + 1][0] - stops[i][0]) * frac
    g = stops[i][1] + (stops[i + 1][1] - stops[i][1]) * frac
    b = stops[i][2] + (stops[i + 1][2] - stops[i][2]) * frac
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _div_color(t: float) -> str:
    """Diverging blue -> white -> red colormap for t in [-1, 1]."""
    t = 0.0 if math.isnan(t) else min(max(t, -1.0), 1.0)
    if t <= 0:
        frac = t + 1.0
        c0, c1 = DIV_STOPS[0], DIV_STOPS[1]
    else:
        frac = t
        c0, c1 = DIV_STOPS[1], DIV_STOPS[2]
    r = c0[0] + (c1[0] - c0[0]) * frac
    g = c0[1] + (c1[1] - c0[1]) * frac
    b = c0[2] + (c1[2] - c0[2]) * frac
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _div_color_rgba(t: float) -> tuple:
    """Same as _div_color but returns an (r, g, b, 255) uint8 tuple, for
    rasterizing directly into a PNG without a hex round-trip."""
    if math.isnan(t):
        return (0, 0, 0, 0)
    t = min(max(t, -1.0), 1.0)
    if t <= 0:
        frac = t + 1.0
        c0, c1 = DIV_STOPS[0], DIV_STOPS[1]
    else:
        frac = t
        c0, c1 = DIV_STOPS[1], DIV_STOPS[2]
    r = c0[0] + (c1[0] - c0[0]) * frac
    g = c0[1] + (c1[1] - c0[1]) * frac
    b = c0[2] + (c1[2] - c0[2]) * frac
    return (int(r * 255), int(g * 255), int(b * 255), 255)


def _seq_range(vals: pd.Series, pct_low: float = None, pct_high: float = None) -> tuple:
    """[p_low, p_high] of the values (default p10/p90), used as the
    sequential colormap domain."""
    pct_low = CMAP_PERCENTILE_LOW if pct_low is None else pct_low
    pct_high = CMAP_PERCENTILE_HIGH if pct_high is None else pct_high
    clean = vals.dropna()
    if len(clean) == 0:
        return 0.0, 1.0
    lo = float(np.percentile(clean, pct_low))
    hi = float(np.percentile(clean, pct_high))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _div_absmax(vals, pct_high: float = None) -> float:
    """p_high (default p90) of |values|, used as the symmetric diverging
    colormap domain."""
    pct_high = CMAP_PERCENTILE_HIGH if pct_high is None else pct_high
    clean = pd.Series(vals).dropna() if not isinstance(vals, pd.Series) else vals.dropna()
    if len(clean) == 0:
        return 1.0
    absmax = float(np.percentile(np.abs(clean), pct_high))
    return absmax if absmax > 0 else 1.0


def _div_normalize(v: float, lo: float, hi: float) -> float:
    """Normalize a single value to [-1, 1] for _div_color, against
    *independent* lo/hi bounds (not forced symmetric around 0) — the
    positive and negative sides of a diverging scale like %Δ traffic
    increase usually have very different magnitudes, and mirroring them
    would either wash out the smaller side or clip the larger one."""
    if v >= 0:
        return 0.0 if hi <= 0 else min(v / hi, 1.0)
    return 0.0 if lo >= 0 else max(v / -lo, -1.0)


def _raster_scale_bounds(
    vals, exclude_zero: bool = False, pct_low: float = None, pct_high: float = None,
) -> tuple:
    """(p_low, p_high) of ``vals`` (default p1/p99, the same wide percentile
    bounds as the "% affected"/"traffic %Δ" AOI layers — see
    WIDE_CMAP_PERCENTILE_*) — each side clamped so it doesn't collapse to
    (or cross) zero.

    ``exclude_zero``: for the raster, most of the grid is exactly 0%
    (edges/pixels with literally no change) — including those in the
    percentile computation would pull p1/p99 toward 0 and wash out the
    color scale for the pixels that actually changed, so the raster call
    site drops exact zeros (along with NaN, already dropped) before
    computing the percentiles."""
    pct_low = WIDE_CMAP_PERCENTILE_LOW if pct_low is None else pct_low
    pct_high = WIDE_CMAP_PERCENTILE_HIGH if pct_high is None else pct_high
    clean = pd.Series(vals).dropna()
    if exclude_zero:
        clean = clean[clean != 0]
    if len(clean) == 0:
        return -1.0, 1.0
    lo = float(np.percentile(clean, pct_low))
    hi = float(np.percentile(clean, pct_high))
    lo = min(lo, -1e-6)
    hi = max(hi, 1e-6)
    return lo, hi


def _raster_band_to_rgba(band: np.ndarray) -> tuple:
    """Colorize a raster band (already reprojected to EPSG:3857 — that
    reprojection now happens once in main.py when the GeoTIFF is written,
    not on every map build) with the diverging colormap, clipped to the
    p1/p99 bounds of its own values (NaN -> transparent). Returns
    (rgba_uint8_array, vmin, vmax) — folium/branca PNG-encodes the array
    directly (no manual PNG step needed — this keeps the raster path
    dependency-free)."""
    band = band.astype(np.float64)
    vmin, vmax = _raster_scale_bounds(
        band.ravel(), exclude_zero=True,
        pct_low=RASTER_CMAP_PERCENTILE_LOW, pct_high=RASTER_CMAP_PERCENTILE_HIGH,
    )

    rgba = np.zeros((*band.shape, 4), dtype=np.uint8)
    finite = np.isfinite(band)
    vals = band[finite]

    # Vectorized color lookup via the same stop table _div_color_rgba uses;
    # each side of zero normalized against its own (not mirrored) bound.
    colors = np.array(
        [_div_color_rgba(_div_normalize(float(v), vmin, vmax)) for v in vals], dtype=np.uint8
    )
    rgba[finite] = colors

    return rgba, vmin, vmax


def _raster_click_points(band: np.ndarray, transform, src_crs) -> list:
    """Every finite (real, non-NaN) pixel's own (lat, lon, value) — this
    raster is only ~0.4% non-NaN (roads are 1-2px wide against a mostly
    transparent background), so shipping every real pixel directly is
    only ~60k points, not the full multi-million-pixel array.

    This replaces an earlier coarse-block-pooled grid: pooling many raw
    pixels into one output cell (even picking the single nearest-to-
    block-center pixel) meant a click's *position within* a block was
    ignored — two different, even opposite-signed, roads (e.g. a road
    gaining traffic right next to a parallel one losing it after a
    closure) can fall in the same block, silently reporting a click on
    one road as the other's value. Shipping every real pixel's exact
    position instead lets the browser do a true nearest-point search
    against the actual click location, with no block boundary to land on
    the wrong side of.
    """
    rows, cols = np.nonzero(np.isfinite(band))
    if len(rows) == 0:
        return []
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    lons, lats = rasterio.warp.transform(src_crs, "EPSG:4326", xs, ys)
    vals = band[rows, cols]
    return [
        [round(float(lat), 6), round(float(lon), 6), round(float(v), 1)]
        for lat, lon, v in zip(lats, lons, vals)
    ]


def build_map(
    choropleth: gpd.GeoDataFrame,
    od_matrices: Dict[str, gpd.GeoDataFrame],
    scenario_names: List[str],
    output_path: str,
    overall_stats: Optional[Dict[str, dict]] = None,
    affected_time_threshold_min: float = 5.0,
    traffic_raster_path: Optional[str] = None,
    route_simplify_tol: float = 5.0,  # meters — coarsened to keep routeLookup's inline JSON small
    route_coord_precision: int = 5,  # ~1m at these latitudes, still well under simplify_tol
    route_top_k_per_origin: int = 15,  # only the K highest-weight destinations per origin ship
    pop_lookup: Optional[dict] = None,
    workplace_lookup: Optional[dict] = None,
    closure_boundaries: Optional[Dict[str, object]] = None,
    closure_centroids: Optional[Dict[str, object]] = None,
    area_boundaries: Optional[Dict[str, object]] = None,
    worst_dest_lookup: Optional[Dict[str, dict]] = None,
) -> None:
    pop_lookup = pop_lookup or {}
    workplace_lookup = workplace_lookup or pop_lookup
    closure_boundaries = closure_boundaries or {}
    closure_centroids = closure_centroids or {}
    area_boundaries = area_boundaries or {}
    worst_dest_lookup = worst_dest_lookup or {}
    choropleth_wgs = choropleth.to_crs(4326)
    overall_stats = overall_stats or {}

    # ------------------------------------------------------------
    # Precompute Python-side colors for every (scenario, layer) pair so the
    # browser only ever does a property lookup, no math/CDN colormap lib.
    # Color-scale bounds are clipped to the 10th-90th percentile of the
    # data so a handful of outliers don't wash out the whole gradient.
    # ------------------------------------------------------------
    value_cols = [f"{s}_{k}" for s in scenario_names for k in LAYER_KEYS]
    color_cols = {}
    ranges = {}

    for s in scenario_names:
        for k in LAYER_KEYS:
            if k == NO_LAYER_KEY:
                continue
            col = f"{s}_{k}"
            if col not in choropleth_wgs.columns:
                continue
            vals = choropleth_wgs[col].astype(float)
            is_diff = k.startswith("diff_") or k in DIVERGING_EXTRA_KEYS
            wide_percentile = k in WIDE_PERCENTILE_KEYS

            if is_diff:
                if wide_percentile:
                    # Not forced symmetric: traffic increase's positive and
                    # negative sides usually have very different
                    # magnitudes, so each is normalized against its own p1/
                    # p99 bound (same asymmetric approach as the raster).
                    lo, hi = _raster_scale_bounds(vals)
                    colors = vals.apply(
                        lambda v: None if pd.isna(v) else _div_color(_div_normalize(v, lo, hi))
                    )
                    ranges[col] = [lo, hi]
                else:
                    absmax = _div_absmax(vals)
                    colors = vals.apply(
                        lambda v: None if pd.isna(v) else _div_color(v / absmax)
                    )
                    ranges[col] = [-absmax, absmax]
            else:
                if wide_percentile:
                    vmin, vmax = _seq_range(vals, WIDE_CMAP_PERCENTILE_LOW, WIDE_CMAP_PERCENTILE_HIGH)
                else:
                    vmin, vmax = _seq_range(vals)
                span = (vmax - vmin) or 1.0
                colors = vals.apply(
                    lambda v: None if pd.isna(v) else _seq_color((v - vmin) / span)
                )
                ranges[col] = [vmin, vmax]

            color_cols[f"color_{col}"] = colors

    for col, series in color_cols.items():
        choropleth_wgs[col] = series

    popup_cols = [
        c for c in ["id", "Name", "type", "source", "population", "workplaces", "node_lon", "node_lat"]
        if c in choropleth_wgs.columns
    ]
    keep_cols = popup_cols + ["geometry"] + value_cols + list(color_cols.keys())
    keep_cols = [c for c in dict.fromkeys(keep_cols) if c in choropleth_wgs.columns]
    geo_gdf = choropleth_wgs[keep_cols].copy()
    for col in value_cols:
        if col in geo_gdf.columns:
            geo_gdf[col] = geo_gdf[col].round(4)
    if "population" in geo_gdf.columns:
        geo_gdf["population"] = geo_gdf["population"].round(0)
    if "workplaces" in geo_gdf.columns:
        geo_gdf["workplaces"] = geo_gdf["workplaces"].round(0)

    geojson_data = json.loads(geo_gdf.to_json())

    # ------------------------------------------------------------
    # Traffic-increase raster overlay (optional): one small PNG per
    # scenario, reprojected + colorized in Python so the browser only ever
    # positions an <img>, no vector geometry shipped for this layer.
    # ------------------------------------------------------------
    traffic_overlays: Dict[str, dict] = {}
    if traffic_raster_path:
        with rasterio.open(traffic_raster_path) as src:
            src_crs = src.crs
            # The GeoTIFF is written directly in EPSG:3857 (Web Mercator —
            # see main.py) because that's the CRS Leaflet actually renders
            # in: an ImageOverlay is placed by linearly stretching the
            # image between two lat/lng corners *in Leaflet's own projected
            # space*, so a source grid that isn't itself linear in Web
            # Mercator comes out warped against the roads/basemap under it
            # (worse away from the equator) even though the corners line
            # up exactly. Old rasters not in 3857 still get reprojected
            # here as a fallback so this doesn't hard-fail on stale files.
            needs_reproject = src_crs is None or src_crs.to_epsg() != 3857

            for i in range(1, src.count + 1):
                band_name = src.descriptions[i - 1] or f"band{i}"
                if band_name not in scenario_names:
                    continue
                band = src.read(i)
                transform, height, width = src.transform, src.height, src.width
                if needs_reproject:
                    band, transform, _ = raster_utils.reproject(
                        band.astype(np.float64), src.transform, src_crs,
                        src_nodata=np.nan, dst_nodata=np.nan, dst_crs=3857,
                    )
                    height, width = band.shape

                rgba, vmin, vmax = _raster_band_to_rgba(band)

                # Corner points only (not densified edges): the source grid
                # is a plain rectangle in EPSG:3857, and Leaflet also
                # stretches the image between exactly these two corners in
                # its own Web Mercator space — using the exact rectangle
                # corners (rather than a bounding box of a densified,
                # curved reprojection) is what makes the two stretches
                # match pixel-for-pixel.
                left_m, top_m = transform * (0, 0)
                right_m, bottom_m = transform * (width, height)
                (left, right), (bottom, top) = rasterio.warp.transform(
                    "EPSG:3857", "EPSG:4326", [left_m, right_m], [bottom_m, top_m]
                )
                bounds = [[bottom, left], [top, right]]
                click_points = _raster_click_points(band, transform, "EPSG:3857")

                traffic_overlays[band_name] = {
                    "rgba": rgba,
                    "bounds": bounds,
                    "range": [vmin, vmax],
                    "click_points": click_points,
                }

    # ------------------------------------------------------------
    # Route lookup: simplified geometry + metrics per (origin,dest) pair,
    # per scenario. Built from the projected (input CRS) OD matrices.
    # ------------------------------------------------------------
    route_lookup: Dict[str, Dict[str, dict]] = {}

    for s in scenario_names:
        od = od_matrices.get(s)
        if od is None or len(od) == 0:
            continue

        # Route "weight" (0-100): the origin's traffic/importance (by
        # population) split among its destinations by workplaces share
        # (same gravity-split definition as scenario.pair_weight), then
        # rescaled 0-100 relative to this scenario's single busiest OD pair
        # so it reads as a relative importance score in the route explorer.
        raw_weight = scenario_module.pair_weight(od, pop_lookup, workplace_lookup)
        max_weight = float(raw_weight.max()) if len(raw_weight) > 0 else 0.0
        weight_0_100 = (raw_weight / max_weight * 100.0) if max_weight > 0 else raw_weight * 0.0

        od_simplified = od.copy()
        od_simplified["geometry"] = od_simplified.geometry.simplify(route_simplify_tol)
        od_wgs = od_simplified.to_crs(4326)
        od_wgs["weight_score"] = weight_0_100.to_numpy()
        od_wgs["origin_population"] = od_wgs["origin_id"].map(pop_lookup).fillna(0.0)
        od_wgs["destination_workplaces"] = od_wgs["destination_id"].map(workplace_lookup).fillna(0.0)

        # Keep only each origin's highest-weight destinations — most pairs
        # have near-zero weight and just bloat routeLookup's inline JSON
        # without being useful in the route explorer.
        od_wgs = (
            od_wgs.sort_values("weight_score", ascending=False)
            .groupby("origin_id", group_keys=False)
            .head(route_top_k_per_origin)
        )

        for row in od_wgs.itertuples(index=False):
            key = f"{row.origin_id}_{row.destination_id}"
            coords = [
                [round(y, route_coord_precision), round(x, route_coord_precision)]
                for x, y in row.geometry.coords
            ]
            entry = {
                "coords": coords,
                "time_min": row.time_min,
                "distance_km": row.distance_km,
                "avg_speed_kmh": row.avg_speed_kmh,
                "co2_kg": row.co2_kg,
                "weight": round(float(row.weight_score), 2),
                "origin_population": round(float(row.origin_population)),
                "destination_workplaces": round(float(row.destination_workplaces)),
            }
            route_lookup.setdefault(key, {})[s] = entry

    # ------------------------------------------------------------
    # Base map
    # ------------------------------------------------------------
    bounds = geo_gdf.total_bounds  # minx, miny, maxx, maxy
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=12, tiles=None)
    bg_current_var, bg_hybrid_var = _add_background_layers(m)

    # Raster overlays: ImageOverlay is non-interactive by default, so it
    # never blocks clicks on the AOI layer even though the AOI polygons are
    # drawn above it. Actual stacking order (raster < AOI < routes) is
    # forced via a negative z-index applied to its <img> element in the
    # deferred JS below, so it doesn't depend on DOM insertion order.
    traffic_overlay_vars: Dict[str, str] = {}
    if traffic_overlays:
        overlays_dir = os.path.join(os.path.dirname(os.path.abspath(output_path)), "overlays")
        os.makedirs(overlays_dir, exist_ok=True)
    for s, overlay in traffic_overlays.items():
        png_name = f"{s}.png"
        png_path = os.path.join(overlays_dir, png_name)
        Image.fromarray(overlay["rgba"], mode="RGBA").save(png_path)
        # Pass a dummy 1x1 array so folium's constructor doesn't try to
        # base64-inline the real raster, then overwrite .url with a
        # relative path to the PNG we just wrote to disk — folium only
        # skips inlining for image args it recognizes as an already-valid
        # URL (http/https/etc.), not a plain relative path, so we bypass
        # image_to_url entirely for the real image.
        img_overlay = folium.raster_layers.ImageOverlay(
            image=np.zeros((1, 1, 4), dtype=np.uint8),
            bounds=overlay["bounds"],
            name=f"Traffic increase ({s})",
            opacity=0.85,
        )
        img_overlay.url = f"overlays/{png_name}"
        img_overlay.add_to(m)
        traffic_overlay_vars[s] = img_overlay.get_name()

    def style_function(feature):
        # No layer active by default — plain outline, no fill (matches
        # featureStyle()'s 'none' branch in the JS below), until the user
        # picks a layer from the dropdown.
        return {
            "fillColor": "#000000",
            "color": "#333333",
            "weight": 1,
            "fillOpacity": 0,
        }

    geojson_layer = folium.GeoJson(
        geojson_data,
        name="AOI",
        style_function=style_function,
    )
    geojson_layer.add_to(m)
    geojson_var = geojson_layer.get_name()

    # City / suburban outline: a static (not scenario-dependent), no-fill
    # black outline around all Stadtteil rows ("city") / all Gemeinde rows
    # ("suburban"), thicker than the AOI layer's own 1px border so it reads
    # as a grouping outline rather than another row border.
    for area_key, geom in area_boundaries.items():
        if geom is None or geom.is_empty:
            continue
        area_data = json.loads(gpd.GeoSeries([geom], crs=4326).to_json())
        style = {"color": "#000000", "weight": 5}
        folium.GeoJson(
            area_data,
            name=f"{area_key.title()} outline",
            # interactive=False (also passed inside the style dict, which
            # Leaflet merges into the per-feature Path options) is critical
            # here: without it, this fillOpacity:0 layer still intercepts
            # every click across its whole area (Leaflet interactive Path
            # layers get pointer-events regardless of visible fill), which
            # silently ate every AOI click/route-selection click since this
            # outline sits on top of the AOI layer and covers most of the
            # map.
            style_function=lambda _f, style=style: {
                "fillOpacity": 0,
                "color": style["color"],
                "weight": style["weight"],
                "interactive": False,
            },
            interactive=False,
        ).add_to(m)

    # Closure-boundary overlay: the blocked-road area for the currently
    # selected scenario, drawn as a plain gray semi-transparent polygon
    # with a thick black border — one static GeoJson per scenario, shown/
    # hidden by the scenario dropdown (there's no data-driven coloring
    # here, so it doesn't need the AOI layer's per-scenario/layer color
    # lookup machinery).
    closure_boundary_vars: Dict[str, str] = {}
    closure_centroid_vars: Dict[str, str] = {}
    for s, geom in closure_boundaries.items():
        if geom is None or geom.is_empty:
            continue
        closure_data = json.loads(gpd.GeoSeries([geom], crs=4326).to_json())
        closure_layer = folium.GeoJson(
            closure_data,
            name=f"Closure area ({s})",
            # Non-interactive for the same reason as the area outlines
            # above — a decorative overlay sitting on top of the AOI layer
            # must never intercept clicks meant for it.
            style_function=lambda _f: {
                "fillColor": "#808080",
                "fillOpacity": 0.4,
                "color": "#000000",
                "weight": 4,
                "interactive": False,
            },
            interactive=False,
        )
        closure_layer.add_to(m)
        closure_boundary_vars[s] = closure_layer.get_name()

        centroid = closure_centroids.get(s)
        if centroid is not None and not centroid.is_empty:
            # "No entry" road sign — red circle with a white horizontal
            # bar (not the diagonal-slash prohibition emoji).
            prohibited_icon = folium.DivIcon(
                html='<div style="font-size:40px; line-height:1; text-align:center;">⛔</div>',
                icon_size=(40, 40),
                icon_anchor=(20, 20),
            )
            centroid_marker = folium.Marker(
                location=[centroid.y, centroid.x],
                icon=prohibited_icon,
                tooltip=f"No cars: {s} closure",
            )
            centroid_marker.add_to(m)
            closure_centroid_vars[s] = centroid_marker.get_name()

    route_group = folium.FeatureGroup(name="Selected routes")
    route_group.add_to(m)
    route_group_var = route_group.get_name()

    worst_routes_group = folium.FeatureGroup(name="Worst-route to slowest destination")
    worst_routes_group.add_to(m)
    worst_routes_group_var = worst_routes_group.get_name()

    marker_group = folium.FeatureGroup(name="Origin/destination markers")
    marker_group.add_to(m)
    marker_group_var = marker_group.get_name()

    map_var = m.get_name()

    # ------------------------------------------------------------
    # Controls + JS
    # ------------------------------------------------------------
    scenario_options_html = "".join(
        f'<option value="{s}">{SCENARIO_LABELS.get(s, s)}</option>' for s in scenario_names
    )
    layer_options_html = "".join(
        f'<option value="{k}"{" selected" if k == NO_LAYER_KEY else ""}>{LAYER_LABELS[k]}</option>'
        for k in LAYER_KEYS
    )

    traffic_checkbox_html = ""
    if traffic_overlay_vars:
        traffic_checkbox_html = """
      <hr style="margin:8px 0;">
      <label><input type="checkbox" id="traffic-raster-toggle"> Show traffic increase (roads)</label>
      <div id="traffic-legend" style="display:none; margin-top:4px;">
        <div style="font-size:11px; color:#333;"><span data-i18n="legend_title">Traffic %Δ vs. current (selected scenario)</span></div>
        <div id="traffic-legend-gradient" style="width:100%; height:12px; border:1px solid #999;"></div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:#333;">
          <span id="traffic-legend-min"></span>
          <span id="traffic-legend-max"></span>
        </div>
      </div>
        """

    controls_html = f"""
    <div id="scenario-controls" style="
        position: fixed; top: 10px; right: 10px; z-index: 9999;
        background: white; padding: 10px 12px; border: 2px solid #666;
        border-radius: 6px; font-size: 13px; width: 230px;">
      <b>Layer</b><br>
      <label><span data-i18n="scenario">Scenario</span></label>
      <select id="scenario-select" style="width:100%; margin-bottom:6px;">
        {scenario_options_html}
      </select>
      <label>Layer</label>
      <select id="layer-select" style="width:100%; margin-bottom:6px;">
        {layer_options_html}
      </select>
      <div id="legend-title" style="font-size:11px; color:#333; margin-bottom:2px;"></div>
      <div id="legend-gradient" style="width:100%; height:14px; border:1px solid #999;"></div>
      <div style="display:flex; justify-content:space-between; font-size:11px; color:#333;">
        <span id="legend-min"></span>
        <span id="legend-max"></span>
      </div>
      {traffic_checkbox_html}
      {_background_controls_html()}
    </div>

    <button id="route-explorer-activate-btn" style="
        position: fixed; bottom: 10px; left: 10px; z-index: 9999;
        padding: 8px 12px; border: 2px solid #666; border-radius: 6px;
        background: white; font-size: 13px; cursor: pointer;">
      Select route
    </button>

    <div id="route-controls" style="
        display:none; position: fixed; bottom: 10px; left: 10px; z-index: 9999;
        background: white; padding: 10px 12px; border: 2px solid #666;
        border-radius: 6px; font-size: 13px; width: 280px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <b>Route explorer</b>
        <button id="route-explorer-close-btn" title="Close" style="
            border:none; background:none; font-size:16px; line-height:1;
            cursor:pointer; padding:0 2px;">&times;</button>
      </div>
      <div style="margin-bottom:4px;">Click origin, then destination.</div>
      <div>Origin: <b id="origin-label" style="color:#1e88e5;">none</b></div>
      <div>Destination: <b id="destination-label" style="color:#e53935;">none</b></div>
      <button id="clear-route-btn" style="width:100%; margin:6px 0;">Clear</button>
      <div id="route-info"></div>
    </div>

    <div id="worst-route-legend" style="
        display:none; position: fixed; top: 90px; left: 10px; z-index: 9999;
        background: white; padding: 8px 12px; border: 2px solid #666;
        border-radius: 6px; font-size: 12px; max-width: 260px;">
    </div>

    <div id="overall-results" style="
        position: fixed; bottom: 10px; right: 10px; z-index: 9999;
        background: white; padding: 8px 12px; border: 2px solid #666;
        border-radius: 6px; font-size: 12px; max-width: 480px;">
      <b>Summary — <span id="overall-scenario-label"></span></b>
      <div id="overall-body"></div>
    </div>
    """ + TOPLEFT_CONTROLS_ROW_CSS

    m.get_root().html.add_child(folium.Element(controls_html))
    m.get_root().html.add_child(folium.Element(_background_controls_js(bg_current_var, bg_hybrid_var, map_var)))
    m.get_root().html.add_child(folium.Element(_scale_control_js(map_var)))

    js_code = f"""
    <script>
    // Deferred to the 'load' event: Folium's own map/layer-creation
    // <script> block is appended to the document AFTER this one, so the
    // geo_json_.../feature_group_.../map_... variables below don't exist
    // yet if this code ran immediately (that would throw a ReferenceError
    // and silently abort the whole block before any listener is attached).
    window.addEventListener('load', function () {{
        var routeLookup = {json.dumps(route_lookup)};
        var worstDestLookup = {json.dumps(worst_dest_lookup)};
        var scenarioRanges = {json.dumps(ranges)};
        var scenarioLabels = {json.dumps(SCENARIO_LABELS)};
        var layerLabels = {json.dumps(LAYER_LABELS)};
        var scenarioColors = {json.dumps(SCENARIO_ROUTE_COLORS)};
        var scenarioOrder = {json.dumps(scenario_names)};
        var seqGradientCss = {json.dumps(SEQ_GRADIENT_CSS)};
        var divGradientCss = {json.dumps(DIV_GRADIENT_CSS)};
        var overallStats = {json.dumps(overall_stats)};
        var affectedThreshold = {json.dumps(affected_time_threshold_min)};
        var trafficRanges = {json.dumps({s: o["range"] for s, o in traffic_overlays.items()})};
        // Every real (non-NaN) raster pixel's own [lat, lon, value], per
        // scenario — used for an exact nearest-pixel lookup on click (see
        // _raster_click_points in map.py for why this isn't a pooled grid).
        var trafficClickData = {json.dumps({
            s: {"points": o["click_points"]} for s, o in traffic_overlays.items()
        })};

        var geojsonLayer = {geojson_var};
        var trafficOverlays = {{{", ".join(f'"{s}": {v}' for s, v in traffic_overlay_vars.items())}}};
        var closureBoundaries = {{{", ".join(f'"{s}": {v}' for s, v in closure_boundary_vars.items())}}};
        var closureCentroidMarkers = {{{", ".join(f'"{s}": {v}' for s, v in closure_centroid_vars.items())}}};
        var routeGroup = {route_group_var};
        var worstRoutesGroup = {worst_routes_group_var};
        var markerGroup = {marker_group_var};
        var leafletMap = {map_var};

        var selected = {{ origin: null, destination: null }};
        var routeExplorerActive = false;
        var TRANSPARENT_FILL_OPACITY = 0.15; // AOI fill dimmed (not fully
            // hidden) while the traffic-increase raster overlay is shown,
            // so the roads underneath are visible but AOI borders/clicks
            // stay usable.

        function currentScenario() {{
            return document.getElementById('scenario-select').value;
        }}

        function currentLayer() {{
            return document.getElementById('layer-select').value;
        }}

        function trafficRasterVisible() {{
            var box = document.getElementById('traffic-raster-toggle');
            return !!(box && box.checked);
        }}

        function featureStyle(feature) {{
            var layer = currentLayer();
            if (layer === 'none') {{
                // Plain outline, no fill at all — for looking at the
                // basemap/raster underneath without any AOI cmap on top.
                return {{
                    fillColor: '#000000',
                    color: '#333333',
                    weight: 1,
                    fillOpacity: 0,
                }};
            }}
            var colorKey = 'color_' + currentScenario() + '_' + layer;
            var color = feature.properties[colorKey];
            var fillOpacity = color ? 0.75 : 0.15;
            if (trafficRasterVisible()) {{
                fillOpacity = Math.min(fillOpacity, TRANSPARENT_FILL_OPACITY);
            }}
            return {{
                fillColor: color || '#cccccc',
                color: '#333333',
                weight: 1,
                fillOpacity: fillOpacity,
            }};
        }}

        function applyStyles() {{
            geojsonLayer.setStyle(function (feature) {{
                var id = String(feature.properties.id);
                var base = featureStyle(feature);
                // Both origin and destination get the same black *dotted*
                // border — which one is which is shown by the distinct
                // start/end markers instead, not by border color. Dotted
                // (vs. the closure area's solid thick border) so the two
                // "black outline" meanings are never visually confused.
                if (id === selected.origin || id === selected.destination) {{
                    return Object.assign({{}}, base, {{ color: '#000000', weight: 3, dashArray: '2,6', fillOpacity: Math.max(base.fillOpacity, 0.35) }});
                }}
                return base;
            }});

            var scenario = currentScenario();
            var showTraffic = trafficRasterVisible();
            for (var s in trafficOverlays) {{
                var overlay = trafficOverlays[s];
                if (showTraffic && s === scenario) {{
                    if (!leafletMap.hasLayer(overlay)) overlay.addTo(leafletMap);
                    // Force the raster BELOW the AOI polygons/routes
                    // (negative z-index sinks it below "auto"-stacked
                    // siblings in the same pane) regardless of DOM
                    // insertion order. ImageOverlay stays non-interactive,
                    // so this never blocks clicks on the AOI layer above.
                    if (overlay._image) overlay._image.style.zIndex = -1;
                }} else if (leafletMap.hasLayer(overlay)) {{
                    leafletMap.removeLayer(overlay);
                }}
            }}

            var legendBox = document.getElementById('traffic-legend');
            if (legendBox) {{
                legendBox.style.display = showTraffic ? 'block' : 'none';
                if (showTraffic) {{
                    var rng = trafficRanges[scenario];
                    document.getElementById('traffic-legend-gradient').style.background = divGradientCss;
                    document.getElementById('traffic-legend-min').innerText = rng ? rng[0].toFixed(1) + '%' : '';
                    document.getElementById('traffic-legend-max').innerText = rng ? rng[1].toFixed(1) + '%' : '';
                }}
            }}

            // Only the closure boundary (+ its "no cars" marker) for the
            // currently selected scenario is shown (there's nothing to
            // show for "current", which has no closure).
            for (var cs in closureBoundaries) {{
                var boundaryLayer = closureBoundaries[cs];
                if (cs === scenario) {{
                    if (!leafletMap.hasLayer(boundaryLayer)) boundaryLayer.addTo(leafletMap);
                }} else if (leafletMap.hasLayer(boundaryLayer)) {{
                    leafletMap.removeLayer(boundaryLayer);
                }}
            }}
            for (var cm in closureCentroidMarkers) {{
                var centroidMarker = closureCentroidMarkers[cm];
                if (cm === scenario) {{
                    if (!leafletMap.hasLayer(centroidMarker)) centroidMarker.addTo(leafletMap);
                }} else if (leafletMap.hasLayer(centroidMarker)) {{
                    leafletMap.removeLayer(centroidMarker);
                }}
            }}
        }}

        function updateMarkers() {{
            markerGroup.clearLayers();
            // Start (origin) and end (destination) use different marker
            // shapes (circle vs. square pin), not different colors, to
            // clarify which is which independently of the AOI border color.
            [
                {{ id: selected.origin, label: 'Origin (start)', shape: 'circle' }},
                {{ id: selected.destination, label: 'Destination (end)', shape: 'square' }},
            ].forEach(function (entry) {{
                if (!entry.id) return;
                var layer = featureById(entry.id);
                if (!layer) return;
                var props = layer.feature.properties;
                if (props.node_lat === undefined || props.node_lat === null) return;

                var marker;
                if (entry.shape === 'circle') {{
                    marker = L.circleMarker([props.node_lat, props.node_lon], {{
                        radius: 8,
                        color: '#ffffff',
                        weight: 2,
                        fillColor: '#000000',
                        fillOpacity: 1,
                    }});
                }} else {{
                    var icon = L.divIcon({{
                        className: '',
                        html: '<div style="width:14px;height:14px;background:#000000;' +
                            'border:2px solid #ffffff;transform:rotate(45deg);"></div>',
                        iconSize: [14, 14],
                        iconAnchor: [7, 7],
                    }});
                    marker = L.marker([props.node_lat, props.node_lon], {{ icon: icon }});
                }}
                marker.bindTooltip(entry.label + ': ' + (props.Name || props.id)).addTo(markerGroup);
            }});
        }}

        function updateLegend() {{
            var scenario = currentScenario();
            var layer = currentLayer();

            document.getElementById('legend-title').innerText = layerLabels[layer] || layer;

            if (layer === 'none') {{
                document.getElementById('legend-gradient').style.background = 'none';
                document.getElementById('legend-min').innerText = '';
                document.getElementById('legend-max').innerText = '';
                return;
            }}

            var rangeKey = scenario + '_' + layer;
            var isDiff = layer.indexOf('diff_') === 0 || layer.indexOf('worst_') === 0 ||
                layer === 'busiest_road_pct_change';

            document.getElementById('legend-gradient').style.background =
                isDiff ? divGradientCss : seqGradientCss;

            var range = scenarioRanges[rangeKey];
            document.getElementById('legend-min').innerText = range ? range[0].toFixed(2) : '';
            document.getElementById('legend-max').innerText = range ? range[1].toFixed(2) : '';
        }}

        function updateOverallBox() {{
            var scenario = currentScenario();
            var stats = overallStats[scenario];
            document.getElementById('overall-scenario-label').innerText = scenarioLabels[scenario] || scenario;
            var box = document.getElementById('overall-body');
            if (!stats) {{
                box.innerHTML = '<i>No data.</i>';
                return;
            }}
            var rows =
                '<tr><td>Time/route</td><td>' + stats.time_min.toFixed(1) + ' min' +
                    (stats.diff_time_min ? ' (Δ ' + (stats.diff_time_min >= 0 ? '+' : '') + stats.diff_time_min.toFixed(2) + ')' : '') + '</td></tr>' +
                '<tr><td>Distance/route</td><td>' + stats.distance_km.toFixed(2) + ' km' +
                    (stats.diff_distance_km ? ' (Δ ' + (stats.diff_distance_km >= 0 ? '+' : '') + stats.diff_distance_km.toFixed(3) + ')' : '') + '</td></tr>' +
                '<tr><td>Speed/route</td><td>' + stats.avg_speed_kmh.toFixed(1) + ' km/h' +
                    (stats.diff_avg_speed_kmh ? ' (Δ ' + (stats.diff_avg_speed_kmh >= 0 ? '+' : '') + stats.diff_avg_speed_kmh.toFixed(2) + ')' : '') + '</td></tr>' +
                '<tr><td>CO2/route</td><td>' + stats.co2_kg.toFixed(3) + ' kg' +
                    (stats.diff_co2_kg ? ' (Δ ' + (stats.diff_co2_kg >= 0 ? '+' : '') + stats.diff_co2_kg.toFixed(4) + ')' : '') + '</td></tr>' +
                '<tr><td>Routes</td><td>' + stats.n_od_pairs.toLocaleString() + '</td></tr>' +
                '<tr><td>Routes >' + affectedThreshold + 'min slower</td><td>' + stats.n_routes_affected +
                    ' (' + stats.pct_routes_affected.toFixed(1) + '%)</td></tr>' +
                '<tr><td>Pop. affected (>' + affectedThreshold + 'min)</td><td>' +
                    stats.pct_people_affected.toFixed(1) + '%</td></tr>' +
                '<tr><td>Max traffic Δ (P99)</td><td>' +
                    (stats.max_traffic_increase_pct >= 0 ? '+' : '') + stats.max_traffic_increase_pct.toFixed(1) + '%</td></tr>' +
                '<tr><td>Max time increase (P99)</td><td>' +
                    (stats.max_time_increase_min >= 0 ? '+' : '') + stats.max_time_increase_min.toFixed(1) + ' min</td></tr>' +
                '<tr><td>Max distance increase (P99)</td><td>' +
                    (stats.max_distance_increase_km >= 0 ? '+' : '') + stats.max_distance_increase_km.toFixed(2) + ' km</td></tr>' +
                '<tr><td>Max speed decrease (P99)</td><td>' +
                    (stats.max_speed_decrease_kmh >= 0 ? '+' : '') + stats.max_speed_decrease_kmh.toFixed(1) + ' km/h</td></tr>' +
                '<tr><td>Avg. traffic Δ — city</td><td>' +
                    (stats.avg_traffic_increase_pct_city >= 0 ? '+' : '') + stats.avg_traffic_increase_pct_city.toFixed(1) + '%</td></tr>' +
                '<tr><td>Avg. traffic Δ — suburban</td><td>' +
                    (stats.avg_traffic_increase_pct_suburban >= 0 ? '+' : '') + stats.avg_traffic_increase_pct_suburban.toFixed(1) + '%</td></tr>';
            box.innerHTML = '<table style="border-collapse:collapse;">' + rows + '</table>';
        }}

        function updateChoropleth() {{
            applyStyles();
            updateLegend();
            updateOverallBox();
        }}

        document.getElementById('scenario-select').addEventListener('change', updateChoropleth);
        document.getElementById('layer-select').addEventListener('change', updateChoropleth);
        var trafficToggle = document.getElementById('traffic-raster-toggle');
        if (trafficToggle) {{
            trafficToggle.addEventListener('change', applyStyles);
        }}

        // Clicking the map while the traffic-increase raster is shown finds
        // the nearest *real* raster pixel (every non-NaN pixel's exact
        // lat/lon/value is shipped, see trafficClickData above) to the
        // click and shows its value — a true nearest-point search, not a
        // lookup into some coarser pre-pooled block that could belong to
        // a different, unrelated road.
        var MAX_CLICK_SNAP_DEG = 0.0009; // ~ this raster's own pixel size
        leafletMap.on('click', function (e) {{
            if (routeExplorerActive || !trafficRasterVisible()) return;
            var data = trafficClickData[currentScenario()];
            if (!data || !data.points || data.points.length === 0) return;

            var lat = e.latlng.lat, lng = e.latlng.lng;
            var bestValue = null, bestDist2 = Infinity;
            for (var i = 0; i < data.points.length; i++) {{
                var p = data.points[i];
                var dLat = p[0] - lat, dLng = p[1] - lng;
                var d2 = dLat * dLat + dLng * dLng;
                if (d2 < bestDist2) {{ bestDist2 = d2; bestValue = p[2]; }}
            }}
            var value = (bestValue !== null && Math.sqrt(bestDist2) <= MAX_CLICK_SNAP_DEG) ? bestValue : null;

            var content = value === null
                ? '<div style="font-size:12px;">No traffic-increase data at this point.</div>'
                : '<div style="font-size:12px;">Traffic %Δ: <b>' +
                    (value >= 0 ? '+' : '') + value.toFixed(1) + '%</b></div>';
            L.popup({{ maxWidth: 260 }}).setLatLng(e.latlng).setContent(content).openOn(leafletMap);
        }});

        updateChoropleth();

        var POPUP_ROAD_ROWS = [
            {{ key: 'time_min', label: 'Time (min)', digits: 1 }},
            {{ key: 'distance_km', label: 'Distance (km)', digits: 2 }},
            {{ key: 'avg_speed_kmh', label: 'Speed (km/h)', digits: 1 }},
            {{ key: 'co2_kg', label: 'CO2 (kg)', digits: 3 }},
            {{ key: 'worst_time_increase', label: 'Worst Δ time (min)', digits: 1, signed: true }},
            {{ key: 'worst_distance_increase', label: 'Worst Δ distance (km)', digits: 2, signed: true }},
            {{ key: 'worst_speed_decrease', label: 'Worst Δ speed (km/h)', digits: 1, signed: true }},
            {{ key: 'pct_people_affected_5min', label: 'Pop. affected >' + affectedThreshold + 'min (%)', digits: 1, suffix: '%' }},
            {{ key: 'busiest_road_pct_change', label: 'Busiest road Δ (%)', digits: 1, suffix: '%', signed: true }},
        ];

        function popupHtml(props) {{
            var html = '<div style="font-size:11px;">';
            html += '<b>' + (props.Name || ('AOI ' + props.id)) + '</b>' +
                '<button onclick="togglePopupTable(this)" title="Show/hide scenario comparison" style="' +
                'font-size:12px; padding:0 4px; margin-left:6px; cursor:pointer; border:1px solid #999; ' +
                'border-radius:3px; background:#fff;">📊</button><br>';
            if (props.type) html += 'Type: ' + props.type + (props.source ? ' (' + props.source + ')' : '') + '<br>';
            html += 'AOI id: ' + props.id + '<br>';
            if (props.population !== undefined && props.population !== null) {{
                html += 'Population: ' + Math.round(props.population).toLocaleString() + '<br>';
            }}
            if (props.workplaces !== undefined && props.workplaces !== null) {{
                html += 'Workplaces: ' + Math.round(props.workplaces).toLocaleString() + '<br>';
            }}

            // Scenario-comparison table is collapsed by default — toggled
            // by the emoji button above without closing the popup (or the
            // worst-route lines it opened with, which live in a separate
            // Leaflet layer group untouched by this).
            html += '<div class="popup-table-wrap" style="display:none;">';

            // One column per scenario (current + closures), one row per
            // road metric — lets the popup be compared at a glance instead
            // of having to flip the scenario dropdown back and forth.
            html += '<table style="font-size:11px; width:100%; border-collapse:collapse;">';
            html += '<tr><th style="text-align:left;"></th>';
            scenarioOrder.forEach(function (s) {{
                html += '<th style="text-align:right; color:' + (scenarioColors[s] || '#000') + ';">' +
                    (scenarioLabels[s] || s) + '</th>';
            }});
            html += '</tr>';

            POPUP_ROAD_ROWS.forEach(function (rowDef) {{
                var anyValue = scenarioOrder.some(function (s) {{
                    var v = props[s + '_' + rowDef.key];
                    return v !== undefined && v !== null;
                }});
                if (!anyValue) return;

                html += '<tr><td>' + rowDef.label + '</td>';
                scenarioOrder.forEach(function (s) {{
                    var v = props[s + '_' + rowDef.key];
                    if (v === undefined || v === null) {{
                        html += '<td style="text-align:right;">–</td>';
                        return;
                    }}
                    var text = (rowDef.signed && v >= 0 ? '+' : '') + v.toFixed(rowDef.digits) + (rowDef.suffix || '');
                    html += '<td style="text-align:right;">' + text + '</td>';
                }});
                html += '</tr>';
            }});
            html += '</table>';
            html += '</div>'; // end scenario-comparison collapsible

            html += '</div>';
            return html;
        }}

        // Inline onclick= handlers run in global scope, not this closure —
        // exposed on window so the popup's button (plain HTML, not a
        // Leaflet-bound listener) can find it.
        window.togglePopupTable = function (btn) {{
            var wrap = btn.parentElement.querySelector('.popup-table-wrap');
            if (!wrap) return;
            wrap.style.display = (wrap.style.display === 'none') ? 'block' : 'none';
        }};

        function updateSelectionLabels() {{
            var originLayer = selected.origin && featureById(selected.origin);
            var destLayer = selected.destination && featureById(selected.destination);
            document.getElementById('origin-label').innerText =
                originLayer ? (originLayer.feature.properties.Name || selected.origin) : 'none';
            document.getElementById('destination-label').innerText =
                destLayer ? (destLayer.feature.properties.Name || selected.destination) : 'none';
        }}

        function featureById(id) {{
            var found = null;
            geojsonLayer.eachLayer(function (layer) {{
                if (String(layer.feature.properties.id) === String(id)) found = layer;
            }});
            return found;
        }}

        function showRoute() {{
            routeGroup.clearLayers();
            document.getElementById('route-info').innerHTML = '';

            if (!selected.origin || !selected.destination) return;

            var key = selected.origin + '_' + selected.destination;
            var entry = routeLookup[key];

            if (!entry) {{
                document.getElementById('route-info').innerHTML = '<i>No route data for this pair.</i>';
                return;
            }}

            var bounds = [];
            var rows = '';

            scenarioOrder.forEach(function (scenario) {{
                var r = entry[scenario];
                if (!r) return;
                var color = scenarioColors[scenario] || '#000000';
                // A slightly wider black line drawn first, underneath the
                // colored line, fakes a border (Leaflet polylines have no
                // native stroke/outline option).
                L.polyline(r.coords, {{ color: '#000000', weight: 7, opacity: 0.85 }}).addTo(routeGroup);
                var line = L.polyline(r.coords, {{ color: color, weight: 4, opacity: 0.95 }});
                line.addTo(routeGroup);
                bounds = bounds.concat(r.coords);

                rows += '<tr>' +
                    '<td style="color:' + color + '; font-weight:bold;">' + scenarioLabels[scenario] + '</td>' +
                    '<td>' + r.time_min.toFixed(1) + '</td>' +
                    '<td>' + r.distance_km.toFixed(2) + '</td>' +
                    '<td>' + r.avg_speed_kmh.toFixed(1) + '</td>' +
                    '<td>' + r.co2_kg.toFixed(3) + '</td>' +
                    '</tr>';
            }});

            var anyRow = scenarioOrder.map(function (s) {{ return entry[s]; }}).find(Boolean);
            var originPopHtml = anyRow ?
                '<div>Origin population: <b>' + Math.round(anyRow.origin_population).toLocaleString() + '</b> · ' +
                'Destination workplaces: <b>' + Math.round(anyRow.destination_workplaces).toLocaleString() + '</b></div>'
                : '';

            document.getElementById('route-info').innerHTML =
                originPopHtml +
                '<table style="font-size:11px; width:100%; border-collapse:collapse;">' +
                '<tr><th></th><th>min</th><th>km</th><th>km/h</th><th>kg CO2</th></tr>' +
                rows + '</table>';

            if (bounds.length > 0) {{
                leafletMap.fitBounds(bounds);
            }}
        }}

        function clearWorstRoutes() {{
            worstRoutesGroup.clearLayers();
            document.getElementById('worst-route-legend').style.display = 'none';
        }}

        // Normal (non-route-explorer) mode: clicking an AOI polygon also
        // draws the route from that polygon to whichever destination is
        // its worst (biggest time-increase) route *in the currently
        // selected scenario* — always the "current" (no-closure) route to
        // that destination, plus the selected scenario's own route to the
        // same destination if a closure scenario is selected (so the two
        // are directly comparable), colored to match the scenario
        // dropdown, with a small legend.
        function showWorstRoutes(originId) {{
            worstRoutesGroup.clearLayers();
            var legendBox = document.getElementById('worst-route-legend');
            var boundsAcc = [];
            var legendRows = '';

            var scenario = currentScenario();
            var destId = (worstDestLookup[scenario] || {{}})[originId];
            if (destId === undefined || destId === null) {{
                legendBox.style.display = 'none';
                return;
            }}
            var entry = routeLookup[originId + '_' + destId];

            var scenariosToShow = scenario === 'current' ? ['current'] : ['current', scenario];
            scenariosToShow.forEach(function (s) {{
                var r = entry && entry[s];
                if (!r) return;
                var color = scenarioColors[s] || '#000000';
                L.polyline(r.coords, {{ color: '#000000', weight: 6, opacity: 0.7 }}).addTo(worstRoutesGroup);
                L.polyline(r.coords, {{ color: color, weight: 3, opacity: 0.95 }}).addTo(worstRoutesGroup);
                boundsAcc = boundsAcc.concat(r.coords);
                legendRows += '<div><span style="display:inline-block; width:10px; height:10px; ' +
                    'background:' + color + '; margin-right:4px;"></span>' +
                    (scenarioLabels[s] || s) + '</div>';
            }});

            if (legendRows) {{
                legendBox.innerHTML = '<b>Worst route (Δ time, ' + (scenarioLabels[scenario] || scenario) + ')</b>' + legendRows;
                legendBox.style.display = 'block';
            }} else {{
                legendBox.style.display = 'none';
            }}
            if (boundsAcc.length > 0) leafletMap.fitBounds(boundsAcc);
        }}

        geojsonLayer.on('click', function (e) {{
            if (!routeExplorerActive) {{
                // While the traffic-increase raster is shown *and* the
                // route explorer isn't active, a click reads the raster's
                // value instead (handled by the map-level click listener
                // below) rather than opening the AOI popup. Route
                // selection (below) always takes priority over this,
                // regardless of the raster checkbox — otherwise leaving
                // the raster on silently breaks route creation.
                if (trafficRasterVisible()) return;

                // A standalone L.popup (not layer.bindPopup) so nothing is
                // left attached to the feature: bindPopup registers
                // Leaflet's own permanent click-to-toggle behavior on the
                // layer, which would keep opening this popup on every
                // future click even after route-explorer mode is turned
                // on — a standalone popup has no such persistent listener.
                var popup = L.popup({{ maxWidth: 220, minWidth: 140 }})
                    .setLatLng(e.latlng)
                    .setContent(popupHtml(e.layer.feature.properties))
                    .openOn(leafletMap);
                // Closing the popup (the × button, pressing Escape, or
                // clicking elsewhere on the map, which auto-closes it)
                // also clears the worst-route lines/legend it opened with.
                popup.on('remove', clearWorstRoutes);
                showWorstRoutes(String(e.layer.feature.properties.id));
                return;
            }}

            worstRoutesGroup.clearLayers();
            document.getElementById('worst-route-legend').style.display = 'none';

            var id = String(e.layer.feature.properties.id);

            if (selected.origin === null) {{
                selected.origin = id;
            }} else if (selected.destination === null && id !== selected.origin) {{
                selected.destination = id;
            }} else {{
                selected.origin = id;
                selected.destination = null;
            }}

            applyStyles();
            updateSelectionLabels();
            updateMarkers();
            showRoute();
        }});

        function clearRouteSelection() {{
            selected.origin = null;
            selected.destination = null;
            applyStyles();
            updateMarkers();
            updateSelectionLabels();
            routeGroup.clearLayers();
            document.getElementById('route-info').innerHTML = '';
        }}

        document.getElementById('clear-route-btn').addEventListener('click', clearRouteSelection);

        document.getElementById('route-explorer-activate-btn').addEventListener('click', function () {{
            routeExplorerActive = true;
            worstRoutesGroup.clearLayers();
            document.getElementById('worst-route-legend').style.display = 'none';
            document.getElementById('route-explorer-activate-btn').style.display = 'none';
            document.getElementById('route-controls').style.display = 'block';
        }});

        document.getElementById('route-explorer-close-btn').addEventListener('click', function () {{
            routeExplorerActive = false;
            clearRouteSelection();
            document.getElementById('route-controls').style.display = 'none';
            document.getElementById('route-explorer-activate-btn').style.display = 'block';
        }});
    }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(js_code))

    m.get_root().html.add_child(folium.Element(toggle_html(_I18N_CONTENT["carfree"])))

    m.save(output_path)


REGION_MAP_FILL_KEYS = ("none", "population", "workplaces", "population_density", "workplace_density")
REGION_MAP_FILL_LABELS = {
    "none": "None",
    "population": "Population",
    "workplaces": "Workplaces",
    "population_density": "Population density (/km²)",
    "workplace_density": "Workplace density (/km²)",
}


def build_region_map(
    aoi: gpd.GeoDataFrame,
    output_path: str,
    city_union: Optional[object] = None,
    suburban_union: Optional[object] = None,
) -> None:
    """Builds the census/region overview map (population, workplaces, and
    their densities per AOI row) — a separate, much simpler map than
    build_map's scenario-comparison one: black-bordered AOI polygons with a
    popup of the census columns, an optional density fill picked from a
    dropdown (default: no fill), thicker/patterned (not colored) outlines
    marking the city and suburban areas, and the same background-layer
    checkboxes as the main map.
    """
    aoi = aoi.copy()
    area_km2 = aoi.geometry.area / 1e6
    aoi["population_density"] = (aoi["population"] / area_km2.replace(0, np.nan)).fillna(0.0)
    aoi["workplace_density"] = (aoi["workplaces"] / area_km2.replace(0, np.nan)).fillna(0.0)

    aoi_wgs = aoi.to_crs(4326)

    color_cols = {}
    ranges = {}
    for key in ("population", "workplaces", "population_density", "workplace_density"):
        vals = aoi_wgs[key].astype(float)
        vmin, vmax = _seq_range(vals)
        span = (vmax - vmin) or 1.0
        color_cols[f"color_{key}"] = vals.apply(lambda v: _seq_color((v - vmin) / span))
        ranges[key] = [vmin, vmax]
    for col, series in color_cols.items():
        aoi_wgs[col] = series

    popup_cols = [
        c for c in ["id", "Name", "type", "source", "population", "workplaces",
                    "population_density", "workplace_density"]
        if c in aoi_wgs.columns
    ]
    geo_gdf = aoi_wgs[popup_cols + ["geometry"] + list(color_cols.keys())].copy()
    geo_gdf["population"] = geo_gdf["population"].round(0)
    geo_gdf["workplaces"] = geo_gdf["workplaces"].round(0)
    geo_gdf["population_density"] = geo_gdf["population_density"].round(1)
    geo_gdf["workplace_density"] = geo_gdf["workplace_density"].round(1)
    geojson_data = json.loads(geo_gdf.to_json())

    bounds = geo_gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    m = folium.Map(location=center, zoom_start=12, tiles=None)
    bg_current_var, bg_hybrid_var = _add_background_layers(m)

    def style_function(feature):
        return {"fillColor": "#000000", "color": "#000000", "weight": 1, "fillOpacity": 0}

    geojson_layer = folium.GeoJson(geojson_data, name="AOI", style_function=style_function)
    geojson_layer.add_to(m)
    geojson_var = geojson_layer.get_name()

    # City/suburban outlines: same black color as the AOI borders, but a
    # thicker solid line for the city and a thicker dashed line (different
    # pattern *and* thickness, not color) for the suburban ring — see the
    # legend box below for what's what.
    for area_key, geom, style in (
        ("city", city_union, {"weight": 5}),
        ("suburban", suburban_union, {"weight": 4, "dashArray": "14,8"}),
    ):
        if geom is None or geom.is_empty:
            continue
        area_data = json.loads(gpd.GeoSeries([geom], crs=aoi.crs).to_crs(4326).to_json())
        folium.GeoJson(
            area_data,
            name=f"{area_key.title()} outline",
            style_function=lambda _f, style=style: {
                "fillOpacity": 0,
                "color": "#000000",
                "weight": style["weight"],
                **({"dashArray": style["dashArray"]} if "dashArray" in style else {}),
                "interactive": False,
            },
            interactive=False,
        ).add_to(m)

    map_var = m.get_name()

    fill_options_html = "".join(
        f'<option value="{k}"{" selected" if k == "none" else ""}>{REGION_MAP_FILL_LABELS[k]}</option>'
        for k in REGION_MAP_FILL_KEYS
    )

    controls_html = f"""
    <div id="region-map-controls" style="
        position: fixed; top: 10px; right: 10px; z-index: 9999;
        background: white; padding: 10px 12px; border: 2px solid #666;
        border-radius: 6px; font-size: 13px; width: 230px;">
      <b>Layer</b><br>
      <select id="fill-select" style="width:100%; margin-bottom:6px;">
        {fill_options_html}
      </select>
      <div id="legend-title" style="font-size:11px; color:#333; margin-bottom:2px;"></div>
      <div id="legend-gradient" style="width:100%; height:14px; border:1px solid #999;"></div>
      <div style="display:flex; justify-content:space-between; font-size:11px; color:#333;">
        <span id="legend-min"></span>
        <span id="legend-max"></span>
      </div>
      <hr style="margin:8px 0;">
      <b>Borders</b>
      <div style="font-size:11px; color:#333; margin-top:4px;">
        <div><span style="display:inline-block; width:22px; border-top:4px solid #000; vertical-align:middle;"></span> City</div>
        <div><span style="display:inline-block; width:22px; border-top:3px dashed #000; vertical-align:middle;"></span> Suburban</div>
      </div>
      {_background_controls_html()}
    </div>
    """ + TOPLEFT_CONTROLS_ROW_CSS

    m.get_root().html.add_child(folium.Element(controls_html))
    m.get_root().html.add_child(folium.Element(_background_controls_js(bg_current_var, bg_hybrid_var, map_var)))
    m.get_root().html.add_child(folium.Element(_scale_control_js(map_var)))

    js_code = f"""
    <script>
    window.addEventListener('load', function () {{
        var scenarioRanges = {json.dumps(ranges)};
        var fillLabels = {json.dumps(REGION_MAP_FILL_LABELS)};
        var seqGradientCss = {json.dumps(SEQ_GRADIENT_CSS)};
        var geojsonLayer = {geojson_var};
        var leafletMap = {map_var};

        function currentFill() {{
            return document.getElementById('fill-select').value;
        }}

        function applyStyle() {{
            var fill = currentFill();
            geojsonLayer.setStyle(function (feature) {{
                if (fill === 'none') {{
                    return {{ fillColor: '#000000', color: '#000000', weight: 1, fillOpacity: 0 }};
                }}
                var color = feature.properties['color_' + fill];
                return {{
                    fillColor: color || '#cccccc',
                    color: '#000000',
                    weight: 1,
                    fillOpacity: color ? 0.75 : 0.15,
                }};
            }});
        }}

        function updateLegend() {{
            var fill = currentFill();
            document.getElementById('legend-title').innerText = fillLabels[fill] || fill;
            if (fill === 'none') {{
                document.getElementById('legend-gradient').style.background = 'none';
                document.getElementById('legend-min').innerText = '';
                document.getElementById('legend-max').innerText = '';
                return;
            }}
            document.getElementById('legend-gradient').style.background = seqGradientCss;
            var range = scenarioRanges[fill];
            document.getElementById('legend-min').innerText = range ? range[0].toFixed(1) : '';
            document.getElementById('legend-max').innerText = range ? range[1].toFixed(1) : '';
        }}

        function update() {{
            applyStyle();
            updateLegend();
        }}

        document.getElementById('fill-select').addEventListener('change', update);
        update();

        function popupHtml(props) {{
            var html = '<div style="font-size:13px;">';
            html += '<b>' + (props.Name || ('AOI ' + props.id)) + '</b><br>';
            if (props.type) html += 'Type: ' + props.type + (props.source ? ' (' + props.source + ')' : '') + '<br>';
            if (props.population !== undefined && props.population !== null) {{
                html += 'Population: ' + Math.round(props.population).toLocaleString() + '<br>';
            }}
            if (props.workplaces !== undefined && props.workplaces !== null) {{
                html += 'Workplaces: ' + Math.round(props.workplaces).toLocaleString() + '<br>';
            }}
            if (props.population_density !== undefined && props.population_density !== null) {{
                html += 'Population density: ' + props.population_density.toLocaleString() + '/km²<br>';
            }}
            if (props.workplace_density !== undefined && props.workplace_density !== null) {{
                html += 'Workplace density: ' + props.workplace_density.toLocaleString() + '/km²<br>';
            }}
            html += '</div>';
            return html;
        }}

        geojsonLayer.on('click', function (e) {{
            L.popup({{ maxWidth: 320 }})
                .setLatLng(e.latlng)
                .setContent(popupHtml(e.layer.feature.properties))
                .openOn(leafletMap);
        }});
    }});
    </script>
    """

    m.get_root().html.add_child(folium.Element(js_code))

    m.get_root().html.add_child(folium.Element(toggle_html(_I18N_CONTENT["region"])))

    m.save(str(output_path))
    print(f"Region map saved -> {output_path}")
    print(f"Map saved -> {output_path}")
