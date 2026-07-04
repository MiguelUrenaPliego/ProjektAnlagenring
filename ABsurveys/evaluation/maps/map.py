# coding: utf-8
"""
map.py

Core engine to generate a highly customized Leaflet map utilizing Folium, with an 
independent, highly responsive floating dashboard overlay, advanced double-decker 
colormapped tooltips, uncertainty interval brackets, and side-by-side click popups 
with fullscreen image viewing capabilities.
"""

from __future__ import annotations
import json
import os
import folium
from branca.element import Element

def generate_custom_html_map(
    points_data: list[dict],
    unique_users: int,
    total_clicks: int,
    metrics_list: list[str],
    default_metric: str,
    output_path: str,
    has_trueskill: bool = True,
    has_streetscore: bool = True,
):
    """
    Generates a single self-contained HTML file (map.html) using Folium as a container, 
    but with a fully custom-engineered responsive dashboard interface built in HTML/CSS/JS.

    Args:
        points_data (list[dict]): List of points containing coordinates, image paths,
                                  and multi-model/multi-metric scores/uncertainties.
        unique_users (int): Total number of unique user IDs.
        total_clicks (int): Total number of survey answers (clicks).
        metrics_list (list[str]): List of metrics (e.g., ['walk', 'bike', 'stay']).
        default_metric (str): Default metric to active on load.
        output_path (str): File path to save the generated map.html.
    """
    # Initialize basic map centered at the average of coordinates
    lats = [p["y"] for p in points_data if p["y"] is not None]
    lons = [p["x"] for p in points_data if p["x"] is not None]
    
    avg_lat = sum(lats) / len(lats) if lats else 50.1109
    avg_lon = sum(lons) / len(lons) if lons else 8.6821
    
    # Base Folium Map (clean canvas, default layercontrol is hidden/not added)
    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=17,
        tiles=None,
        control_scale=True,
        maxZoom=24  # Set maximum zoom on the map to 24
    )
    
    # Add Google Hybrid Tile Layer
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid Map",
        name="Google Hybrid",
        opacity=0.65,
        max_zoom=24,         # <--- Let Leaflet zoom the layer up to level 24
        max_native_zoom=21,  # <--- Stop requesting new tiles at 21, stretching them instead
    ).add_to(m)

    # Convert our points_data to a clean JSON string for embedding
    points_json = json.dumps(points_data, ensure_ascii=False)
    metrics_json = json.dumps(metrics_list, ensure_ascii=False)

    # Load external map.css and map.js files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(current_dir, "map.css")
    js_path = os.path.join(current_dir, "map.js")

    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
    else:
        css_content = "/* map.css not found */"

    if os.path.exists(js_path):
        with open(js_path, "r", encoding="utf-8") as f:
            js_content = f.read()
    else:
        js_content = "/* map.js not found */"

    # Wrap the loaded CSS into style tags
    custom_css = f"<style>\n{css_content}\n</style>"

    # Minimalist HTML panels
    custom_html = f"""
    <!-- HTML Control Panel -->
    <div class="custom-control-panel" id="main-control-panel">
        <div class="panel-header">
            <span data-i18n="perception">perception</span>
            <button type="button" id="lang-toggle-btn" class="lang-toggle-btn" onclick="toggleLanguage()" style="margin-left:auto; cursor:pointer; border:none; border-radius:999px; padding:3px 10px; font-size:11px; font-weight:600; background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid rgba(56,189,248,0.3);">🌐 DE/EN</button>
        </div>

        <!-- Model Switch: built dynamically by JS based on available models -->
        <div class="switch-group" id="model-switch-group"></div>

        <!-- Display Mode Switch (score vs uncertainty) -->
        <div class="switch-group" id="mode-switch-group">
            <div class="switch-container" id="mode-switch" data-active="left" onclick="toggleMode()">
                <div class="switch-slider"></div>
                <div class="switch-option active" id="opt-score" data-i18n="score">score</div>
                <div class="switch-option" id="opt-uncertainty" data-i18n="uncertainty">uncertainty</div>
            </div>
        </div>

        <!-- Split Filters Checkboxes: built dynamically by JS if split column exists -->
        <div class="switch-group" id="split-filter-group" style="display: none;"></div>

        <!-- Metric Selector Buttons -->
        <div class="switch-group">
            <div class="switch-title" data-i18n="metric">metric</div>
            <div class="metrics-grid" id="metric-buttons-grid"></div>
        </div>

        <!-- Colormap Legend Card -->
        <div class="legend-card">
            <div class="legend-title">
                <span id="legend-min-title" data-i18n="bad">bad</span>
                <span id="legend-max-title" data-i18n="good">good</span>
            </div>
            <div class="legend-bar-container">
                <span class="legend-label" id="legend-min-val">0</span>
                <div class="legend-bar score-track" id="legend-color-bar"></div>
                <span class="legend-label" id="legend-max-val">10</span>
            </div>
        </div>
    </div>

    <!-- Survey Statistics Card (Bottom Left) -->
    <div class="survey-stats-panel">
        <div class="stats-item">
            <span class="stats-value">{total_clicks:,}</span>
            <span class="stats-label" data-i18n="clicks">clicks</span>
        </div>
        <div class="stats-item" style="border-left: 1px solid rgba(255,255,255,0.15); padding-left: 10px;">
            <span class="stats-value">{unique_users:,}</span>
            <span class="stats-label" data-i18n="respondents">respondents</span>
        </div>
    </div>

    <!-- Image Fullscreen Modal -->
    <div class="fullscreen-image-modal" id="fs-modal" onclick="closeFullscreen()">
        <span class="fullscreen-close-btn">&times;</span>
        <img id="fs-image" src="" alt="Fullscreen view" />
    </div>
    """

    # Wrap the loaded JS content and inject data
    js_content_processed = js_content.replace("'__DEFAULT_METRIC__'", f"'{default_metric}'")
    js_content_processed = js_content_processed.replace("__HAS_TRUESKILL__", "true" if has_trueskill else "false")
    js_content_processed = js_content_processed.replace("__HAS_STREETSCORE__", "true" if has_streetscore else "false")
    custom_js = f"""
    <script>
        // Data injected from Python
        const mapPoints = {points_json};
        const availableMetrics = {metrics_json};
        
        {js_content_processed}
    </script>
    """

    # Injecting CSS, HTML and JS directly into Folium Map elements
    m.get_root().header.add_child(Element(custom_css))
    m.get_root().html.add_child(Element(custom_html))
    m.get_root().html.add_child(Element(custom_js))

    # Save to path
    m.save(output_path)
    print(f"[MapEngine] Saved interactive perceptual map HTML to: {output_path}")
