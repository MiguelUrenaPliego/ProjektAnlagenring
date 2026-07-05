/* map.js */

// ============================================================================
// I18N (DE/EN language toggle)
// ============================================================================
const translations = {
    en: {
        perception: "perception",
        score: "score",
        uncertainty: "uncertainty",
        metric: "metric",
        clicks: "clicks",
        respondents: "respondents",
        bad: "bad",
        good: "good",
        low_unc: "low unc",
        high_unc: "high unc",
        diff_min: "ts < ss",
        diff_max: "ts > ss",
        streetscore: "streetscore",
        trueskill: "trueskill",
        difference: "difference",
        filter_splits: "filter splits",
        other: "Other"
    },
    de: {
        perception: "Wahrnehmung",
        score: "Bewertung",
        uncertainty: "Unsicherheit",
        metric: "Metrik",
        clicks: "Klicks",
        respondents: "Teilnehmende",
        bad: "schlecht",
        good: "gut",
        low_unc: "geringe Unsich.",
        high_unc: "hohe Unsich.",
        diff_min: "ts < ss",
        diff_max: "ts > ss",
        streetscore: "Streetscore",
        trueskill: "Trueskill",
        difference: "Differenz",
        filter_splits: "Splits filtern",
        other: "Sonstige"
    }
};

function resolveInitialLanguage() {
    try {
        const urlLang = new URLSearchParams(location.search).get('lang');
        if (urlLang === 'de' || urlLang === 'en') return urlLang;
        const storedLang = localStorage.getItem('site_lang');
        if (storedLang === 'de' || storedLang === 'en') return storedLang;
    } catch (e) {
        // localStorage/URLSearchParams may be unavailable in some contexts
    }
    return 'de';
}

let currentLang = resolveInitialLanguage();

// Generic lookup helper for dynamically-generated strings
function t(key) {
    const dict = translations[currentLang] || translations.en;
    if (Object.prototype.hasOwnProperty.call(dict, key)) return dict[key];
    return (translations.en[key] !== undefined) ? translations.en[key] : key;
}

// Sweep the DOM for any [data-i18n] element and set its text from the current dict
function applyTranslations() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        el.textContent = t(key);
    });
    const langBtn = document.getElementById('lang-toggle-btn');
    if (langBtn) langBtn.textContent = currentLang === 'de' ? '🌐 DE/EN' : '🌐 EN/DE';
}

function setLanguage(lang) {
    if (lang !== 'de' && lang !== 'en') lang = 'en';
    currentLang = lang;
    try {
        localStorage.setItem('site_lang', lang);
    } catch (e) {
        // ignore storage errors (e.g. privacy mode)
    }
    applyTranslations();
    // Re-render dynamic UI pieces whose strings are set imperatively (not via data-i18n sweep)
    if (typeof updateLegend === 'function' && markersGroup) updateLegend();
}

function toggleLanguage() {
    setLanguage(currentLang === 'de' ? 'en' : 'de');
}

// Active selections state
let currentModel = 'streetscore'; // 'streetscore', 'trueskill', or 'difference'
let currentMode = 'score'; // 'score' or 'uncertainty'
let currentMetric = '__DEFAULT_METRIC__'; // Will be replaced in python
const hasTrueskill = __HAS_TRUESKILL__;
const hasStreetscore = __HAS_STREETSCORE__;

// Active splits filter state
let uniqueSplits = []; // all unique splits found in the dataset
let activeSplits = {}; // splitName -> boolean

// Dynamic limits for colormaps
let diffLimit = 1.0;
let uncertaintyMin = 0.0;
let uncertaintyLimit = 3.0;

function getPercentile(arr, q) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
        return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
    } else {
        return sorted[base];
    }
}

function getDiffValue(point, metric) {
    const metricData = point.metrics[metric];
    if (!metricData || !metricData.trueskill || !metricData.streetscore) return null;
    const ts = metricData.trueskill.score;
    const ss = metricData.streetscore.score;
    if (ts === null || ts === undefined || ss === null || ss === undefined) return null;
    return ts - ss;
}

function calculateDiffBounds() {
    const diffs = [];
    mapPoints.forEach(point => {
        const val = getDiffValue(point, currentMetric);
        if (val !== null && !isNaN(val)) {
            diffs.push(val);
        }
    });
    if (diffs.length > 0) {
        const p10 = getPercentile(diffs, 0.10);
        const p90 = getPercentile(diffs, 0.90);
        const limit = Math.max(Math.abs(p10), Math.abs(p90));
        diffLimit = limit > 0 ? limit : 1.0;
    } else {
        diffLimit = 1.0;
    }
}

function calculateUncertaintyBounds() {
    const uncs = [];
    mapPoints.forEach(point => {
        // Find uncertainty for the current model & metric
        const metricData = point.metrics[currentMetric];
        if (metricData && metricData[currentModel]) {
            const val = metricData[currentModel].uncertainty;
            if (val !== null && !isNaN(val)) {
                uncs.push(val);
            }
        }
    });
    if (uncs.length > 0) {
        const p10 = getPercentile(uncs, 0.10);
        const p90 = getPercentile(uncs, 0.90);
        uncertaintyMin = p10;
        uncertaintyLimit = p90 > p10 ? p90 : p10 + 1.0;
    } else {
        uncertaintyMin = 0.0;
        uncertaintyLimit = 3.0;
    }
}

// Color helper functions
function getScoreColor(val) {
    if (val === null || val === undefined || isNaN(val)) return '#64748b';
    val = Math.max(0, Math.min(10, val));
    let r, g, b;
    if (val < 5) {
        let ratio = val / 5;
        r = 239;
        g = Math.round(68 + (204 - 68) * ratio);
        b = Math.round(68 + (21 - 68) * ratio);
    } else {
        let ratio = (val - 5) / 5;
        r = Math.round(250 + (34 - 250) * ratio);
        g = Math.round(204 + (197 - 204) * ratio);
        b = Math.round(21 + (94 - 21) * ratio);
    }
    return `rgb(${r}, ${g}, ${b})`;
}

function getUncertaintyColor(val) {
    if (val === null || val === undefined || isNaN(val)) return '#64748b';
    const minVal = typeof uncertaintyMin !== 'undefined' ? uncertaintyMin : 0;
    val = Math.max(minVal, Math.min(uncertaintyLimit, val));
    const denom = uncertaintyLimit - minVal;
    let ratio = denom > 0 ? (val - minVal) / denom : 0;
    let r = Math.round(186 + (2 - 186) * ratio);
    let g = Math.round(230 + (132 - 230) * ratio);
    let b = Math.round(253 + (199 - 253) * ratio);
    return `rgb(${r}, ${g}, ${b})`;
}

function getDiffColor(val) {
    if (val === null || val === undefined || isNaN(val)) return '#64748b';
    // Clamp to symmetric bounds [-diffLimit, diffLimit]
    const clamped = Math.max(-diffLimit, Math.min(diffLimit, val));
    
    let r, g, b;
    if (clamped < 0) {
        // Interpolate between Dark Blue rgb(26, 54, 153) and White rgb(255, 255, 255)
        const ratio = (clamped - (-diffLimit)) / diffLimit; // 0 (dark blue) to 1 (white)
        r = Math.round(26 + (255 - 26) * ratio);
        g = Math.round(54 + (255 - 54) * ratio);
        b = Math.round(153 + (255 - 153) * ratio);
    } else {
        // Interpolate between White rgb(255, 255, 255) and Dark Yellow rgb(180, 140, 10)
        const ratio = clamped / diffLimit; // 0 (white) to 1 (dark yellow)
        r = Math.round(255 + (180 - 255) * ratio);
        g = Math.round(255 + (140 - 255) * ratio);
        b = Math.round(255 + (10 - 255) * ratio);
    }
    return `rgb(${r}, ${g}, ${b})`;
}

function getDiffTextColor(val) {
    if (val === null || val === undefined || isNaN(val)) return '#ffffff';
    if (Math.abs(val) < (diffLimit * 0.25)) {
        return '#0f172a';
    }
    return '#ffffff';
}

// Reference to the map object initialized by Folium
let mapObject = null;
let markersGroup = null;

// Initialize markers on Leaflet
function initLeafletOverlays() {
    const mapContainer = document.querySelector('.folium-map');
    if (!mapContainer) {
        setTimeout(initLeafletOverlays, 100);
        return;
    }
    const mapContainerId = mapContainer.id;
    mapObject = window[mapContainerId];
    
    if (!mapObject) {
        setTimeout(initLeafletOverlays, 100);
        return;
    }

    // Disable standard layer controls to avoid layout clash
    const defaultControls = document.querySelectorAll('.leaflet-control-layers');
    defaultControls.forEach(ctrl => ctrl.style.display = 'none');

    markersGroup = L.featureGroup().add_to ? L.featureGroup().add_to(mapObject) : L.featureGroup().addTo(mapObject);
    renderMarkers();
}

// Get value for current state
function getPointValue(point, model, mode, metric) {
    if (model === 'difference') {
        if (!hasTrueskill || !hasStreetscore) return null;
        return getDiffValue(point, metric);
    }
    if (model === 'streetscore' && !hasStreetscore) return null;
    if (model === 'trueskill' && !hasTrueskill) return null;
    
    const metricData = point.metrics[metric];
    if (!metricData || !metricData[model]) return null;
    return metricData[model][mode];
}

// Draw geometric arrows with bearing
function getArrowSvg(color, bearing) {
    return `
        <svg width="24" height="24" viewBox="0 0 24 24" style="transform: rotate(${bearing}deg); overflow: visible;">
            <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="1" stdDeviation="1" flood-opacity="0.5"/>
            </filter>
            <path d="M12,2 L19,18 L12,14 L5,18 Z" fill="black" stroke="black" stroke-width="2" filter="url(#shadow)"/>
            <path d="M12,3 L18,17 L12,13 L6,17 Z" fill="${color}" stroke="${color}" stroke-width="1"/>
        </svg>
    `;
}

// Format a number safely
function fmt(v) {
    if (v === null || v === undefined) return '—';
    return Number(v).toFixed(2);
}

// Generate Minimalist Double-Decker Comparison Tooltip Layout
function generateTooltipHtml(point, targetMetric) {
    const data = point.metrics[targetMetric] || {};
    const ss = data.streetscore || { score: null, uncertainty: null };
    const ts = data.trueskill || { score: null, uncertainty: null, n_answers: null };

    // Calculate percentage positions for markers (0 to 10 scale map to 0% to 100%)
    const ssPos = ss.score !== null ? (ss.score * 10) : 50;
    const tsPos = ts.score !== null ? (ts.score * 10) : 50;

    // Uncertainty bounds for brackets
    const ssUnc = ss.uncertainty !== null ? ss.uncertainty : 0;
    const tsUnc = ts.uncertainty !== null ? ts.uncertainty : 0;

    const ssLeft = Math.max(0, (ss.score - ssUnc) * 10);
    const ssWidth = Math.min(100 - ssLeft, (ssUnc * 2) * 10);

    const tsLeft = Math.max(0, (ts.score - tsUnc) * 10);
    const tsWidth = Math.min(100 - tsLeft, (tsUnc * 2) * 10);

    const diffVal = getDiffValue(point, targetMetric);
    const diffStr = diffVal !== null ? `${diffVal > 0 ? '+' : ''}${fmt(diffVal)}` : '—';
    const diffBg = diffVal !== null ? getDiffColor(diffVal) : 'transparent';
    const diffTextColor = diffVal !== null ? getDiffTextColor(diffVal) : '#ffffff';

    const splitBadgeHtml = point.split ? `
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 9px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 4px; font-family: sans-serif;">
            <span style="color: #94a3b8; font-weight: 600;">split</span>
            <span style="background: rgba(56, 189, 248, 0.1); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.2); padding: 1px 4px; border-radius: 3px; font-weight: 700; font-size: 8px; text-transform: uppercase;">${point.split}</span>
        </div>
    ` : '';

    return `
        <div class="custom-tooltip-wrapper">
            <div class="tooltip-header">
                <span>id: ${point.id}</span>
                <span style="text-transform: uppercase; font-size: 9px; color: #38bdf8; font-weight: 700;">${targetMetric}</span>
            </div>

            ${splitBadgeHtml}

            ${(hasTrueskill && hasStreetscore && diffVal !== null) ? `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 9px; font-weight: 700; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 4px;">
                <span style="color: #94a3b8;">difference (ts - ss)</span>
                <span style="background: ${diffBg}; color: ${diffTextColor}; padding: 1px 4px; border-radius: 3px; font-family: monospace;">
                    ${diffStr}
                </span>
            </div>
            ` : ''}

            <!-- STREETSCORE ROW -->
            ${hasStreetscore ? `
            <div class="comparison-bar-group">
                <div class="bar-label-row">
                    <span>streetscore</span>
                    <span class="bar-label-highlight">
                        ${ss.score !== null ? `${fmt(ss.score)} ± ${fmt(ssUnc)}` : 'no pred'}
                    </span>
                </div>
                <div class="gradient-track-wrapper">
                    <span style="position: absolute; left: 0; bottom: 12px; font-size: 7px; color: #64748b;">0</span>
                    <div class="gradient-track score-track"></div>
                    <span style="position: absolute; right: 0; bottom: 12px; font-size: 7px; color: #64748b;">10</span>
                    ${ss.score !== null ? `
                    <div class="uncertainty-bracket" style="left: ${ssLeft}%; width: ${ssWidth}%;"></div>
                    <div class="score-marker-pin" style="left: ${ssPos}%;">
                        <div class="score-marker-bubble">${fmt(ss.score)}</div>
                        <div class="score-marker-line"></div>
                    </div>
                    ` : ''}
                </div>
            </div>
            ` : ''}

            <!-- TRUESKILL ROW -->
            ${hasTrueskill ? `
            <div class="comparison-bar-group" style="${hasStreetscore ? 'border-top: 1px solid rgba(255,255,255,0.05); padding-top: 6px; margin-top: 4px;' : ''}">
                <div class="bar-label-row">
                    <span>trueskill</span>
                    <span class="bar-label-highlight">
                        ${ts.score !== null ? `${fmt(ts.score)} ± ${fmt(tsUnc)}` : 'no data'}
                    </span>
                </div>
                <div class="gradient-track-wrapper">
                    <span style="position: absolute; left: 0; bottom: 12px; font-size: 7px; color: #64748b;">0</span>
                    <div class="gradient-track score-track"></div>
                    <span style="position: absolute; right: 0; bottom: 12px; font-size: 7px; color: #64748b;">10</span>
                    ${ts.score !== null ? `
                    <div class="uncertainty-bracket" style="left: ${tsLeft}%; width: ${tsWidth}%;"></div>
                    <div class="score-marker-pin" style="left: ${tsPos}%;">
                        <div class="score-marker-bubble">${fmt(ts.score)}</div>
                        <div class="score-marker-line"></div>
                    </div>
                    ` : ''}
                </div>
            </div>
            ` : ''}

            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 4px;">
                <span class="answers-count">
                    ${ts.n_answers !== null ? `${ts.n_answers} answers` : '0 answers'}
                </span>
                <span style="font-size: 8px; color: #475569;">click to zoom</span>
            </div>
        </div>
    `;
}

// Generate Side-by-Side Detailed Click Popup Layout
function generatePopupHtml(point, targetMetric) {
    const leftTooltip = generateTooltipHtml(point, targetMetric);
    const imgPath = point.img_path;

    return `
        <div class="popup-split-container">
            <div class="popup-left-info">
                ${leftTooltip}
            </div>
            <div class="popup-right-img" onclick="openFullscreen('${imgPath}')">
                <img src="${imgPath}" class="popup-img-tag" onerror="this.src='https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?q=80&w=600&auto=format&fit=crop'" />
                <div class="popup-fullscreen-hint">zoom image</div>
            </div>
        </div>
    `;
}

// Render markers based on active choices
function renderMarkers() {
    if (!markersGroup) return;
    markersGroup.clearLayers();

    // Recalculate dynamic colormap bounds
    calculateDiffBounds();
    calculateUncertaintyBounds();
    updateLegend();

    mapPoints.forEach(point => {
        if (point.x === null || point.y === null) return;

        // Apply Split Filtering if splits are available
        if (uniqueSplits.length > 0) {
            const ptSplit = point.split || 'Other';
            if (activeSplits[ptSplit] === false) {
                return; // skipped as unchecked
            }
        }

        const val = getPointValue(point, currentModel, currentMode, currentMetric);
        if (val === null) return;

        let markerColor;
        if (currentModel === 'difference') {
            markerColor = getDiffColor(val);
        } else if (currentMode === 'score') {
            markerColor = getScoreColor(val);
        } else {
            markerColor = getUncertaintyColor(val);
        }

        let markerElement;

        if (point.bearing !== null && point.bearing !== undefined && !isNaN(point.bearing)) {
            const arrowIcon = L.divIcon({
                html: getArrowSvg(markerColor, point.bearing),
                className: '',
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            });
            
            markerElement = L.marker([point.y, point.x], {
                icon: arrowIcon
            });
        } else {
            markerElement = L.circleMarker([point.y, point.x], {
                radius: 6,
                fillColor: markerColor,
                color: '#000000',
                weight: 1.2,
                opacity: 1,
                fillOpacity: 0.9
            });
        }

        markerElement.bindTooltip(generateTooltipHtml(point, currentMetric), {
            direction: 'top',
            className: 'custom-tooltip-reset',
            sticky: true,
            opacity: 1
        });

        markerElement.bindPopup(generatePopupHtml(point, currentMetric), {
            maxWidth: 450,
            className: 'custom-popup-reset'
        });

        markerElement.addTo(markersGroup);
    });
}

// Switch Action UI Handlers
function setModel(modelName) {
    currentModel = modelName;
    
    const switchEl = document.getElementById('model-switch');
    const optSS = document.getElementById('opt-streetscore');
    const optTS = document.getElementById('opt-trueskill');
    const optDiff = document.getElementById('opt-difference');
    const modeSwitchGroup = document.getElementById('mode-switch-group');

    if (optSS) optSS.classList.remove('active');
    if (optTS) optTS.classList.remove('active');
    if (optDiff) optDiff.classList.remove('active');

    // Count available models
    const availableCount = (hasStreetscore ? 1 : 0) + (hasTrueskill ? 1 : 0) + ((hasStreetscore && hasTrueskill) ? 1 : 0);

    if (modelName === 'streetscore') {
        if (switchEl) switchEl.setAttribute('data-active', 'left');
        if (optSS) optSS.classList.add('active');
        if (modeSwitchGroup) modeSwitchGroup.style.display = 'block';
    } else if (modelName === 'trueskill') {
        if (switchEl) {
            if (availableCount === 3) {
                switchEl.setAttribute('data-active', 'middle');
            } else {
                switchEl.setAttribute('data-active', 'right');
            }
        }
        if (optTS) optTS.classList.add('active');
        if (modeSwitchGroup) modeSwitchGroup.style.display = 'block';
    } else if (modelName === 'difference') {
        if (switchEl) switchEl.setAttribute('data-active', 'right');
        if (optDiff) optDiff.classList.add('active');
        
        // Force mode to score since difference doesn't have uncertainty
        currentMode = 'score';
        const modeSwitch = document.getElementById('mode-switch');
        if (modeSwitch) modeSwitch.setAttribute('data-active', 'left');
        const optScore = document.getElementById('opt-score');
        const optUnc = document.getElementById('opt-uncertainty');
        if (optScore) optScore.classList.add('active');
        if (optUnc) optUnc.classList.remove('active');
        if (modeSwitchGroup) modeSwitchGroup.style.display = 'none';
    }
    
    renderMarkers();
}

function toggleMode() {
    const switchEl = document.getElementById('mode-switch');
    const optScore = document.getElementById('opt-score');
    const optUnc = document.getElementById('opt-uncertainty');

    if (currentMode === 'score') {
        currentMode = 'uncertainty';
        switchEl.setAttribute('data-active', 'right');
        optScore.classList.remove('active');
        optUnc.classList.add('active');
    } else {
        currentMode = 'score';
        switchEl.setAttribute('data-active', 'left');
        optScore.classList.add('active');
        optUnc.classList.remove('active');
    }

    renderMarkers();
}

function selectMetric(metricName) {
    currentMetric = metricName;
    
    const buttons = document.querySelectorAll('.metric-btn');
    buttons.forEach(btn => {
        if (btn.getAttribute('data-metric') === metricName) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    renderMarkers();
}

// Dynamic legend color and text update
function updateLegend() {
    const minTitle = document.getElementById('legend-min-title');
    const maxTitle = document.getElementById('legend-max-title');
    const minVal = document.getElementById('legend-min-val');
    const maxVal = document.getElementById('legend-max-val');
    const colorBar = document.getElementById('legend-color-bar');

    if (currentModel === 'difference') {
        minTitle.removeAttribute('data-i18n');
        maxTitle.removeAttribute('data-i18n');
        minTitle.innerText = t('diff_min');
        maxTitle.innerText = t('diff_max');
        minVal.innerText = `-${diffLimit.toFixed(2)}`;
        maxVal.innerText = `+${diffLimit.toFixed(2)}`;
        colorBar.className = "legend-bar difference-track";
    } else if (currentMode === 'score') {
        minTitle.removeAttribute('data-i18n');
        maxTitle.removeAttribute('data-i18n');
        minTitle.innerText = t('bad');
        maxTitle.innerText = t('good');
        minVal.innerText = "0";
        maxVal.innerText = "10";
        colorBar.className = "legend-bar score-track";
    } else {
        minTitle.removeAttribute('data-i18n');
        maxTitle.removeAttribute('data-i18n');
        minTitle.innerText = t('low_unc');
        maxTitle.innerText = t('high_unc');
        minVal.innerText = `${uncertaintyMin.toFixed(2)}`;
        maxVal.innerText = `${uncertaintyLimit.toFixed(2)}`;
        colorBar.className = "legend-bar uncertainty-track";
    }
}

// Generate metric selection buttons dynamically
function setupMetricGrid() {
    const grid = document.getElementById('metric-buttons-grid');
    if (!grid) return;
    grid.innerHTML = '';

    availableMetrics.forEach(metric => {
        const btn = document.createElement('button');
        btn.className = `metric-btn ${metric === currentMetric ? 'active' : ''}`;
        btn.innerText = metric;
        btn.setAttribute('data-metric', metric);
        btn.onclick = () => selectMetric(metric);
        grid.appendChild(btn);
    });
}

// Fullscreen Modal Handlers
function openFullscreen(imagePath) {
    const modal = document.getElementById('fs-modal');
    const modalImg = document.getElementById('fs-image');
    if (modal && modalImg) {
        modal.style.display = 'flex';
        modalImg.src = imagePath;
    }
}

// Close popup on map click or modal close
function closeFullscreen() {
    const modal = document.getElementById('fs-modal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Dynamically build model switcher depending on model availability
function setupModelSwitch() {
    const group = document.getElementById('model-switch-group');
    if (!group) return;

    // Check how many models are available
    const available = [];
    if (hasStreetscore) available.push('streetscore');
    if (hasTrueskill) available.push('trueskill');
    if (hasStreetscore && hasTrueskill) available.push('difference');

    if (available.length <= 1) {
        // Only one model or none, we don't need a model switch!
        group.style.display = 'none';
        // Set currentModel to the only available one
        if (available.length === 1) {
            currentModel = available[0];
        }
        return;
    }

    // Otherwise, construct a custom styled switch
    let containerClass = 'switch-container';
    if (available.length === 3) {
        containerClass += ' three-way';
    }

    let optionsHtml = '';
    available.forEach((mod, idx) => {
        const activeClass = (mod === currentModel) ? 'active' : '';
        optionsHtml += `<div class="switch-option ${activeClass}" id="opt-${mod}" data-i18n="${mod}" onclick="setModel('${mod}')">${t(mod)}</div>`;
    });

    group.innerHTML = `
        <div class="${containerClass}" id="model-switch" data-active="left">
            <div class="switch-slider"></div>
            ${optionsHtml}
        </div>
    `;

    // Set initial model safely
    if (!available.includes(currentModel)) {
        currentModel = available[0];
    }
}

// Dynamically construct split filter check list
function setupSplitFilter() {
    const group = document.getElementById('split-filter-group');
    if (!group) return;

    // Scan mapPoints to collect all unique split values
    const splitsSet = new Set();
    mapPoints.forEach(p => {
        if (p.split) {
            splitsSet.add(p.split);
        }
    });

    uniqueSplits = Array.from(splitsSet).sort();
    
    if (uniqueSplits.length === 0) {
        group.style.display = 'none';
        return;
    }

    // Add 'Other' category if there are some images with split and some without
    const hasAnyWithoutSplit = mapPoints.some(p => !p.split);
    if (hasAnyWithoutSplit) {
        uniqueSplits.push('Other');
    }

    // Initialize all active by default
    uniqueSplits.forEach(s => {
        activeSplits[s] = true;
    });

    // Build the UI checkboxes
    let checkboxesHtml = `<div class="switch-title" data-i18n="filter_splits" style="margin-bottom: 6px;">${t('filter_splits')}</div>`;
    checkboxesHtml += '<div style="display: flex; flex-direction: column; gap: 6px; padding: 4px 0;">';

    uniqueSplits.forEach(s => {
        const labelAttr = (s === 'Other') ? ' data-i18n="other"' : '';
        const labelText = (s === 'Other') ? t('other') : s;
        checkboxesHtml += `
            <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; font-size: 11px; color: #cbd5e1; user-select: none; margin-bottom: 2px;">
                <input type="checkbox" id="chk-split-${s}" checked onchange="toggleSplitFilter('${s}')" style="accent-color: #38bdf8; cursor: pointer;" />
                <span${labelAttr}>${labelText}</span>
            </label>
        `;
    });
    checkboxesHtml += '</div>';

    group.innerHTML = checkboxesHtml;
    group.style.display = 'block';
}

function toggleSplitFilter(splitName) {
    const chk = document.getElementById(`chk-split-${splitName}`);
    if (chk) {
        activeSplits[splitName] = chk.checked;
    }
    renderMarkers();
}

// Initialize overall dashboard on load
window.addEventListener('DOMContentLoaded', () => {
    setupModelSwitch();
    setupSplitFilter();
    setupMetricGrid();
    updateLegend();
    applyTranslations();
    initLeafletOverlays();
});
