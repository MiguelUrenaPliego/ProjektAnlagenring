"""Dependency-free language-toggle widget shared by every generated folium map.

Import with a sys.path.insert(0, "<repo_root>/shared") from any generator
script, then call toggle_html(translations) and inject the returned string
into the map via m.get_root().html.add_child(folium.Element(toggle_html(...))).
"""

import json


def toggle_html(translations: dict, default: str = "de") -> str:
    """Build a top-right DE/EN toggle button + i18n script block.

    translations: {"de": {key: text, ...}, "en": {key: text, ...}}
    Elements tagged data-i18n="key" anywhere in the page get their
    innerHTML swapped when the language changes.

    Language resolution order: ?lang= URL param -> localStorage.site_lang
    -> default. The resolved language is written back to localStorage so
    same-origin pages stay in sync; the URL param keeps things working
    under file:// where localStorage is not shared across pages.
    """

    translations_json = json.dumps(translations, ensure_ascii=False)

    return f"""
    <style>
      #langSwitcher {{
        position: fixed;
        top: 16px;
        right: 16px;
        background: rgba(8, 25, 55, 0.75);
        backdrop-filter: blur(4px);
        color: #fff;
        padding: 7px 12px;
        border-radius: 18px;
        cursor: pointer;
        font-size: 13px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
        z-index: 10000;
        border: 1px solid rgba(255,255,255,0.35);
        user-select: none;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      }}
      #langSwitcher:hover {{ background: rgba(8, 25, 55, 0.92); }}
    </style>
    <div id="langSwitcher" onclick="__mapToggleLanguage()" title="Sprache wechseln / Switch language">
      \U0001F310 <span id="currentLangLabel"></span>
    </div>
    <script>
    (function() {{
      var TRANSLATIONS = {translations_json};
      var DEFAULT_LANG = {json.dumps(default)};

      function resolveInitialLang() {{
        try {{
          var params = new URLSearchParams(window.location.search);
          var fromUrl = params.get("lang");
          if (fromUrl && TRANSLATIONS[fromUrl]) return fromUrl;
        }} catch (e) {{}}
        try {{
          var fromStorage = window.localStorage.getItem("site_lang");
          if (fromStorage && TRANSLATIONS[fromStorage]) return fromStorage;
        }} catch (e) {{}}
        return DEFAULT_LANG;
      }}

      // Exposed so a map's own script can look up strings itself for
      // content it re-renders dynamically (tables, legends, dropdown
      // option text) — not just the static data-i18n elements handled
      // here automatically.
      window.__mapTranslations = TRANSLATIONS;
      window.t = function(key) {{
        var dict = TRANSLATIONS[window.__mapCurrentLang] || TRANSLATIONS[DEFAULT_LANG];
        return (dict && dict[key] !== undefined) ? dict[key] : key;
      }};

      window.__mapApplyLanguage = function(lang) {{
        var dict = TRANSLATIONS[lang] || TRANSLATIONS[DEFAULT_LANG];
        document.querySelectorAll("[data-i18n]").forEach(function(el) {{
          var key = el.getAttribute("data-i18n");
          if (dict[key] !== undefined) el.innerHTML = dict[key];
        }});
        var label = document.getElementById("currentLangLabel");
        if (label) label.textContent = lang.toUpperCase();
        document.documentElement.lang = lang;
        window.__mapCurrentLang = lang;
        try {{ window.localStorage.setItem("site_lang", lang); }} catch (e) {{}}
        // Lets a map's own script re-render anything it built dynamically
        // (legends, tables, dropdown labels) in the new language.
        window.dispatchEvent(new CustomEvent("maplangchange", {{ detail: {{ lang: lang }} }}));
      }};

      window.__mapToggleLanguage = function() {{
        var next = window.__mapCurrentLang === "de" ? "en" : "de";
        window.__mapApplyLanguage(next);
      }};

      window.__mapApplyLanguage(resolveInitialLang());
    }})();
    </script>
    """
