# Serving the maps

`carfree_zones_map.html` and `region_map.html` are static files, but
`carfree_zones_map.html` fetches its route geometry from external JSON
files (`routes/routes_<scenario>.json`) instead of embedding them inline —
that's what keeps the HTML itself a few MB instead of hundreds of MB.

Browsers block `fetch()` of local files when a page is opened directly via
a `file://` URL (no HTTP status to work around, just an unconditional
security restriction), so **double-clicking `carfree_zones_map.html`
will show route data as unavailable** with an explanation in the UI —
that's expected, not a bug. It needs to be served over http(s) instead:

```bash
cd routing
python3 -m http.server 8000
```

Then open `http://localhost:8000/carfree_zones_map.html`. Any other static
file server works the same way (GitHub Pages, nginx, `npx serve`, etc.) —
nothing needs to run server-side, it's just a security restriction on the
`file://` protocol specifically.

`region_map.html` has no external data files and works fine opened
directly via `file://` too, but serving both the same way keeps testing
consistent with how they'll actually be hosted.
