# Previewing the presentation locally

Serve the repo over local HTTP rather than opening `presentation.html` via
a `file://` URL — some browser behavior (e.g. iframe handling) differs
between `file://` and the `https://` GitHub Pages serves in production, so
testing over HTTP is closer to what visitors actually see:

```bash
cd /home/miguel/Documents/UNI/Master/2/ProjektVerkehr
python3 -m http.server 8791
```

Then open, in a private/incognito window (to avoid stale cache):

```
http://localhost:8791/marp/presentation.html
```

Navigate through several slides (not just the one you land on) and check
on both desktop and mobile if possible. Stop the server with Ctrl+C when
done.

# Regenerating PDF-page images

Slides that show a page "directly from the PDF" use a pre-rendered JPG
(`pdf_pages/page-N.jpg`), not a live embed of `presentation_base.pdf` —
mobile browsers (iOS Safari, Android Chrome) have no built-in PDF viewer
plugin for iframes, so a live embed just showed a fallback "open
presentation_base.pdf" link there instead of the page.

This means the images go stale whenever `presentation_base.pdf` changes.
After editing the PDF (or adding/removing/reordering which pages appear —
see the config comment at the top of `presentation.md`), regenerate them:

```bash
python3 marp/render_pdf_pages.py
```

It reads every `data-pdf-page="N"` in `presentation.md` and re-renders
`pdf_pages/page-N.jpg` from `presentation_base.pdf` accordingly. Requires
`pdftoppm` (poppler-utils): `apt install poppler-utils` / `brew install
poppler`.

Then rebuild `presentation.html` from `presentation.md`:

```bash
marp presentation.md --theme-set theme.css -o presentation.html --html --allow-local-files
```

(`--theme-set` must come before the input filename — it's an array flag
that otherwise greedily swallows the next argument too.)

# Converting new cover images to JPG

`assets/*.png` covers with no transparency are kept as JPG instead (JPG
compresses photo-like images much better than PNG; flat/text screenshots
can go the other way, so check before committing). After adding a new PNG
to `assets/`:

```bash
python3 marp/convert_assets_to_jpg.py
```

It writes a `.jpg` next to any PNG with no alpha channel and prints the
size delta. Only keep the JPG (and update references, then delete the
PNG) if it actually saved space.

# Map slides

Right after the title slide there are 5 plain map slides, one per project
map, each a `custom-slide map-slide` with a single
`<iframe class="map-embed" src="...">` pointing at the map's HTML file.
These embed the map as-is — no layer control or postMessage wiring, the
map opens on its own default state exactly like opening the file
directly.
