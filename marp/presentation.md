---
marp: true
theme: FUAS
paginate: true
html: true
---

<!--
Configuration
=============
- Every slide taken "directly from the PDF" shows a given page of
  presentation_base.pdf as a pre-rendered JPG via
  <img class="pdf-embed" src="pdf_pages/page-N.jpg" data-pdf-page="N">.
  This used to be a live <iframe> straight into the PDF, but mobile
  browsers (iOS Safari, Android Chrome) have no built-in PDF viewer plugin
  for iframes — they just showed a fallback "open presentation_base.pdf"
  link there instead of the page. Pre-rendered images work everywhere.
  This DOES mean the images go stale when presentation_base.pdf changes —
  run `python3 marp/render_pdf_pages.py` afterwards to regenerate every
  pdf_pages/page-N.jpg referenced by a data-pdf-page="N" attribute below.
- To choose which pages appear, just add/remove/reorder the slides below —
  each one is a single <img class="pdf-embed" ...data-pdf-page="N"> line,
  plus the slide-class comment above it. After adding/removing a page
  number, rerun render_pdf_pages.py so its JPG exists.
- presentation_base.pdf currently has 9 pages. The title slide always
  shows page 2 (title/authors/logos).
- Map slides: right after the title slide there are 5 plain map slides,
  one per project map, each a "custom-slide map-slide" with a single
  <iframe class="map-embed" src="..."> pointing at the map's HTML file
  (relative to marp/). These just embed the map as-is — no layer control
  or postMessage wiring, the map opens on its own default state exactly
  like opening the file directly.
-->

<!-- _class: title -->
<img class="pdf-embed" src="pdf_pages/page-2.jpg" data-pdf-page="2" alt="">

---

<!-- _class: custom-slide map-slide -->
<div class="map-frame">
  <iframe class="map-embed" src="../maps/intersections/intersections_map.html"></iframe>
</div>

---

<!-- _class: custom-slide map-slide -->
<div class="map-frame">
  <iframe class="map-embed" src="../maps/points_of_interest/points_of_interest.html"></iframe>
</div>

---

<!-- _class: custom-slide map-slide -->
<div class="map-frame">
  <iframe class="map-embed" src="../ABsurveys/index.html"></iframe>
</div>

---

<!-- _class: custom-slide map-slide -->
<div class="map-frame">
  <iframe class="map-embed" src="../routing/region_map.html"></iframe>
</div>

---

<!-- _class: custom-slide map-slide -->
<div class="map-frame">
  <iframe class="map-embed" src="../routing/carfree_zones_map.html"></iframe>
</div>

---

<!-- _class: pdf-page -->
<img class="pdf-embed" src="pdf_pages/page-3.jpg" data-pdf-page="3" alt="">

---

<!-- _class: pdf-page -->
<img class="pdf-embed" src="pdf_pages/page-4.jpg" data-pdf-page="4" alt="">

---

<!-- _class: pdf-page -->
<img class="pdf-embed" src="pdf_pages/page-5.jpg" data-pdf-page="5" alt="">

---

<!-- _class: pdf-page -->
<img class="pdf-embed" src="pdf_pages/page-6.jpg" data-pdf-page="6" alt="">

---

<!-- _class: pdf-page -->
<img class="pdf-embed" src="pdf_pages/page-7.jpg" data-pdf-page="7" alt="">

---

<!-- _class: pdf-page -->
<img class="pdf-embed" src="pdf_pages/page-8.jpg" data-pdf-page="8" alt="">

---

<!-- _class: pdf-page -->
<img class="pdf-embed" src="pdf_pages/page-9.jpg" data-pdf-page="9" alt="">
