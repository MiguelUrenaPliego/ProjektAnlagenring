# Durchgängigkeit: Jane Jacobs in den Karten

Kurzer Überblick, wie das Durchgängigkeit-Konzept aus `report/main.tex`
(Jane Jacobs, `jacobs1961`) in den beiden interaktiven Karten operationalisiert wurde.

## Jane Jacobs, kurz

Ein Park bleibt nur lebendig und sicher, wenn er (a) durchgehend und ohne
Barrieren an sein Umfeld angebunden ist und (b) über den ganzen Tag und
die Woche hinweg unterschiedliche Nutzendengruppen anzieht statt nur eine
Gruppe zu einer Uhrzeit. Beide Punkte wurden je in einer Karte umgesetzt.

## `maps/points_of_interest` — Nutzungsdiversität

- POIs werden nach der Tageszeit/Wochentag klassifiziert, zu der sie
  hauptsächlich genutzt werden, in 4 Gruppen: Sport (früh/abends),
  Familien (nachmittags/Wochenende), Ruhe & Erholung (ganztags),
  Gastronomie & Kultur (abends/Wochenende).
- Ein H3-Hexagonraster (Radius ≈1,22 km, Auflösung 7) über `aoi.gpkg`
  zeigt pro Hexagon, wie viele dieser 4 Gruppen dort vertreten sind
  (0 = einseitig genutzt, 4 = ganztägig durchmischt) — direkte
  Operationalisierung von Jacobs' Diversitätsargument.
- Die AOI wird als hellgrüner Hintergrund dargestellt, die Hexagone als
  literale (unbeschnittene) Sechsecke, und die POI-Legende ist nach den
  4 Nutzergruppen gegliedert statt nach einzelnen Typen.

## `maps/intersections` — Isoliertheit / Barrierewirkung

- Für jedes Straßensegment und jeden Zebrastreifenpunkt existiert bereits
  ein Abstandswert (Distanz zum nächsten Zebrastreifen).
- Dasselbe H3-Raster (Radius ≈1,22 km) über `aoi.gpkg` aggregiert diese
  Werte pro Hexagon als gewichteten Mittelwert (Straßenlänge als Gewicht),
  gefärbt auf derselben Blau-Gelb-Rot-Skala wie die Straßenlinien. Hohe
  Werte markieren Bereiche, deren Übergänge/Kreuzungen weit von sicheren
  Querungen entfernt und damit potenzielle Barrieren sind.
- Auch hier: AOI als hellgrüner Hintergrund, Hexagone unbeschnitten.

## Verbindung

Beide Karten liefern zusammen die räumliche Evidenz für die
Durchgängigkeit-Sektion des Berichts: hohe Nutzungsdiversität an einem Ort
(POI-Karte) nützt wenig, wenn die Wege dorthin durch isolierende
Kreuzungen unterbrochen sind (Intersections-Karte) — genau das Muster, das
die Ergebnisse-Sektion des Berichts als \"Inseln\" beschreibt.
