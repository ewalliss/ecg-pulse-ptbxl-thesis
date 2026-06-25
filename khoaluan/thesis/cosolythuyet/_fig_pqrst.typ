#import "@preview/cetz:0.3.4"
#set page(width: auto, height: auto, margin: 10pt, fill: white)
#set text(font: "Helvetica", size: 10pt)

#cetz.canvas(length: 1cm, {
  import cetz.draw: *

  let sx = 13.0   // horizontal scale (one beat across)
  let sy = 2.4    // vertical scale

  // one normal P-Q-R-S-T beat, points in (t in [0,1], amplitude)
  let beat = (
    (0.00, 0), (0.05, 0),
    // P wave
    (0.07, 0.04), (0.09, 0.12), (0.11, 0.15), (0.13, 0.12), (0.15, 0.04), (0.17, 0),
    // PR segment
    (0.25, 0),
    // QRS complex (sharp)
    (0.265, -0.09), (0.285, 1.00), (0.305, -0.32), (0.325, 0),
    // ST segment
    (0.43, 0.015),
    // T wave
    (0.47, 0.08), (0.53, 0.27), (0.59, 0.30), (0.65, 0.22), (0.71, 0.06), (0.75, 0),
    // TP
    (1.00, 0),
  )
  let pts = beat.map(p => (p.at(0) * sx, p.at(1) * sy))

  // baseline (isoelectric)
  line((0, 0), (sx, 0), stroke: (paint: luma(180), thickness: 0.6pt, dash: "dashed"))

  // ECG trace
  line(..pts, stroke: (paint: rgb("#b3123a"), thickness: 1.8pt))

  // ---- wave labels ----
  let lab(t, y, s, col) = content((t * sx, y), text(11pt, weight: "bold", fill: col)[#s])
  lab(0.11, 0.15 * sy + 0.35, "P", rgb("#2f9e44"))
  lab(0.285, 1.00 * sy + 0.30, "QRS", rgb("#b3123a"))
  lab(0.57, 0.30 * sy + 0.35, "T", rgb("#9c36b5"))

  // small segment tags
  content((0.205 * sx, 0.34), text(8.5pt, fill: rgb("#e8930c"), weight: "bold")[PR])
  content((0.38 * sx, 0.30), text(8.5pt, fill: rgb("#7048e8"), weight: "bold")[ST])

  // ---- interval brackets (below) ----
  let yb = -0.55
  let bracket(t1, t2, y, s) = {
    line((t1 * sx, y + 0.12), (t1 * sx, y - 0.05), stroke: 0.7pt)
    line((t2 * sx, y + 0.12), (t2 * sx, y - 0.05), stroke: 0.7pt)
    line((t1 * sx, y), (t2 * sx, y), stroke: 0.7pt)
    content(((t1 + t2) / 2 * sx, y - 0.28), text(9pt, weight: "bold")[#s])
  }
  bracket(0.05, 0.285, yb, "Khoảng PR")
  bracket(0.285, 0.73, yb, "Khoảng QT")
  // TP markers (rest between beats)
  content((0.02 * sx, 0.55), text(8.5pt, fill: rgb("#1971c2"), weight: "bold")[TP])
  content((0.97 * sx, 0.55), text(8.5pt, fill: rgb("#1971c2"), weight: "bold")[TP])
})
