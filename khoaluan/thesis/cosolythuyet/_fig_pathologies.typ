#import "@preview/cetz:0.3.4"
#set page(width: 21cm, height: auto, margin: 12pt, fill: white)
#set text(font: "Helvetica", size: 10pt)

// ---- one normal beat (t in [0,1], amplitude), reused as a template ----
#let NORM = (
  (0.00, 0), (0.05, 0),
  (0.07, 0.04), (0.09, 0.12), (0.11, 0.15), (0.13, 0.12), (0.15, 0.04), (0.17, 0),
  (0.25, 0),
  (0.265, -0.09), (0.285, 0.95), (0.305, -0.30), (0.325, 0),
  (0.43, 0.015),
  (0.47, 0.08), (0.53, 0.27), (0.59, 0.30), (0.65, 0.22), (0.71, 0.06), (0.75, 0),
  (1.00, 0),
)

// MI: pathologic Q + elevated, coved ST merging into T
#let MI = (
  (0.00, 0), (0.06, 0),
  (0.08, 0.06), (0.11, 0.12), (0.14, 0.05), (0.18, 0),
  (0.24, 0),
  (0.255, -0.24), (0.275, 0.80), (0.295, 0.02),
  (0.34, 0.34), (0.42, 0.40), (0.50, 0.46), (0.58, 0.40), (0.66, 0.16), (0.72, 0.03),
  (1.00, 0),
)

// INVT: normal P-QRS, but T wave inverted (downward)
#let INVT = (
  (0.00, 0), (0.05, 0),
  (0.07, 0.04), (0.09, 0.12), (0.11, 0.15), (0.13, 0.12), (0.15, 0.04), (0.17, 0),
  (0.25, 0),
  (0.265, -0.09), (0.285, 0.95), (0.305, -0.30), (0.325, 0),
  (0.43, -0.02),
  (0.47, -0.08), (0.53, -0.27), (0.59, -0.30), (0.65, -0.22), (0.71, -0.06), (0.75, 0),
  (1.00, 0),
)

// draw a panel: a sequence of beats (each an array of pts), placed left to right
#let panel(beats, highlight: none, hl-col: rgb("#d61f1f")) = cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let bw = 1.55          // beat width in cm
  let sy = 1.35
  let total = beats.len() * bw
  // baseline
  line((0, 0), (total, 0), stroke: (paint: luma(190), thickness: 0.5pt, dash: "dashed"))
  // beats
  for (i, b) in beats.enumerate() {
    let pts = b.map(p => (p.at(0) * bw + i * bw, p.at(1) * sy))
    line(..pts, stroke: (paint: rgb("#b3123a"), thickness: 1.4pt))
  }
  // highlight ellipse (cx, cy, rx, ry) in cm
  if highlight != none {
    circle((highlight.at(0), highlight.at(1)), radius: (highlight.at(2), highlight.at(3)),
      stroke: (paint: hl-col, thickness: 1.4pt, dash: "dashed"))
  }
})

#let head(t) = text(9.5pt, weight: "bold")[#t]
#let note(t) = text(8.5pt, fill: luma(60))[#t]

#align(center)[#text(13pt, weight: "bold")[Biểu hiện hình học đặc trưng của các nhóm bệnh lý trên sóng ECG]]
#v(8pt)

#grid(columns: (1fr, 1fr, 1fr), column-gutter: 10pt, align: center + top, row-gutter: 6pt,
  // ---- Diagnostic: MI ----
  [#head[(1) Chẩn đoán — Nhồi máu cơ tim (MI)]
   #v(4pt)
   #panel((MI, MI), highlight: (0.55, 0.55, 0.42, 0.45))
   #v(2pt)
   #note[Đoạn ST chênh lên dạng vòm và sóng Q bệnh lý (đạo trình đối diện vùng hoại tử).]
  ],
  // ---- Rhythm: AFIB ----
  [#head[(2) Nhịp — Rung nhĩ (AFIB)]
   #v(4pt)
   #cetz.canvas(length: 1cm, {
     import cetz.draw: *
     let sy = 1.35
     line((0,0), (4.65,0), stroke: (paint: luma(190), thickness: 0.5pt, dash: "dashed"))
     // fibrillatory wavy baseline (no P waves)
     let base = range(0, 48).map(k => {
       let x = k / 47 * 4.65
       let y = 0.05 * calc.sin(x * 7) + 0.03 * calc.sin(x * 17 + 1)
       (x, y * sy)
     })
     line(..base, stroke: (paint: rgb("#b3123a"), thickness: 1.0pt))
     // irregular QRS spikes at uneven spacing
     for cx in (0.55, 1.7, 2.55, 3.7) {
       line((cx - 0.06, 0.0), (cx - 0.02, -0.22 * sy), (cx, 0.95 * sy), (cx + 0.04, -0.30 * sy), (cx + 0.08, 0.0),
         stroke: (paint: rgb("#b3123a"), thickness: 1.4pt))
     }
     // mark irregular RR
     content((2.3, -0.62), text(7.5pt, fill: rgb("#1971c2"))[R–R bất quy tắc])
   })
   #v(2pt)
   #note[Mất hoàn toàn sóng P (thay bằng sóng f lăn tăn); khoảng R–R bất quy tắc.]
  ],
  // ---- Form: INVT ----
  [#head[(3) Hình thái — Sóng T đảo ngược (INVT)]
   #v(4pt)
   #panel((INVT, INVT), highlight: (0.56, -0.42, 0.34, 0.34))
   #v(2pt)
   #note[Sóng T đảo chiều (âm) — biến đổi tái cực khu trú; thuộc nhóm hình thái (Form).]
  ],
)
