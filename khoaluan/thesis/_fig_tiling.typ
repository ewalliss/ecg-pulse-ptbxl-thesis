#set page(width: 22cm, height: auto, margin: 10mm)
#set text(font: "Helvetica", size: 11pt, lang: "en")
#set par(leading: 0.85em)

#let leadcell(name) = box(width: 100%, height: 100%, stroke: 0.4pt + luma(170), inset: 5pt)[
  #align(center + horizon)[#text(8.5pt, weight: "bold")[#name]]
]

#align(center)[#text(13pt, weight: "bold")[AnyRes tiling actually used on the 12-lead ECG image]]
#v(2pt)
#align(center)[#text(9pt)[1344×672 → *downscale* → *672×336* → *2×1* cut + 1 global = *3 tiles* · *1728* tokens]]
#v(10pt)

// ---------- Step 1: original ----------
#grid(columns: (5.2cm, 1.4cm, 6.6cm), align: horizon + center, column-gutter: 4pt,
  // original 1344x672 (4 lead cols x 3 rows + rhythm)
  stack(dir: ttb, spacing: 3pt,
    align(center)[#text(8.5pt, weight: "bold")[original 1344 × 672]],
    box(stroke: 1pt + luma(60), width: 5.2cm, height: 2.9cm)[
      #grid(columns: (1fr,1fr,1fr,1fr), rows: (1fr,1fr,1fr,0.85fr), inset: 0pt,
        leadcell("I"), leadcell("aVR"), leadcell("V1"), leadcell("V4"),
        leadcell("II"), leadcell("aVL"), leadcell("V2"), leadcell("V5"),
        leadcell("III"), leadcell("aVF"), leadcell("V3"), leadcell("V6"),
        grid.cell(colspan: 4, box(width:100%, height:100%, fill: rgb("#fff3e6"), stroke: 0.4pt+luma(170), inset: 5pt)[#align(center + horizon)[#text(7.5pt, weight:"bold")[II — rhythm strip]]]),
      )
    ],
  ),
  align(center)[#text(9pt)[downscale \ ×½ \ →]],
  // downscaled 672x336 with single vertical cut
  stack(dir: ttb, spacing: 5pt,
    align(center)[#text(8.5pt, weight: "bold")[downscaled 672 × 336  —  one vertical cut (2×1)]],
    // tile labels ABOVE each half (no overlap with the lead cells)
    box(width: 6.6cm)[#grid(columns: (1fr, 1fr), align: center + bottom,
      text(7.5pt, fill: rgb("#1f6fd0"), weight: "bold")[tile 1 (left half) ↓],
      text(7.5pt, fill: rgb("#1f6fd0"), weight: "bold")[tile 2 (right half) ↓],
    )],
    box(stroke: 1pt + luma(60), width: 6.6cm, height: 2.0cm)[
      #grid(columns: (1fr,1fr,1fr,1fr), rows: (1fr,1fr,1fr,0.85fr), inset: 0pt,
        leadcell("I"), leadcell("aVR"), leadcell("V1"), leadcell("V4"),
        leadcell("II"), leadcell("aVL"), leadcell("V2"), leadcell("V5"),
        leadcell("III"), leadcell("aVF"), leadcell("V3"), leadcell("V6"),
        grid.cell(colspan: 4, box(width:100%, height:100%, fill: rgb("#fff3e6"), stroke: 0.4pt+luma(170), inset: 5pt)[#align(center + horizon)[#text(7pt, weight:"bold")[II — rhythm strip]]]),
      )
      // the single vertical tile cut at the middle (x = 336)
      #place(top + left, dx: 3.3cm, line(start: (0pt,0pt), end: (0pt, 2.0cm), stroke: (paint: rgb("#d61f1f"), thickness: 1.8pt, dash: "dashed")))
    ],
  ),
)

#v(8pt)
#align(center)[#text(9pt, weight: "bold")[↓  the 3 tiles CLIP actually sees (each 336 × 336)]]
#v(4pt)

#grid(columns: (1fr, 1fr, 1fr), column-gutter: 12pt, align: center + top,
  // global
  stack(dir: ttb, spacing: 3pt,
    box(stroke: 0.8pt + luma(90), width: 100%, height: 3cm, fill: luma(244), inset: 6pt)[
      #set text(6.5pt)
      #grid(columns: (1fr,1fr,1fr,1fr), rows: (1fr,1fr,1fr,0.8fr), gutter: 1pt,
        ..("I","aVR","V1","V4","II","aVL","V2","V5","III","aVF","V3","V6").map(x => align(center+horizon)[#x]),
        grid.cell(colspan: 4, align(center+horizon)[II strip]),
      )
    ],
    align(center)[#text(8pt, weight: "bold")[tile 0 — global]],
    align(center)[#text(7pt)[whole image squashed to 336×336]],
  ),
  // tile 1
  stack(dir: ttb, spacing: 3pt,
    box(stroke: 0.8pt + luma(90), width: 100%, height: 3cm, fill: rgb("#eef4fb"), inset: 6pt)[
      #set text(7pt)
      #grid(columns: (1fr,1fr), rows: (1fr,1fr,1fr,0.8fr), gutter: 1pt,
        align(center+horizon)[I], align(center+horizon)[aVR],
        align(center+horizon)[II], align(center+horizon)[aVL],
        align(center+horizon)[III], align(center+horizon)[aVF],
        grid.cell(colspan: 2, align(center+horizon)[II strip (left half)]),
      )
    ],
    align(center)[#text(8pt, weight: "bold")[tile 1 — left half]],
    align(center)[#text(7pt)[limb leads (I, II, III, aVR, aVL, aVF)]],
  ),
  // tile 2
  stack(dir: ttb, spacing: 3pt,
    box(stroke: 0.8pt + luma(90), width: 100%, height: 3cm, fill: rgb("#eef4fb"), inset: 6pt)[
      #set text(7pt)
      #grid(columns: (1fr,1fr), rows: (1fr,1fr,1fr,0.8fr), gutter: 1pt,
        align(center+horizon)[V1], align(center+horizon)[V4],
        align(center+horizon)[V2], align(center+horizon)[V5],
        align(center+horizon)[V3], align(center+horizon)[V6],
        grid.cell(colspan: 2, align(center+horizon)[II strip (right half)]),
      )
    ],
    align(center)[#text(8pt, weight: "bold")[tile 2 — right half]],
    align(center)[#text(7pt)[precordial leads (V1–V6)]],
  ),
)
