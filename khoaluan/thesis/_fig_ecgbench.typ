#set page(width: 28cm, height: auto, margin: 10mm)
#set text(font: "Times New Roman", size: 11pt, lang: "en")

// ---------- palette (muted, echoing PULSE) ----------
#let c_abn  = rgb("#f3d9b8")   // Abnormality Detection — orange-ish
#let c_rep  = rgb("#cfe0f0")   // Report Generation — blue-ish
#let c_mmmu = rgb("#cfe6cf")   // MMMU-ECG — green-ish
#let c_aren = rgb("#ddd2ec")   // ECG Arena — purple-ish
#let c_flow = rgb("#efefef")   // mini-flow boxes
#let c_foot = luma(245)        // footer strip

// ---------- mini-flow helpers ----------
#let fbox(label, fill: c_flow) = box(
  fill: fill, stroke: 0.6pt + luma(120), radius: 3pt, inset: (x: 6pt, y: 4pt),
)[#text(size: 8.5pt)[#label]]
#let farr = box(inset: (x: 3pt))[#text(size: 12pt)[→]]

// ---------- card helper ----------
#let card(title, body, fill: c_flow, accent: false) = box(
  width: 100%,
  fill: white,
  stroke: if accent { 1.4pt + rgb("#c0392b") } else { 0.8pt + luma(90) },
  radius: 4pt,
  inset: 0pt,
  clip: true,
)[
  // header bar
  #box(width: 100%, fill: fill, inset: (x: 7pt, y: 5pt))[
    #grid(columns: (1fr, auto), align: (left + horizon, right + horizon),
      text(weight: "bold", size: 10.5pt)[#title],
      if accent {
        box(fill: rgb("#c0392b"), radius: 2pt, inset: (x: 4pt, y: 2pt))[
          #text(size: 7pt, fill: white, weight: "bold")[task formulation adapted here]
        ]
      } else { [] },
    )
  ]
  // body
  #box(width: 100%, inset: (x: 7pt, y: 6pt))[#text(size: 8pt)[#body]]
]

// ---------- footer strip inside a card body ----------
#let foot(body) = box(width: 100%, fill: c_foot, stroke: 0.5pt + luma(180),
  radius: 2pt, inset: (x: 5pt, y: 4pt))[#text(size: 7pt)[#body]]

// ---------- field label helper ----------
#let fl(name, value) = [#text(weight: "bold")[#name:] #value]

// ---------- dashed outer panel ----------
#let panel(title, body) = box(
  width: 100%,
  stroke: (paint: luma(60), thickness: 0.9pt, dash: "dashed"),
  radius: 5pt, inset: 9pt,
)[
  #align(center)[#text(weight: "bold", size: 12pt)[#title]]
  #v(6pt)
  #body
]

// ============================================================
//  TITLE
// ============================================================
#align(center)[#text(size: 14pt, weight: "bold")[ECGBench Task Taxonomy]]
#v(7pt)

// ============================================================
//  PANEL (A) — Repurposed Tasks
// ============================================================
#panel([(A) Repurposed Tasks from Diagnosis and Reports])[

  // mini-flow
  #align(center)[
    #box(inset: (bottom: 6pt))[
      #fbox[ECG Datasets]
      #farr
      #fbox[Diagnosis & Reports + Synthesized ECG Images]
      #box(inset: (x: 3pt))[#stack(dir: ttb, spacing: 1pt,
        text(size: 12pt)[→], text(size: 7pt, style: "italic")[Repurposing])]
      #fbox(fill: c_flow)[ECG-related Tasks]
    ]
  ]
  #v(2pt)

  #grid(columns: (1fr, 1fr), gutter: 8pt,

    // ---- Abnormality Detection (HIGHLIGHT) ----
    card("Abnormality Detection", fill: c_abn, accent: true)[
      #fl("Query")[“Please determine the appropriate diagnosis for the ECG
      image: right bundle branch block, ST depression, ventricular ectopics,
      1st degree AV block, ST elevation, left bundle branch block, atrial
      fibrillation, premature atrial contraction, normal ECG.”]

      #v(4pt)
      #fl("Answer")[“atrial fibrillation; right bundle branch block”]

      #v(5pt)
      #foot[
        #fl("Question type")[Diagnosis Classification; Close-ended] \
        #fl("Image")[12×1 layout; Rotation; Wrinkles; Colored] \
        #fl("Source")[CPSC (out-of-domain)]
      ]
    ],

    // ---- Report Generation ----
    card("Report Generation", fill: c_rep)[
      #fl("Query")[“Please write a clinical report based on this ECG image.”]

      #v(4pt)
      #fl("Answer")[“Premature ventricular contractions are present. ...
      likely atrial flutter with 2:1 AV block. Left ventricular hypertrophy.
      Non-specific ST-T changes.”]

      #v(5pt)
      #foot[
        #fl("Question type")[Report generation; Open-ended] \
        #fl("Image")[4×3 layout; w/o distortions] \
        #fl("Source")[PTB-XL (in-domain)]
      ]
    ],
  )
]

#v(9pt)

// ============================================================
//  PANEL (B) — Created Tasks
// ============================================================
#panel([(B) Created Tasks from External ECG-related Resources])[

  // mini-flow
  #align(center)[
    #box(inset: (bottom: 6pt))[
      #fbox[Resource Selection]
      #farr
      #fbox[Question Creation]
      #farr
      #fbox[Quality Control]
    ]
  ]
  #v(2pt)

  #grid(columns: (1fr, 1fr), gutter: 8pt,

    // ---- MMMU-ECG ----
    card("MMMU-ECG", fill: c_mmmu)[
      #fl("Question")[“What is the rhythm shown in this ECG?”]

      #v(4pt)
      (A) Sinus tachycardia with ventricular tachycardia \
      (B) Atrial fibrillation with right bundle branch aberrancy \
      (C) Atrial tachycardia with right bundle branch aberrancy \
      (D) Polymorphic ventricular tachycardia

      #v(4pt)
      #fl("Answer")[(D)]

      #v(5pt)
      #foot[
        #fl("Question type")[Multi-choice; Close-ended] \
        #fl("Image")[6×2 layout; real-world] \
        #fl("Source")[Online Quiz]
      ]
    ],

    // ---- ECG Arena ----
    card("ECG Arena", fill: c_aren)[
      #fl("Question")[“Can you describe the features observed in this ECG
      (rhythm, waveforms, intervals, notable findings)?”]

      #v(3pt)
      #fl("Follow-up")[“Given the ST-segment changes and dual-chamber pacing,
      what is the diagnosis?”]

      #v(3pt)
      #fl("Answer")[“dual-chamber paced rhythm at 60 bpm; ST elevation in
      leads II...”]

      #v(5pt)
      #foot[
        #fl("Question type")[Multi-turn; Open-ended] \
        #fl("Image")[4×3 layout; real-world] \
        #fl("Source")[Textbook]
      ]
    ],
  )
]
