#set page(width: 28cm, height: auto, margin: 10mm)
#set text(font: "Times New Roman", size: 11pt, lang: "en")

// ---------- palette ----------
#let c_data = rgb("#bcd6ef")   // datasets / signals  (blue-grey)
#let c_dist = rgb("#cfe3f2")   // distortion samples
#let c_task = rgb("#c2dbb2")   // ECG tasks (green)
#let c_type = rgb("#d7ead0")   // task types
#let c_inst = rgb("#bcd6ef")   // instruction curation (blue)
#let c_orig = rgb("#e8dcc0")   // original diagnosis / reports
#let c_egi  = rgb("#e0b07a")   // ECGInstruct (orange)
#let c_qc   = rgb("#cfcfd9")   // quality checking (grey)
#let c_exp  = rgb("#a8d8d0")   // clinical experts

// ---------- helpers ----------
#let node(title, sub: none, fill: c_data, w: auto) = box(
  width: w, fill: fill, stroke: 0.7pt + luma(70), radius: 4pt, inset: (x: 7pt, y: 6pt),
)[
  #set align(center)
  #grid(columns: (1fr,), row-gutter: 2pt,
    [#text(weight: "bold", size: 10pt)[#title]],
    if sub != none { text(size: 8pt, fill: luma(70))[#sub] },
  )
]

// horizontal arrow with optional small blue label above
#let arrow(label) = align(horizon)[
  #stack(dir: ttb, spacing: 1pt,
    if label != none { align(center)[#text(size: 7.5pt, fill: rgb("#3a6ea5"))[#label]] },
    align(center)[#text(13pt)[→]],
  )
]

// dashed numbered panel
#let dpanel(title, body) = box(
  stroke: (paint: luma(40), thickness: 0.9pt, dash: "dashed"),
  radius: 4pt, inset: 8pt, width: 100%,
)[
  #text(weight: "bold", size: 12pt)[#title]
  #v(6pt)
  #body
]

// small list box for a task category
#let catbox(head, items, fill: c_task) = box(
  width: 100%, fill: fill, stroke: 0.6pt + luma(90), radius: 3pt, inset: (x: 6pt, y: 5pt),
)[
  #text(size: 9pt, weight: "bold")[#head]
  #set text(size: 7.8pt)
  #for it in items [ \ #text[• #it] ]
]

// small pill
#let pill(t, fill: c_type) = box(fill: fill, stroke: 0.5pt + luma(110), radius: 3pt, inset: (x: 5pt, y: 3pt))[#text(size: 8pt)[#t]]


// ============================================================
//  PANEL (1)
// ============================================================
#dpanel([(1) ECG Image Synthesis with Various Distortions],
  stack(dir: ttb, spacing: 6pt,

    // top flow: datasets -> signals -> synthesis
    grid(columns: (auto, auto, auto, auto, auto), align: horizon, column-gutter: 4pt,
      node("ECG Datasets", fill: c_data, w: 4.2cm),
      arrow[Extraction],
      node("ECG Signals", fill: c_data, w: 4.2cm),
      arrow[Filtering],
      node("Image Synthesis", fill: c_data, w: 4.2cm),
    ),

    align(center)[#text(13pt)[↓]],

    // distortions row
    box(fill: luma(249), stroke: 0.5pt + luma(170), radius: 3pt, inset: 6pt, width: 100%)[
      #align(center)[#text(size: 9.5pt, weight: "bold", fill: luma(50))[Various Distortions]]
      #v(4pt)
      #grid(columns: (1fr, 1fr, 1fr, 1fr, 1fr, auto), column-gutter: 6pt, align: horizon,
        node("Standard",  fill: c_dist, w: 100%),
        node("Rotation",  fill: c_dist, w: 100%),
        node("Noises",    fill: c_dist, w: 100%),
        node("Layout",    fill: c_dist, w: 100%),
        node("Wrinkles",  fill: c_dist, w: 100%),
        align(horizon)[#text(13pt, weight: "bold")[…]],
      )
    ],
  )
)

#v(8pt)

// ============================================================
//  PANEL (2)
// ============================================================
#dpanel([(2) Diverse Task Construction with Clinical Experts' Insights],
  stack(dir: ttb, spacing: 6pt,

    // ---- top row: ECG Tasks  ->  Instruction Curation  ->  ECGInstruct ----
    grid(columns: (6.8cm, 3.4cm, 7.2cm, auto, 7.2cm), align: horizon, column-gutter: 4pt,

      // LEFT: ECG Tasks (from clinical experts)
      stack(dir: ttb, spacing: 4pt,
        align(center)[#box(fill: c_exp, stroke: 0.6pt + luma(90), radius: 3pt, inset: (x: 6pt, y: 3pt))[#text(size: 8pt, weight: "bold")[Clinical Experts]]],
        align(center)[#text(12pt)[↓]],
        box(stroke: 0.7pt + luma(80), radius: 4pt, inset: 6pt, width: 100%)[
          #align(center)[#text(size: 9.5pt, weight: "bold")[ECG Tasks]]
          #v(4pt)
          #stack(dir: ttb, spacing: 4pt,
            catbox("Basic Feature Recognition", ("basic waveforms", "intervals", "rate")),
            catbox("Heart Rhythm Analysis", ("arrhythmias", "pacing patterns")),
            catbox("Morphology & Pathology", ("ischemia", "infarction", "pericarditis")),
            catbox("Clinical Report Generation", ("report generation",)),
          )
        ],
        v(2pt),
        box(stroke: 0.6pt + luma(100), radius: 4pt, inset: 6pt, width: 100%)[
          #align(center)[#text(size: 9pt, weight: "bold")[Diverse Task Types]]
          #v(4pt)
          #align(center)[#stack(dir: ltr, spacing: 5pt,
            pill("MCQ"), pill("Fill-in-the-blank"), pill("Close-ended QA"), pill("Open-ended QA"),
          )]
        ],
      ),

      // arrow with original inputs feeding in
      stack(dir: ttb, spacing: 3pt,
        node("Original Diagnosis & Reports", fill: c_orig, w: 100%),
        align(center)[#text(11pt)[↓]],
        arrow[Llama-3-70B],
      ),

      // MIDDLE: Instruction Curation prompt
      box(fill: c_inst, stroke: 0.7pt + luma(80), radius: 4pt, inset: 7pt, width: 100%)[
        #align(center)[#text(size: 9.5pt, weight: "bold")[Instruction Curation]]
        #v(2pt)
        #align(center)[#text(size: 7.5pt, fill: luma(60), style: "italic")[prompt template]]
        #v(4pt)
        #set text(size: 8pt)
        #stack(dir: ttb, spacing: 3pt,
          box(fill: white, stroke: 0.4pt + luma(150), radius: 2pt, inset: 4pt, width: 100%)[• guidelines for task creation],
          box(fill: white, stroke: 0.4pt + luma(150), radius: 2pt, inset: 4pt, width: 100%)[• original clinical report],
          box(fill: white, stroke: 0.4pt + luma(150), radius: 2pt, inset: 4pt, width: 100%)[• task description],
          box(fill: white, stroke: 0.4pt + luma(150), radius: 2pt, inset: 4pt, width: 100%)[• task type],
        )
      ],

      arrow[Llama-3-70B],

      // RIGHT: ECGInstruct example
      box(fill: c_egi, stroke: 0.7pt + luma(80), radius: 4pt, inset: 7pt, width: 100%)[
        #align(center)[#text(size: 9.5pt, weight: "bold")[ECGInstruct]]
        #v(4pt)
        #set text(size: 8pt)
        #stack(dir: ttb, spacing: 4pt,
          box(fill: white, stroke: 0.4pt + luma(150), radius: 2pt, inset: 5pt, width: 100%)[
            #text(weight: "bold")[Instruction:] "Please describe the P wave of the given ECG image."
          ],
          box(fill: white, stroke: 0.4pt + luma(150), radius: 2pt, inset: 5pt, width: 100%)[
            #text(weight: "bold")[Response:] "...p sinistrocardiale, suggesting possible left atrial enlargement... broad and notched P wave in lead II..."
          ],
        )
      ],
    ),

    // ---- feedback loop: Prompt Refining ----
    align(center)[
      #box(stroke: (paint: rgb("#3a6ea5"), thickness: 0.8pt, dash: "dashed"), radius: 3pt, inset: (x: 7pt, y: 4pt))[
        #text(size: 8.5pt, fill: rgb("#3a6ea5"))[↺ Prompt Refining (Clinical Experts) — feedback into Instruction Curation]
      ]
    ],

    align(center)[#text(11pt)[↓] #h(4pt) #text(size: 7.5pt, fill: rgb("#3a6ea5"))[Llama 3]],

    // ---- final: Quality Checking ----
    align(center)[
      #node([Quality Checking], sub: [score the answer 0–5 by guidelines], fill: c_qc, w: 9cm)
    ],
  )
)
