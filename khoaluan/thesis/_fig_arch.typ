#set page(width: 28cm, height: auto, margin: 10mm)
#set text(font: "Helvetica", size: 11pt, lang: "en")
#set par(leading: 0.85em)

// ---------- palette ----------
#let c_img  = rgb("#e0883a")
#let c_vis  = rgb("#bcd6ef")
#let c_proj = rgb("#c2dbb2")
#let c_llm  = rgb("#e8dcc0")
#let c_head = rgb("#a8d8d0")
#let c_cat  = rgb("#f0c267")
#let c_eval = rgb("#b6b6c6")
#let c_frz  = rgb("#d9d9d9")
#let c_tr   = rgb("#ffd8a8")

// ---------- helpers ----------
#let node(title, sub: none, fill: c_vis, w: 6.6cm, tag: none) = box(
  width: w, fill: fill, stroke: 0.7pt + luma(70), radius: 4pt, inset: (x: 11pt, y: 9pt),
)[
  #set align(center)
  #grid(columns: (1fr,), row-gutter: 2pt,
    [#text(weight: "bold", size: 10.5pt)[#title] #if tag != none [#h(3pt) #box(fill: white, stroke: 0.5pt, inset: 2pt, radius: 2pt)[#text(size: 7.5pt)[#tag]]]],
    if sub != none { text(size: 8.5pt)[#sub] },
  )
]
#let sh(label) = align(center)[#text(size: 8.5pt, fill: rgb("#3a6ea5"))[#label]]
#let ar = align(center)[#text(size: 14pt)[↓]]
#let arl(label) = stack(dir: ttb, spacing: 4pt, ar, sh(label))

#let dnode(title, sub: none, fill: c_vis) = box(width: 100%, fill: fill, stroke: 0.6pt + luma(90), radius: 3pt, inset: 8pt)[
  #set align(center); #text(size: 8.5pt, weight: "bold")[#title]
  #if sub != none [ \ #text(size: 7.5pt)[#sub] ]
]
#let dpanel(title, body) = box(stroke: 0.8pt + luma(40), radius: 4pt, inset: 11pt, width: 100%)[
  #align(center)[#text(weight: "bold", size: 10pt)[#title]] #v(4pt)
  #body
]
#let ddown = align(center)[#text(size: 11pt)[↓]]

// ============================================================
//  DETAILS PANEL
// ============================================================
#align(center)[#text(size: 13pt, weight: "bold")[Details]
  #h(8pt) #box(fill: c_frz, stroke: 0.5pt, inset: 3pt, radius: 2pt)[#text(7.5pt)[frozen]]
  #h(3pt) #box(fill: c_tr, stroke: 0.5pt, inset: 3pt, radius: 2pt)[#text(7.5pt)[QLoRA-tuned]]
]
#v(4pt)

#grid(columns: (1fr, 1.15fr, 1.2fr, 1.25fr, 1.05fr), column-gutter: 7pt, align: center + top,
  dpanel([Projector (mlp2x\_gelu)], stack(dir: ttb, spacing: 4pt,
    dnode("Linear  1024 → 4096", fill: c_frz), ddown,
    dnode("GELU", fill: c_frz), ddown,
    dnode("Linear  4096 → 4096", fill: c_frz),
    align(center)[#text(7pt, style: "italic")[frozen (freeze\_mm\_mlp\_adapter)]],
  )),
  dpanel("ViT block (CLIP-L, ×24)", stack(dir: ttb, spacing: 4pt,
    dnode("LayerNorm", fill: c_frz), ddown,
    dnode("Multi-Head Self-Attention", fill: c_frz), ddown,
    dnode([⊕  residual]), ddown,
    dnode("LayerNorm + MLP (GELU)", fill: c_frz), ddown,
    dnode([⊕  residual]),
  )),
  dpanel("Vicuna decoder layer (×32)", stack(dir: ttb, spacing: 4pt,
    dnode("RMSNorm", fill: c_llm), ddown,
    dnode([Self-Attention (RoPE) #h(2pt) #box(fill: c_tr, inset: 1pt)[#text(7pt)[+LoRA]]], fill: c_llm), ddown,
    dnode([⊕  residual]), ddown,
    dnode([RMSNorm + SwiGLU MLP #h(2pt) #box(fill: c_tr, inset: 1pt)[#text(7pt)[+LoRA]]], fill: c_llm), ddown,
    dnode([⊕  residual]),
  )),
  dpanel("QLoRA adapter (4-bit NF4)", stack(dir: ttb, spacing: 8pt,
    dnode([$W_0$  (4-bit NF4, FROZEN)], fill: c_frz),
    align(center)[#text(11pt)[+]],
    dnode([$(alpha/r) dot B A x$  •  $r=16, alpha=32$], fill: c_tr),
    align(center)[#text(8pt)[$A: 4096 -> 16$, #h(2pt) $B: 16 -> 4096$]],
    align(center)[#text(8pt, style: "italic")[$h = W_0 x + (alpha/r) B A x$]],
    align(center)[#text(7pt)[on q,k,v,o,gate,up,down (find\_all\_linear)]],
  )),
  dpanel("Teacher-forced readout", stack(dir: ttb, spacing: 4pt,
    box(stroke: 0.6pt, inset: 4pt, radius: 2pt)[#text(8pt)[`NORM` #box(fill: rgb("#ffe0e6"), inset: 2pt)[`:`] `0` … `(K codes)`]],
    align(center)[#text(8pt)[pink cell = readout position]],
    align(center)[#text(8pt, style: "italic")[$P(y_k=1) = "softmax"(z_("id0"), z_("id1"))_1$]],
  )),
)

#v(10pt)
#line(length: 100%, stroke: 0.5pt + luma(150))
#v(8pt)

// ============================================================
//  MAIN DIAGRAM
// ============================================================
#grid(columns: (1fr, 1fr, 1fr), align: center + top, column-gutter: 6pt,
  align(center)[#text(13pt, weight: "bold")[Vision encoding (frozen)]],
  align(center)[#text(13pt, weight: "bold")[Fusion & Language model]],
  align(center)[#text(13pt, weight: "bold")[Output & Evaluation]],
)
#v(6pt)

#grid(columns: (1fr, 1fr, 1fr), column-gutter: 10pt, align: center + top,

  // -------- COL 1: vision --------
  stack(dir: ttb, spacing: 8pt,
    node("ECG image", sub: [12-lead plot rendered to RGB, 1344×672], fill: c_img, w: 6.8cm),
    arl[1344 × 672 × 3],
    node("AnyRes tiling (flat)", sub: [best-fit 672×336 → DOWNSCALE, then \ 2×1 = 2 local tiles + 1 global = 3 tiles, each 336×336], fill: c_cat, tag: "anyres", w: 6.8cm),
    arl[3 × (336 × 336 × 3)],
    node("CLIP ViT-L/14 @ 336", sub: [frozen vision encoder \ patch 14 → 24×24 = 576 tokens/tile, drop CLS, layer −2], fill: c_frz, tag: "frozen", w: 6.8cm),
    arl[3 × 576 × 1024],
    node("MLP projector (mlp2x_gelu)", sub: [1024 → 4096 → 4096 — frozen], fill: c_frz, w: 6.8cm),
    arl[3 × 576 = 1728 visual tokens × 4096],
    v(4pt),
    node([Task prompt $q_t$ + ⟨image⟩ token], sub: [diagnostic / rhythm / form; ⟨image⟩ = placeholder (index −200)], fill: c_img, w: 6.8cm),
    arl[text token sequence (one ⟨image⟩ token)],
    node("Tokenizer + Embedding (Vicuna)", sub: [embedding table 32000 × 4096], fill: c_frz, w: 6.8cm),
    arl[T × 4096 (text tokens)],
  ),

  // -------- COL 2: fusion + LLM --------
  stack(dir: ttb, spacing: 8pt,
    v(8pt),
    node("Multimodal fusion", sub: [REPLACE the ⟨image⟩ token with 1728 visual tokens: \ [text before] ⊕ [visual tokens] ⊕ [text after]], fill: c_cat, tag: "fusion", w: 8cm),
    arl[(1728 + T) × 4096  •  image positions set to IGNORE_INDEX in loss],
    node("Vicuna-7B — 32 Transformer decoder layers", sub: [hidden 4096, 32 heads, intermediate 11008, context 4096 \ 4-bit NF4 base FROZEN + QLoRA tuning (r=16, α=32)], fill: c_llm, tag: "QLoRA", w: 8cm),
    arl[(1728 + T) × 4096],
    node("LM head", sub: [Linear 4096 → 32000, untied — frozen], fill: c_frz, w: 8cm),
    arl[(1728 + T) × 32000 (vocabulary logits)],
    node("Extract logits at label-block positions", sub: [fixed teacher-forced label block; take logit pair (id\"0\", id\"1\")], fill: c_cat, tag: "TF", w: 8cm),
    arl[K_t × 2 (one logit pair per code)],
  ),

  // -------- COL 3: output / eval --------
  stack(dir: ttb, spacing: 8pt,
    v(8pt),
    node([Training: next-token cross-entropy], sub: [PULSE's native loss on label tokens \ (optional class-balanced reweighting)], fill: c_eval, w: 7cm),
    v(10pt),
    align(center)[#text(9pt, style: "italic")[— at inference / evaluation —]],
    v(6pt),
    node([Two-way softmax per code], sub: [$P(y_k=1) = e^(z_1) \/ (e^(z_0)+e^(z_1))$], fill: c_head, w: 7cm),
    arl[K_t probabilities ∈ [0,1]],
    node([Multi-label probability vector], sub: [diagnostic 44 · rhythm 12 · form 19], fill: c_head, w: 7cm),
    arl[grouped per task],
    node([Evaluation: macro-AUC (primary) · macro-F1], sub: [val-frozen per-class threshold], fill: c_eval, w: 7cm),
  ),
)
