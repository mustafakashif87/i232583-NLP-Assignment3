# 🧠 NLP Assignment 3
### Transformer RAG Pipeline for Amazon Review Understanding

> Build a Transformer from scratch. Make it retrieve. Make it explain itself.
---

## What This Does

Takes an Amazon review → predicts sentiment → retrieves similar reviews → writes a one-sentence explanation. All in PyTorch, all from scratch.

```
"Battery died after one week..."
        │
        ▼
  [Encoder]  →  Negative · Medium-length
        │
        ▼
  [Retrieval]  →  3 similar bad-battery reviews
        │
        ▼
  [Decoder]  →  "this review is negative because the reviewer
                 expresses negative opinions in a medium review"
```

---

## The Three Parts

### ⚡ Part A — Encoder
Custom encoder-only Transformer. Two tasks, one backbone.

- `d_model=128` · `n_heads=4` · `n_layers=2` · Pre-LayerNorm
- **Task 1:** Sentiment (Neg / Neu / Pos) — weighted F1 **~0.84**
- **Task 2:** Review length bucket (Short / Med / Long) — weighted F1 **~0.85**
- Loss: `L = 1.0 × L_sentiment + 0.5 × L_derived`
- CLS token output → 128-dim embedding used for retrieval

> Neutral class is the hardest (F1 ~0.50) — 3-star reviews are genuinely ambiguous.

---

### 🔍 Part B — Retrieval
Cosine similarity over all 25,200 training embeddings. No FAISS, no libraries — pure NumPy matrix multiply.

```
sim(q, c) = (q · c) / (||q|| × ||c||)
```

- Top **k=3** retrieved reviews fed into the decoder context
- Pre-normalised corpus matrix → single matmul per query
- Top-1 retrieved review matches query sentiment the **majority of the time**

---

### ✍️ Part C — Decoder
GPT-style causal Transformer conditioned on a structured prompt.

```
[REVIEW] ... [SENTIMENT] ... [FEATURE] ... [CONTEXT] ... [EXPLANATION] → generated here
```

- Causal masking · greedy decoding · loss masked to explanation tokens only
- **Full RAG** achieves lower test perplexity than the no-retrieval baseline ✅

---

## Results at a Glance

| System | Sentiment F1 | Length F1 | Decoder PPL |
|--------|-------------|-----------|-------------|
| Encoder | ~0.84 | ~0.85 | — |
| RAG (k=3) | — | — | lower ✅ |
| Baseline (no retrieval) | — | — | higher ❌ |

---

## Repo Structure

```
├── i232583-NLP-Assignment3.ipynb   ← run this top-to-bottom
├── models/
│   ├── encoder.pt
│   ├── decoder.pt
│   └── decoder_baseline.pt
└── results/
    ├── train_embeddings.pkl        ← 25,200 CLS vectors
    └── vocab.pkl
```

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| Pre-LayerNorm | More stable gradients than Post-LN at shallow depth |
| CLS pooling | Clean fixed-size embedding for both heads + retrieval |
| Cosine similarity | Magnitude-invariant; better than Euclidean for dense vectors |
| k=3 retrieval | Balances diversity vs. decoder context budget |
| Loss masking | Decoder only trains on the explanation, not the prompt |
| Warmup + cosine LR | Best stability across all 15 hyperparameter runs |

---

*Department of Computer Science — Natural Language Processing*
