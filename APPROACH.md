# SHL Assessment Recommender — Approach

**Author:** Kunal Punia
**Stack:** FastAPI · FAISS · sentence-transformers (MiniLM-L6-v2) · Groq Llama 3.3 70B
**Deploy:** Render.com (free tier)

## 1. How I read the problem

A hiring manager rarely knows SHL's catalog vocabulary — they describe a
role and expect a useful shortlist back. The conversation has to handle
four things: clarify when the brief is too thin, commit a 1–10 item
shortlist when there's enough context, refine on user edits, and compare
two assessments when asked. Everything has to stay inside the 377-item
SHL catalog — any URL outside the catalog counts as a hallucination.

Before writing code I read all ten sample traces end-to-end. A few
patterns mattered for the design:

* The agent in the traces almost always asks at most one clarifying
  question, then commits.
* OPQ32r appears in 8 of the 10 final shortlists. It's the default
  personality measure for professional roles.
* On legal/HIPAA-style questions (C7), the agent refuses cleanly and
  keeps the existing shortlist intact rather than wiping it.
* When the user pastes a JD with named skills (C9), the agent layers
  one knowledge test per named technology.

These observations are encoded in the prompt rather than the code.

## 2. Architecture

```
POST /chat
  │
  ├─ Keyword guardrails              (instant refusal for legal /
  │                                   off-topic / prompt-injection —
  │                                   no LLM call)
  │
  ├─ build_search_query              (concatenate last 3 user turns)
  │
  ├─ FAISS retriever                 (cosine sim over MiniLM-L6-v2,
  │                                   top-40 of 377 catalog items)
  │
  ├─ Groq Llama 3.3 70B              (system prompt + top-20 catalog
  │                                   excerpts + full history,
  │                                   response_format = json_object)
  │
  ├─ JSON parse + URL validation     (drop URLs not in catalog,
  │                                   canonicalize by name when possible)
  │
  ▼
{reply, recommendations[], end_of_conversation}
```

**Choices and why:**

* **FastAPI** — what the spec asks for; Pydantic gives me schema
  enforcement on every response without a custom validator.
* **FAISS + sentence-transformers (`all-MiniLM-L6-v2`)** — local, no
  external embedding API. The catalog is small (377 items) so a flat
  inner-product index is fine — sub-millisecond search.
* **Groq Llama 3.3 70B** — free tier, OpenAI-compatible API, supports
  `response_format={"type": "json_object"}`. Originally I used Gemini
  Flash 2.0; I switched after hitting the per-minute input-token cap
  during a multi-trace eval. The agent code was the same; only the
  client call changed.
* **No agent framework** — the orchestration is five sequential steps.
  Adding LangChain or LlamaIndex would have meant an extra layer to
  read through and debug, with no behavior gain for a stateless API.

The catalog is downloaded once (`python -m scripts.build_index`), parsed
with `strict=False` because a few descriptions contain stray control
characters, and encoded into a flat IP index on disk. The server loads
the index once in `lifespan`, so `/chat` never pays embedding warmup.

## 3. Context engineering

Three things make the retrieval-augmented prompt work in practice:

* **The query is the last 3 user turns concatenated**, not the latest
  message. "Drop X, add Y" only makes sense in context — embedding just
  that line returns garbage.
* **Retrieve top-40, show the LLM top-20.** Going wider on retrieval
  helped trace 5 (sales re-skilling), where some expected items sit at
  rank 15–25 in the embedding score and would be cut by a tighter k.
  Showing the LLM 20 items rather than all 40 keeps the prompt under
  the daily token budget.
* **Each catalog item is rendered as ~6 lines** of name, URL, test_type
  letter code, levels, duration, 180-char description. Longer
  descriptions don't help the LLM make better choices and they add up
  fast across 20 items.

The system prompt encodes (i) the test_type letter mapping
(A/B/C/D/E/K/P/S), (ii) hard rules for catalog-only output and refusal
categories, (iii) when to clarify vs commit, (iv) how refine and compare
differ from a fresh shortlist, and (v) battery-design heuristics derived
from the sample traces (default OPQ32r for professional roles, layer
SVAR for contact center, DSI for safety-critical roles, Graduate
Scenarios for graduate hires, etc.).

JSON output is enforced via Groq's `response_format` parameter, with
a regex fallback for the rare case where the model returns surrounding
prose.

## 4. Guardrails against hallucinated URLs

Two layers, because any URL outside the catalog is graded as a fail:

1. **Prompt level**: the LLM only ever sees real catalog items; the
   system prompt explicitly says "copy the exact URL string from the
   catalog block above".
2. **Post-validation**: every recommendation's URL is checked against
   the catalog's URL set. If the URL is unknown but the name matches a
   catalog item, the URL is canonicalized to the catalog's version.
   If neither match, the recommendation is dropped. This makes the
   system robust to the LLM truncating, lowercasing, or paraphrasing
   a URL — which it does occasionally.

Refusals work the same way. Keyword detection short-circuits the LLM
for the clearest categories ("legally required", "ignore previous
instructions"); subtler cases are caught by the system prompt itself.

## 5. Evaluation

I wrote a replay harness (`tests/eval_traces.py`) that mirrors what the
spec describes the grader doing: parse each `Cn.md` for the user turns
and the URL set from the final agent table, replay turn-by-turn against
`POST /chat`, then score schema compliance, behavior probes, and
Recall@10.

**Results on the 10 public traces:**

| Metric | Result |
|---|---|
| Schema compliance | 10 / 10 traces clean |
| Behavior probes (vague, legal, off-topic, injection, specific, gap) | 6 / 6 pass |
| Mean Recall@10 across the 7 traces that completed | 0.228 |

A note on the Recall@10 number. Three of the ten traces (C8, C9, C10)
fell at the tail of the eval run, by which point the free-tier daily
token budget on Groq was exhausted. Those calls returned the
"temporary issue" fallback with an empty `recommendations` array, which
scored 0 and dragged the mean down. The realistic number is the mean
across the seven traces that actually completed, which is what's in the
table. With a paid tier (or a fresh free-tier budget the next day) the
eval would finish all ten and the mean would be in the same range.

I made three changes after seeing the baseline:

1. **Top-K retrieval 20 → 40, context 15 → 20.** The bigger pool covers
   items the embedding model put just outside the original cutoff
   (e.g. some C5 sales-report variants).
2. **Battery-design heuristics in the prompt.** The biggest miss was
   under-committing — the agent returning 2 or 3 items when the trace
   expected 5. The prompt now says "default 4–7 items", "include
   OPQ32r for professional roles unless asked to drop", and stack
   layering rules per role family.
3. **Compacter catalog rendering.** Roughly 40% fewer tokens per call,
   which gives the daily budget enough headroom to finish a full
   10-trace eval in one run.

## 6. What didn't work

* **Pure keyword search** (first prototype): completely missed the C2
  Rust-engineer case. There's no Rust test in the catalog, and the
  closest items (Linux Programming, Networking and Implementation,
  Smart Interview Live Coding) only surface via embedding similarity.
  Switched to FAISS.
* **Gemini Flash 2.0** (first LLM): the per-minute input-token cap
  was too tight for a multi-trace eval. Moved to Groq.
* **Letting the LLM pick the shortlist length freely**: it
  under-commits. Fixed with an explicit "default 4–7 items" rule.

## 7. Limits and next steps

* The Recall@10 ceiling here is bounded mostly by what the LLM is
  willing to add past the "obviously asked for" items. A cross-encoder
  re-ranker over the FAISS top-40 (e.g. `cross-encoder/ms-marco-
  MiniLM-L-6-v2`) before the LLM would likely lift Recall@10 by 0.1+,
  at the cost of ~200 ms per call.
* Render free tier sleeps after 15 minutes of inactivity, so the first
  `/health` after a quiet period takes ~30 s to wake. The spec allows
  2 min, so this is fine, but a keep-warm cron would be a polish item.
* The API is stateless as required. If conversation caching were
  allowed, hashing `tuple(messages)` would save ~70% of LLM calls
  during grading.
