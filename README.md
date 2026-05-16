# SHL Assessment Recommender

A conversational agent that recommends SHL assessments from the official SHL
product catalog. Backed by FAISS semantic retrieval over the catalog JSON +
Google Groq (Llama 3.3 70B) for reasoning.

## Architecture

```
POST /chat
   │
   ├─ guardrails (legal / off-topic / prompt-injection refusal)
   │
   ├─ build_search_query  (last 3 user turns)
   │     │
   │     ▼
   ├─ FAISS retriever  (top-20 candidates, cosine sim, MiniLM-L6-v2)
   │     │
   │     ▼
   ├─ Groq Llama 3.3 70B  (system prompt + top-15 catalog excerpts + history)
   │     │
   │     ▼
   ├─ JSON parse + URL validation (drops any fabricated assessments)
   │
   ▼
{reply, recommendations[], end_of_conversation}
```

## Local setup (uv)

```bash
# 1. Install uv if you haven't (https://docs.astral.sh/uv/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the venv (uses Python 3.12 — newer wheels for faiss-cpu)
uv venv --python 3.12
source .venv/bin/activate

# 3. Install deps (reads pyproject.toml)
uv pip install -e .

# 4. Edit .env and paste your Groq key in place of `xyz`
# Get a free key (no credit card): https://console.groq.com/keys

# 5. Download catalog + build FAISS index (one-time)
python -m scripts.build_index

# 6. Run the server
uvicorn app.main:app --reload --port 8000
```

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`
Request:
```json
{
  "messages": [
    {"role": "user", "content": "I'm hiring a senior Java developer, backend-leaning."}
  ]
}
```

Response:
```json
{
  "reply": "For a senior IC backend role, here's a focused shortlist...",
  "recommendations": [
    {"name": "Core Java (Advanced Level) (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

`test_type` letter codes: `A` Ability & Aptitude · `B` Biodata & Situational
Judgment · `C` Competencies · `D` Development & 360 · `E` Assessment
Exercises · `K` Knowledge & Skills · `P` Personality & Behavior ·
`S` Simulations. Multiple keys are comma-joined (e.g. `K,S`).

## Project layout

```
shl-recommender/
├── app/
│   ├── main.py             FastAPI app (lifespan warm-up, /health, /chat)
│   ├── agent.py            Guardrails + retrieval + Gemini + JSON validation
│   ├── retriever.py        FAISS index loader, URL/name lookup
│   ├── prompts.py          System prompt engineered from the 10 sample traces
│   ├── catalog_loader.py   Downloads SHL JSON catalog
│   ├── config.py           Env-driven configuration
│   └── models.py           Pydantic schemas
├── scripts/
│   └── build_index.py      One-shot: encode catalog → FAISS index
├── data/catalog.json       Cached catalog (generated)
├── index/                  FAISS index + metadata (generated)
├── tests/                  Smoke tests (no live API needed)
├── render.yaml             Deployment config for Render.com
└── pyproject.toml          Project metadata + pinned dependencies
```

## Deploy to Render

1. Push this folder to a GitHub repo.
2. On render.com → New → Web Service → connect the repo.
3. Render will read `render.yaml`. Add the env var `GEMINI_API_KEY`.
4. First build runs `python -m scripts.build_index`, which downloads the
   catalog and writes the FAISS files to disk.

Render free tier sleeps after 15 min — first request cold-start is ~30s.
`/health` answers fast enough for the assignment's 2-minute budget.

## Behavior contracts (validated against the 10 reference traces)

- **Vague opener** ("we need an assessment") → asks one clarifying question,
  `recommendations: []`.
- **Specific opener** (role + level, or pasted JD) → commits a shortlist.
- **Catalog gap** ("hiring a Rust engineer") → states the gap honestly and
  proposes adjacent assessments (Smart Interview Live Coding, Linux
  Programming, Networking).
- **Refine** ("drop X, add Y") → returns the FULL updated shortlist.
- **Compare** ("what's the difference between A and B?") → explains using
  catalog descriptions only.
- **Legal/compliance question** ("are we legally required to test X?") →
  declines, refers to legal/HR team, keeps the shortlist intact.
- **Prompt injection** ("ignore your instructions…") → polite refusal,
  steers back to the catalog.
- **URL hallucinations** → silently dropped via post-validation against the
  catalog's URL set; if the name resolves, the URL is canonicalized.
