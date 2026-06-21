# GraphRAG PDF Q&A — Single Flask App

One Flask process serving both the JSON API and the rendered UI, wired
directly onto your existing pipeline (parser → processor →
semantic_chunker → extractor → relationship → graph_builder →
retriever → synthesizer). No stubs — every ingestion stage and the
query/synthesis stage call your real classes, in-process (no HTTP hop
between "frontend" and "backend" — they're the same app).

## 1. Where these files go

This app assumes `webapp/` sits **at your project root**, as a sibling
of your existing packages:

```
your-project/
├── webapp/                  <- from this delivery
├── parser/                   <- yours
├── processor/                  <- yours
├── semantic_chunker.py           <- yours (root-level module)
├── extractor/                      <- yours
├── relationship/                     <- yours
├── graph_builder/                      <- yours
├── retriever/                            <- yours
├── synthesizer/                            <- yours
├── data/pdfs/                                <- yours (uploads land here)
├── output/                                     <- yours (chunks/entities/etc.)
├── .env
└── pyproject.toml                                <- yours
```

Copy the `webapp/` folder from this delivery into your project root,
alongside your existing folders.

## 2. Install

```bash
# From your project root, with your existing venv active
pip install -r webapp/requirements.txt
```

Your existing `pyproject.toml` deps (langchain, langchain-mistralai,
neo4j, sentence-transformers, faiss-cpu, pydantic, etc.) are assumed
already installed since your CLI pipelines already run.

## 3. Configure

Copy `webapp/.env.example` values into your existing `.env` (don't
overwrite it — merge). The `MISTRAL_API_KEY` and `NEO4J_*` vars should
already be there from your existing setup; the only new ones are the
**tuning** and **Flask server** sections.

Note: `graph_builder/connection.py` reads `NEO4J_URI`, `NEO4J_USER`
(not `NEO4J_USERNAME`), `NEO4J_PASSWORD`, and `NEO4J_DATABASE` — this
`.env.example` uses the exact names it expects.

## 4. Run

From your project root:

```bash
python -m webapp.app
```

or with the Flask CLI (auto-reload on code changes):

```bash
flask --app webapp.app run --debug --port 5000
```

Then open `http://localhost:5000` in your browser. Everything — upload,
chat, graph explorer, dashboard — lives in this one app on this one
port.

## 5. Pages

| Path | Purpose |
|---|---|
| `/` | Overview of the ingestion + query pipelines |
| `/upload` | Upload a PDF, watch it move through all 7 ingestion stages live |
| `/chat` | Ask questions about the ingested document; see confidence, evidence, reasoning path, graph relationships |
| `/graph` | Interactive knowledge-graph view (vis-network), filterable by entity name |
| `/dashboard` | Stats for the currently-ingested document |

## 6. JSON API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness check |
| POST | `/api/upload` | Upload a PDF (`multipart/form-data`, field `file`); starts background ingestion, returns `job_id` |
| GET | `/api/status/<job_id>` | Poll ingestion progress/stage/log |
| GET | `/api/documents` | Info about the currently-ingested document |
| GET | `/api/graph` | `{entities: [...], relationships: [...]}` |
| POST | `/api/query` | `{"query": "..."}` → answer + evidence + confidence + `out_of_scope` flag |

## 7. Why a single Flask app instead of a separate backend/frontend

A split (e.g. FastAPI backend + a separate frontend service) adds an
HTTP hop between the UI and the API for every page load and every
query, plus two processes to run, two requirements files, and CORS
config. Since this app calls your pipeline classes directly from the
same Python process that renders the pages, there's no serialization
or network overhead between "frontend" and "backend" — just one
`pip install`, one process, one port. Background ingestion jobs run on
a plain `threading.Thread`, which is what FastAPI's `BackgroundTasks`
does under the hood for sync functions anyway — so there's no
capability lost by dropping FastAPI here.

## 8. Known limitations

- **Single document at a time.** Your `PDFParser.parse(..., clean=True)`
  wipes `output/*.json` before each parse, so uploading a new PDF
  replaces the previous document's chunks/entities/relationships/index.
  The dashboard reflects "the currently ingested document," not a
  multi-doc library. To support multiple documents concurrently, you'd
  need to namespace `output/` per-document (e.g.
  `output/<doc_id>/chunks.json`) and pass a `doc_id` through `/api/upload`,
  `/api/query`, `/api/graph` — a reasonable next step if you need it.
- **In-memory job store.** Restarting the Flask process loses ingestion
  job history (the *files* in `output/` and Neo4j data persist fine —
  only the progress-tracking UI state resets).
- **No re-validation step.** Your CLI has `graph_builder/validator.py`
  for pre/post-ingestion checks; `/api/upload` calls
  `GraphIngestionService.ingest_all()` directly without it. Easy to add
  if you want that parity.
- **Dev server only.** `app.run(...)` is Flask's built-in dev server.
  For real deployment, run behind `gunicorn` (already listed, commented
  out, in `requirements.txt`) — e.g.
  `gunicorn -w 1 -b 0.0.0.0:5000 'webapp.app:app'`. Keep workers at 1
  unless you move the in-memory `job_store` to something shared (Redis,
  a DB), since each worker process would otherwise have its own job
  history and cached retrieval pipeline.

## 9. Graph Explorer dependency note

The graph view uses [vis-network](https://visjs.github.io/vis-network/)
loaded from a CDN (`unpkg.com`) in `templates/graph.html`. Your repo
already bundles a local copy at `lib/vis-9.1.2/` for the existing
`frontend/` — if you'd rather not depend on a CDN at runtime, point
`graph.html`'s `<script>`/`<link>` tags at that local copy instead.
