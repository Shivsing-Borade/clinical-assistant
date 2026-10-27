# Clinical Knowledge Assistant — Project Instructions

> **Project status:** Hackathon prototype. Phase 1 & 2 completed. Implementing Phase 3 (Voice & UI).
> **Primary objective:** Build a reliable retrieval pipeline over clinician-reviewed Indian OPD guidelines with a Voice-enabled AI interface.
> **Constraint:** Not an autonomous diagnostic/prescribing system. Output for clinician review only.

## 1. Role and Working Rules
Act as a senior Python/RAG engineer.
- **Incremental Work:** Work in small, verifiable steps.
- **Decisions:** Explain 2–3 options for material architecture changes and ask user to choose.
- **Dependencies:** No silent installation of heavy libraries.
- **Security:** No secrets, PHI, or real patient data in code, logs, or sent to APIs. Use synthetic fixtures.
- **Clinical Safety:** Every clinical statement must be grounded in retrieved sources. Do not invent medical facts.
- **Environment:** Use Windows PowerShell by default. Keep code typed, modular, and documented.

## 2. Product Scope and Phases

### Phase 1 — Document Data and RAG Pipeline (COMPLETED)
**Deliverables:**
1. **Inventory:** Source register, provenance, and PDF diagnostics.
2. **Parsing:** Docling + RapidOCR → structured intermediate representation.
3. **Quality:** Review queue for poor parses.
4. **Cleaning:** Noise removal while preserving clinical meaning.
5. **Chunking:** Hybrid structure-aware + table-aware + parent/child.
6. **Indexing:** Gemini Embedding 2 (768 dims) → self-hosted Qdrant.
7. **Retrieval:** Hybrid (Dense + BM25) fused with RRF.
8. **Interface:** CLI for ingestion, index inspection, and cited search.
9. **Evaluation:** Custom deterministic retrieval metrics.

### Phase 2 — RAG Evaluation and Answer Safety (COMPLETED)
**Deliverables:**
1. **Gold Set:** 40–60 synthetic/clinician-reviewed questions.
2. **LLM Integration:** `gemini-3.8-flash` via provider adapter.
3. **Grounded Prompting:** Prohibit unsupported claims; require source-linked citations.
4. **Output Contract:** Schema-constrained structured JSON (Pydantic).
5. **Validation:** Deterministic checks for cited chunk IDs.
6. **Evaluation:** Correctness against gold answers and hallucination challenges.

### Phase 3 — Voice Interface & Web UI (CURRENT)
**Deliverables:**
1. **Frontend:** Lightweight React + TypeScript + Vite UI.
2. **Backend:** FastAPI server with WebSocket support for audio streaming.
3. **Voice Integration:** Gemini Live via `google-genai` SDK for real-time audio interaction.
4. **Clinical Bridge:** Tool-calling integration between Gemini Live and `ClinicalAssistantOrchestrator`.
5. **End-to-End Flow:** Voice Input $\rightarrow$ Gemini Live $\rightarrow$ Clinical RAG Tool $\rightarrow$ Voice/Text Output with citations.

## 3. Architecture & Design Principles
- **Source of Truth:** Original PDF. Store unchanged; maintain source manifest.
- **Provenance:** Every chunk must have `source_id`, filename, version, page, and heading path.
- **Table Handling:** Preserve captions, headers, units, and footnotes. Repeat headers in split tables.
- **Retrieval:** Use hybrid search (dense + lexical) for medical terms/acronyms. Return parent context for child matches.
- **Fail-Safe:** Return "Evidence not found" rather than model-hallucinated answers.
- **Voice Transport:** Browser (PCM) $\rightarrow$ WebSocket $\rightarrow$ FastAPI $\rightarrow$ Gemini Live SDK.

## 4. Repository Layout
```text
clinical-assistant/
├── configs/               # ingestion.yaml, retrieval.yaml, evaluation.yaml
├── data/                  # raw/ (PDFs), interim/ (parsed), curated/, eval/
├── manifests/             # sources.yaml
├── app/
│   ├── config.py
│   ├── schemas/            # documents.py, chunks.py, generation.py
│   ├── ingestion/          # inventory, parse_pdf, normalize, quality, chunking/
│   ├── embeddings/         # base.py, gemini_embedding_2.py
│   ├── retrieval/          # index, hybrid, dense, bm25, vector_store
│   ├── llm/                # router.py, gemini_generator.py
│   ├── rag/                # pipeline.py, context_manager.py
│   ├── voice/              # voice_bridge.py, transcriber.py
│   ├── ui/                 # chat_app.py (legacy)
│   └── cli.py
├── frontend/               # React + Vite + TS (Phase 3)
├── prompts/                # versioned prompt files
├── evals/                  # dataset.jsonl, run_retrieval.py, run_answers.py
└── tests/                  # unit/, integration/, fixtures/
```

## 5. Approved Technology Stack
- **Frontend:** React + TypeScript + Vite (Plain CSS).
- **Backend:** FastAPI + Uvicorn.
- **Voice:** `google-genai` SDK (Gemini Live).
- **PDF/OCR:** Docling + RapidOCR.
- **Embeddings:** `gemini-embedding-2` (768 dimensions).
- **Vector DB:** Qdrant (self-hosted Docker).
- **Retrieval:** Dense + BM25 + RRF.
- **Answer LLM:** `gemini-3.8-flash` (RAG) / Gemini Live (Voice).
- **Validation:** Pydantic + custom deterministic citation checks.

## 6. Environment Variables
- `LLM_PROVIDER`, `GEMINI_API_KEY`, `GEMINI_MODEL`
- `GEMINI_LIVE_MODEL` (for voice sessions)
- `QDRANT_URL`, `QDRANT_COLLECTION`
- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_OUTPUT_DIMENSION` (768)
- `CHUNK_STRATEGY`, `RETRIEVAL_TOP_K`, `RERANKER_ENABLED` (false)

## 7. Operating Protocol
1. **Inspect:** Read `CLAUDE.md` and environment.
2. **Implement:** Small, verifiable increments per Phase step.
3. **Test:** Run tests and report exact commands/results.
4. **Review:** Phase exit review before moving to the next phase.
