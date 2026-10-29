# ⚡ NexusRAG — Enterprise Multimodal Agentic Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/React-18.3+-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 18"/>
  <img src="https://img.shields.io/badge/TypeScript-5.7+-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript"/>
  <img src="https://img.shields.io/badge/ChromaDB-HNSW_Cosine-FF6F00?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Tests-100%25_Offline_Passing-success?style=for-the-badge&logo=githubactions&logoColor=white" alt="Tests"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License MIT"/>
</p>

---

## 🎯 Executive Summary & Problem Statement

> **Most standard RAG implementations break the moment real-world documents enter the pipeline.**
> Naive chunking flattens complex visual plots into illegible text strings, mangles relational tables into unparseable tokens, fails on audio recordings or video keyframes, and suffers from semantic drift and hallucinated citations.

**NexusRAG** is a production-grade, state-of-the-art **Multimodal Agentic Retrieval-Augmented Generation (RAG)** platform. Built to parse, cross-examine, and synthesize complex multi-format document archives in real time, it couples:
- **Hierarchical Parent-Child Ingestion**: Extracts multi-column text, high-resolution visual diagrams (via PyMuPDF & Gemini Vision), structured tables (via pdfplumber & Pandas), audio transcripts, and video keyframes.
- **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**: Combines dense semantic vector search (ChromaDB) and sparse keyword search (BM25Okapi) with neural cross-encoder re-ranking.
- **Agentic Query Routing & HyDE**: Dynamically classifies query intent, executes zero-shot hypothetical passage generation, and decomposes comparative questions into multi-step sub-queries.
- **Grounded Multimodal Synthesis**: Generates streaming token-by-token responses with strict, interactive footnote citations linking directly to visual figures and source excerpts.
- **Academic Evaluation Framework**: Impartial LLM-as-a-Judge benchmarking across Faithfulness, Answer Relevancy, Context Precision, and Context Recall.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["1. Multi-Format Hierarchical Ingestion Pipeline"]
        A[Multi-Format Uploads<br/>PDF, DOCX, PPTX, CSV, XLSX, Audio, Video, Image] --> B[Modality Router & Extractors]
        B --> C1[PyMuPDF Vector & Raster<br/>Diagram Extractor @ 150 DPI]
        B --> C2[pdfplumber & Pandas<br/>Tabular Schema Extractor]
        B --> C3[Gemini Vision Dense<br/>Captioning & Visual OCR]
        B --> C4[Timestamped Audio & Video<br/>Keyframe Chunking]
        
        C1 & C2 & C3 & C4 --> D[Hierarchical Parent-Child Chunker<br/>Child: 400 chars | Parent: 1200 chars]
    end

    subgraph DUAL_INDEX ["2. Dual Inverted & Vector Indices"]
        D --> E1[(ChromaDB Dense Vectors<br/>all-MiniLM-L6-v2 Cosine Space)]
        D --> E2[(BM25Okapi Sparse Index<br/>In-Memory Inverted Corpus)]
    end

    subgraph AGENTIC_ROUTER ["3. Agentic Routing & Query Expansion"]
        U[User Query + Optional Image Attachment] --> R{Agentic Query Router}
        R -->|Direct Greeting / Meta| DIR[Direct Conversational Engine]
        R -->|Quantitative / Relational| TAB[Tabular Schema Router]
        R -->|Multi-Part / Comparative| DEC[Multi-Step Query Decomposer]
        R -->|Factual / Research| HYD[HyDE Hypothetical<br/>Document Expansion]
    end

    subgraph RETRIEVAL_FUSION ["4. Hybrid Retrieval & Neural Re-Ranking"]
        DEC & HYD & TAB --> F1[Dense Cosine Retrieval<br/>Candidate Pool k=15]
        DEC & HYD & TAB --> F2[Sparse BM25 Keyword Search<br/>Candidate Pool k=15]
        
        F1 & F2 --> G[Reciprocal Rank Fusion RRF<br/>k=60 | Dense w=0.6 | Sparse w=0.4]
        G --> H[Cross-Encoder Neural Re-Ranker<br/>ms-marco-MiniLM-L-6-v2]
    end

    subgraph SYNTHESIS ["5. Grounded Multimodal Synthesis & Delivery"]
        H --> S[Multimodal Synthesizer Engine<br/>Gemini Multimodal Context Window]
        S --> SSE[Real-Time Server-Sent Events<br/>Token-by-Token Streaming]
        S --> CIT[Interactive Footnote Badges<br/>Doc, Fig, Table, Audio]
        S --> MEM[(Persistent SQLite Sliding Memory<br/>4-Turn Context + Rolling Summary)]
        S --> EVAL[Automated LLM-as-a-Judge<br/>Faithfulness, Relevancy, Precision, Recall]
    end

    CIT --> UI[React 18 SPA + Marginalia Figure Drawer]
    SSE --> UI
```

---

## ⚖️ NexusRAG vs. Traditional Naive RAG

| Architectural Dimension | Naive / Tutorial RAG | NexusRAG Enterprise Architecture |
|---|---|---|
| **Supported File Formats** | Plain `.txt` or raw text-only `.pdf` | **17 formats**: `.pdf`, `.docx`, `.pptx`, `.csv`, `.xlsx`, `.png`, `.jpg`, `.webp`, `.mp3`, `.wav`, `.m4a`, `.mp4`, `.webm`, `.mov` |
| **Visual Charts & Figures** | Discarded or lost in extraction | Extracted at **150 DPI**, captioned via **Gemini Vision OCR**, displayed in **interactive Marginalia drawer** |
| **Tabular Data** | Flattened into unstructured text strings | Preserved as **Markdown tables & HTML schemas** with row/column coordinate metadata |
| **Chunking Strategy** | Fixed character count (e.g. 500 chars) | **Hierarchical Parent-Child**: Granular 400-char retrieval child linked to 1200-char parent synthesis context |
| **Retrieval Mechanism** | Vector search only (semantic drift on IDs/codes) | **Hybrid Search**: Dense vectors + BM25Okapi inverted index fused via **Reciprocal Rank Fusion (RRF)** |
| **Ranking Pipeline** | Naive top-$k$ cosine similarity | Two-stage: Candidate generation pool ($k=15$) re-ranked via full **Cross-Encoder Neural Attention** |
| **Query Routing** | Blind retrieval on every prompt | **Agentic Router**: Routes between `DIRECT`, `RETRIEVAL`, `MULTI_STEP` decomposition, and `TABLE_LOOKUP` |
| **Footnote Verification** | Hallucinated or generic file mentions | Strict grounded citations: `[Doc 1 (Pg 2)]`, `[Fig. 1 (Pg 3)]`, `[Table 2]`, `[Audio 1]` |
| **Streaming Protocol** | Blocking HTTP request or plain text stream | **SSE Protocol**: Typed event stream (`route` $\rightarrow$ `token` $\rightarrow$ `done` with structured citations) |
| **Session Memory** | Stateless or unbounded memory leak | **Persistent SQLite Sliding Window**: Last 4 turns in context + rolling asynchronous background summary |
| **Resilience & Testing** | Fails on upstream 503/504 errors; unmocked tests | **Grounded fallback on 503/504 outages**; **100% offline automated test suite** running in 0.03s |

---

## 🔬 Core Engineering Highlights & Mathematical Rigor

### 1. 🗂️ Multi-Format Hierarchical Ingestion Pipeline
- **Dual-Stream PDF Parsing**: High-resolution text extraction combined with `pdfplumber` for tabular structures and `PyMuPDF` (`pymupdf`) for embedded graphics. Vector drawings with sufficient stroke complexity are rasterized at 150 DPI.
- **Dense Visual Captioning & OCR**: All extracted technical figures are processed through vision LLMs with structured prompts extracting: Visual Type (e.g., Bar Chart, Architecture, Scatter Plot), Numerical Trends & Axis Bounds, and Verbatim OCR Formulae.
- **Parent-Child Chunking**:
  - **Child Chunks** (400 chars, 80 overlap): Optimized for dense embedding vector specificity and precise BM25 keyword matching.
  - **Parent Chunks** (1200 chars, 150 overlap): Injected into LLM synthesis context to ensure complete paragraph continuity without truncation.

### 2. 🔍 Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
Vector search alone struggles with exact part numbers, model names, and domain-specific acronyms. NexusRAG unifies dense semantic vectors and sparse BM25Okapi using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF\_score}(d) = \frac{w_{\text{dense}}}{k_{\text{rrf}} + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{k_{\text{rrf}} + \text{rank}_{\text{sparse}}(d)}$$

Where:
- $k_{\text{rrf}} = 60$ (smoothing constant preventing top-rank skew)
- $w_{\text{dense}} = 0.6$, $w_{\text{sparse}} = 0.4$
- Candidate pools of top-$15$ documents from each channel are merged and re-ranked via a neural cross-encoder (`ms-marco-MiniLM-L-6-v2`) to eliminate semantic false-positives.

### 3. 🤖 Agentic Query Router & HyDE Expansion
- **Dynamic Query Classification**:
  - `DIRECT`: Greetings or meta-inquiries resolved instantaneously without database overhead.
  - `RETRIEVAL`: Single-step factual inquiry over multimodal document index.
  - `MULTI_STEP`: Decomposes complex comparative questions into sub-queries, executes parallel retrieval, and synthesizes across evidence.
  - `TABLE_LOOKUP`: Targets structured tabular chunks for quantitative calculations.
- **HyDE (Hypothetical Document Embeddings)**: Zero-shot hypothetical passage generation bridges the semantic gap between concise user questions and long document passages.

### 4. 🛡️ Resilience & Outage Protection (503/504 Handling)
Upstream LLM providers occasionally experience 503 Service Unavailable, 504 Gateway Timeouts, or rate limit throttling. NexusRAG features an automated **Grounded Fallback Engine**:
- Intercepts 503/504 errors mid-stream.
- Formats an extractive synthesis directly from the top retrieved citations and figures.
- Seamlessly closes the SSE event stream with citations intact, ensuring the user interface remains fully functional even during upstream outages.

### 5. 📊 Academic RAG Evaluation Framework
NexusRAG includes a built-in **LLM-as-a-Judge** scoring engine evaluating 4 standard academic metrics:
1. **Faithfulness**: Verifies whether each generated claim is directly supported by retrieved context.
2. **Answer Relevancy**: Assesses how directly and completely the response answers the user's inquiry.
3. **Context Precision**: Evaluates the signal-to-noise ratio in retrieved context chunks.
4. **Context Recall**: Measures coverage of required ground-truth evidence.

---

## 💻 Modern React 18 + TypeScript SPA

The frontend is a dedicated Single Page Application crafted with **React 18**, **TypeScript**, **Vite**, and **Tailwind CSS**:
- **Real-time SSE Streaming**: Live token-by-token rendering with glowing cursor animations and route indicators.
- **Interactive Footnotes**: Clickable citation chips (`[Doc 1 (Pg 2)]`, `[Fig. 1 (Pg 3)]`, `[Table 1]`, `[Audio 1]`).
- **Marginalia Drawer**: Slides out from the right to display high-resolution source diagrams, page excerpts, and relevance scores without losing chat context.
- **Fullscreen Lightbox**: Pinch-and-zoom inspection modal for complex technical architecture diagrams and plots.
- **Evaluation Benchmark Dashboard**: Real-time modal tracking system RAG scores and benchmark history.
- **KaTeX & PrismJS**: Full rendering for mathematical equations ($\LaTeX$) and syntax-highlighted code snippets.

---

## ⚡ Quickstart Guide

### Option 1: Docker (Recommended — 1 Command)

```bash
# 1. Clone repository
git clone https://github.com/Driv1216/NexusRag.git
cd NexusRag

# 2. Configure API key
cp .env.example .env
# Edit .env and insert your GOOGLE_API_KEY

# 3. Launch with Docker Compose
docker compose up --build
```
Open **`http://localhost:8000`** in your browser.

---

### Option 2: Local Development Setup

#### Prerequisites
- **Python 3.11+**
- **Node.js 20+** and **npm**
- Google Gemini API Key

#### 1. Clone & Configure Environment
```bash
git clone https://github.com/Driv1216/NexusRag.git
cd NexusRag

# Create & activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and insert your GOOGLE_API_KEY
```

#### 2. Build Frontend Single Page Application
```bash
cd frontend
npm ci
npm run build
cd ..
```

#### 3. Launch Backend API Server
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Navigate to **`http://localhost:8000`** to access the complete application!

---

## 🧪 Automated Testing & Verification

NexusRAG includes a **100% Offline Automated Test Suite** that runs without any external API keys or network access:

```bash
# Run the complete offline verification suite
python3 -m unittest tests/test_offline_suite.py -v
```

### Test Suite Coverage:
| Test ID | Module | Verifications | Result |
|---|---|---|---|
| `test_01` | Hierarchical Chunker | Parent-Child linking, UUID references, character boundaries | `PASSED` |
| `test_02` | Tabular Chunker | Markdown table conversion, schema metadata preservation | `PASSED` |
| `test_03` | BM25 Store | Inverted index tokenization, multi-word scoring, empty queries | `PASSED` |
| `test_04` | RRF Math & Reranker | Reciprocal Rank Fusion scoring ($k=60$), weight prioritization | `PASSED` |
| `test_05` | Agent Router Rules | Deterministic heuristics (`DIRECT`, `TABLE_LOOKUP`, `MULTI_STEP`) | `PASSED` |
| `test_06` | SQLite Memory | Sliding window (4 turns), session isolation, message ordering | `PASSED` |
| `test_07` | Pydantic Schemas | Data contracts for `QueryRequest`, `QueryResponse`, `Citation` | `PASSED` |
| `test_08` | FastAPI Endpoints | `/api/health`, `/api/documents`, `/api/gallery`, schema validation | `PASSED` |

*Execution time: **~0.03 seconds** across all 8 test modules.*

---

## 📡 REST & SSE API Reference

| Method | Endpoint | Description | Payload / Parameters |
|---|---|---|---|
| `GET` | `/api/health` | Healthcheck & system status | Returns model, document count, vector chunk count |
| `GET` | `/api/documents` | List all ingested documents | Returns filename, size, format, figure counts |
| `GET` | `/api/gallery` | Retrieve extracted figures/diagrams | Returns image URLs, captions, source page numbers |
| `POST` | `/api/upload` | Upload & ingest document | `multipart/form-data` with file |
| `POST` | `/api/query` | Synchronous multimodal query | `{"question": "...", "session_id": "...", "top_k": 5}` |
| `POST` | `/api/query/stream` | Real-time SSE token stream | `QueryRequest` payload via Server-Sent Events |
| `GET` | `/api/sessions/{session_id}` | Fetch session conversation history | Returns multi-turn message history |
| `POST` | `/api/evaluate` | Trigger automated RAG evaluation | Optional `test_questions: string[]` |
| `GET` | `/api/evaluate/results` | Fetch evaluation score history | Returns Faithfulness, Relevancy, Precision, Recall |

---

## 📁 Repository Structure

```
NexusRag/
├── .github/
│   ├── workflows/
│   │   └── ci.yml               # Automated GitHub Actions CI pipeline
│   ├── ISSUE_TEMPLATE/          # Bug report and feature request templates
│   └── PULL_REQUEST_TEMPLATE.md # Standardized PR review checklist
├── backend/
│   ├── agent/                   # Engine orchestrator, Query Router, HyDE, SQLite Memory
│   ├── embeddings/              # Dense sentence embeddings & CLIP representations
│   ├── evaluation/              # Academic RAG evaluation judge & benchmark metrics
│   ├── ingestion/               # PDF, Office, Spreadsheet, Audio, Video ingestors
│   ├── retrieval/               # ChromaDB store, BM25Okapi store, RRF, Cross-Encoder
│   ├── vision/                  # Gemini Vision captioner & diagram OCR
│   ├── config.py                # Central application configuration & paths
│   ├── models.py                # Pydantic v2 data models & validation schemas
│   └── main.py                  # FastAPI application & SSE streaming endpoints
├── frontend/
│   ├── src/
│   │   ├── components/          # ChatFeed, MarginaliaDrawer, LightboxModal, Sidebar
│   │   ├── App.tsx              # Master state management & SSE event handler
│   │   └── types.ts             # TypeScript interface contracts
│   ├── package.json             # React 18, Vite, Tailwind CSS dependencies
│   └── vite.config.ts           # Vite configuration & proxy rules
├── tests/
│   ├── test_offline_suite.py    # 100% offline automated test suite
│   └── test_nexus_suite.py      # End-to-end integration test suite
├── Dockerfile                   # Multi-stage production container build
├── docker-compose.yml           # 1-command container deployment
├── requirements.txt             # Python dependency specification
├── portfolio.md                 # In-depth Engineering Case Study
├── .env.example                 # Documented environment template
├── .gitignore                   # Production repository ignore rules
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation & architecture
```

---

## 🏆 Engineering Highlights for Technical Recruiters

- **Zero Monolithic Bloat**: Strictly decoupled into dedicated ingestion, retrieval, agentic routing, and synthesis modules.
- **Production Defense**: Resilience against upstream LLM 503/504 errors via deterministic grounded fallback synthesis.
- **Mathematical Integrity**: Dual-stage Reciprocal Rank Fusion ($RRF$) with neural Cross-Encoder re-ranking instead of naive linear score additions.
- **Enterprise DevOps**: Fully containerized with multi-stage Docker builds, automated GitHub Actions CI, and 100% offline verification tests.

---

## 📄 License
This project is open-source software licensed under the **[MIT License](LICENSE)**.
