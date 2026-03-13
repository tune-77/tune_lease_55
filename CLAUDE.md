# CLAUDE.md — Warm Water Lease Screening AI (温水式リース審査AI)

AI assistant reference for the `tune_lease_55` repository. Read this before making any changes.

---

## Project Overview

This is a **lease application evaluation and risk-assessment system** for a Japanese leasing company. It combines rule-based financial scoring, machine learning (LightGBM), qualitative analysis, and AI-powered consultation (Ollama/Gemini) into a Streamlit dashboard.

- **Primary user**: Kobayashi (Senior Lease Officer)
- **Language**: Bilingual — English for technical code, Japanese for domain terms, comments, and UI
- **Approval threshold**: **71 points** (defined as `APPROVAL_LINE` in `config.py`)

---

## Repository Layout

```
tune_lease_55/
├── lease_logic_sumaho11/       ← ACTIVE: current modular version
│   ├── lease_logic_sumaho11.py   Main Streamlit app (~4,380 lines)
│   ├── config.py                 Constants, paths, env vars
│   ├── scoring_core.py           Scoring calculations (no Streamlit)
│   ├── indicators.py             Financial ratio computation
│   ├── analysis_regression.py    Regression & ML model training
│   ├── charts.py                 Plotly/Matplotlib visualizations
│   ├── ai_chat.py                Ollama + Gemini integration
│   ├── knowledge.py              Knowledge base search/retrieval
│   ├── data_cases.py             JSONL case file I/O (no Streamlit)
│   ├── auth_logic.py             Password + face recognition auth
│   ├── report_pdf.py             PDF report generation (ReportLab)
│   ├── web_services.py           Industry benchmark web services
│   ├── data_holder.py            Shared state container
│   ├── append_sample_cases.py    Demo data generator for testing
│   ├── scoring/                  ML scoring submodule
│   │   ├── model.py              CreditScoringModel (LightGBM/LogReg/XGBoost)
│   │   ├── feature_engineering_custom.py  CustomFinancialFeatures
│   │   ├── industry_hybrid_model.py       IndustrySpecificHybridModel
│   │   ├── predict_one.py        Single-record prediction utility
│   │   └── models/industry_specific/  *.pkl trained model files
│   └── web/                      Flask web app (simplified scoring UI)
│       ├── app.py                Flask routes (GET /, POST /result, GET /health)
│       ├── requirements.txt      flask, numpy, waitress
│       └── templates/            index.html, result.html, base.html, etc.
│
├── lease_logic_sumaho10(X)/    ← LEGACY (previous iteration)
├── lease_logic_sumaho9/        ← LEGACY
├── lease_logic_sumaho8.py      ← LEGACY monolith (do not edit)
│
├── past_cases.jsonl            Production case log (NEVER truncate)
├── past_cases_sample.jsonl     Demo/test case data
├── consultation_memory.jsonl   AI consultation history
├── debate_logs.jsonl           AI debate argument logs
├── case_news.jsonl             Case-related news items
├── knowledge_base.json         Structured knowledge (FAQ, manuals, case studies)
├── lease_logic_knowledge.json  Legacy knowledge reference
├── industry_benchmarks.json    Industry financial ratios by sector
├── web_industry_benchmarks.json  Web-scraped industry averages
├── industry_assets_benchmarks.json  Asset ratio norms
├── industry_trends_extended.json    Market trend data
├── sales_band_benchmarks.json       Sales bracket bands
├── byoki_list.json             Disease/health issue DB (non-profit sector)
├── data/
│   ├── coeff_overrides.json    User-edited coefficient adjustments
│   ├── business_rules.example.json
│   └── ai_teach_rules.example.json
│
├── requirements.txt            Root dependencies (Streamlit, LightGBM, ReportLab)
├── run_lease_app.sh            Launch Streamlit (8505) + Flask (5050)
├── run_with_ngrok.sh           Launch with ngrok tunnel
├── .streamlit/config.toml      Theme (Navy/Grey/Gold professional)
├── .github/workflows/daily_predict.yml  S&P 500 forecast CI (daily)
│
├── MEMORY.md                   Long-term session memory (main sessions only)
├── SOUL.md                     Assistant behavior guidelines
├── AGENTS.md                   Workspace conventions
├── TO_CURSOR.md                Critical development policies
├── IMPROVEMENTS_LEASE_LOGIC.md  Refactoring roadmap
└── CLAUDE.md                   ← This file
```

---

## Running the App

```bash
# Streamlit dashboard (primary)
streamlit run lease_logic_sumaho11/lease_logic_sumaho11.py
# → http://localhost:8501

# Streamlit + Flask backend (production mode)
./run_lease_app.sh
# → Streamlit on :8505, Flask on :5050

# With ngrok tunnel for remote access
./run_with_ngrok.sh

# Flask web app only
cd lease_logic_sumaho11/web
pip install -r requirements.txt
python app.py
# → http://localhost:5050
```

---

## Architecture & Scoring Pipeline

```
User Input (financials + qualitative ratings)
    │
    ▼
config.py → coefficient set selection (by industry + customer type)
    │
    ▼
scoring_core.py → logit score calculation
    │         → sigmoid → approval probability (0–100 pts)
    │
    ▼
APPROVAL_LINE = 71 pts
    ├─ ≥71 → Approved (承認圏内)
    └─ <71 → Rejected (否決)
    │
    ▼
indicators.py → financial ratio analysis vs. industry benchmarks
    │
    ▼
charts.py → radar, gauge, waterfall, benchmark scatter visualizations
    │
    ▼
ai_chat.py → Ollama / Gemini consultation (optional)
```

### Score Weights (defaults from `config.py`)

| Dimension | Default Weight |
|-----------|---------------|
| Borrower (借手) | 85% |
| Asset/Property (物件) | 15% |
| Quantitative (定量) | 60% |
| Qualitative (定性) | 40% |

Weights are adjustable at runtime. Coefficients can be retrained via `analysis_regression.py`.

### Industry Classifications

Scoring coefficients are selected by JSIC major code:
- `D建設` — Construction
- `E製造` — Manufacturing
- `H運輸` — Transportation
- `P医療` — Healthcare/Medical
- `サービス業` — Services
- `全体` — Fallback (all industries)

Customer types: `既存先` (existing) vs `新規先` (new).

---

## Module Conventions

### Critical Rule: Streamlit Separation

**Core business logic modules must NOT import Streamlit:**
- `scoring_core.py` — pure Python, callable from Flask/CLI
- `data_cases.py` — pure Python, no `st.*` calls
- `indicators.py` — pure Python
- `analysis_regression.py` — pure Python

**Streamlit-dependent modules** (use `st.*` internally):
- `lease_logic_sumaho11.py` — main app
- `charts.py`, `ai_chat.py`, `web_services.py`

### Path Resolution

Always use `BASE_DIR` from `config.py` for data file paths:

```python
from config import BASE_DIR
data_path = os.path.join(BASE_DIR, "past_cases.jsonl")
```

`BASE_DIR` = repository root (parent of `lease_logic_sumaho11/`). This keeps paths consistent across sumaho8/9/11 and the web app.

### Configuration Centralization

All constants belong in `config.py`:
- File paths, model names, approval threshold
- `CHART_STYLE` dict for color scheme
- `STRENGTH_TAG_OPTIONS` for qualitative tags
- `TEIREI_BYOKI_DEFAULT` for AI complaint messages

Do not hardcode file paths, model names, or thresholds elsewhere.

---

## Data Persistence

### JSONL (Append-Only Logs)

Used for case history and AI memory. **Never truncate or overwrite** these files:

| File | Purpose |
|------|---------|
| `past_cases.jsonl` | Production lease case records |
| `past_cases_sample.jsonl` | Demo/test cases |
| `consultation_memory.jsonl` | AI consultation history |
| `debate_logs.jsonl` | AI debate argument logs |
| `case_news.jsonl` | Case-related news |

Append pattern (from `data_cases.py`):
```python
with open(CASES_FILE, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

Read pattern (skip bad lines silently):
```python
with open(CASES_FILE, encoding="utf-8") as f:
    for line in f:
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError:
            continue
```

### JSON (Structured Config)

- `data/coeff_overrides.json` — user-edited coefficient deltas
- `industry_benchmarks.json` and related — static reference data
- `knowledge_base.json` — knowledge retrieval source

### PKL (ML Models)

Serialized LightGBM/XGBoost models at `scoring/models/industry_specific/*.pkl`.
Path configurable via `LEASE_SCORING_MODELS_DIR` environment variable.

---

## AI Integration

### Supported Backends

| Backend | Model | Config |
|---------|-------|--------|
| Ollama (local) | `lease-anna` | `OLLAMA_MODEL` env var or `st.secrets` |
| Gemini (cloud) | `gemini-2.0-flash` | `GEMINI_API_KEY` env var or `st.secrets` |

### API Key Management

Priority order (in `ai_chat.py`):
1. `st.secrets["GEMINI_API_KEY"]`
2. `GEMINI_API_KEY` environment variable
3. Sidebar input (dev/override only — do not rely on this in production)

Never log or print API keys.

### Thread Safety

AI calls run in background threads. Result handoff uses a global holder:
```python
_chat_result_holder = {"result": None, "done": False}
```
**Known limitation**: This is per-process, not per-session. The system is designed for single-user use. Document this if enabling multi-user deployment.

### Ollama Connection

Default endpoint: `http://localhost:11434` (override with `OLLAMA_HOST` env var).
The app tests connectivity on startup and shows a warning if Ollama is unreachable.

---

## Coding Conventions

### Naming

- **Domain identifiers**: Use Japanese-derived names for financial inputs: `nenshu` (売上高/sales), `rieki` (利益/profit), `yosan` (予算/budget)
- **Private functions**: Prefix with `_` (e.g., `_safe_sigmoid()`, `_load_benchmarks()`)
- **Constants**: SCREAMING_SNAKE_CASE (`APPROVAL_LINE`, `CHART_STYLE`, `BASE_DIR`)
- **Classes**: CamelCase (`CreditScoringModel`, `IndustrySpecificHybridModel`)

### Type Hints

Add type hints to all public functions:
```python
def run_quick_scoring(inputs: dict) -> dict: ...
def load_all_cases() -> list[dict]: ...
def get_score_weights() -> tuple[float, float, float, float]: ...
```

### Error Handling

- Catch specific exceptions, not bare `except:` or broad `except Exception`:
  ```python
  except json.JSONDecodeError:   # not: except:
  except (FileNotFoundError, PermissionError):
  ```
- Use `st.error()` to surface errors in Streamlit context
- Return empty defaults (`{}`, `[]`, `None`) on file-not-found rather than crashing

### Caching

```python
@st.cache_data(ttl=3600)   # 1h TTL for benchmark JSON files
def load_json_data(path: str) -> dict: ...

# No cache for case logs (must reflect new entries immediately)
def load_all_cases() -> list[dict]: ...
```

### Streamlit State

```python
# Read with default
value = st.session_state.get("ollama_model", OLLAMA_MODEL)

# Set
st.session_state["authenticated"] = True
```

---

## Authentication

Implemented in `auth_logic.py`:

- **Password**: Default `"123456"` — **development/demo only**, change before any external deployment
- **Face recognition**: Optional; requires `admin_face.jpg` and `face_recognition` package
- **Session flag**: `st.session_state.authenticated` — checked at app entry

---

## CRITICAL: Do NOT Change

From `TO_CURSOR.md` — these constraints are intentional:

1. **Do not remove input form fields** (including qualitative items). Even if a field is not used in the current scoring formula, it accumulates data for a planned qualitative-only regression model that will be ensembled with the quantitative model.

2. **Do not remove authentication logic** in `auth_logic.py` or at the top of `lease_logic_sumaho11.py`. Face recognition is actively being developed.

3. **Do not truncate JSONL data files**. Every record is training data. Deleting records degrades future model quality.

---

## Testing

There is no pytest suite. Testing approach:

1. **Generate demo data**: `python lease_logic_sumaho11/append_sample_cases.py` creates synthetic cases in `past_cases_sample.jsonl`
2. **Manual UI testing**: Run Streamlit and exercise each tab
3. **Flask health check**: `curl http://localhost:5050/health` → `{"status": "ok"}`
4. **GitHub Actions**: `.github/workflows/daily_predict.yml` runs daily S&P 500 forecast — check Actions tab for CI status

When adding new scoring logic, verify with the sample data generator and confirm the approval line behavior is preserved at 71 points.

---

## Git Workflow

- **Active branch**: `master` for production; `claude/*` for AI-generated feature branches
- Commit message style: imperative, concise (e.g., `Fix invalid industry benchmark values`)
- Never force-push to `master`
- Data files (`past_cases.jsonl`, benchmarks) are committed alongside code — this is intentional

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_MODEL` | `lease-anna` | Local LLM model name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server endpoint |
| `GEMINI_API_KEY` | *(empty)* | Google Gemini API key |
| `LEASE_SCORING_MODELS_DIR` | `scoring/models/industry_specific` | Path to trained PKL models |
| `DASHBOARD_IMAGES_ASSETS` | *(empty)* | Dashboard image assets directory |

---

## Known Issues & Roadmap

From `IMPROVEMENTS_LEASE_LOGIC.md`:

**High priority (in progress)**
- Replace bare `except:` with `except json.JSONDecodeError:` in case loading
- Move hardcoded absolute paths to `BASE_DIR`-relative paths
- Mandate `st.secrets` for API keys (remove sidebar key input in production)

**Medium priority**
- Session-state-scoped AI result holder (replace process-global `_chat_result_holder`)
- Retry/timeout mechanism for external benchmark web scraping
- Cache policy: short TTL for case logs, long TTL for static benchmarks

**Planned features**
- Qualitative-only regression model ensembled with quantitative model
- Real-time industry trend scraping
- Discord/LINE webhook notifications for case status
- Multi-user session isolation

---

## Streamlit Theme

Defined in `.streamlit/config.toml`. Professional finance/banking palette:

| Token | Color | Use |
|-------|-------|-----|
| `primary` | `#1e3a5f` | Navy — primary actions |
| `bg` | `#f8fafc` | Light gray — page background |
| `secondary` | `#ffffff` | White — card background |
| `text` | `#334155` | Slate — body text |
| `warning` | `#b45309` | Amber — caution indicators |
| `danger` | `#b91c1c` | Red — rejection/risk |
| `good` | `#0d9488` | Teal — approval/positive |

Match this palette when adding new chart colors or UI elements.
