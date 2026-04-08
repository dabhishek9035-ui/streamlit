# 🧠 Hallucination Detector

A Streamlit-based AI tool that detects hallucinations in LLM outputs by verifying claims against Wikipedia, arXiv, and live web data, enhanced with semantic similarity, Natural Language Inference (NLI), and Gemini 1.5 Flash scoring.

## 🚀 Features
- **Claim-by-claim verification** — splits LLM output into sentences
- **Semantic similarity scoring** — uses `intfloat/e5-base-v2` embeddings
- **Natural Language Inference (NLI)** — `facebook/bart-large-mnli` model
- **Multi-source evidence retrieval** — Wikipedia, arXiv, Tavily web search
- **Gemini 1.5 Flash scoring** — AI-powered verdict and explanations
- **Blended scoring** — 60% embedding+NLI + 40% Gemini (configurable)
- **Visual dashboard** — color-coded claims, metric cards, progress indicators
- **Downloadable reports** — text export with detailed analysis
- **History tracking** — stores queries and scores locally

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd streamlit
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 🔑 API Setup

Edit the configuration section in `newcopy.py`:

```python
HF_TOKEN   = ""      # HuggingFace token (for model access)
TAVILY_KEY = ""      # Tavily API key (web search)
GEMINI_KEY = ""      # Google Gemini API key
```

## ▶️ Run

```bash
streamlit run newcopy.py
```

## 🧠 How It Works

1. **Split LLM output into claims** — uses NLTK sentence tokenization
2. **Fetch context** — retrieves evidence from Wikipedia, arXiv, and web (parallel)
3. **Chunk context** — splits text into overlapping chunks (120 words, top-3 retrieval)
4. **Compute embedding similarity** — uses `intfloat/e5-base-v2` to score grounding
5. **Run NLI** — checks entailment/contradiction with `facebook/bart-large-mnli`
6. **Gemini re-scoring** — optional AI verdict with JSON-structured output
7. **Blend scores** — combines embedding+NLI with Gemini (default 30% Gemini weight)
8. **Generate final report** — shows per-claim breakdown and overall score

## 📊 Scoring Methodology

### Base Score (Embedding + NLI)
- **Embedding score**: cosine similarity between claim and top context chunks (0.0–1.0)
- **NLI adjustment**:
  - **Entailment**: +0.30 (max 1.0)
  - **Neutral**: ×0.70
  - **Contradiction**: ×0.30

### Gemini Re-scoring
- **Supported** (0.85–1.0): claim is clearly and directly supported
- **Partial** (0.60–0.84): claim is partially supported or implied
- **Insufficient** (0.35–0.59): context does not clearly confirm or deny
- **Contradicted** (0.00–0.34): claim is contradicted by or absent from context

### Final Score
- Blends base score (60%) with Gemini score (40%)
- Configurable via `GEMINI_WEIGHT` (0.0–1.0)

## 🎯 Final Verdicts

| Score Range | Label | Meaning |
|---|---|---|
| ≥ 0.75 | **Grounded** | Reliable, well-supported claim |
| 0.50–0.75 | **Uncertain** | Partially grounded, needs verification |
| < 0.50 | **Hallucinated** | Contradicted or unsupported by evidence |

## 📁 Project Structure

```
streamlit/
├── newcopy.py              # Main Streamlit app
├── requirements.txt        # Dependencies
├── history.json           # Local query history (auto-created)
└── README.md              # This file
```

## ⚙️ Configuration

| Parameter | Default | Purpose |
|---|---|---|
| `SCORE_RELIABLE` | 0.75 | Threshold for "Grounded" |
| `SCORE_PARTIAL` | 0.50 | Threshold for "Uncertain" |
| `CHUNK_SIZE` | 120 | Words per context chunk |
| `TOP_K_CHUNKS` | 3 | Top chunks to use per claim |
| `MAX_CONTEXT_WDS` | 300 | Max words for Gemini context |
| `GEMINI_WEIGHT` | 0.30 | Gemini influence (0–1) |

## 🔧 Dependencies

- **streamlit** — web framework
- **sentence-transformers** — embedding model (`intfloat/e5-base-v2`)
- **transformers** — NLI model (`facebook/bart-large-mnli`)
- **wikipedia** — Wikipedia API
- **arxiv** — arXiv paper search
- **tavily-python** — web search
- **google-generativeai** — Gemini API
- **nltk** — sentence tokenization
- **huggingface-hub** — model authentication

## 📜 License

MIT