import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from huggingface_hub import login
import wikipedia
import arxiv
import nltk
from nltk.tokenize import sent_tokenize
from tavily import TavilyClient
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
import re

# CONFIG 

HISTORY_FILE = "history.json"
HF_TOKEN     = ""
TAVILY_KEY   = ""      
GEMINI_KEY   = ""    

SCORE_RELIABLE  = 0.75
SCORE_PARTIAL   = 0.50
CHUNK_SIZE      = 120
TOP_K_CHUNKS    = 3
MAX_CONTEXT_WDS = 300

# Gemini re-scoring weight: how much Gemini's verdict influences the final score
# 0.0 = ignore Gemini, 1.0 = use Gemini only, 0.4 = blend 40% Gemini / 60% embedding+NLI
GEMINI_WEIGHT = 0.30

# PAGE CONFIG 

st.set_page_config(
    page_title="Hallucination Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS 

st.markdown("""
<style>
/* ── layout ── */
.main { padding: 1.5rem 2rem; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* ── claim cards ── */
.claim-card {
    padding: 14px 18px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 15px;
    line-height: 1.55;
    border: 1px solid transparent;
}
.claim-ok      { background: #e9f9ef; border-color: #a3d9b1; color: #1a5c31; }
.claim-warn    { background: #fef9ec; border-color: #f5d97a; color: #7a5500; }
.claim-bad     { background: #fdecea; border-color: #f5a8a1; color: #7a1f1a; }

/* ── badge pills ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: .02em;
}
.badge-ok     { background: #d1f5de; color: #1a5c31; }
.badge-warn   { background: #fef3c7; color: #7a5500; }
.badge-bad    { background: #fde8e8; color: #7a1f1a; }
.badge-nli    { background: #ede9fe; color: #3730a3; }
.badge-gemini { background: #e0f2fe; color: #0369a1; }

/* ── score gauge ── */
.gauge-wrap { text-align: center; padding: 1rem 0; }
.gauge-num  { font-size: 48px; font-weight: 700; line-height: 1; }
.gauge-label{ font-size: 14px; color: #888; margin-top: 4px; }

/* ── history card ── */
.hist-card {
    background: #f9f9f9;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 13px;
    border: 1px solid #e5e5e5;
}
.hist-card .query { font-weight: 600; color: #222; margin-bottom: 2px; }
.hist-card .score { color: #555; }

/* ── separator ── */
hr.thin { border: none; border-top: 1px solid #eee; margin: 12px 0; }

/* ── metric row ── */
.metric-row { display: flex; gap: 12px; margin: 8px 0; }
.metric-box {
    flex: 1;
    background: #f5f5f7;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.metric-box .val { font-size: 20px; font-weight: 600; color: #222; }
.metric-box .lbl { font-size: 12px; color: #888; margin-top: 2px; }

/* ── gemini summary box ── */
.gemini-summary {
    background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 100%);
    border: 1px solid #7dd3fc;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
    font-size: 14px;
    line-height: 1.6;
    color: #0c4a6e;
}
.gemini-summary .gem-title {
    font-weight: 700;
    font-size: 13px;
    color: #0369a1;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
</style>
""", unsafe_allow_html=True)

# NLTK 

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# AUTH 

if HF_TOKEN:
    login(HF_TOKEN)

# GEMINI SETUP 

@st.cache_resource(show_spinner=False)
def get_gemini_model():
    if not GEMINI_KEY:
        return None
    genai.configure(api_key=GEMINI_KEY)
    return genai.GenerativeModel("gemini-2.5-flash-latest")

gemini_model = get_gemini_model()

def gemini_available() -> bool:
    return gemini_model is not None

# HISTORY 

def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

def save_history(history: list) -> None:
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except OSError as e:
        st.warning(f"Could not save history: {e}")

if "history" not in st.session_state:
    st.session_state.history = load_history()

if "gemini_errors" not in st.session_state:
    st.session_state.gemini_errors = []

# MODELS 

@st.cache_resource(show_spinner="Loading embedding & NLI models…")
def load_models():
    embed = SentenceTransformer("intfloat/e5-base-v2")
    nli   = pipeline("text-classification", model="facebook/bart-large-mnli")
    return embed, nli

embed_model, nli_model = load_models()

# TEXT HELPERS 

def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 10]

def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def truncate(text: str, max_words: int = MAX_CONTEXT_WDS) -> str:
    return " ".join(text.split()[:max_words])

#  SCORING 

def get_top_chunks(sentence: str, chunks: list[str]) -> tuple[list[str], float]:
    s_emb = embed_model.encode(sentence, convert_to_tensor=True)
    c_emb = embed_model.encode(chunks,   convert_to_tensor=True)
    scores = util.cos_sim(s_emb, c_emb)[0]
    top_idx = scores.argsort(descending=True)[:TOP_K_CHUNKS]
    return [chunks[i] for i in top_idx], float(scores.max())

def nli_check(sentence: str, context: str) -> str:
    result = nli_model(f"{truncate(context)} </s></s> {sentence}")
    return result[0]["label"].lower()   # "entailment" | "neutral" | "contradiction"

def nli_to_score_delta(nli_label: str) -> float:
    return {"entailment": +0.30, "neutral": -0.10, "contradiction": -0.40}.get(nli_label, 0)

def base_score(grounding: float, nli_label: str) -> float:
    """Original scoring without Gemini."""
    if nli_label == "entailment":
        return min(1.0, grounding + 0.30)
    if nli_label == "neutral":
        return grounding * 0.70
    if nli_label == "contradiction":
        return grounding * 0.30
    return grounding

# GEMINI FUNCTIONS 

def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from Gemini's response, which may include
    markdown fences, leading/trailing text, or other noise.
    Strategy: try plain parse → strip fences → regex extract first {...} block.
    """
   
    text = text.strip()

    # 1. Direct parse (best case — Gemini obeyed the instruction)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip any ```json ... ``` or ``` ... ``` fences
    fenced = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # 3. Pull the first {...} block out of whatever Gemini returned
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in Gemini response: {text[:200]!r}")


def gemini_score_claim(sentence: str, context: str) -> tuple[float, str]:
    """
    Ask Gemini to score how well the claim is supported by the context.
    Returns (score 0.0–1.0, verdict label).
    Uses response_mime_type='application/json' to force structured output.
    """
    if not gemini_available():
        return 0.5, "unavailable"

    prompt = f"""You are a precise fact-checking assistant.

Given the CONTEXT below, evaluate how well the CLAIM is supported.

CONTEXT:
{truncate(context, 200)}

CLAIM:
{sentence}

Scoring guide:
- 0.85 to 1.00 → claim is clearly and directly supported by the context
- 0.60 to 0.84 → claim is partially supported or implied
- 0.35 to 0.59 → context does not clearly confirm or deny the claim
- 0.00 to 0.34 → claim is contradicted by or absent from the context

Return a JSON object with exactly two keys:
  "score"   : a float between 0.0 and 1.0
  "verdict" : one of "supported", "partial", "insufficient", "contradicted"
"""
    try:
        # Use JSON mime type so Gemini returns clean JSON without markdown fences
        json_model = genai.GenerativeModel(
            "gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"},
        )
        response = json_model.generate_content(prompt)
        data = _extract_json(response.text)
        score = float(max(0.0, min(1.0, data.get("score", 0.5))))
        verdict_label = str(data.get("verdict", "partial")).lower()
        # Normalise any unexpected verdict strings
        if verdict_label not in ("supported", "partial", "insufficient", "contradicted"):
            verdict_label = "partial"
        return score, verdict_label
    except Exception as e:
        # Store error details for sidebar debug panel
        err_msg = str(e)
        if hasattr(st, "session_state"):
            st.session_state.gemini_errors.append(err_msg)
        return 0.5, "error"


def gemini_explain_claim(sentence: str, context: str, gemini_verdict: str) -> str:
    """Ask Gemini for a human-readable explanation of its verdict."""
    if not gemini_available():
        return "(Gemini not configured — set GEMINI_KEY)"

    prompt = f"""You are a fact-checking assistant. Explain in 2–3 sentences why this claim is "{gemini_verdict}" based on the provided context.

CONTEXT:
{truncate(context, 150)}

CLAIM:
{sentence}

Be specific. Mention what the context does or does not say. Do not use bullet points."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"(Gemini explanation error: {e})"


def gemini_overall_summary(query: str, llm_output: str, sentences: list[str],
                            scores: list[float], gemini_verdicts: list[str],
                            overall: float) -> str:
    """Ask Gemini for a final natural-language summary of the full analysis."""
    if not gemini_available():
        return ""

    claim_lines = "\n".join(
        f"{i+1}. [{gemini_verdicts[i].upper()}] (score {scores[i]:.2f}) {s}"
        for i, s in enumerate(sentences)
    )
    prompt = f"""You are a hallucination analysis expert. A user asked: "{query}"

An LLM produced this output:
\"\"\"{llm_output[:500]}\"\"\"

Here is the per-claim analysis (score 0–1, verdict):
{claim_lines}

Overall score: {overall*100:.0f}%

Write a concise 3–5 sentence report for the user explaining:
1. Which parts of the LLM output are well-supported
2. Which parts are uncertain or hallucinated
3. What the user should watch out for

Be direct and specific. No bullet points. Plain paragraphs."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"(Summary unavailable: {e})"


def final_score_with_gemini(base: float, gemini_score: float) -> float:
    """Blend the base embedding+NLI score with Gemini's score."""
    return (1 - GEMINI_WEIGHT) * base + GEMINI_WEIGHT * gemini_score

# CONTEXT RETRIEVAL 

def _fetch_wikipedia(query: str) -> str:
    try:
        return "[WIKI] " + wikipedia.summary(query, sentences=10)
    except Exception:
        return ""

def _fetch_arxiv(query: str) -> str:
    try:
        results = arxiv.Search(query=query, max_results=2).results()
        return "[ARXIV] " + " ".join(p.summary for p in results)
    except Exception:
        return ""

def _fetch_web(query: str) -> str:
    if not TAVILY_KEY:
        return ""
    try:
        client = TavilyClient(api_key=TAVILY_KEY)
        res = client.search(query=query, max_results=3)
        return "[WEB] " + " ".join(r["content"] for r in res["results"])
    except Exception:
        return ""

def get_context(query: str) -> str:
    parts = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_fetch_wikipedia, query): "wiki",
            pool.submit(_fetch_arxiv,     query): "arxiv",
            pool.submit(_fetch_web,       query): "web",
        }
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                parts.append(result)
    return " ".join(parts).strip()

# ─── VERDICT HELPERS ─────────────────────────────────────────────────────────

def verdict(score: float) -> tuple[str, str, str]:
    if score >= SCORE_RELIABLE:
        return "claim-ok",   "badge-ok",   "Grounded"
    if score >= SCORE_PARTIAL:
        return "claim-warn", "badge-warn", "Uncertain"
    return "claim-bad",  "badge-bad",  "Hallucinated"

def overall_verdict(score: float) -> tuple[str, str]:
    if score >= SCORE_RELIABLE:
        return "success", "Reliable output"
    if score >= SCORE_PARTIAL:
        return "warning", "Partially grounded"
    return "error", "Likely hallucinated"

GEMINI_VERDICT_COLOR = {
    "supported":    "#1a5c31",
    "partial":      "#7a5500",
    "insufficient": "#7a5500",
    "contradicted": "#7a1f1a",
    "unavailable":  "#888",
    "error":        "#888",
}

# SIDEBAR 

with st.sidebar:
    st.markdown("## 🔑 API Status")
    st.markdown(
        f"**Gemini:** {'✅ Connected' if gemini_available() else '❌ Not configured — set GEMINI_KEY'}"
    )
    st.markdown(
        f"**Tavily:** {'✅ Set' if TAVILY_KEY else '⚠️ Not set (web search disabled)'}"
    )
    st.divider()

    # Gemini debug panel — shown only when errors occurred in the last run
    if st.session_state.gemini_errors:
        with st.expander("⚠️ Gemini debug log", expanded=True):
            for err in st.session_state.gemini_errors[-5:]:
                st.code(err, language=None)
        if st.button("Clear errors", use_container_width=True):
            st.session_state.gemini_errors = []
            st.rerun()
        st.divider()

    st.markdown("## History")
    if st.session_state.history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            save_history([])
            st.rerun()
        st.divider()
        for item in reversed(st.session_state.history[-20:]):
            score_pct = item["score"] * 100
            color = "#1a5c31" if item["score"] >= SCORE_RELIABLE else (
                    "#7a5500" if item["score"] >= SCORE_PARTIAL else "#7a1f1a")
            st.markdown(f"""
            <div class="hist-card">
                <div class="query">{item["query"][:55]}{"…" if len(item["query"]) > 55 else ""}</div>
                <div class="score" style="color:{color}">Score: {score_pct:.0f}%</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No analyses yet.")

# HEADER 

st.markdown("## 🧠 Hallucination Detector")
st.caption(
    "Verify LLM outputs against live web, Wikipedia, and arXiv evidence — "
    + ("enhanced with **Gemini 1.5 Flash** scoring ✨" if gemini_available()
       else "⚠️ Add your `GEMINI_KEY` for enhanced accuracy")
)

# INPUT FORM 

with st.container():
    col_q, col_out = st.columns([1, 2], gap="medium")
    with col_q:
        query = st.text_input(
            "User query",
            placeholder="e.g. What causes lightning?",
            help="The original question or topic the LLM was answering.",
        )
    with col_out:
        llm_output = st.text_area(
            "LLM output to verify",
            height=130,
            placeholder="Paste the model's response here…",
        )

    run = st.button("Analyze", type="primary", use_container_width=True)

# ANALYSIS 

if run:
    if not query.strip() or not llm_output.strip():
        st.warning("Please fill in both fields before analyzing.")
        st.stop()

    sentences = split_sentences(llm_output)
    if not sentences:
        st.warning("Could not detect any sentences in the LLM output.")
        st.stop()

    # Clear any errors from previous run
    st.session_state.gemini_errors = []

    # Fetch context
    with st.spinner("Fetching evidence from Wikipedia, arXiv, and the web…"):
        t0 = time.perf_counter()
        context = get_context(query)
        fetch_time = time.perf_counter() - t0

    if not context:
        st.error("No context could be retrieved. Check your API keys and network.")
        st.stop()

    context_chunks = chunk_text(context)

    # Analyse each sentence
    st.markdown("---")
    st.markdown("### Claim-by-claim analysis")

    scores          = []
    gemini_scores   = []
    gemini_verdicts = []

    with st.spinner("Scoring claims…"):
        for i, sent in enumerate(sentences, 1):
            # --- Embedding + NLI ---
            top_chunks, g_score = get_top_chunks(sent, context_chunks)
            combined  = " ".join(top_chunks)
            nli_label = nli_check(sent, combined)
            b_score   = base_score(g_score, nli_label)

            # --- Gemini re-scoring ---
            gem_score, gem_verdict = gemini_score_claim(sent, combined)
            gemini_scores.append(gem_score)
            gemini_verdicts.append(gem_verdict)

            # --- Blend ---
            score = final_score_with_gemini(b_score, gem_score) if gemini_available() else b_score
            scores.append(score)

            card_cls, badge_cls, label = verdict(score)
            gem_color = GEMINI_VERDICT_COLOR.get(gem_verdict, "#888")

            st.markdown(f"""
            <div class="claim-card {card_cls}">
                <span class="badge {badge_cls}">{label}</span>
                <span class="badge badge-nli" style="margin-left:6px">{nli_label}</span>
                {'<span class="badge badge-gemini" style="margin-left:6px">Gemini: ' + gem_verdict + '</span>' if gemini_available() else ''}
                &nbsp; <strong>#{i}</strong> &nbsp; {sent}
            </div>
            <div class="metric-row">
                <div class="metric-box"><div class="val">{g_score:.2f}</div><div class="lbl">Embedding</div></div>
                <div class="metric-box"><div class="val">{b_score:.2f}</div><div class="lbl">Base score</div></div>
                {'<div class="metric-box"><div class="val" style="color:' + gem_color + '">' + f"{gem_score:.2f}" + '</div><div class="lbl">Gemini score</div></div>' if gemini_available() else ''}
                <div class="metric-box"><div class="val"><strong>{score:.2f}</strong></div><div class="lbl">Final score</div></div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Supporting context chunks"):
                for chunk in top_chunks:
                    st.markdown(f"> {chunk}")

            with st.expander("🔍 Gemini explanation" if gemini_available() else "Explanation"):
                with st.spinner("Generating explanation…"):
                    explanation = gemini_explain_claim(sent, combined, gem_verdict) if gemini_available() else "(Set GEMINI_KEY for explanations)"
                    st.write(explanation)

            st.markdown('<hr class="thin">', unsafe_allow_html=True)

    # Overall result
    overall = sum(scores) / len(scores)
    kind, label = overall_verdict(overall)
    color = {"success": "#1a5c31", "warning": "#7a5500", "error": "#7a1f1a"}[kind]

    st.markdown("---")
    st.markdown("### Overall result")

    r1, r2 = st.columns([3, 1], gap="medium")
    with r1:
        st.progress(overall)
        getattr(st, kind)(label)
    with r2:
        st.markdown(f"""
        <div class="gauge-wrap">
            <div class="gauge-num" style="color:{color}">{overall*100:.0f}%</div>
            <div class="gauge-label">overall score</div>
        </div>""", unsafe_allow_html=True)

    # Gemini summary
    if gemini_available():
        with st.spinner("Gemini is writing the analysis summary…"):
            summary = gemini_overall_summary(
                query, llm_output, sentences,
                scores, gemini_verdicts, overall
            )
        if summary:
            st.markdown(f"""
            <div class="gemini-summary">
                <div class="gem-title">✨ Gemini Analysis Summary</div>
                {summary.replace(chr(10), '<br>')}
            </div>""", unsafe_allow_html=True)

    # Stats row
    n_good = sum(1 for s in scores if s >= SCORE_RELIABLE)
    n_warn = sum(1 for s in scores if SCORE_PARTIAL <= s < SCORE_RELIABLE)
    n_bad  = sum(1 for s in scores if s < SCORE_PARTIAL)

    st.markdown(f"""
    <div class="metric-row" style="margin-top:12px">
        <div class="metric-box"><div class="val" style="color:#1a5c31">{n_good}</div><div class="lbl">Grounded</div></div>
        <div class="metric-box"><div class="val" style="color:#7a5500">{n_warn}</div><div class="lbl">Uncertain</div></div>
        <div class="metric-box"><div class="val" style="color:#7a1f1a">{n_bad}</div><div class="lbl">Hallucinated</div></div>
        <div class="metric-box"><div class="val">{len(sentences)}</div><div class="lbl">Total claims</div></div>
        <div class="metric-box"><div class="val">{fetch_time:.1f}s</div><div class="lbl">Fetch time</div></div>
    </div>""", unsafe_allow_html=True)

    # Download
    report_lines = [
        f"Query: {query}",
        f"Overall score: {overall*100:.1f}%  ({label})",
        f"Grounded: {n_good}  Uncertain: {n_warn}  Hallucinated: {n_bad}",
        f"Gemini enhanced: {'yes' if gemini_available() else 'no'}",
        "",
        "── Claim detail ──",
    ]
    for i, (sent, score) in enumerate(zip(sentences, scores), 1):
        _, _, lbl = verdict(score)
        gem_v = gemini_verdicts[i-1] if gemini_available() else "n/a"
        report_lines.append(f"[{i}] ({lbl} {score:.2f} | gemini:{gem_v}) {sent}")

    st.download_button(
        "Download report (.txt)",
        data="\n".join(report_lines),
        file_name="hallucination_report.txt",
        mime="text/plain",
    )

    # Save to history
    st.session_state.history.append({
        "query":      query,
        "llm_output": llm_output,
        "score":      round(overall, 4),
    })
    save_history(st.session_state.history)

# FOOTER 

st.markdown("---")
st.caption(
    "Score ≥ 0.75 = Grounded · 0.50–0.75 = Uncertain · < 0.50 = Hallucinated  "
    "| Scoring = 60% embedding+NLI + 40% Gemini 1.5 Flash"
)
