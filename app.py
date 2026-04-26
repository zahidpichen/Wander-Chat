import os
import io
import json
import faiss
import pickle
import pdfplumber
import numpy as np
import tempfile, os
import streamlit as st
from docx import Document
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
load_dotenv()
# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kerala Tourism AI",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #0d1f3c 60%, #0a2e1a 100%);
    border-right: 1px solid #1e3a5f;
}

[data-testid="stSidebar"] .stRadio label {
    color: #a0b4c8 !important;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}

[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #a0b4c8;
}

/* Sidebar nav items */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 3px 0;
    transition: all 0.2s ease;
    cursor: pointer;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(46, 160, 67, 0.12);
    border-color: rgba(46, 160, 67, 0.3);
}

/* ── Main Content ── */
.block-container {
    padding: 2rem 3rem 3rem 3rem;
    max-width: 1100px;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0a2e1a 0%, #0d3b22 40%, #0a1628 100%);
    border: 1px solid #1a4d2e;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(46,160,67,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #e6edf3;
    margin: 0 0 0.3rem 0;
    line-height: 1.2;
}

.hero-title span {
    color: #3fb950;
}

.hero-subtitle {
    color: #7d8fa3;
    font-size: 0.95rem;
    font-weight: 300;
    margin: 0;
    letter-spacing: 0.02em;
}

.hero-badge {
    display: inline-block;
    background: rgba(46, 160, 67, 0.15);
    border: 1px solid rgba(46, 160, 67, 0.3);
    color: #3fb950;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Section Titles ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #e6edf3;
    margin: 0 0 0.25rem 0;
}

.section-subtitle {
    color: #7d8fa3;
    font-size: 0.85rem;
    margin: 0 0 1.5rem 0;
}

/* ── Cards ── */
.info-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #3fb950;
    font-size: 1.8rem;
    font-weight: 600;
}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #7d8fa3;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #2d4a3e;
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #3fb950;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.55rem 1.4rem;
    transition: all 0.2s ease;
    letter-spacing: 0.02em;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(46, 160, 67, 0.3);
}

.stButton > button[kind="secondary"] {
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
}

.stButton > button[kind="secondary"]:hover {
    background: #2d333b;
    border-color: #3fb950;
    box-shadow: 0 4px 12px rgba(46, 160, 67, 0.15);
}

/* ── Download Button ── */
[data-testid="stDownloadButton"] > button {
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    transition: all 0.2s;
}

[data-testid="stDownloadButton"] > button:hover {
    border-color: #3fb950;
    color: #3fb950;
    box-shadow: 0 4px 12px rgba(46, 160, 67, 0.15);
}

/* ── Radio Buttons ── */
.stRadio > div {
    gap: 0.5rem;
}

.stRadio [data-testid="stMarkdownContainer"] p {
    color: #7d8fa3;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Progress Bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #238636, #3fb950);
    border-radius: 4px;
}

.stProgress > div > div {
    background: #21262d;
    border-radius: 4px;
}

/* ── Text Input ── */
.stTextInput > div > div > input {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #e6edf3;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    padding: 0.6rem 0.9rem;
    transition: border-color 0.2s;
}

.stTextInput > div > div > input:focus {
    border-color: #3fb950;
    box-shadow: 0 0 0 3px rgba(46, 160, 67, 0.15);
}

.stTextInput label {
    color: #7d8fa3 !important;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.03em;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    transition: border-color 0.2s;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #3fb950;
    box-shadow: 0 0 0 3px rgba(46, 160, 67, 0.1);
}

[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    display: none;
}

/* ── Success / Info / Warning ── */
.stSuccess {
    background: rgba(46, 160, 67, 0.1);
    border: 1px solid rgba(46, 160, 67, 0.3);
    border-radius: 8px;
    color: #3fb950;
}

.stAlert {
    border-radius: 8px;
}

/* ── Divider ── */
hr {
    border-color: #21262d;
    margin: 1.5rem 0;
}

/* ── Status container (pipeline log) ── */
.pipeline-status {
    background: #0d1117;
    border: 1px solid #21262d;
    border-left: 3px solid #3fb950;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #7d8fa3;
    font-style: italic;
}

/* ── Indexed Files List ── */
.file-pill {
    display: inline-block;
    background: rgba(46, 160, 67, 0.08);
    border: 1px solid rgba(46, 160, 67, 0.2);
    color: #3fb950;
    font-size: 0.8rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 3px 4px 3px 0;
}

/* ── Selectbox / Dropdown ── */
[data-testid="stSelectbox"] > div > div {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #e6edf3;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3fb950; }

/* ── Sidebar logo area ── */
.sidebar-logo {
    padding: 1.5rem 1rem 1rem 1rem;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 1rem;
}

.sidebar-logo-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #e6edf3;
    margin: 0.5rem 0 0.2rem 0;
}

.sidebar-logo-sub {
    font-size: 0.75rem;
    color: #4a6fa5;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.sidebar-nav-label {
    font-size: 0.7rem;
    color: #4a6fa5;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 600;
    padding: 0 0.5rem;
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
        <div class="sidebar-logo">
            <div style="font-size: 2rem;">🌴</div>
            <div class="sidebar-logo-title">Kerala Tourism AI</div>
            <div class="sidebar-logo-sub">Intelligent Travel Assistant</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)

    page = st.radio(
        label="",
        options=["Knowledge Base", "Chat", "Configuration"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="padding: 0 0.5rem;">
            <div style="font-size: 0.75rem; color: #2d4a3e; border-top: 1px solid #1e3a5f; padding-top: 1rem;">
                Powered by MoA Pipeline · RAG · Web Search
            </div>
        </div>
    """, unsafe_allow_html=True)

# ── Session State Defaults ────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "chunk_store" not in st.session_state:
    st.session_state.chunk_store = []

if "index_meta" not in st.session_state:
    st.session_state.index_meta = {}

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


# ── Pipeline Helpers ──────────────────────────────────────────────────────────

MOA_MODEL_1 = "inclusionai/ling-2.6-flash:free"
MOA_MODEL_2 = "nvidia/nemotron-3-nano-30b-a3b:free"
MOA_MODEL_3 = "nvidia/nemotron-nano-9b-v2:free"
MOA_LAYER3_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"
WEB_QUERY_MODEL = "inclusionai/ling-2.6-flash:free"
REASONING_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
FORMATTER_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"
FORMAT_EVALUATOR_MODEL = "liquid/lfm-2.5-1.2b-instruct:free"
STREAMING_MODEL = "inclusionai/ling-2.6-flash:free"

MOA_LAYER1_PROMPT_WITH_MEMORY = """You are a knowledgeable Kerala tourism assistant.

        Conversation History:
        {conversation_history}

        Retrieved Context:
        {retrieved_context}

        User Query:
        {user_query}

        Give a detailed and informative response covering what you know about this topic.
"""

MOA_LAYER2_PROMPT = """You are a Kerala tourism expert refining and improving responses.

        You are given 3 responses from other assistants for the same user query. Collect the best points from all 3, remove redundancy, fix inaccuracies, and give an improved response.

        User Query:
        {user_query}

        Response 1:
        {response_1}

        Response 2:
        {response_2}

        Response 3:
        {response_3}

        Give a single improved response that combines the best of all three.
        """

MOA_LAYER3_PROMPT = """You are a Kerala tourism expert. Your job is to synthesize 3 refined responses into one single, coherent, final response.

    User Query:
    {user_query}

    Refined Response 1:
    {response_1}

    Refined Response 2:
    {response_2}

    Refined Response 3:
    {response_3}

    Combine these into one clear, complete, and well-structured final response. Do not add new information. Only synthesize what is already there.
"""

WEB_SURFER_PROMPT = """You are analyzing a tourism-related response to identify what real-time or recent information is missing or could be outdated.

User Query:
{user_query}

LLM Response:
{llm_response}

Your tasks:
1. Identify key entities in the query and response: places, landmarks, activities, food, events, etc.
2. Identify what information is likely outdated or missing: weather, traffic, restaurant recommendations, entry fees, events, timings, current conditions.
3. Generate specific search queries to fill those gaps.

Output a JSON object in this exact format:
{{
  "entities": {{"places": [], "activities": [], "food": [], "other": []}},
  "missing_or_outdated": [],
  "search_queries": []
}}

Only output the JSON. Nothing else."""

REASONING_PROMPT = """You are a verification expert for a Kerala tourism assistant.

Your job is to check if the web search results cover the missing or outdated information that was identified, and then produce a final merged response.

User Query:
{user_query}

Original LLM Response:
{llm_response}

Missing/Outdated Information That Was Searched For:
{missing_info}

Web Search Results:
{web_results}

Tasks:
1. Check if the web search results address the missing or outdated information.
2. If YES: merge the LLM response and web search results into one complete response. Output:
{{
  "status": "pass",
  "final_response": "<merged response here>"
}}
3. If NO (web results are insufficient or irrelevant): Output:
{{
  "status": "retry",
  "missing_still": "<what is still missing>"
}}

Only output the JSON. Nothing else."""



FORMATTER_PROMPT = """You are a response formatter for a Kerala tourism assistant.

Your job is to present the final response in a clean, human-friendly way based on what the user is asking.

User Query:
{user_query}

Final Response:
{final_response}

Follow these two rules:

RULE 1 — If the user is asking for a list of places (e.g. "list places", "places to visit", "what to see"):
- Write each place as a heading followed by 2-3 sentences describing it in a natural, conversational tone.
- Example: "Marine Drive is a scenic walkway along the backwaters of Kochi. It is one of the best spots to catch a sunset and the area also has a lot of street food stalls along the way."
- If the place has quick facts like timings or entry fee, add them in a single compact line below the description.
- Example: "Open: All day | Entry: Free | Best time: Evening"

RULE 2 — If the user is asking for a specific fact (e.g. timings, weather, entry fee, distance, temperature):
- Skip prose entirely. Just return the facts as clean labeled lines.
- Example:
  Opening Time: 9:00 AM – 5:00 PM
  Entry Fee: ₹30 (adults), ₹10 (children)
  Best Season: October – February

Do not mix these two rules. Do not add categories like "Dining recommendations" or "Weather advice" as bullet points. Only include information that is present in the final response.
"""


FORMAT_EVALUATOR_PROMPT = """You are a format quality checker for a Kerala tourism assistant.

Original Data:
{original_data}

Formatter Output:
{formatter_output}

Check for:
1. Any factual data present in the original that is missing from the formatted output.
2. Any data that was incorrectly formatted or misrepresented.
3. Any fields that are empty but shouldn't be, or fields with wrong values.

If everything is correct, output:
{{
  "status": "pass"
}}

If there are issues, output:
{{
  "status": "fail",
  "issues": "<describe what is wrong>",
  "fix": "<describe exactly what needs to be fixed>"
}}

Only output the JSON. Nothing else."""


def get_clients():
    openrouter_key = st.session_state.get("openrouter_key", "") or os.getenv("OPENROUTER_API_KEY", "")
    tavily_key = st.session_state.get("tavily_key", "") or os.getenv("TAVILY_API_KEY", "")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_key)
    tavily_client = TavilyClient(api_key=tavily_key)
    return client, tavily_client

def llm_request(client, model, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

def stream_final_response(final_response):
    for word in final_response.split(" "):
        yield word + " "

def retrieve(query, top_k=5):
    index = st.session_state.faiss_index
    chunk_store = st.session_state.chunk_store
    if index is None or not chunk_store:
        return "No knowledge base loaded."
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    results = [chunk_store[i] for i in indices[0] if i < len(chunk_store)]
    return "\n\n".join(results)


def format_history():
    history = st.session_state.conversation_history
    if not history:
        return "No previous conversation."
    return "\n".join([f"User: {t['user']}\nAgent: {t['agent']}" for t in history])


def add_to_memory(user_input, agent_response):
    st.session_state.conversation_history.append({"user": user_input, "agent": agent_response})
    if len(st.session_state.conversation_history) > 5:
        st.session_state.conversation_history.pop(0)


# ── ANSI color codes ──────────────────────────────────────────────────────────

CYAN    = "\033[36m"
YELLOW  = "\033[33m"
GREEN   = "\033[32m"
MAGENTA = "\033[35m"
GRAY    = "\033[90m"
RESET   = "\033[0m"
BOLD    = "\033[1m"

STAGE_COLORS = {
    "Retrieving":  CYAN,
    "MoA":         YELLOW,
    "Web":         MAGENTA,
    "Evaluator":   GREEN,
    "Formatter":   CYAN,
}

def terminal_log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    color = GRAY
    for keyword, c in STAGE_COLORS.items():
        if msg.startswith(keyword):
            color = c
            break
    print(f"{GRAY}{timestamp}{RESET}  {color}{BOLD}{msg}{RESET}")


def run_pipeline(user_query, status):

    def log(msg):
        status.markdown(f"""
            <div class="pipeline-status">
                <span style="color:#3fb950; margin-right:8px;">▶</span>{msg}
            </div>
        """, unsafe_allow_html=True)
        terminal_log(msg)

    client, tavily_client = get_clients()

    print(f"\n{GRAY}{'─' * 60}{RESET}")
    print(f"{BOLD}  Query:{RESET} {user_query}")
    print(f"{GRAY}{'─' * 60}{RESET}\n")

    log("Retrieving context from knowledge base...")
    retrieved_context = retrieve(user_query)

    history = format_history()
    l1_prompt = MOA_LAYER1_PROMPT_WITH_MEMORY.format(
        conversation_history=history,
        retrieved_context=retrieved_context,
        user_query=user_query
    )

    log("MoA — Layer 1: Agent 1 generating response...")
    l1_r1 = llm_request(client, MOA_MODEL_1, "You are a helpful Kerala tourism assistant.", l1_prompt)

    log("MoA — Layer 1: Agent 2 generating response...")
    l1_r2 = llm_request(client, MOA_MODEL_2, "You are a helpful Kerala tourism assistant.", l1_prompt)

    log("MoA — Layer 1: Agent 3 generating response...")
    l1_r3 = llm_request(client, MOA_MODEL_3, "You are a helpful Kerala tourism assistant.", l1_prompt)

    l2_prompt = MOA_LAYER2_PROMPT.format(user_query=user_query, response_1=l1_r1, response_2=l1_r2, response_3=l1_r3)

    log("MoA — Layer 2: Agent 1 refining...")
    l2_r1 = llm_request(client, MOA_MODEL_1, "You are a Kerala tourism expert.", l2_prompt)

    log("MoA — Layer 2: Agent 2 refining...")
    l2_r2 = llm_request(client, MOA_MODEL_2, "You are a Kerala tourism expert.", l2_prompt)

    log("MoA — Layer 2: Agent 3 refining...")
    l2_r3 = llm_request(client, MOA_MODEL_3, "You are a Kerala tourism expert.", l2_prompt)

    l3_prompt = MOA_LAYER3_PROMPT.format(user_query=user_query, response_1=l2_r1, response_2=l2_r2, response_3=l2_r3)

    log("MoA — Layer 3: Synthesizing final response...")
    moa_response = llm_request(client, MOA_LAYER3_MODEL, "You are a Kerala tourism expert.", l3_prompt)

    log("Web researcher: Identifying missing information...")
    raw = llm_request(
        client, WEB_QUERY_MODEL,
        "You are a web search query builder for a Kerala tourism assistant.",
        WEB_SURFER_PROMPT.format(user_query=user_query, llm_response=moa_response)
    )
    parsed = json.loads(raw)
    queries = parsed["search_queries"]
    missing_info = parsed.get("missing_or_outdated", None)

    web_data = []
    for i, q in enumerate(queries):
        log(f"Web researcher: Searching [{i+1}/{len(queries)}] — {q}")
        results = tavily_client.search(q, max_results=3)
        web_data.extend(results.get("results", []))

    log("Evaluator: Verifying and merging web results...")
    evaluated_response = llm_request(
    client, REASONING_MODEL,
    "You are a verification expert for a Kerala tourism assistant. Merge the original response and web search results into one complete, accurate response. Just return the merged response directly as plain text, nothing else.",
    f"User Query:\n{user_query}\n\nOriginal Response:\n{moa_response}\n\nWeb Search Results:\n{web_data}"
    )

    log("Formatter: Structuring final response...")
    formatted = llm_request(
        client, FORMATTER_MODEL,
        "You are a data formatter for a Kerala tourism assistant.",
        FORMATTER_PROMPT.format(user_query=user_query, final_response=evaluated_response)
    )

    log("Formatter: Evaluating output quality...")
    eval_raw = llm_request(
        client, FORMAT_EVALUATOR_MODEL,
        "You are a format quality checker for a Kerala tourism assistant.",
        FORMAT_EVALUATOR_PROMPT.format(original_data=evaluated_response, formatter_output=formatted)
    )
    eval_parsed = json.loads(eval_raw)

    if eval_parsed.get("status") == "pass":
        final_response = formatted
    else:
        log("Formatter: Fixing formatting issues...")
        final_response = llm_request(
            client, FORMATTER_MODEL,
            "You are a data formatter for a Kerala tourism assistant.",
            FORMATTER_PROMPT.format(user_query=user_query, final_response=evaluated_response) +
            f"\n\nPrevious output had issues: {eval_parsed.get('issues')}\nFix: {eval_parsed.get('fix')}\n\nPrevious wrong output:\n{formatted}"
        )

    print(f"\n{GREEN}{BOLD}  ✓ Pipeline complete.{RESET}\n{GRAY}{'─' * 60}{RESET}\n")

    return final_response, client


# ── Chat Page ─────────────────────────────────────────────────────────────────

if page == "Chat":

    # Hero
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">🌴 Live Demo</div>
            <div class="hero-title">Ask about <span>Kerala</span></div>
            <div class="hero-subtitle">MoA pipeline · RAG retrieval · Real-time web search</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("🗑️", use_container_width=True, help="Clear conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about Kerala tourism..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.empty()
            final_response, client = run_pipeline(prompt, status)
            status.empty()
            streamed = st.write_stream(stream_final_response(final_response))

        st.session_state.messages.append({"role": "assistant", "content": streamed})
        add_to_memory(prompt, streamed)


# ── Knowledge Base Page ───────────────────────────────────────────────────────

elif page == "Knowledge Base":

    st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">📚 Data Ingestion</div>
            <div class="hero-title">Knowledge <span>Base</span></div>
            <div class="hero-subtitle">Upload documents or load a pre-built vector index</div>
        </div>
    """, unsafe_allow_html=True)

    if "index_source" not in st.session_state:
        st.session_state.index_source = None

    mode = st.radio(
        "Choose an option",
        ["Upload Documents", "Upload Existing Index"],
        horizontal=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if mode == "Upload Documents":

        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("⚙️  Build Index", type="primary"):

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading embedding model...")
            progress_bar.progress(5)
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            status_text.text("Preparing text extraction pipeline...")
            progress_bar.progress(10)

            all_chunks = []
            total_files = len(uploaded_files)

            def extract_pdf(file):
                text = ""
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text

            def extract_docx(file):
                doc = Document(file)
                return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

            def chunk_text(text, chunk_size=500, overlap=50):
                words = text.split()
                chunks = []
                start = 0
                while start < len(words):
                    end = start + chunk_size
                    chunks.append(" ".join(words[start:end]))
                    start += chunk_size - overlap
                return chunks

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Extracting text [{i+1}/{total_files}]: {file.name}")
                progress_bar.progress(10 + int((i / total_files) * 30))

                if file.name.endswith(".pdf"):
                    text = extract_pdf(file)
                else:
                    text = extract_docx(file)

                chunks = chunk_text(text)
                all_chunks.extend(chunks)

                status_text.text(f"Chunked {file.name} → {len(chunks)} chunks (total so far: {len(all_chunks)})")

            status_text.text(f"Embedding {len(all_chunks)} chunks... this may take a moment")
            progress_bar.progress(50)

            embeddings = model.encode(all_chunks, show_progress_bar=False)
            embeddings = np.array(embeddings).astype("float32")

            status_text.text("Building vector index...")
            progress_bar.progress(85)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)

            status_text.text("Saving index to session...")
            progress_bar.progress(95)

            st.session_state.faiss_index = index
            st.session_state.chunk_store = all_chunks
            st.session_state.index_meta = {
                "total_chunks": len(all_chunks),
                "total_files": total_files,
                "file_names": [f.name for f in uploaded_files],
                "dimensions": dim,
                "index_type": "FlatL2"
            }

            progress_bar.progress(100)
            status_text.text("Done!")
            st.success(f"✅  Index built — {total_files} file(s) · {len(all_chunks)} total chunks")
            st.session_state.index_source = "build"


    elif mode == "Upload Existing Index":

        col1, col2 = st.columns(2)

        with col1:
            index_file = st.file_uploader("Upload FAISS index (.index)", type=["index"])
        with col2:
            chunk_file = st.file_uploader("Upload chunk store (.pkl)", type=["pkl"])

        if index_file and chunk_file and st.button("📂  Load Index", type="primary"):

            status_text = st.empty()
            progress_bar = st.progress(0)

            status_text.text("Loading FAISS index...")
            progress_bar.progress(30)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".index") as tmp:
                tmp.write(index_file.read())
                tmp_path = tmp.name

            index = faiss.read_index(tmp_path)
            os.remove(tmp_path)

            status_text.text("Loading chunk store...")
            progress_bar.progress(70)

            chunk_store = pickle.load(chunk_file)

            st.session_state.faiss_index = index
            st.session_state.chunk_store = chunk_store
            st.session_state.index_meta = {
                "total_chunks": len(chunk_store),
                "total_files": "N/A (loaded from index)",
                "file_names": [],
                "dimensions": index.d,
                "index_type": "FlatL2"
            }

            progress_bar.progress(100)
            status_text.text("Done!")
            st.success("✅  Index loaded successfully!")
            st.session_state.index_source = "load"


    if st.session_state.faiss_index is not None and (
        (mode == "Upload Documents" and st.session_state.index_source == "build") or
        (mode == "Upload Existing Index" and st.session_state.index_source == "load")
    ):
        st.divider()

        st.markdown("""
            <div class="section-title">Vector Database Info</div>
            <div class="section-subtitle">Index metadata and statistics</div>
        """, unsafe_allow_html=True)

        meta = st.session_state.index_meta

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks", meta.get("total_chunks", 0))
        col2.metric("Embedding Dimensions", meta.get("dimensions", 0))
        col3.metric("Index Type", meta.get("index_type", ""))

        if meta.get("file_names"):
            st.markdown("<br>**Indexed Files**", unsafe_allow_html=True)
            pills_html = "".join([f'<span class="file-pill">📄 {name}</span>' for name in meta["file_names"]])
            st.markdown(pills_html, unsafe_allow_html=True)

        if st.session_state.index_source == "build":
            st.divider()

            st.markdown("""
                <div class="section-title">Export Index</div>
                <div class="section-subtitle">Save your built index for reuse</div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                index_folder_path = "faiss_index"
                if not os.path.exists(index_folder_path):
                    os.makedirs(index_folder_path)
                index_file_path = os.path.join(index_folder_path, "vector.index")
                faiss.write_index(st.session_state.faiss_index, index_file_path)
                with open(index_file_path, "rb") as f:
                    st.download_button(
                        "⬇️  Download FAISS Index",
                        data=f.read(),
                        file_name="vector.index",
                        mime="application/octet-stream"
                    )

            with col2:
                pkl_bytes = pickle.dumps(st.session_state.chunk_store)
                st.download_button(
                    "⬇️  Download Chunk Store",
                    data=pkl_bytes,
                    file_name="chunk_store.pkl",
                    mime="application/octet-stream"
                )


# ── Configuration Page ────────────────────────────────────────────────────────

elif page == "Configuration":

    st.markdown("""
        <div class="hero-banner">
            <div class="hero-badge">⚙️ Setup</div>
            <div class="hero-title">Configur<span>ation</span></div>
            <div class="hero-subtitle">Set your API keys to power the pipeline</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="section-title">API Keys</div>
        <div class="section-subtitle">Keys are stored in session memory only — never persisted</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            value=st.session_state.get("openrouter_key", ""),
            type="password",
            placeholder="sk-or-..."
        )

    with col2:
        tavily_key = st.text_input(
            "Tavily API Key",
            value=st.session_state.get("tavily_key", ""),
            type="password",
            placeholder="tvly-..."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("💾  Save Configuration", type="primary"):
        st.session_state.openrouter_key = openrouter_key
        st.session_state.tavily_key = tavily_key
        st.success("✅  API keys saved for this session!")

    st.divider()

    st.markdown("""
        <div class="section-title">Pipeline Overview</div>
        <div class="section-subtitle">What happens when you send a query</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    steps = [
        ("🔍", "RAG Retrieval", "Fetches relevant chunks from the FAISS vector index using semantic search"),
        ("🤖", "MoA Layer 1", "Three independent agents generate initial responses in parallel"),
        ("🔄", "MoA Layer 2", "Three agents refine and merge the Layer 1 outputs"),
        ("✨", "MoA Layer 3", "Final synthesis into one coherent response"),
        ("🌐", "Web Search", "Tavily searches for real-time data to fill gaps"),
        ("✅", "Evaluator", "Verifies and merges web results with the MoA response"),
        ("📝", "Formatter", "Structures the output for clarity and evaluates quality"),
    ]

    for icon, title, desc in steps:
        st.markdown(f"""
            <div class="info-card" style="display:flex; align-items:flex-start; gap:1rem;">
                <div style="font-size:1.4rem; margin-top:2px;">{icon}</div>
                <div>
                    <div style="color:#e6edf3; font-weight:600; font-size:0.95rem; margin-bottom:0.25rem;">{title}</div>
                    <div style="color:#7d8fa3; font-size:0.85rem;">{desc}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)