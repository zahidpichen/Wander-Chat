import streamlit as st
import faiss
import numpy as np
import pickle
import io
from openai import OpenAI
from tavily import TavilyClient
import json

with st.sidebar:
    page = st.radio(
        label="Navigation",
        options=["Chat", "Knowledge Base", "Configuration"]
    )

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

MOA_MODEL_1 = "arcee-ai/trinity-mini:free"
MOA_MODEL_2 = "nvidia/nemotron-3-nano-30b-a3b:free"
MOA_MODEL_3 = "arcee-ai/trinity-large-preview:free"
MOA_LAYER3_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"
WEB_QUERY_MODEL = "arcee-ai/trinity-mini:free"
REASONING_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
FORMATTER_MODEL = "liquid/lfm-2.5-1.2b-thinking:free"
FORMAT_EVALUATOR_MODEL = "liquid/lfm-2.5-1.2b-instruct:free"
STREAMING_MODEL = "arcee-ai/trinity-mini:free"

MOA_LAYER1_PROMPT_WITH_MEMORY = """You are a knowledgeable Kerala tourism assistant.

Conversation History:
{conversation_history}

Retrieved Context:
{retrieved_context}

User Query:
{user_query}

Give a detailed and informative response covering what you know about this topic."""

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

Give a single improved response that combines the best of all three."""

MOA_LAYER3_PROMPT = """You are a Kerala tourism expert. Your job is to synthesize 3 refined responses into one single, coherent, final response.

User Query:
{user_query}

Refined Response 1:
{response_1}

Refined Response 2:
{response_2}

Refined Response 3:
{response_3}

Combine these into one clear, complete, and well-structured final response. Do not add new information. Only synthesize what is already there."""

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

FORMATTER_PROMPT = """You are a data formatter for a Kerala tourism assistant.

You will receive a final response about a Kerala tourism topic. Your job is to extract and present only the structured factual data from it.

Final Response:
{final_response}

Extract any structured facts that are present (only include fields that actually exist in the response):
- Distances
- Opening time
- Closing time
- Entry fees
- Best time to visit
- Restaurants / food recommendations
- Weather / current conditions
- Traffic conditions
- Events / festivals

Present them in a clean structured format. For any narrative or descriptive information, keep it as a short paragraph under "About".

Do not add any information that is not in the response."""

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
    openrouter_key = st.session_state.get("openrouter_key", "")
    tavily_key = st.session_state.get("tavily_key", "")
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


def stream_final_response(client, final_response):
    """Generator that streams the final response token by token."""
    stream = client.chat.completions.create(
        model=STREAMING_MODEL,
        messages=[
            {"role": "system", "content": "You are a Kerala tourism assistant. Present the following information clearly and naturally."},
            {"role": "user", "content": final_response}
        ],
        stream=True
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield token


def retrieve(query, top_k=5):
    index = st.session_state.faiss_index
    chunk_store = st.session_state.chunk_store
    if index is None or not chunk_store:
        return "No knowledge base loaded."
    from sentence_transformers import SentenceTransformer
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


def run_pipeline(user_query, status):
    client, tavily_client = get_clients()

    # ── Retrieve ──────────────────────────────────────────────────────────────
    status.write("_Retrieving context from knowledge base..._")
    retrieved_context = retrieve(user_query)

    # ── MoA Layer 1 ───────────────────────────────────────────────────────────
    history = format_history()
    l1_prompt = MOA_LAYER1_PROMPT_WITH_MEMORY.format(
        conversation_history=history,
        retrieved_context=retrieved_context,
        user_query=user_query
    )

    status.write("_MoA — Layer 1: Agent 1 generating response..._")
    l1_r1 = llm_request(client, MOA_MODEL_1, "You are a helpful Kerala tourism assistant.", l1_prompt)

    status.write("_MoA — Layer 1: Agent 2 generating response..._")
    l1_r2 = llm_request(client, MOA_MODEL_2, "You are a helpful Kerala tourism assistant.", l1_prompt)

    status.write("_MoA — Layer 1: Agent 3 generating response..._")
    l1_r3 = llm_request(client, MOA_MODEL_3, "You are a helpful Kerala tourism assistant.", l1_prompt)

    # ── MoA Layer 2 ───────────────────────────────────────────────────────────
    l2_prompt = MOA_LAYER2_PROMPT.format(user_query=user_query, response_1=l1_r1, response_2=l1_r2, response_3=l1_r3)

    status.write("_MoA — Layer 2: Agent 1 refining..._")
    l2_r1 = llm_request(client, MOA_MODEL_1, "You are a Kerala tourism expert.", l2_prompt)

    status.write("_MoA — Layer 2: Agent 2 refining..._")
    l2_r2 = llm_request(client, MOA_MODEL_2, "You are a Kerala tourism expert.", l2_prompt)

    status.write("_MoA — Layer 2: Agent 3 refining..._")
    l2_r3 = llm_request(client, MOA_MODEL_3, "You are a Kerala tourism expert.", l2_prompt)

    # ── MoA Layer 3 ───────────────────────────────────────────────────────────
    l3_prompt = MOA_LAYER3_PROMPT.format(user_query=user_query, response_1=l2_r1, response_2=l2_r2, response_3=l2_r3)

    status.write("_MoA — Layer 3: Synthesizing final response..._")
    moa_response = llm_request(client, MOA_LAYER3_MODEL, "You are a Kerala tourism expert.", l3_prompt)

    # ── Web Search ────────────────────────────────────────────────────────────
    status.write("_Web researcher: Identifying missing information..._")
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
        status.write(f"_Web researcher: Searching [{i+1}/{len(queries)}] — {q}_")
        results = tavily_client.search(q, max_results=3)
        web_data.extend(results.get("results", []))

    # ── Evaluator ─────────────────────────────────────────────────────────────
    status.write("_Evaluator: Verifying and merging web results..._")
    raw = llm_request(
        client, REASONING_MODEL,
        "You are a verification expert for a Kerala tourism assistant.",
        REASONING_PROMPT.format(
            user_query=user_query,
            llm_response=moa_response,
            missing_info=missing_info,
            web_results=web_data
        )
    )
    parsed = json.loads(raw)

    if parsed.get("status") == "pass":
        evaluated_response = parsed["final_response"]
    else:
        status.write("_Evaluator: Retrying with additional search..._")
        web_data = []
        for q in queries:
            results = tavily_client.search(q, max_results=3)
            web_data.extend(results.get("results", []))
        raw = llm_request(
            client, REASONING_MODEL,
            "You are a verification expert for a Kerala tourism assistant.",
            REASONING_PROMPT.format(
                user_query=user_query,
                llm_response=moa_response,
                missing_info=parsed.get("missing_still"),
                web_results=web_data
            )
        )
        parsed = json.loads(raw)
        evaluated_response = parsed["final_response"]

    # ── Formatter ─────────────────────────────────────────────────────────────
    status.write("_Formatter: Structuring final response..._")
    formatted = llm_request(
        client, FORMATTER_MODEL,
        "You are a data formatter for a Kerala tourism assistant.",
        FORMATTER_PROMPT.format(final_response=evaluated_response)
    )

    status.write("_Formatter: Evaluating output quality..._")
    eval_raw = llm_request(
        client, FORMAT_EVALUATOR_MODEL,
        "You are a format quality checker for a Kerala tourism assistant.",
        FORMAT_EVALUATOR_PROMPT.format(original_data=evaluated_response, formatter_output=formatted)
    )
    eval_parsed = json.loads(eval_raw)

    if eval_parsed.get("status") == "pass":
        final_response = formatted
    else:
        status.write("_Formatter: Fixing formatting issues..._")
        final_response = llm_request(
            client, FORMATTER_MODEL,
            "You are a data formatter for a Kerala tourism assistant.",
            FORMATTER_PROMPT.format(final_response=evaluated_response) +
            f"\n\nPrevious output had issues: {eval_parsed.get('issues')}\nFix: {eval_parsed.get('fix')}\n\nPrevious wrong output:\n{formatted}"
        )

    return final_response, client


# ── Chat Page ─────────────────────────────────────────────────────────────────

if page == "Chat":

    st.markdown("""
        <style>
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about Kerala tourism..."):

        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Run pipeline with live status, then stream response
        with st.chat_message("assistant"):
            status = st.empty()

            final_response, client = run_pipeline(prompt, status)

            # Clear status and stream response
            status.empty()
            streamed = st.write_stream(stream_final_response(client, final_response))

        # Save to history
        st.session_state.messages.append({"role": "assistant", "content": streamed})
        add_to_memory(prompt, streamed)


# ── Knowledge Base Page ───────────────────────────────────────────────────────

elif page == "Knowledge Base":

    st.title("Knowledge Base")

    mode = st.radio(
        "Choose an option",
        ["Upload Documents", "Upload Existing Index"],
        horizontal=True
    )

    if mode == "Upload Documents":

        uploaded_files = st.file_uploader(
            "Upload PDF or DOCX files",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Build Index", type="primary"):

            import pdfplumber
            from docx import Document
            from sentence_transformers import SentenceTransformer

            progress_bar = st.progress(0)
            status_text = st.empty()

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
                status_text.text(f"Processing file [{i+1}/{total_files}]: {file.name}")
                progress_bar.progress(int((i / total_files) * 40))

                if file.name.endswith(".pdf"):
                    text = extract_pdf(file)
                else:
                    text = extract_docx(file)

                chunks = chunk_text(text)
                all_chunks.extend(chunks)

            status_text.text("Embedding chunks...")
            progress_bar.progress(50)

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            embeddings = model.encode(all_chunks, show_progress_bar=False)
            embeddings = np.array(embeddings).astype("float32")

            status_text.text("Building vector database...")
            progress_bar.progress(80)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)

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
            st.success("Index built successfully!")

    elif mode == "Upload Existing Index":

        col1, col2 = st.columns(2)

        with col1:
            index_file = st.file_uploader("Upload FAISS index (.index)", type=["index"])
        with col2:
            chunk_file = st.file_uploader("Upload chunk store (.pkl)", type=["pkl"])

        if index_file and chunk_file and st.button("Load Index", type="primary"):

            status_text = st.empty()
            progress_bar = st.progress(0)

            status_text.text("Loading FAISS index...")
            progress_bar.progress(30)

            with open("/tmp/uploaded.index", "wb") as f:
                f.write(index_file.read())
            index = faiss.read_index("/tmp/uploaded.index")

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
            st.success("Index loaded successfully!")

    if st.session_state.faiss_index is not None:

        st.divider()
        st.subheader("Vector Database Info")

        meta = st.session_state.index_meta

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Chunks", meta.get("total_chunks", 0))
        col2.metric("Embedding Dimensions", meta.get("dimensions", 0))
        col3.metric("Index Type", meta.get("index_type", ""))

        if meta.get("file_names"):
            st.markdown("**Indexed Files:**")
            for name in meta["file_names"]:
                st.markdown(f"- {name}")

        st.divider()
        st.subheader("Export Index")

        col1, col2 = st.columns(2)

        with col1:
            faiss.write_index(st.session_state.faiss_index, "/tmp/export.index")
            with open("/tmp/export.index", "rb") as f:
                st.download_button(
                    "Download FAISS Index",
                    data=f.read(),
                    file_name="vector.index",
                    mime="application/octet-stream"
                )

        with col2:
            pkl_bytes = pickle.dumps(st.session_state.chunk_store)
            st.download_button(
                "Download Chunk Store",
                data=pkl_bytes,
                file_name="chunk_store.pkl",
                mime="application/octet-stream"
            )


# ── Configuration Page ────────────────────────────────────────────────────────

elif page == "Configuration":

    st.title("Configuration")

    st.subheader("API Keys")

    openrouter_key = st.text_input(
        "OpenRouter API Key",
        value=st.session_state.get("openrouter_key", ""),
        type="password"
    )

    tavily_key = st.text_input(
        "Tavily API Key",
        value=st.session_state.get("tavily_key", ""),
        type="password"
    )

    if st.button("Save", type="primary"):
        st.session_state.openrouter_key = openrouter_key
        st.session_state.tavily_key = tavily_key
        st.success("API keys saved!")