import streamlit as st
import time
import json
from typing import List, Dict, Any, Tuple, Optional
from app.rag.pipeline import ClinicalAssistantOrchestrator, OrchestratedResponse
from app.storage.conversation_store import ConversationStore
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration ---
st.set_page_config(
    page_title="Clinical Assistant",
    page_icon="⚕️",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        .stChatMessage {
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }

        .main {
            background-color: #fcfcfc;
        }

        .stExpander {
            background-color: #f0f2f6;
            border: none;
            border-radius: 10px;
        }

        .stAlert {
            border-radius: 10px;
        }

        div[data-testid="stChatMessageContent"] {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        /*
           Fix New Chat button
           Streamlit often wraps buttons in a div. We target the button element
           within the sidebar that has the primary style.
        */
        div[data-testid="stSidebar"] button[kind="primary"],
        div[data-testid="stSidebar"] button[kind="primary"] div {
            background-color: #123B6D !important;
            color: #FFFFFF !important;
        }

        /* Target the inner button text specifically if needed */
        div[data-testid="stSidebar"] button[kind="primary"] p {
            color: #FFFFFF !important;
        }

        div[data-testid="stSidebar"] button[kind="primary"]:hover {
            background-color: #0B2A50 !important;
            color: #FFFFFF !important;
        }

        /*
           Fix Diagnosis and View Sources labels
           We target all text-containing elements within the expander header.
        */
        div[data-testid="stExpander"] header,
        div[data-testid="stExpander"] header p,
        div[data-testid="stExpander"] header span,
        div[data-testid="stExpander"] header div {
            color: #1E293B !important;
        }

        div[data-testid="stExpander"] header p {
            font-weight: 600 !important;
        }

        div[data-testid="stExpander"] header:hover p,
        div[data-testid="stExpander"] header:hover span {
            color: #123B6D !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Backend Initialization ---
@st.cache_resource
def get_orchestrator(mode: str):
    return ClinicalAssistantOrchestrator(retriever_mode=mode)

@st.cache_resource
def get_store():
    return ConversationStore()

# --- Session State ---
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "transcription_buffer" not in st.session_state:
    st.session_state.transcription_buffer = ""

# --- Sidebar ---
store = get_store()
with st.sidebar:
    st.title("⚕️ Clinical Assistant")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_thread_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Conversations")

    threads = store.list_threads()
    for t in threads:
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            if st.button(t['title'], key=f"t_{t['thread_id']}", use_container_width=True,
                         type="secondary" if st.session_state.current_thread_id != t['thread_id'] else "primary"):
                st.session_state.current_thread_id = t['thread_id']
                st.session_state.messages = store.get_messages(t['thread_id'])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{t['thread_id']}", help="Delete thread"):
                store.delete_thread(t['thread_id'])
                if st.session_state.current_thread_id == t['thread_id']:
                    st.session_state.current_thread_id = None
                    st.session_state.messages = []
                st.rerun()

    st.divider()
    st.title("⚙️ Settings")
    retrieval_mode = st.selectbox("Retrieval Mode", ["hybrid", "dense"], index=0)

    st.caption("History is stored locally on your device and can be deleted at any time.")
    st.caption("**Disclaimer:** Clinical answers are guideline-grounded references and do not replace clinical judgment.")

# --- Main Interface ---
st.title("Clinical Assistant")

# Welcome Message
if not st.session_state.current_thread_id and not st.session_state.messages:
    st.markdown("""
        <div style="text-align: center; color: #666; margin-top: 10vh; padding: 2rem; border: 1px dashed #ccc; border-radius: 15px;">
            <h3>Welcome to the Clinical Knowledge Assistant</h3>
            <p>Start a new conversation to retrieve grounded information from reviewed OPD guidelines.</p>
        </div>
    """, unsafe_allow_html=True)

# Render History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        # Recover metadata from store (SQLite stored as flat row)
        if message.get("route") == "clinical_rag":
            # Note: citations are derived from the result set at runtime in previous version
            # To persist citations, we must store them in the message record.
            # For now, we display the caution and review flag.
            if message.get("clinician_review_required"):
                st.warning("⚠️ **Clinician Review Required**")
            if message.get("caution"):
                st.info(f"ℹ️ {message['caution']}")

# Chat Input
# --- Voice Input Section ---
voice_col1, voice_col2 = st.columns([0.8, 0.2])
with voice_col1:
    # Changed format to "wav" as it's more likely to be compatible with the browser's
    # audio capture and required for the transcription pipeline.
    audio_data = mic_recorder(
        start_prompt="🎙️ Record Query",
        stop_prompt="🛑 Stop",
        format="wav",
        key="clinical_mic"
    )

if audio_data:
    # Note: In a full Gemini Live implementation, we would stream these bytes.
    # For this milestone, we use the recorder's output and a simplified transcription call.
    if "last_audio" not in st.session_state or st.session_state.last_audio != audio_data['bytes']:
        st.session_state.last_audio = audio_data['bytes']
        with st.spinner("Transcribing..."):
            # Simplified transcription for Milestone 1 integration
            # Real Gemini Live would use the async stream in app/voice/transcriber.py
            import base64
            from google import genai
            from app.config import settings

            try:
                client = genai.Client(api_key=settings.GEMINI_API_KEY)
                # Using the transcription model to process the recorded audio
                response = client.models.generate_content(
                    model=settings.TRANSCRIPTION_MODEL,
                    contents=["Transcribe this clinical audio accurately. Output only the transcription.",
                              {"mime_type": "audio/wav", "data": audio_data['bytes']}]
                )
                st.session_state.transcription_buffer = response.text
            except Exception as e:
                st.error(f"Transcription failed: {e}")

if st.session_state.transcription_buffer:
    st.markdown("### 📝 Review Transcription")
    edited_transcript = st.text_area("Edit if necessary:", value=st.session_state.transcription_buffer, key="transcript_edit")
    if st.button("✅ Use Transcript"):
        st.session_state.transcript_submitted = edited_transcript
        st.session_state.transcription_buffer = ""

# Resolve final prompt (either from chat_input or voice)
prompt = st.chat_input("Enter clinical query...")
if st.session_state.get("transcript_submitted"):
    prompt = st.session_state.transcript_submitted
    st.session_state.transcript_submitted = None

if prompt:
    # Ensure we have a thread
    if st.session_state.current_thread_id is None:
        # Create new thread using first message as title
        title = prompt[:50] + "..." if len(prompt) > 50 else prompt
        st.session_state.current_thread_id = store.create_thread(title)

    # 1. Display & Save User Message
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    store.add_message(st.session_state.current_thread_id, "user", prompt)

    # 2. Process with Orchestrator
    with st.chat_message("assistant"):
        with st.spinner("Processing clinical query..."):
            try:
                orchestrator = get_orchestrator(retrieval_mode)

                # Pass current history for multi-turn context
                # Convert stored messages to the format expected by bound_context: List[Dict[str, Any]]
                response, results = orchestrator.process_query(prompt, history=st.session_state.messages)

                answer_text = response.answer
                st.markdown(answer_text)

                # Clinical metadata
                metadata = {
                    "route": response.route,
                    "clinician_review_required": response.clinician_review_required,
                    "caution": response.caution,
                    "metrics": response.metrics
                }

                if response.route == "clinical_rag":
                    citations = []
                    for cid in response.cited_chunk_ids:
                        res = next((r for r in results if r['chunk_id'] == cid), None)
                        if res:
                            meta = res.get('metadata', {})
                            citations.append(f"{meta.get('source_file')} (Page {meta.get('page_start')}, Section: {meta.get('section_path')})")
                        else:
                            citations.append(f"ID {cid}: [Citation not found]")

                    with st.expander("📚 View Sources"):
                        for cit in citations:
                            st.markdown(f"- {cit}")

                    if response.clinician_review_required:
                        st.warning("⚠️ **Clinician Review Required**")
                    if response.caution:
                        st.info(f"ℹ️ {response.caution}")

                # Diagnostics
                with st.expander("⏱️ Diagnostics"):
                    m = response.metrics
                    diag = response.context_diagnostics
                    st.text(f"Total: {m.get('total_ms', 0):.2f}ms | Turns: {diag.get('history_turns', 0)}")

                # Save Assistant Message
                store.add_message(
                    st.session_state.current_thread_id,
                    "assistant",
                    answer_text,
                    route=response.route,
                    evidence_status=response.evidence_status,
                    clinician_review_required=response.clinician_review_required,
                    caution=response.caution,
                    metrics_json=json.dumps(response.metrics)
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": answer_text,
                    "route": response.route,
                    "clinician_review_required": response.clinician_review_required,
                    "caution": response.caution
                })

            except Exception as e:
                st.error("An error occurred.")
                st.exception(e)
