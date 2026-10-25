import streamlit as st
import time
from typing import List, Dict, Any, Tuple
from app.rag.pipeline import ClinicalAssistantOrchestrator, OrchestratedResponse

# --- Page Configuration ---
st.set_page_config(
    page_title="Clinical Assistant",
    page_icon="⚕️",
    layout="centered"
)

# --- Custom CSS for Clinical Aesthetics ---
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
    </style>
""", unsafe_allow_html=True)

# --- Backend Initialization ---
@st.cache_resource
def get_orchestrator(mode: str):
    """
    Caches the orchestrator to avoid rebuilding indices
    and re-initializing clients on every rerun.
    """
    return ClinicalAssistantOrchestrator(retriever_mode=mode)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")

    # Retrieval Mode Selector
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["hybrid", "dense"],
        index=0,
        help="Hybrid uses both Dense (semantic) and BM25 (keyword) search. Dense uses only embeddings."
    )

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "**Disclaimer:** Clinical answers are guideline-grounded references and "
        "do not replace clinical judgment. Always verify with primary sources."
    )

# --- Main Interface ---
st.title("⚕️ Clinical Assistant")

# Welcome Message
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-top: 10vh; padding: 2rem; border: 1px dashed #ccc; border-radius: 15px;">
            <h3>Welcome to the Clinical Knowledge Assistant</h3>
            <p>Ask a clinical question to retrieve grounded information from reviewed OPD guidelines.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metadata" in message:
            meta = message["metadata"]
            # Display Sources for clinical RAG
            if meta.get("route") == "clinical_rag":
                with st.expander("📚 View Sources"):
                    if meta.get("citations"):
                        for cit in meta["citations"]:
                            st.markdown(f"- {cit}")
                    else:
                        st.markdown("No specific citations available.")

            # Display Review Warnings
            if meta.get("clinician_review_required"):
                st.warning("⚠️ **Clinician Review Required**")
            if meta.get("caution"):
                st.info(f"ℹ️ {meta['caution']}")

# Chat Input
if prompt := st.chat_input("Enter clinical query..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Assistant Response
    with st.chat_message("assistant"):
        # Subtle loading indicator
        with st.spinner("Processing clinical query..."):
            try:
                # Initialize orchestrator with current mode
                orchestrator = get_orchestrator(retrieval_mode)

                # Single call to the orchestrator
                response, results = orchestrator.process_query(prompt)

                # Extract main answer
                answer_text = response.answer
                st.markdown(answer_text)

                # Prepare metadata for session state and display
                metadata = {
                    "route": response.route,
                    "clinician_review_required": response.clinician_review_required,
                    "caution": response.caution,
                    "metrics": response.metrics
                }

                # Handle Citations specifically for clinical RAG
                if response.route == "clinical_rag":
                    citations = []
                    for cid in response.cited_chunk_ids:
                        res = next((r for r in results if r['chunk_id'] == cid), None)
                        if res:
                            meta = res.get('metadata', {})
                            file = meta.get('source_file', 'Unknown')
                            page = meta.get('page_start', 'Unknown')
                            section = meta.get('section_path', 'Unknown')
                            citations.append(f"{file} (Page {page}, Section: {section})")
                        else:
                            citations.append(f"ID {cid}: [Citation not found in evidence]")
                    metadata["citations"] = citations

                    # Display Sources Expandable
                    with st.expander("📚 View Sources"):
                        if citations:
                            for cit in citations:
                                st.markdown(f"- {cit}")
                        else:
                            st.markdown("No specific citations available.")

                    # Display Warnings/Caution
                    if response.clinician_review_required:
                        st.warning("⚠️ **Clinician Review Required**")
                    if response.caution:
                        st.info(f"ℹ️ {response.caution}")

                # Diagnostics (Hidden in expander)
                with st.expander("⏱️ Diagnostics"):
                    m = response.metrics
                    st.text(
                        f"Routing: {m.get('routing_ms', 0):.2f}ms | "
                        f"Retrieval: {m.get('retrieval_ms', 0):.2f}ms | "
                        f"Generation: {m.get('generation_ms', 0):.2f}ms | "
                        f"Total: {m.get('total_ms', 0):.2f}ms"
                    )

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "metadata": metadata
                })

            except Exception as e:
                st.error("An error occurred while processing your request.")
                # Log diagnostic details to console/internal logs (simplified here)
                # In a real app, use logger.exception(e)
                st.expander("Error Details").write(str(e))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I encountered an error processing that request. Please try again.",
                    "metadata": {"error": True}
                })
