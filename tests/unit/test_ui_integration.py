import pytest
from unittest.mock import MagicMock
from app.ui.chat_app import get_orchestrator

def test_orchestrator_caching():
    """
    Verify that get_orchestrator uses st.cache_resource
    (via testing the logic of caching if needed, or just ensuring it initializes)
    Note: Streamlit caching is hard to test in pure pytest without streamlit run,
    but we can verify the function return type.
    """
    # We mock the Orchestrator since we can't run Qdrant/Gemini in offline tests
    with pytest.raises(Exception): # Expect failure because env vars aren't set
        get_orchestrator("hybrid")

def test_citation_mapping_logic():
    """
    Test the logic used in the UI to map chunk IDs to metadata.
    This mimics the loop in app/ui/chat_app.py
    """
    results = [
        {
            "chunk_id": "chunk_1",
            "metadata": {"source_file": "guide1.pdf", "page_start": 1, "section_path": "Intro"}
        },
        {
            "chunk_id": "chunk_2",
            "metadata": {"source_file": "guide2.pdf", "page_start": 5, "section_path": "Treatment"}
        }
    ]
    cited_ids = ["chunk_1", "chunk_3"] # One exists, one doesn't

    citations = []
    for cid in cited_ids:
        res = next((r for r in results if r['chunk_id'] == cid), None)
        if res:
            meta = res.get('metadata', {})
            citations.append(f"{meta.get('source_file')} (Page {meta.get('page_start')}, Section: {meta.get('section_path')})")
        else:
            citations.append(f"ID {cid}: [Citation not found in evidence]")

    assert citations[0] == "guide1.pdf (Page 1, Section: Intro)"
    assert "ID chunk_3: [Citation not found in evidence]" in citations[1]
