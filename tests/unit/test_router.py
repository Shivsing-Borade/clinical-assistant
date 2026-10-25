import pytest
from unittest.mock import MagicMock, patch
from app.llm.router import IntentRouter, RoutingResponse
from app.rag.pipeline import ClinicalAssistantOrchestrator, OrchestratedResponse

@pytest.fixture
def mock_genai_client():
    with patch("google.genai.Client") as mock:
        yield mock

def test_router_greeting(mock_genai_client):
    # Setup mock response
    mock_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.parsed = RoutingResponse(route="greeting", response="Hello! How can I help you today?")
    mock_instance.models.generate_content.return_value = mock_response

    router = IntentRouter()
    res = router.route_query("Hi there!")

    assert res.route == "greeting"
    assert "Hello" in res.response
    mock_instance.models.generate_content.assert_called_once()

def test_router_general_chat(mock_genai_client):
    mock_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.parsed = RoutingResponse(route="general_chat", response="The capital of France is Paris.")
    mock_instance.models.generate_content.return_value = mock_response

    router = IntentRouter()
    res = router.route_query("What is the capital of France?")

    assert res.route == "general_chat"
    assert "Paris" in res.response

def test_router_clinical_rag(mock_genai_client):
    mock_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.parsed = RoutingResponse(route="clinical_rag", response=None)
    mock_instance.models.generate_content.return_value = mock_response

    router = IntentRouter()
    res = router.route_query("What is the recommended dose for Paracetamol in adults?")

    assert res.route == "clinical_rag"
    assert res.response is None

def test_router_patient_data(mock_genai_client):
    mock_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.parsed = RoutingResponse(route="patient_data", response="I don't have access to patient records yet.")
    mock_instance.models.generate_content.return_value = mock_response

    router = IntentRouter()
    res = router.route_query("Show me the lab results for Patient ID 123.")

    assert res.route == "patient_data"
    assert "access to patient records" in res.response

def test_router_mixed_greeting_clinical(mock_genai_client):
    mock_instance = mock_genai_client.return_value
    mock_response = MagicMock()
    mock_response.parsed = RoutingResponse(route="clinical_rag", response=None)
    mock_instance.models.generate_content.return_value = mock_response

    router = IntentRouter()
    res = router.route_query("Hello! Could you tell me the symptoms of Dengue?")

    assert res.route == "clinical_rag"

def test_router_error_fallback(mock_genai_client):
    mock_instance = mock_genai_client.return_value
    mock_instance.models.generate_content.side_effect = Exception("API Error")

    router = IntentRouter()
    res = router.route_query("Any query")

    # Should fallback to clinical_rag for safety
    assert res.route == "clinical_rag"

@patch("app.rag.pipeline.IntentRouter")
@patch("app.rag.pipeline.ClinicalRAGPipeline")
def test_orchestrator_routing_logic(mock_rag_class, mock_router_class):
    # Mock Router
    mock_router = mock_router_class.return_value
    # Mock RAG
    mock_rag = mock_rag_class.return_value

    orchestrator = ClinicalAssistantOrchestrator()

    # Scenario 1: Greeting -> No RAG call
    mock_router.route_query.return_value = RoutingResponse(route="greeting", response="Hi!")
    resp, results = orchestrator.process_query("Hello")
    assert resp.route == "greeting"
    assert resp.answer == "Hi!"
    mock_rag.ask.assert_not_called()

    # Scenario 2: Clinical -> RAG call
    mock_router.route_query.return_value = RoutingResponse(route="clinical_rag", response=None)
    mock_rag.ask.return_value = (
        MagicMock(answer="Dose is 500mg", evidence_status="supported", cited_chunk_ids=["c1"], clinician_review_required=False, caution=None),
        [{"chunk_id": "c1"}],
        {"retrieval_ms": 10.0, "generation_ms": 20.0, "total_ms": 30.0}
    )

    resp, results = orchestrator.process_query("What is the dose?")
    assert resp.route == "clinical_rag"
    assert resp.answer == "Dose is 500mg"
    mock_rag.ask.assert_called_once()
