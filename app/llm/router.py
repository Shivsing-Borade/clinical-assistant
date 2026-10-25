import logging
from typing import Literal, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from app.config import settings

logger = logging.getLogger(__name__)

class RoutingResponse(BaseModel):
    """Structured response for intent routing."""
    route: Literal['greeting', 'general_chat', 'clinical_rag', 'patient_data'] = Field(
        ...,
        description="The classified intent of the user query."
    )
    response: Optional[str] = Field(
        None,
        description="The final answer for non-clinical routes. Leave null for 'clinical_rag'."
    )
    standalone_question: Optional[str] = Field(
        None,
        description="For 'clinical_rag', a rewritten version of the query that is standalone and contains all necessary context from history for retrieval."
    )

class IntentRouter:
    """Classifies user intent and provides immediate responses for non-RAG routes."""

    def __init__(self):
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self.model_id = settings.GEMINI_GENERATION_MODEL

    def route_query(self, query: str, history: List[Dict[str, Any]] = None) -> RoutingResponse:
        """
        Routes the query to the appropriate handler.
        Returns a RoutingResponse containing the route and an optional immediate answer.
        """
        system_instructions = (
            "You are an intent router for a clinical assistant. Your goal is to classify the user's message "
            "into exactly one of the following routes:\n\n"
            "1. 'greeting': Greeting-only or simple social conversation.\n"
            "2. 'general_chat': Non-clinical knowledge questions and ordinary conversation.\n"
            "3. 'clinical_rag': Medical knowledge questions requiring evidence from clinical guidelines. "
            "This includes questions about medication doses, treatment, diagnosis, contraindications, "
            "clinical thresholds, or guideline recommendations.\n"
            "4. 'patient_data': Requests about a particular patient's reports, lab results, history, or records.\n\n"
            "SAFETY AND ROUTING RULES:\n"
            "- A greeting attached to a clinical question does NOT make the entire message a greeting. "
            "Route these to 'clinical_rag'.\n"
            "- Requests for a specific person's records belong to 'patient_data', NOT general chat and NOT guideline RAG.\n"
            "- When uncertain whether a question needs clinical evidence, prefer 'clinical_rag'.\n"
            "- For 'greeting', 'general_chat', and 'patient_data', provide a concise, natural response in the 'response' field.\n"
            "- For 'patient_data', explain briefly that patient-record access is not connected yet.\n"
            "- For 'clinical_rag', leave the 'response' field null. "
            "Crucially, if the user's query is a follow-up (e.g., 'what about children?'), "
            "use the provided conversation history to rewrite it as a standalone retrieval question in the 'standalone_question' field.\n"
            "- DO NOT fabricate patient information or guideline claims."
        )

        # Construct context from history
        history_text = ""
        if history:
            history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history])

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=f"{system_instructions}\n\nRECENT HISTORY:\n{history_text}\n\nUSER QUERY: {query}",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=250,
                    response_mime_type="application/json",
                    response_schema=RoutingResponse,
                ),
            )
            return response.parsed
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            # Fallback to clinical_rag for safety when routing fails
            return RoutingResponse(route='clinical_rag', response=None)
