import logging
from typing import Any, Dict, List
from app.rag.pipeline import ClinicalAssistantOrchestrator, OrchestratedResponse

logger = logging.getLogger(__name__)

class VoiceBridge:
    """
    Bridges the Gemini Live tool-calling system to the existing
    ClinicalAssistantOrchestrator RAG pipeline.
    """
    def __init__(self, orchestrator: ClinicalAssistantOrchestrator):
        self.orchestrator = orchestrator

    async def get_clinical_guidelines(self, question: str, history: List[Dict[str, Any]] = None) -> str:
        """
        Retrieves clinical guidelines and answers for a specific medical question.
        This is the primary tool called by Gemini Live when a clinical query is detected.

        Args:
            question: The clinical question to answer.
            history: Conversation history for context.

        Returns:
            A string representation of the clinical answer, citations, and any warnings.
        """
        logger.info(f"VoiceBridge: processing clinical query: {question}")

        try:
            # Use the existing orchestrator logic
            response, results = self.orchestrator.process_query(question, history=history)

            # Construct a detailed response for the LLM to read back to the user
            # and for the UI to render.
            response_text = f"Answer: {response.answer}\n"

            if response.caution:
                response_text += f"Caution: {response.caution}\n"

            if response.cited_chunk_ids:
                sources = [res['chunk_id'] for res in results if res['chunk_id'] in response.cited_chunk_ids]
                response_text += f"Sources: {', '.join(sources)}\n"

            if response.clinician_review_required:
                response_text += "Note: This response requires clinician review.\n"

            return response_text

        except Exception as e:
            logger.error(f"VoiceBridge error: {e}")
            return f"I encountered an error while retrieving the guidelines: {str(e)}"

# Tool definition for Gemini Live
CLINICAL_TOOL_DEFINITION = {
    "function_declarations": [
        {
            "name": "get_clinical_guidelines",
            "description": "Retrieves evidenced-based clinical guidelines for Indian OPD settings. Use this whenever the user asks a medical or clinical question about guidelines, dosages, or treatment protocols.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "question": {
                        "type": "STRING",
                        "description": "The specific clinical question to be answered from the guidelines."
                    },
                    "history": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "role": {"type": "STRING"},
                                "parts": {"type": "ARRAY", "items": {"type": "STRING"}}
                            }
                        },
                        "description": "The conversation history."
                    }
                },
                "required": ["question"]
            }
        }
    ]
}
