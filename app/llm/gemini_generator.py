import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types
from app.config import settings
from app.schemas.generation import ClinicalAnswer

logger = logging.getLogger(__name__)

class GeminiGenerator:
    """Handles clinical answer generation using Gemini with structured output."""

    def __init__(self):
        api_key = settings.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self.model_id = settings.GEMINI_GENERATION_MODEL
        self.temperature = settings.GENERATION_TEMPERATURE

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]], history: List[Dict[str, Any]] = None) -> ClinicalAnswer:
        """
        Generates a grounded clinical answer based on the provided retrieved chunks and recent history.
        """
        # Format context for the prompt
        context_text = "\n\n".join([
            f"[Chunk ID: {c['chunk_id']}] Content: {c['content']}"
            for c in context_chunks
        ])

        # Format history for the prompt
        history_text = ""
        if history:
            history_text = "\n".join([f"{m['role']}: {m['text']}" for m in history])

        prompt = (
            "You are a clinical reference assistant. Answer the user's question ONLY using the provided evidence.\n\n"
            "RULES:\n"
            "1. If the evidence is insufficient to answer the question, set evidence_status to 'insufficient_evidence' "
            "and answer 'The available indexed guidelines do not establish the answer.'\n"
            "2. If sources conflict, set evidence_status to 'conflicting_evidence', state the conflict concisely, "
            "and set clinician_review_required=True.\n"
            "3. Every clinical claim must be linked to a chunk_id in cited_chunk_ids.\n"
            "4. DO NOT invent doses, thresholds, recommendations, or clinical facts.\n"
            "5. Keep the answer spoken-style, concise (max 40 words).\n\n"
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"EVIDENCE:\n{context_text}\n\n"
            f"QUESTION:\n{query}\n"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=settings.GENERATION_MAX_TOKENS,
                    response_mime_type="application/json",
                    response_schema=ClinicalAnswer,
                ),
            )

            # The SDK returns the parsed Pydantic model if response_schema is provided
            return response.parsed
        except Exception as e:
            logger.error(f"Error generating clinical answer: {e}")
            raise
