from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ClinicalAnswer(BaseModel):
    """Schema for a grounded clinical answer response."""
    answer: str = Field(
        ...,
        description="Concise, natural spoken answer. Target 1-3 sentences, max 40 words."
    )
    evidence_status: Literal['supported', 'insufficient_evidence', 'conflicting_evidence'] = Field(
        ...,
        description="Whether the answer is grounded in evidence, lacks evidence, or finds contradictions."
    )
    cited_chunk_ids: List[str] = Field(
        default_factory=list,
        description="The IDs of the retrieved chunks that directly support the answer."
    )
    clinician_review_required: bool = Field(
        ...,
        description="Whether the answer contains conflicts or requires professional verification."
    )
    caution: Optional[str] = Field(
        None,
        description="Brief text for an important limitation, contradiction, or validation warning."
    )
