from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date

class SourceMetadata(BaseModel):
    """Provenance metadata for a clinical guideline PDF."""
    source_id: str = Field(..., description="Unique identifier for the source")
    filename: str = Field(..., description="Original filename of the PDF")
    issuing_authority: str = Field(default="unknown", description="The organization that issued the guideline")
    title: str = Field(default="unknown", description="Full title of the guideline")
    version: str = Field(default="unknown", description="Edition or version number")
    publication_date: Optional[str] = Field(default="unknown", description="Publication date (ISO format if possible)")
    url: Optional[str] = Field(default="unknown", description="Source URL if available")
    checksum: str = Field(..., description="SHA-256 hash of the file")
    file_size_bytes: int = Field(..., description="Size of the file in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages in the PDF")
    extraction_status: str = Field(default="pending_review", description="Status of the parsing quality (e.g., success, partial, failed)")
    parsing_warnings: List[str] = Field(default_factory=list, description="Warnings encountered during parsing/inventory")
    review_status: str = Field(default="pending_review", description="Clinician review status")

class SourceRegistry(BaseModel):
    """A collection of source metadata."""
    sources: List[SourceMetadata]
