from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CleanedDocument(BaseModel):
    """A document after noise removal and normalization."""
    source_id: str
    filename: str
    content: str
    metadata: Dict[str, Any]

class ChunkMetadata(BaseModel):
    """Metadata for a single text chunk."""
    source_id: str
    source_file: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_path: Optional[str] = None
    content_type: str = Field(..., description="prose, list, or table")
    parent_id: Optional[str] = None
    child_id: Optional[str] = None
    chunk_index: int
    table_id: Optional[str] = None

class Chunk(BaseModel):
    """A piece of clinical text for retrieval."""
    chunk_id: str
    content: str
    metadata: ChunkMetadata

class ChunkingReport(BaseModel):
    """Summary of the chunking process per document."""
    source_id: str
    parent_count: int
    child_count: int
    table_count: int
    page_coverage: float
    warnings: List[str] = []
