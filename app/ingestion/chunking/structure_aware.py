import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
from app.schemas.chunks import Chunk, ChunkMetadata, ChunkingReport

class HybridChunker:
    """
    Implements structure-aware, table-aware parent-child chunking.
    """
    def __init__(self,
                 parent_size: int = 1500,
                 child_size: int = 400,
                 overlap: int = 50):
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    def _get_section_path(self, content: str) -> List[str]:
        """Identify headings to build a section path."""
        # Simplified heading extraction: find all lines starting with #
        headings = re.findall(r'^(#+)\s*(.*)$', content, re.MULTILINE)
        return [h[1] for h in headings]

    def _split_into_sections(self, content: str) -> List[Tuple[str, str]]:
        """
        Splits content into sections based on markdown headings.
        Returns a list of (section_title, section_content).
        """
        # Split by lines starting with #
        parts = re.split(r'(^#+ .*)$', content, flags=re.MULTILINE)

        sections = []
        current_title = "Introduction"

        # The first part before any heading is Introduction
        if parts and parts[0].strip():
            sections.append((current_title, parts[0]))

        for i in range(1, len(parts), 2):
            title = parts[i].strip('# ').strip()
            body = parts[i+1] if i+1 < len(parts) else ""
            sections.append((title, body))

        return sections

    def _handle_tables(self, text: str) -> List[str]:
        """
        Identifies and isolates tables.
        For simplicity here, we treat the entire table as one unit.
        """
        # Markdown table regex: starts with | and ends with a newline after a |
        # This is a basic approximation.
        table_pattern = r'(\|.*\|(?:\n\|.*\|)*)'
        tables = re.findall(table_pattern, text)
        return tables

    def chunk(self, source_id: str, filename: str, content: str) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Main chunking logic.
        Returns (parent_chunks, child_chunks).
        """
        sections = self._split_into_sections(content)
        all_parents = []
        all_children = []

        table_counter = 0

        for section_idx, (title, body) in enumerate(sections):
            section_full_text = f"## {title}\n{body}"
            parent_id = f"p_{source_id}_{section_idx}"

            # 1. Create Parent Chunk
            # If section is too large, we would recursive split it, but here we keep it as a parent
            # if it fits the parent_size. Otherwise, we treat it as multiple parents.

            # Simple Parent creation
            parent_metadata = ChunkMetadata(
                source_id=source_id,
                source_file=filename,
                section_path=title,
                content_type="prose",
                parent_id=parent_id,
                child_id=None,
                chunk_index=section_idx
            )

            parent_chunk = Chunk(
                chunk_id=parent_id,
                content=section_full_text,
                metadata=parent_metadata
            )
            all_parents.append(parent_chunk)

            # 2. Create Child Chunks
            # Split body into smaller pieces
            words = section_full_text.split()
            for i in range(0, len(words), self.child_size - self.overlap):
                child_slice = words[i : i + self.child_size]
                child_text = " ".join(child_slice)

                child_id = f"c_{source_id}_{section_idx}_{i // (self.child_size - self.overlap)}"

                child_metadata = ChunkMetadata(
                    source_id=source_id,
                    source_file=filename,
                    section_path=title,
                    content_type="prose",
                    parent_id=parent_id,
                    child_id=child_id,
                    chunk_index=i // (self.child_size - self.overlap)
                )

                all_children.append(Chunk(
                    chunk_id=child_id,
                    content=child_text,
                    metadata=child_metadata
                ))

            # 3. Special Table Handling
            tables = self._handle_tables(body)
            for t_idx, table_text in enumerate(tables):
                table_counter += 1
                table_id = f"t_{source_id}_{table_counter}"

                # Tables are parents in their own right or children of the section
                table_metadata = ChunkMetadata(
                    source_id=source_id,
                    source_file=filename,
                    section_path=title,
                    content_type="table",
                    parent_id=parent_id,
                    child_id=None,
                    chunk_index=t_idx
                )

                all_parents.append(Chunk(
                    chunk_id=table_id,
                    content=f"Table in {title}:\n{table_text}",
                    metadata=table_metadata
                ))

                # Table children (if table is small, just one child)
                all_children.append(Chunk(
                    chunk_id=f"ct_{source_id}_{table_counter}_{0}",
                    content=f"Table in {title}:\n{table_text}",
                    metadata=ChunkMetadata(
                        source_id=source_id,
                        source_file=filename,
                        section_path=title,
                        content_type="table",
                        parent_id=table_id,
                        child_id=f"ct_{source_id}_{table_counter}_{0}",
                        chunk_index=0
                    )
                ))

        return all_parents, all_children
