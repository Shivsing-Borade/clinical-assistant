import re
import unicodedata
from typing import List, Dict, Any
from pathlib import Path
import json
from app.schemas.documents import SourceMetadata

class ClinicalCleaner:
    """
    Deterministic cleaner for clinical guideline text.
    Focuses on removing noise while preserving clinical meaning.
    """
    def __init__(self):
        # Noise patterns to remove (case insensitive)
        self.noise_patterns = [
            r'<!-- image -->',
            r'Page \d+ of \d+',
            r'\(Page \d+\)',
            # Remove recurring footer-like noise (simplified for now)
            # Add specific patterns as they are discovered in documents
        ]

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # 1. Normalize unicode
        text = unicodedata.normalize('NFKC', text)

        # 2. Remove identified noise patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 3. Fix broken line wrapping (merge lines that don't end in punctuation/headings)
        # This is a heuristic: if a line ends with a letter/number and next starts with one, merge.
        lines = text.splitlines()
        cleaned_lines = []

        if not lines:
            return ""

        current_line = lines[0]
        for i in range(1, len(lines)):
            line = lines[i]
            # Basic heuristic: if line is essentially empty, skip
            if not line.strip():
                continue

            # If current line doesn't end with common sentence terminators or is a heading,
            # and the next line isn't a heading, we might merge them.
            if (current_line.strip() and
                not current_line.strip().endswith(('.', ':', '?', '!', ')' )) and
                not line.strip().startswith('#')):
                current_line = current_line.rstrip() + " " + line.lstrip()
            else:
                cleaned_lines.append(current_line)
                current_line = line

        cleaned_lines.append(current_line)

        # 4. Normalize whitespace
        text = "\n".join(cleaned_lines)
        text = re.sub(r'[ \t]+', ' ', text) # collapse horizontal whitespace
        # Ensure we don't collapse meaningful double newlines

        return text.strip()

def process_documents(raw_dir: str, interim_dir: str, processed_dir: str, sources_manifest: List[Dict[str, Any]]):
    """
    Reads interim markdown/json files, cleans them, and saves results.
    """
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    cleaner = ClinicalCleaner()
    results = []

    for source in sources_manifest:
        source_id = source['source_id']
        filename = source['filename']

        md_file = interim_path / f"{Path(filename).stem}.md"
        if not md_file.exists():
            print(f"Warning: Markdown file not found for {filename}")
            continue

        content = md_file.read_text(encoding="utf-8")
        cleaned_content = cleaner.clean_text(content)

        output_file = processed_path / f"{source_id}.md"
        output_file.write_text(cleaned_content, encoding="utf-8")

        results.append({
            "source_id": source_id,
            "filename": filename,
            "status": "success"
        })

    return results
