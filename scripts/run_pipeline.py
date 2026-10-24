import yaml
import json
from pathlib import Path
from typing import List, Dict, Any
from app.ingestion.normalize import process_documents
from app.ingestion.chunking.structure_aware import HybridChunker
from app.schemas.chunks import ChunkingReport

def run_pipeline():
    # Config
    RAW_DIR = "data/raw"
    INTERIM_DIR = "data/interim"
    PROCESSED_DIR = "data/processed"
    MANIFEST_FILE = "manifests/sources.yaml"

    # 1. Load Manifest
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        sources = yaml.safe_load(f)

    # 2. Cleaning
    print("Cleaning documents...")
    cleaning_results = process_documents(RAW_DIR, INTERIM_DIR, PROCESSED_DIR, sources)

    # 3. Chunking
    print("Chunking documents...")
    chunker = HybridChunker()

    all_parents = []
    all_children = []
    reports = []

    for source in sources:
        source_id = source['source_id']
        filename = source['filename']

        cleaned_file = Path(PROCESSED_DIR) / f"{source_id}.md"
        if not cleaned_file.exists():
            continue

        content = cleaned_file.read_text(encoding="utf-8")
        parents, children = chunker.chunk(source_id, filename, content)

        # Convert Pydantic models to dict for JSON storage
        all_parents.extend([p.model_dump() for p in parents])
        all_children.extend([c.model_dump() for c in children])

        # Table count
        table_count = sum(1 for p in parents if p.metadata.content_type == "table")

        reports.append(ChunkingReport(
            source_id=source_id,
            parent_count=len(parents),
            child_count=len(children),
            table_count=table_count,
            page_coverage=1.0, # Simplified for now
            warnings=[]
        ).model_dump())

    # 4. Save Outputs
    # Save chunks
    with open(Path(PROCESSED_DIR) / "parents.json", "w", encoding="utf-8") as f:
        json.dump(all_parents, f, indent=2)

    with open(Path(PROCESSED_DIR) / "children.json", "w", encoding="utf-8") as f:
        json.dump(all_children, f, indent=2)

    with open(Path(PROCESSED_DIR) / "chunking_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    print(f"Pipeline complete. Results saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    run_pipeline()
