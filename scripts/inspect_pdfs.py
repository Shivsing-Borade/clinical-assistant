import os
import hashlib
import yaml
from pathlib import Path
from pydantic import ValidationError
from pypdf import PdfReader
from app.schemas.documents import SourceMetadata, SourceRegistry

def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def analyze_pdf(file_path: Path):
    """Performs basic PDF diagnostics to identify scanned pages or missing text."""
    warnings = []
    page_count = None
    has_text = False

    try:
        reader = PdfReader(file_path)
        page_count = len(reader.pages)

        # Check a few pages for extractable text to detect scanned PDFs
        # We check the first 5 pages or all if fewer.
        pages_to_check = reader.pages[:5]
        for i, page in enumerate(pages_to_check):
            text = page.extract_text()
            if text and text.strip():
                has_text = True
                break

        if not has_text:
            warnings.append("Possible scanned PDF: No extractable text found in the first 5 pages.")

    except Exception as e:
        warnings.append(f"Error reading PDF: {str(e)}")

    return page_count, warnings

def inventory_raw_pdfs(raw_dir: str, output_manifest: str):
    raw_path = Path(raw_dir)
    pdf_files = list(raw_path.glob("*.pdf"))

    registry = SourceRegistry(sources=[])

    for pdf in pdf_files:
        print(f"Processing {pdf.name}...")

        checksum = calculate_sha256(pdf)
        file_size = pdf.stat().st_size
        page_count, warnings = analyze_pdf(pdf)

        # source_id is generated as a simple slug from filename for now
        source_id = pdf.stem.lower().replace(" ", "_").replace("-", "_")

        try:
            meta = SourceMetadata(
                source_id=source_id,
                filename=pdf.name,
                checksum=checksum,
                file_size_bytes=file_size,
                page_count=page_count,
                parsing_warnings=warnings,
                # Other metadata defaults to "unknown" as per requirements
            )
            registry.sources.append(meta)
        except ValidationError as e:
            print(f"Validation error for {pdf.name}: {e}")

    # Save to YAML
    with open(output_manifest, "w") as f:
        yaml.dump([s.model_dump() for s in registry.sources], f, sort_keys=False)

    return registry

if __name__ == "__main__":
    # Default paths for local execution
    RAW_DATA_DIR = "data/raw"
    MANIFEST_PATH = "manifests/sources.yaml"

    inventory_raw_pdfs(RAW_DATA_DIR, MANIFEST_PATH)
    print(f"\nInventory complete. Manifest saved to {MANIFEST_PATH}")
