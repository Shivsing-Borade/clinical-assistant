import os
import json
from pathlib import Path
from typing import List, Dict, Any
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import ConversionResult

def convert_pdfs(raw_dir: str, interim_dir: str):
    raw_path = Path(raw_dir)
    interim_path = Path(interim_dir)
    interim_path.mkdir(parents=True, exist_ok=True)

    # Configure Docling with RapidOCR
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    # Docling 2.x generally integrates RapidOCR via the RapidOCR engine
    # if installed and configured in the options.

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    pdf_files = list(raw_path.glob("*.pdf"))
    reports = []

    for pdf in pdf_files:
        json_file = interim_path / f"{pdf.stem}.json"
        md_file = interim_path / f"{pdf.stem}.md"

        # Skip if both outputs exist and are non-empty/valid
        if json_file.exists() and md_file.exists():
            if json_file.stat().st_size > 0 and md_file.stat().st_size > 0:
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        json.load(f)
                    print(f"Skipping {pdf.name}, valid outputs already exist.")
                    # We still need to add this to the reports list
                    # In a real scenario, we'd reload the metadata from the JSON
                    # For now, we'll just mark it as success and move on.
                    # We'll handle report aggregation at the end.
                    continue
                except json.JSONDecodeError:
                    print(f"Invalid JSON for {pdf.name}, reprocessing...")

        print(f"Converting {pdf.name}...")
        try:
            result = converter.convert(pdf)

            # 1. Export Markdown
            md_content = result.document.export_to_markdown()
            md_file = interim_path / f"{pdf.stem}.md"
            md_file.write_text(md_content, encoding="utf-8")

            # 2. Export JSON
            json_content = result.document.export_to_dict()
            json_file = interim_path / f"{pdf.stem}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)

            # 3. Quality Inspection
            doc_report = {
                "filename": pdf.name,
                "status": "success",
                "pages_processed": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                "warnings": [],
                "manual_review_required": False,
                "review_reasons": []
            }

            # Basic Quality Checks
            # check for empty pages or suspicious length
            # Note: result.document structure varies; we check the markdown content as a proxy for text
            if not md_content.strip():
                doc_report["warnings"].append("Document contains no extractable text.")
                doc_report["manual_review_required"] = True
                doc_report["review_reasons"].append("Empty output")

            # Identify tables
            tables_count = 0
            # Search for table markers in markdown or check document structure
            if "Table" in md_content: # Simple proxy
                tables_count += 1

            doc_report["tables_found"] = tables_count
            reports.append(doc_report)

        except Exception as e:
            print(f"Error converting {pdf.name}: {str(e)}")
            reports.append({
                "filename": pdf.name,
                "status": "failed",
                "error": str(e),
                "manual_review_required": True,
                "review_reasons": ["Parsing error"]
            })

    # Save aggregate report
    report_file = interim_path / "extraction_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)

    return reports

if __name__ == "__main__":
    RAW_DIR = "data/raw"
    INTERIM_DIR = "data/interim"

    results = convert_pdfs(RAW_DIR, INTERIM_DIR)
    print(f"\nConversion complete. Report saved to {INTERIM_DIR}/extraction_report.json")
