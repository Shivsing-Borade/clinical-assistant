import pytest
from pathlib import Path
from scripts.inspect_pdfs import calculate_sha256, analyze_pdf

def test_calculate_sha256(tmp_path):
    # Create a synthetic file
    f = tmp_path / "test.txt"
    f.write_text("hello world")

    # Expected SHA-256 for "hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert calculate_sha256(f) == expected

def test_analyze_pdf_non_existent(tmp_path):
    f = tmp_path / "non_existent.pdf"
    # pypdf usually raises an error or we catch it in analyze_pdf
    page_count, warnings = analyze_pdf(f)
    assert page_count is None
    assert any("Error reading PDF" in w for w in warnings)

# Note: For a full PDF test, we would need a synthetic PDF fixture.
# Since we cannot easily create a valid PDF from scratch without heavy deps,
# we rely on the integration check against existing raw files.
