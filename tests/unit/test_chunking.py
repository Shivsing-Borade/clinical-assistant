import pytest
import json
from pathlib import Path

def test_processed_files_exist():
    processed_dir = Path("data/processed")
    assert processed_dir.exists()
    assert (processed_dir / "parents.json").exists()
    assert (processed_dir / "children.json").exists()
    assert (processed_dir / "chunking_report.json").exists()

def test_all_sources_represented():
    with open("manifests/sources.yaml", "r") as f:
        import yaml
        sources = yaml.safe_load(f)

    with open("data/processed/chunking_report.json", "r") as f:
        reports = json.load(f)

    source_ids = [s['source_id'] for s in sources]
    report_ids = [r['source_id'] for r in reports]

    for sid in source_ids:
        assert sid in report_ids

def test_parent_child_links():
    with open("data/processed/children.json", "r") as f:
        children = json.load(f)

    with open("data/processed/parents.json", "r") as f:
        parents = json.load(f)

    parent_ids = {p['chunk_id'] for p in parents}

    for child in children:
        # Every child must have a parent_id
        assert child['metadata']['parent_id'] is not None
        # Parent ID must exist in the parents list
        assert child['metadata']['parent_id'] in parent_ids

def test_no_empty_chunks():
    with open("data/processed/children.json", "r") as f:
        children = json.load(f)

    for child in children:
        assert len(child['content'].strip()) > 0

def test_medical_content_preservation():
    # Check a few known terms in processed files
    with open("data/processed/hypertension.md", "r") as f:
        content = f.read()

    # Check for units and medical terms
    assert "mmHg" in content
    assert "systolic blood pressure" in content
    assert "anti-hypertensives" in content
