import json
from pathlib import Path
from unittest.mock import MagicMock
import fitz
import pytest
from src.downloader.pdf_downloader import PdfDownloader
from src.models.candidate import PaperCandidate
from src.pipeline.ingestion import PaperIngestionPipeline
from src.storage.metadata_store import MetadataStore
from src.validator.pdf_validator import PdfValidator


def create_mock_pdf_file(target_path: Path, text: str = "Sample research paper content text exceeding 100 characters requirement for validation test. " * 3):
    """Helper to create a valid test PDF file."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(str(target_path))
    doc.close()


def test_full_pipeline_ingestion_and_resumability(tmp_path):
    """Test full candidate ingestion workflow and verify resumability."""
    papers_json = tmp_path / "papers.json"
    log_json = tmp_path / "collection_log.json"
    papers_json.write_text("[]", encoding="utf-8")
    log_json.write_text("[]", encoding="utf-8")

    metadata_store = MetadataStore(papers_json_path=papers_json, log_json_path=log_json)

    # Mock downloader to return a created test PDF
    temp_pdf = tmp_path / "mock_download.pdf.tmp"

    downloader = MagicMock()
    def mock_download(url, temp_dir):
        create_mock_pdf_file(temp_pdf)
        return temp_pdf

    downloader.download_temp_pdf.side_effect = mock_download
    downloader.cleanup_temp_file.side_effect = lambda p: p.unlink() if p.exists() else None

    pipeline = PaperIngestionPipeline(
        metadata_store=metadata_store,
        downloader=downloader,
        validator=PdfValidator(min_text_length=50),
        base_dir=tmp_path,
    )

    candidate = PaperCandidate(
        title="Retrieval Augmented Generation for Knowledge Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez"],
        domain="artificial_intelligence",
        subtopic="retrieval_augmented_generation",
        search_query="retrieval augmented generation",
        source="arXiv",
        doi="10.48550/arXiv.2005.11401",
        arxiv_id="2005.11401",
        source_id="2005.11401",
        source_url="https://arxiv.org/abs/2005.11401",
        pdf_url="https://arxiv.org/pdf/2005.11401.pdf",
        publication_year=2020,
    )

    # First Ingestion Run -> Should accept paper as AI001
    accepted, metadata, msg = pipeline.process_candidate(candidate)

    assert accepted is True
    assert metadata is not None
    assert metadata.paper_id == "AI001"
    assert metadata.domain == "artificial_intelligence"
    assert "AI001" in msg

    # Check file relocated to data/papers/artificial_intelligence/AI001.pdf
    expected_pdf = tmp_path / "data" / "papers" / "artificial_intelligence" / "AI001.pdf"
    assert expected_pdf.exists()

    # Check papers.json persisted
    with open(papers_json, "r", encoding="utf-8") as f:
        stored_papers = json.load(f)
    assert len(stored_papers) == 1
    assert stored_papers[0]["paper_id"] == "AI001"

    # Resumability Test: Process same candidate again -> Should be detected as duplicate
    accepted_again, metadata_again, msg_again = pipeline.process_candidate(candidate)

    assert accepted_again is False
    assert metadata_again is None
    assert "Duplicate detected" in msg_again

    # Check papers.json still has only 1 paper (AI001)
    with open(papers_json, "r", encoding="utf-8") as f:
        stored_papers_after = json.load(f)
    assert len(stored_papers_after) == 1
    assert stored_papers_after[0]["paper_id"] == "AI001"
