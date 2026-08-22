import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from src.discovery.service import CandidateDiscoveryService
from src.models.candidate import PaperCandidate


def test_candidate_discovery_service(tmp_path):
    """Test CandidateDiscoveryService with mocked connectors and temporary audit log."""
    log_file = tmp_path / "test_collection_log.json"

    # Create mock connector returning 1 candidate
    mock_connector = MagicMock()
    mock_connector.search_candidates.return_value = [
        PaperCandidate(
            title="Mocked Paper Title",
            authors=["Author A"],
            domain="artificial_intelligence",
            subtopic="retrieval_augmented_generation",
            search_query="retrieval-augmented generation RAG",
            source="mock_arxiv",
            doi="10.1000/123",
            source_id="2005.11401",
        )
    ]

    service = CandidateDiscoveryService(
        log_path=log_file,
        connectors={"arxiv": mock_connector}
    )

    candidates = service.discover_candidates_for_subtopic(
        domain_name="artificial_intelligence",
        subtopic_canonical="retrieval_augmented_generation",
        source_names=["arxiv"],
        max_results_per_query=2,
        log_events=True,
    )

    assert len(candidates) > 0
    cand = candidates[0]
    assert cand.title == "Mocked Paper Title"
    assert cand.source == "mock_arxiv"

    # Check that collection_log.json contains candidate_discovered events
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        log_entries = json.load(f)

    assert len(log_entries) > 0
    event = log_entries[0]
    assert event["action"] == "candidate_discovered"
    assert event["domain"] == "artificial_intelligence"
    assert event["subtopic"] == "retrieval_augmented_generation"
    assert event["title"] == "Mocked Paper Title"
    assert "paper_id" not in event or event["paper_id"] is None
