import pytest
from pydantic import ValidationError
from src.models.paper import PaperMetadata
from src.models.collection_event import CollectionEvent, EventAction


def test_valid_paper_metadata():
    """Test creating a valid PaperMetadata object with required and optional fields."""
    paper = PaperMetadata(
        paper_id="AI001",
        domain="artificial_intelligence",
        subtopic="retrieval_augmented_generation",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        authors=["Patrick Lewis", "Ethan Perez"],
        publication_year=2020,
        source="arXiv",
        local_path="data/papers/artificial_intelligence/AI001.pdf",
        file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        collection_date="2026-07-28T10:00:00Z",
        status="downloaded",
        doi="10.48550/arXiv.2005.11401",
        arxiv_id="2005.11401",
    )
    assert paper.paper_id == "AI001"
    assert paper.domain == "artificial_intelligence"
    assert paper.abstract is None  # Optional field defaults to None
    assert paper.doi == "10.48550/arXiv.2005.11401"
    assert paper.open_access is True


@pytest.mark.parametrize(
    "paper_id,domain",
    [
        ("AI001", "artificial_intelligence"),
        ("CY042", "cybersecurity"),
        ("AG100", "agriculture"),
        ("HC007", "healthcare"),
        ("CL015", "climate"),
    ],
)
def test_valid_domain_id_prefix_combinations(paper_id, domain):
    """Test that paper_id matches expected domain prefix."""
    paper = PaperMetadata(
        paper_id=paper_id,
        domain=domain,
        subtopic="test_subtopic",
        title="Test Paper Title",
        authors=["Author One"],
        publication_year=2024,
        source="Test Source",
        local_path=f"data/papers/{domain}/{paper_id}.pdf",
        file_hash="dummyhash1234567890abcdef1234567890abcdef",
        collection_date="2026-07-28T10:00:00Z",
    )
    assert paper.paper_id == paper_id


@pytest.mark.parametrize(
    "paper_id,domain",
    [
        ("HC001", "artificial_intelligence"),  # HC prefix for AI domain
        ("AI001", "cybersecurity"),             # AI prefix for CY domain
        ("123", "healthcare"),                  # Missing prefix
        ("AG_001", "agriculture"),              # Invalid underscore format
    ],
)
def test_invalid_domain_id_prefix_fails(paper_id, domain):
    """Test that paper_id with wrong prefix for domain raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        PaperMetadata(
            paper_id=paper_id,
            domain=domain,
            subtopic="test_subtopic",
            title="Test Paper Title",
            authors=["Author One"],
            publication_year=2024,
            source="Test Source",
            local_path=f"data/papers/{domain}/{paper_id}.pdf",
            file_hash="dummyhash1234567890abcdef1234567890abcdef",
            collection_date="2026-07-28T10:00:00Z",
        )
    assert "Invalid paper_id" in str(exc_info.value)


def test_empty_authors_fails():
    """Test that empty authors list is rejected."""
    with pytest.raises(ValidationError):
        PaperMetadata(
            paper_id="AI001",
            domain="artificial_intelligence",
            subtopic="test_subtopic",
            title="Test Paper Title",
            authors=[],
            publication_year=2024,
            source="Test Source",
            local_path="data/papers/artificial_intelligence/AI001.pdf",
            file_hash="dummyhash",
            collection_date="2026-07-28T10:00:00Z",
        )


def test_collection_event_creation():
    """Test creating CollectionEvent audit trail records."""
    event = CollectionEvent(
        timestamp="2026-07-28T10:05:00Z",
        domain="healthcare",
        subtopic="medical_image_analysis",
        source="PubMed Central",
        action=EventAction.DUPLICATE_DETECTED,
        doi="10.1016/j.media.2023.102890",
        title="Deep learning in medical image segmentation",
        reason="Exact DOI match with HC003",
    )
    assert event.action == EventAction.DUPLICATE_DETECTED
    assert event.paper_id is None
    assert event.reason == "Exact DOI match with HC003"
