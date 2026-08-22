from unittest.mock import MagicMock, patch
import pytest
import requests
from src.connectors.arxiv import ArxivConnector
from src.connectors.pubmed import EuropePmcConnector
from src.connectors.semantic_scholar import SemanticScholarConnector
from src.models.candidate import PaperCandidate


# Mock arXiv Atom XML response
SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2005.11401v2</id>
    <published>2020-05-22T17:59:00Z</published>
    <title>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</title>
    <summary>Large pre-trained language models have been shown to store implicit knowledge...</summary>
    <author><name>Patrick Lewis</name></author>
    <author><name>Ethan Perez</name></author>
    <arxiv:doi>10.48550/arXiv.2005.11401</arxiv:doi>
  </entry>
</feed>
"""

# Mock Europe PMC JSON response
SAMPLE_EUROPE_PMC_JSON = {
    "resultList": {
        "result": [
            {
                "id": "PMC7123456",
                "pmcid": "PMC7123456",
                "pmid": "32109876",
                "doi": "10.1016/j.artmed.2020.101850",
                "title": "Clinical Decision Support Systems Powered by Machine Learning",
                "authorList": {
                    "author": [
                        {"fullName": "Jane Doe"},
                        {"fullName": "John Smith"}
                    ]
                },
                "abstractText": "This study evaluates machine learning algorithms for clinical decision support...",
                "pubYear": "2020",
                "firstPublicationDate": "2020-03-15",
                "journalTitle": "Artificial Intelligence in Medicine",
                "isOpenAccess": "Y",
                "fullTextUrlList": {
                    "fullTextUrl": [
                        {"documentStyle": "pdf", "url": "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC7123456&blobtype=pdf"}
                    ]
                }
            }
        ]
    }
}

# Mock Semantic Scholar JSON response
SAMPLE_S2_JSON = {
    "data": [
        {
            "paperId": "s2_987654321",
            "title": "Deep Learning for Crop Disease Classification",
            "authors": [{"name": "Alice Farmer"}, {"name": "Bob Green"}],
            "abstract": "Deep convolutional networks applied to plant pathology imagery...",
            "year": 2022,
            "publicationDate": "2022-08-10",
            "venue": "Computers and Electronics in Agriculture",
            "externalIds": {"DOI": "10.1016/j.compag.2022.107000", "ArXiv": "2208.01234"},
            "url": "https://www.semanticscholar.org/paper/s2_987654321",
            "isOpenAccess": True,
            "openAccessPdf": {"url": "https://arxiv.org/pdf/2208.01234.pdf"}
        }
    ]
}


def test_paper_candidate_model_validation():
    """Verify PaperCandidate model creation and clean fields."""
    cand = PaperCandidate(
        title="Test Title",
        authors=[" Author One ", "Author Two "],
        domain="artificial_intelligence",
        subtopic="machine_learning",
        search_query="test query",
        source="arXiv",
    )
    assert cand.title == "Test Title"
    assert cand.authors == ["Author One", "Author Two"]
    assert cand.abstract is None
    assert cand.doi is None


def test_arxiv_connector_xml_parsing():
    """Verify ArxivConnector parses Atom XML into normalized PaperCandidate objects."""
    connector = ArxivConnector()
    candidates = connector.parse_xml_response(
        SAMPLE_ARXIV_XML,
        domain="artificial_intelligence",
        subtopic="retrieval_augmented_generation",
        search_query="retrieval augmented generation"
    )
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
    assert cand.authors == ["Patrick Lewis", "Ethan Perez"]
    assert cand.arxiv_id == "2005.11401"
    assert cand.publication_year == 2020
    assert cand.doi == "10.48550/arXiv.2005.11401"
    assert cand.pdf_url == "https://arxiv.org/pdf/2005.11401.pdf"
    assert cand.source == "arXiv"


def test_europe_pmc_connector_json_parsing():
    """Verify EuropePmcConnector parses JSON into normalized PaperCandidate objects."""
    connector = EuropePmcConnector()
    candidates = connector.parse_json_response(
        SAMPLE_EUROPE_PMC_JSON,
        domain="healthcare",
        subtopic="clinical_decision_support",
        search_query="clinical decision support"
    )
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.title == "Clinical Decision Support Systems Powered by Machine Learning"
    assert cand.authors == ["Jane Doe", "John Smith"]
    assert cand.publication_year == 2020
    assert cand.doi == "10.1016/j.artmed.2020.101850"
    assert cand.pdf_url == "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC7123456&blobtype=pdf"
    assert cand.source == "Europe PMC"


def test_semantic_scholar_connector_json_parsing():
    """Verify SemanticScholarConnector parses JSON into normalized PaperCandidate objects."""
    connector = SemanticScholarConnector()
    candidates = connector.parse_json_response(
        SAMPLE_S2_JSON,
        domain="agriculture",
        subtopic="crop_disease_detection",
        search_query="crop disease detection"
    )
    assert len(candidates) == 1
    cand = candidates[0]
    assert cand.title == "Deep Learning for Crop Disease Classification"
    assert cand.authors == ["Alice Farmer", "Bob Green"]
    assert cand.publication_year == 2022
    assert cand.doi == "10.1016/j.compag.2022.107000"
    assert cand.arxiv_id == "2208.01234"
    assert cand.pdf_url == "https://arxiv.org/pdf/2208.01234.pdf"
    assert cand.source == "Semantic Scholar"


def test_malformed_xml_handling():
    """Verify ArxivConnector handles malformed XML string appropriately."""
    connector = ArxivConnector()
    with pytest.raises(ValueError) as exc_info:
        connector.parse_xml_response("NOT VALID XML", domain="ai", subtopic="ml", search_query="test")
    assert "Failed to parse arXiv Atom XML" in str(exc_info.value)


@patch("src.connectors.base.requests.Session.get")
def test_connector_retry_on_failure(mock_get):
    """Verify BaseConnector retries transient network errors before raising RuntimeError."""
    mock_get.side_effect = requests.RequestException("Network Error")
    connector = ArxivConnector(max_retries=2, backoff_factor=0.01)

    with pytest.raises(RuntimeError) as exc_info:
        connector.search_candidates("query", domain="ai", subtopic="ml")

    assert "request failed after 2 attempts" in str(exc_info.value)
    assert mock_get.call_count == 2
