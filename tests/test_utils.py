from src.utils.text import normalize_title, levenshtein_similarity
from src.utils.dedup import evaluate_candidate_deduplication, DeduplicationSignalType


def test_title_normalization():
    """Test string title normalization removing punctuation, capitalization, and extra whitespace."""
    raw_title = "  Retrieval-Augmented Generation: For Knowledge-Intensive NLP Tasks!! "
    expected = "retrieval augmented generation for knowledge intensive nlp tasks"
    assert normalize_title(raw_title) == expected


def test_levenshtein_similarity():
    """Test fuzzy title similarity computation."""
    t1 = "Retrieval Augmented Generation for Knowledge Intensive NLP Tasks"
    t2 = "Retrieval Augmented Generation for Knowledge-Intensive NLP Task"
    sim = levenshtein_similarity(t1, t2)
    assert sim > 0.95

    t3 = "Deep Convolutional Neural Networks for Image Classification"
    assert levenshtein_similarity(t1, t3) < 0.5


def test_dedup_exact_doi():
    """Test exact DOI match returns strong duplicate signal."""
    res = evaluate_candidate_deduplication(
        candidate_title="Some New Title",
        candidate_doi="10.1000/182",
        existing_dois={"10.1000/182": "AI005"},
    )
    assert res.is_duplicate is True
    assert res.requires_flag is False
    assert res.signal_type == DeduplicationSignalType.EXACT_DOI
    assert res.matching_paper_id == "AI005"


def test_dedup_exact_source_id():
    """Test exact source ID match returns strong duplicate signal."""
    res = evaluate_candidate_deduplication(
        candidate_title="Another Title",
        candidate_source_id="2005.11401",
        existing_source_ids={"2005.11401": "AI001"},
    )
    assert res.is_duplicate is True
    assert res.signal_type == DeduplicationSignalType.EXACT_SOURCE_ID
    assert res.matching_paper_id == "AI001"


def test_dedup_exact_title():
    """Test exact normalized title match returns strong duplicate signal."""
    res = evaluate_candidate_deduplication(
        candidate_title="Precision Agriculture Technologies and Applications!",
        existing_normalized_titles={"precision agriculture technologies and applications": "AG002"},
    )
    assert res.is_duplicate is True
    assert res.signal_type == DeduplicationSignalType.EXACT_TITLE
    assert res.matching_paper_id == "AG002"


def test_dedup_fuzzy_title_match_flags_candidate():
    """Verify that fuzzy title similarity flags candidate without auto-deleting it."""
    res = evaluate_candidate_deduplication(
        candidate_title="Deep Learning for Plant Leaf Disease Detection in Smart Agriculture",
        existing_normalized_titles={"deep learning for plant leaf disease detection in smart farming": "AG009"},
        fuzzy_threshold=0.80,
    )
    # Requirement constraint check: requires_flag must be True, is_duplicate must be False
    assert res.is_duplicate is False
    assert res.requires_flag is True
    assert res.signal_type == DeduplicationSignalType.FUZZY_TITLE_MATCH
    assert res.matching_paper_id == "AG009"
    assert res.similarity_score is not None and res.similarity_score >= 0.80
