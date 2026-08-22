from pathlib import Path
import pytest
from pydantic import ValidationError
from src.config.loader import load_domain_config, AppConfig, DomainConfig, SubtopicConfig


def test_load_default_domain_config():
    """Verify that default config/domains.json loads cleanly and passes all validations."""
    config = load_domain_config()
    assert isinstance(config, AppConfig)
    assert len(config.domains) == 5

    # Check 5 canonical domains
    expected_domains = {
        "artificial_intelligence": "AI",
        "cybersecurity": "CY",
        "agriculture": "AG",
        "healthcare": "HC",
        "climate": "CL",
    }

    for canonical_name, expected_prefix in expected_domains.items():
        assert canonical_name in config.domains
        domain_cfg = config.domains[canonical_name]
        assert domain_cfg.canonical_name == canonical_name
        assert domain_cfg.id_prefix == expected_prefix
        assert domain_cfg.target_count >= 50
        assert len(domain_cfg.subtopics) == 10

        # Check each subtopic has search queries
        for subtopic in domain_cfg.subtopics:
            assert len(subtopic.search_queries) > 0
            assert subtopic.target_count >= 5


def test_missing_required_canonical_domain_fails():
    """Verify that AppConfig rejects configurations missing any of the 5 canonical domains."""
    invalid_data = {
        "domains": {
            "artificial_intelligence": {
                "canonical_name": "artificial_intelligence",
                "display_name": "Artificial Intelligence",
                "id_prefix": "AI",
                "target_count": 50,
                "subtopics": [
                    {
                        "name": "Machine Learning",
                        "canonical": "machine_learning",
                        "target_count": 5,
                        "search_queries": ["query1"],
                    }
                ],
            }
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        AppConfig.model_validate(invalid_data)

    assert "Missing required canonical domains" in str(exc_info.value)


def test_duplicate_id_prefix_fails():
    """Verify that AppConfig rejects duplicate ID prefixes across domains."""
    invalid_data = {
        "domains": {
            "artificial_intelligence": {
                "canonical_name": "artificial_intelligence",
                "display_name": "AI",
                "id_prefix": "AI",
                "target_count": 50,
                "subtopics": [{"name": "S1", "canonical": "s1", "search_queries": ["q1"]}],
            },
            "cybersecurity": {
                "canonical_name": "cybersecurity",
                "display_name": "Cyber",
                "id_prefix": "AI",  # Duplicate prefix!
                "target_count": 50,
                "subtopics": [{"name": "S2", "canonical": "s2", "search_queries": ["q2"]}],
            },
            "agriculture": {
                "canonical_name": "agriculture",
                "display_name": "Agri",
                "id_prefix": "AG",
                "target_count": 50,
                "subtopics": [{"name": "S3", "canonical": "s3", "search_queries": ["q3"]}],
            },
            "healthcare": {
                "canonical_name": "healthcare",
                "display_name": "Health",
                "id_prefix": "HC",
                "target_count": 50,
                "subtopics": [{"name": "S4", "canonical": "s4", "search_queries": ["q4"]}],
            },
            "climate": {
                "canonical_name": "climate",
                "display_name": "Climate",
                "id_prefix": "CL",
                "target_count": 50,
                "subtopics": [{"name": "S5", "canonical": "s5", "search_queries": ["q5"]}],
            },
        }
    }
    with pytest.raises(ValidationError) as exc_info:
        AppConfig.model_validate(invalid_data)

    assert "Duplicate ID prefix" in str(exc_info.value)
