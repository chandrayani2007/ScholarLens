"""
Unit and Regression Test Suite for Strict Claim-Level Citation Verification

Tests:
1. Core invariant: Topical similarity alone fails citation verification.
2. Direct support verification for valid claims.
3. Multi-citation independent verification ([E1][E2]).
4. Online academic citation verification ([O1], [O2]).
"""

import pytest
from src.pipeline.rag import ClaimGroundingValidator, EvidenceItem


class TestClaimLevelCitationVerification:

    def test_1_topical_similarity_alone_fails_citation_verification(self):
        """
        Question: 'What are the limitations of Agentic RAG?'
        Claim: 'Agentic RAG suffers from excessive tool calls and high computational cost [E1].'
        Evidence E1: 'Agentic RAG evaluation practices have limitations.'
        Expected: Citation E1 FAILS verification for the claim because E1 lacks evidence for tool calls / cost.
        """
        claim = "Agentic RAG suffers from excessive tool calls and high computational cost [E1]."
        e1_text = "Agentic RAG evaluation practices have limitations in benchmark datasets."
        
        evidence_map = {
            "E1": EvidenceItem(
                citation_id="E1",
                unit_id="AI001_U01",
                parent_chunk_id="AI001_C01",
                chunk_id="AI001_C01",
                paper_id="AI001",
                section_id="SEC_01",
                section_name="Introduction",
                domain="artificial_intelligence",
                subtopic="rag",
                page_start=1,
                page_end=1,
                text=e1_text
            )
        }
        
        is_valid, direct_sup, aspect_sup, score, rationale = ClaimGroundingValidator.verify_claim_against_passage(
            claim, e1_text, question="What are the limitations of Agentic RAG?"
        )
        assert is_valid is False
        assert direct_sup is False
        
        grounded_ans, unsupported_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(
            claim, evidence_map, question="What are the limitations of Agentic RAG?"
        )
        assert unsupported_count >= 1
        assert "E1" not in active_tags
        assert "[E1]" not in grounded_ans

    def test_2_direct_support_passes_citation_verification(self):
        """
        Valid claim directly supported by evidence passage passes verification.
        """
        claim = "Research indicates that Agentic RAG evaluation practices have limitations [E1]."
        e1_text = "Agentic RAG evaluation practices have limitations in benchmark datasets and metrics."
        
        evidence_map = {
            "E1": EvidenceItem(
                citation_id="E1",
                unit_id="AI001_U01",
                parent_chunk_id="AI001_C01",
                chunk_id="AI001_C01",
                paper_id="AI001",
                section_id="SEC_01",
                section_name="Introduction",
                domain="artificial_intelligence",
                subtopic="rag",
                page_start=1,
                page_end=1,
                text=e1_text
            )
        }
        
        is_valid, direct_sup, aspect_sup, score, rationale = ClaimGroundingValidator.verify_claim_against_passage(
            claim, e1_text, question="What are the limitations of Agentic RAG?"
        )
        assert is_valid is True
        assert direct_sup is True
        
        grounded_ans, unsupported_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(
            claim, evidence_map, question="What are the limitations of Agentic RAG?"
        )
        assert unsupported_count == 0
        assert "E1" in active_tags
        assert "[E1]" in grounded_ans

    def test_3_multi_citation_independent_verification(self):
        """
        Sentence has [E1][E2].
        E1 directly supports the claim. E2 is unrelated.
        Expected: E1 retained, E2 removed.
        """
        claim = "Retrieval-augmented generation enhances output quality [E1][E2]."
        e1_text = "Retrieval-augmented generation enhances output quality significantly."
        e2_text = "Faithfulness is an evaluation dimension used in NLP benchmarks."
        
        evidence_map = {
            "E1": EvidenceItem(
                citation_id="E1",
                unit_id="AI001_U01",
                parent_chunk_id="AI001_C01",
                chunk_id="AI001_C01",
                paper_id="AI001",
                section_id="SEC_01",
                section_name="Introduction",
                domain="artificial_intelligence",
                subtopic="rag",
                page_start=1,
                page_end=1,
                text=e1_text
            ),
            "E2": EvidenceItem(
                citation_id="E2",
                unit_id="AI002_U01",
                parent_chunk_id="AI002_C01",
                chunk_id="AI002_C01",
                paper_id="AI002",
                section_id="SEC_01",
                section_name="Introduction",
                domain="artificial_intelligence",
                subtopic="rag",
                page_start=1,
                page_end=1,
                text=e2_text
            )
        }
        
        grounded_ans, unsupported_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(
            claim, evidence_map
        )
        assert "E1" in active_tags
        assert "E2" not in active_tags
        assert "[E1]" in grounded_ans
        assert "[E2]" not in grounded_ans

    def test_4_online_academic_citation_verification(self):
        """
        Online academic citation [O1] verified against online abstract text.
        """
        claim = "Transformer models utilize self-attention mechanisms [O1]."
        o1_text = "Transformer models utilize self-attention mechanisms to process sequential data."
        
        evidence_map = {
            "O1": EvidenceItem(
                citation_id="O1",
                unit_id="2106.00001_U01",
                parent_chunk_id="2106.00001_C01",
                chunk_id="2106.00001_C01",
                paper_id="arXiv:2106.00001",
                section_id="SEC_ABSTRACT",
                section_name="Abstract",
                domain="artificial_intelligence",
                subtopic="deep_learning",
                page_start=1,
                page_end=1,
                text=o1_text
            )
        }
        
        grounded_ans, unsupported_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(
            claim, evidence_map
        )
        assert unsupported_count == 0
        assert "O1" in active_tags
        assert "[O1]" in grounded_ans
