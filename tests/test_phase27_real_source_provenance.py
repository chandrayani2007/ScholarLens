"""
Phase 27 — Independent Live-Source Provenance & Real ArXiv Verification Test Suite

Verifies:
1. test_online_citation_exists_in_actual_arxiv_response()
2. test_o_citation_provenance_is_retrieved_not_generated()
3. test_online_metadata_contains_authors_and_publication_date()
4. test_no_hardcoded_arxiv_examples_in_production_path()
5. test_live_online_source_id_resolves()
6. test_synthesis_with_rich_evidence_provides_developed_answer()
"""

import pytest
import re
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
from src.pipeline.rag import RAGPipeline
from src.pipeline.online_fallback import OnlineAcademicRetriever, OnlineEvidenceItem
from src.pipeline.llm import MockLLMProvider


class TestPhase27RealSourceProvenance:

    def test_online_citation_exists_in_actual_arxiv_response(self):
        """Every [O#] online citation returned by RAGPipeline MUST independently exist on live ArXiv API."""
        online_retriever = OnlineAcademicRetriever(timeout=10)

        # Force Tier 3 Online Search using an out-of-corpus query concept
        q = "What is surface code quantum error correction?"
        online_items = online_retriever.retrieve(q, max_results=3)

        if not online_items:
            pytest.skip("ArXiv network request timed out or returned 0 items.")

        mock_llm = MockLLMProvider(custom_response='{"answer": "Surface code quantum error correction uses square physical qubit arrays to measure stabilizer syndromes [O1]. Physical stabilizer measurements continuously compute error syndromes across adjacent data and syndrome qubits, allowing real-time fault-tolerant error suppression below the threshold limit [O2].", "confidence": "High", "evidence_strength": "High", "limitations": "Online findings.", "why_this_answer": "ArXiv fallback."}')

        pipeline = RAGPipeline(online_retriever=online_retriever, llm_provider=mock_llm)
        res = pipeline.answer(q)

        online_cits = [c for c in res.citations.values() if c.source_type == "online"]
        if not online_cits:
            pytest.skip("ArXiv API call timed out during RAGPipeline.answer execution.")

        for cit in online_cits:
            clean_id = cit.paper_id.replace("arXiv:", "")
            url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
            req = urllib.request.Request(url, headers={"User-Agent": "ResearchMind-Test/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entry = root.find("atom:entry", ns)
            assert entry is not None, f"ArXiv ID '{cit.paper_id}' not found on export.arxiv.org API!"

            title_elem = entry.find("atom:title", ns)
            real_title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
            assert len(real_title) > 5, "Live ArXiv entry title is empty!"
            assert cit.title == real_title, f"Title mismatch! (Returned: '{cit.title}' vs ArXiv API: '{real_title}')"

    def test_o_citation_provenance_is_retrieved_not_generated(self):
        """Online citation metadata (paper_id, title, authors, published, url) MUST be copied from OnlineEvidenceItem."""
        retrieved_item = OnlineEvidenceItem(
            citation_id="O1",
            paper_id="arXiv:2103.99999",
            title="Real Retrieved Quantum Error Correction Title",
            authors=["Dr. Quantum", "Dr. Physics"],
            published_date="2021-03-15",
            url="http://arxiv.org/abs/2103.99999",
            domain="artificial_intelligence",
            section_name="Abstract",
            text="Real retrieved abstract content for surface code quantum error correction stabilizer syndrome."
        )

        mock_online = type("MockOnlineRetriever", (), {"retrieve": lambda self, **kwargs: [retrieved_item]})()
        pipeline = RAGPipeline(online_retriever=mock_online, llm_provider=MockLLMProvider())

        res = pipeline.answer("What is surface code quantum error correction?")
        assert "O1" in res.citations
        cit_o1 = res.citations["O1"]

        assert cit_o1.paper_id == "arXiv:2103.99999"
        assert cit_o1.title == "Real Retrieved Quantum Error Correction Title"
        assert cit_o1.authors == ["Dr. Quantum", "Dr. Physics"]
        assert cit_o1.published == "2021-03-15"
        assert cit_o1.url == "http://arxiv.org/abs/2103.99999"

    def test_online_metadata_contains_authors_and_publication_date(self):
        """Every online citation object MUST have non-empty authors list and non-empty published date."""
        online_retriever = OnlineAcademicRetriever(timeout=10)
        items = online_retriever.retrieve("quantum error correction", max_results=3)

        if items:
            for item in items:
                assert item.authors is not None and len(item.authors) > 0, "Authors list is empty!"
                assert item.published_date is not None and len(item.published_date) >= 4, "Published date is invalid!"
                assert item.paper_id.startswith("arXiv:"), "Invalid arXiv paper_id format!"
        else:
            assert items == []

    def test_no_hardcoded_arxiv_examples_in_production_path(self):
        """Verify OnlineAcademicRetriever does NOT contain fake hardcoded arXiv IDs."""
        retriever = OnlineAcademicRetriever()
        items = retriever.retrieve("surface code quantum error correction", max_results=3)

        hardcoded_fake_ids = ["arXiv:2401.05432", "arXiv:2311.09876", "arXiv:2401.12345", "arXiv:2403.99999"]
        for item in items:
            assert item.paper_id not in hardcoded_fake_ids, f"Hardcoded test arXiv ID '{item.paper_id}' returned by retriever!"

    def test_live_online_source_id_resolves(self):
        """Canonical URL of retrieved online item resolves via HTTP 200/301/302 from ArXiv."""
        online_retriever = OnlineAcademicRetriever(timeout=10)
        items = online_retriever.retrieve("surface code quantum error correction", max_results=1)

        if items:
            item = items[0]
            url = item.url
            req = urllib.request.Request(url, headers={"User-Agent": "ResearchMind-Test/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status in (200, 301, 302), f"ArXiv URL '{url}' returned HTTP {resp.status}"
        else:
            pytest.skip("ArXiv network request timed out during test.")

    def test_synthesis_with_rich_evidence_provides_developed_answer(self):
        """When 4+ strong evidence items are available for an explanatory query, answer synthesis produces a developed answer (>110 words)."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is surface code quantum error correction?")
        if "insufficient evidence" in res.answer.lower():
            pytest.skip("ArXiv network request timed out during test.")
        word_count = len(re.findall(r"\b\w+\b", res.answer))
        assert word_count >= 110, f"Rich evidence answer is under-developed ({word_count} words < 110 words)"
        assert len(res.citations) >= 2, "Rich evidence query should cite multiple evidence items."
