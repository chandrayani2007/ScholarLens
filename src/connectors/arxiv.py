import re
import xml.etree.ElementTree as ET
from typing import List, Optional
from src.connectors.base import BaseConnector
from src.models.candidate import PaperCandidate

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivConnector(BaseConnector):
    """Connector for querying the official arXiv API (http://export.arxiv.org/api/query)."""

    BASE_URL = "http://export.arxiv.org/api/query"

    @property
    def source_name(self) -> str:
        return "arXiv"

    def search_candidates(
        self,
        query: str,
        domain: str,
        subtopic: str,
        max_results: int = 10
    ) -> List[PaperCandidate]:
        # Formulate search query parameter for arXiv API
        # e.g., search all fields or title/abstract: search_query=all:"query"
        clean_query = query.strip()
        params = {
            "search_query": f'all:"{clean_query}"',
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        response = self._make_request(self.BASE_URL, params=params)
        return self.parse_xml_response(response.text, domain=domain, subtopic=subtopic, search_query=query)

    def parse_xml_response(
        self,
        xml_text: str,
        domain: str,
        subtopic: str,
        search_query: str
    ) -> List[PaperCandidate]:
        candidates: List[PaperCandidate] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse arXiv Atom XML response: {e}")

        for entry in root.findall(f"{ATOM_NS}entry"):
            # Raw id tag: http://arxiv.org/abs/2005.11401v2
            raw_id_elem = entry.find(f"{ATOM_NS}id")
            if raw_id_elem is None or not raw_id_elem.text:
                continue

            raw_id = raw_id_elem.text.strip()
            arxiv_id_match = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+|[\w-]+/?[0-9]+)", raw_id)
            if not arxiv_id_match:
                continue

            arxiv_id = arxiv_id_match.group(1).split("v")[0]  # Strip version suffix (e.g. 2005.11401v2 -> 2005.11401)
            versioned_id = raw_id.split("/")[-1]

            # Title
            title_elem = entry.find(f"{ATOM_NS}title")
            title = title_elem.text.replace("\n", " ").strip() if title_elem is not None and title_elem.text else ""
            if not title:
                continue

            # Authors
            authors = []
            for author_elem in entry.findall(f"{ATOM_NS}author"):
                name_elem = author_elem.find(f"{ATOM_NS}name")
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Abstract
            summary_elem = entry.find(f"{ATOM_NS}summary")
            abstract = summary_elem.text.replace("\n", " ").strip() if summary_elem is not None and summary_elem.text else None

            # Publication Date / Year
            published_elem = entry.find(f"{ATOM_NS}published")
            pub_date = None
            pub_year = None
            if published_elem is not None and published_elem.text:
                pub_date_raw = published_elem.text.strip()
                if len(pub_date_raw) >= 10:
                    pub_date = pub_date_raw[:10]  # YYYY-MM-DD
                    try:
                        pub_year = int(pub_date[:4])
                    except ValueError:
                        pub_year = None

            # DOI (if available in arXiv atom feed)
            doi_elem = entry.find(f"{ARXIV_NS}doi")
            doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

            # Venue / Journal Ref
            journal_elem = entry.find(f"{ARXIV_NS}journal_ref")
            venue = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else None

            # PDF and Source URL
            source_url = f"https://arxiv.org/abs/{arxiv_id}"
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            candidate = PaperCandidate(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_year=pub_year,
                publication_date=pub_date,
                venue=venue,
                doi=doi,
                arxiv_id=arxiv_id,
                source_id=versioned_id,
                source=self.source_name,
                source_url=source_url,
                pdf_url=pdf_url,
                open_access=True,
                domain=domain,
                subtopic=subtopic,
                search_query=search_query,
            )
            candidates.append(candidate)

        return candidates
