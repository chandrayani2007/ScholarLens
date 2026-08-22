import os
from typing import List, Optional, Dict, Any
from src.connectors.base import BaseConnector
from src.models.candidate import PaperCandidate


class SemanticScholarConnector(BaseConnector):
    """Connector for querying the Semantic Scholar Graph API (https://api.semanticscholar.org/graph/v1/paper/search)."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    @property
    def source_name(self) -> str:
        return "Semantic Scholar"

    def search_candidates(
        self,
        query: str,
        domain: str,
        subtopic: str,
        max_results: int = 10
    ) -> List[PaperCandidate]:
        clean_query = query.strip()
        params = {
            "query": clean_query,
            "limit": max_results,
            "fields": "title,authors,abstract,year,publicationDate,venue,externalIds,url,openAccessPdf,isOpenAccess",
        }

        headers = {}
        api_key = os.getenv("S2_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        response = self._make_request(self.BASE_URL, params=params, headers=headers)
        data = response.json()
        return self.parse_json_response(data, domain=domain, subtopic=subtopic, search_query=query)

    def parse_json_response(
        self,
        data: Dict[str, Any],
        domain: str,
        subtopic: str,
        search_query: str
    ) -> List[PaperCandidate]:
        candidates: List[PaperCandidate] = []
        papers = data.get("data", [])

        for paper in papers:
            title = paper.get("title", "").strip()
            if not title:
                continue

            # Authors
            authors = []
            for auth in paper.get("authors", []):
                name = auth.get("name")
                if name:
                    authors.append(name.strip())

            # Abstract
            abstract = paper.get("abstract", "").strip() or None

            # Year / Date
            pub_year = paper.get("year")
            pub_date = paper.get("publicationDate") or None

            # Venue
            venue = paper.get("venue") or None

            # External IDs
            ext_ids = paper.get("externalIds") or {}
            doi = ext_ids.get("DOI") or None
            arxiv_id = ext_ids.get("ArXiv") or None
            source_id = paper.get("paperId") or None

            # Source URL
            source_url = paper.get("url") or None

            # PDF URL and open access status
            open_access_pdf_info = paper.get("openAccessPdf") or {}
            pdf_url = open_access_pdf_info.get("url") or None
            is_open_access = paper.get("isOpenAccess", False) or (pdf_url is not None)

            candidate = PaperCandidate(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_year=pub_year,
                publication_date=pub_date,
                venue=venue,
                doi=doi,
                arxiv_id=arxiv_id,
                source_id=source_id,
                source=self.source_name,
                source_url=source_url,
                pdf_url=pdf_url,
                open_access=is_open_access,
                domain=domain,
                subtopic=subtopic,
                search_query=search_query,
            )
            candidates.append(candidate)

        return candidates
