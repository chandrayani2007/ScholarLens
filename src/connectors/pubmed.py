from typing import List, Optional, Dict, Any
from src.connectors.base import BaseConnector
from src.models.candidate import PaperCandidate


class EuropePmcConnector(BaseConnector):
    """Connector for querying the Europe PMC REST API (https://www.ebi.ac.uk/europepmc/webservices/rest/search)."""

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    @property
    def source_name(self) -> str:
        return "Europe PMC"

    def search_candidates(
        self,
        query: str,
        domain: str,
        subtopic: str,
        max_results: int = 10
    ) -> List[PaperCandidate]:
        clean_query = query.strip()
        # Query for open access papers
        formatted_query = f'"{clean_query}" OPEN_ACCESS:y'
        params = {
            "query": formatted_query,
            "format": "json",
            "pageSize": max_results,
            "resultType": "core",
        }

        response = self._make_request(self.BASE_URL, params=params)
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
        result_list = data.get("resultList", {}).get("result", [])

        for item in result_list:
            title = item.get("title", "").strip().rstrip(".")
            if not title:
                continue

            # Authors
            authors = []
            author_list = item.get("authorList", {}).get("author", [])
            for auth in author_list:
                name = auth.get("fullName") or f"{auth.get('lastName', '')} {auth.get('firstName', '')}".strip()
                if name:
                    authors.append(name)

            # Abstract
            abstract = item.get("abstractText", "").strip() or None

            # Publication year / date
            pub_year_str = item.get("pubYear")
            pub_year = None
            if pub_year_str:
                try:
                    pub_year = int(pub_year_str)
                except ValueError:
                    pub_year = None

            pub_date = item.get("firstPublicationDate") or None

            # Identifiers
            doi = item.get("doi") or None
            pmid = item.get("pmid") or None
            pmcid = item.get("pmcid") or None
            source_id = pmcid or pmid or item.get("id") or None

            # Venue
            journal_info = item.get("journalInfo", {})
            venue = journal_info.get("journal", {}).get("title") or item.get("journalTitle") or None

            # Source URL
            if pmcid:
                source_url = f"https://europepmc.org/article/PMC/{pmcid}"
            elif pmid:
                source_url = f"https://europepmc.org/article/MED/{pmid}"
            elif doi:
                source_url = f"https://doi.org/{doi}"
            else:
                source_url = None

            # PDF URL resolution from fullTextUrlList
            pdf_url = None
            full_text_urls = item.get("fullTextUrlList", {}).get("fullTextUrl", [])
            for ft in full_text_urls:
                if ft.get("documentStyle") == "pdf" and ft.get("url"):
                    pdf_url = ft.get("url")
                    break

            if not pdf_url and pmcid:
                # Construct official PMC open access renderer link
                pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"

            is_open_access = item.get("isOpenAccess") == "Y" or pdf_url is not None

            candidate = PaperCandidate(
                title=title,
                authors=authors,
                abstract=abstract,
                publication_year=pub_year,
                publication_date=pub_date,
                venue=venue,
                doi=doi,
                arxiv_id=None,
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
