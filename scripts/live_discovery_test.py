import json
from src.connectors.arxiv import ArxivConnector

def main():
    connector = ArxivConnector()
    print("Executing small live discovery test against arXiv API...")
    candidates = connector.search_candidates(
        query="retrieval-augmented generation RAG",
        domain="artificial_intelligence",
        subtopic="retrieval_augmented_generation",
        max_results=3
    )

    print(f"\nRetrieved {len(candidates)} candidates from arXiv:\n")
    for idx, cand in enumerate(candidates, 1):
        print(f"--- Candidate #{idx} ---")
        print(f"Title:            {cand.title}")
        print(f"Authors:          {', '.join(cand.authors[:3])}{' et al.' if len(cand.authors) > 3 else ''}")
        print(f"Publication Year: {cand.publication_year}")
        print(f"arXiv ID:         {cand.arxiv_id}")
        print(f"DOI:              {cand.doi or 'N/A'}")
        print(f"Source URL:       {cand.source_url}")
        print(f"PDF URL:          {cand.pdf_url}")
        print(f"Domain:           {cand.domain}")
        print(f"Subtopic:         {cand.subtopic}")
        print()

if __name__ == "__main__":
    main()
