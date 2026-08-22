from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional
from src.config.loader import AppConfig, DomainConfig, load_domain_config
from src.connectors.base import BaseConnector
from src.connectors.arxiv import ArxivConnector
from src.connectors.pubmed import EuropePmcConnector
from src.connectors.semantic_scholar import SemanticScholarConnector
from src.models.candidate import PaperCandidate
from src.models.collection_event import CollectionEvent, EventAction


class CandidateDiscoveryService:
    """Service for querying scholarly APIs and logging discovered paper candidates."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        log_path: Optional[Path] = None,
        connectors: Optional[Dict[str, BaseConnector]] = None,
    ):
        self.config = config or load_domain_config()

        base_dir = Path(__file__).resolve().parent.parent.parent
        self.log_path = log_path or (base_dir / "data" / "metadata" / "collection_log.json")

        # Default source connector map if not provided
        self.connectors = connectors or {
            "arxiv": ArxivConnector(),
            "europe_pmc": EuropePmcConnector(),
            "semantic_scholar": SemanticScholarConnector(),
        }

    def discover_candidates_for_subtopic(
        self,
        domain_name: str,
        subtopic_canonical: str,
        source_names: List[str],
        max_results_per_query: int = 5,
        log_events: bool = True,
    ) -> List[PaperCandidate]:
        """
        Discover candidates for a given domain and subtopic across specified connectors.
        """
        if domain_name not in self.config.domains:
            raise ValueError(f"Unknown domain '{domain_name}'")

        domain_cfg = self.config.domains[domain_name]
        subtopic_cfg = next((s for s in domain_cfg.subtopics if s.canonical == subtopic_canonical), None)
        if not subtopic_cfg:
            raise ValueError(f"Subtopic '{subtopic_canonical}' not found in domain '{domain_name}'")

        discovered_candidates: List[PaperCandidate] = []

        for query in subtopic_cfg.search_queries:
            for s_name in source_names:
                connector = self.connectors.get(s_name)
                if not connector:
                    continue

                candidates = connector.search_candidates(
                    query=query,
                    domain=domain_name,
                    subtopic=subtopic_canonical,
                    max_results=max_results_per_query,
                )
                discovered_candidates.extend(candidates)

                if log_events:
                    self._log_discovered_events(candidates)

        return discovered_candidates

    def _log_discovered_events(self, candidates: List[PaperCandidate]) -> None:
        """Write candidate_discovered audit events to collection_log.json."""
        if not candidates:
            return

        events = []
        now_str = datetime.now(timezone.utc).isoformat()

        for cand in candidates:
            event = CollectionEvent(
                timestamp=now_str,
                domain=cand.domain,
                subtopic=cand.subtopic,
                source=cand.source,
                action=EventAction.CANDIDATE_DISCOVERED,
                source_id=cand.source_id,
                doi=cand.doi,
                title=cand.title,
            )
            events.append(event.model_dump())

        # Append to log file
        existing_log = []
        if self.log_path.exists():
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    existing_log = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_log = []

        existing_log.extend(events)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(existing_log, f, indent=2)
