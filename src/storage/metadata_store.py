from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Set
from src.config.loader import AppConfig, load_domain_config
from src.models.paper import PaperMetadata, DOMAIN_PREFIX_MAP
from src.models.collection_event import CollectionEvent

logger = logging.getLogger(__name__)


class MetadataStore:
    """Atomic store and index manager for Research Mind metadata files."""

    def __init__(
        self,
        papers_json_path: Optional[Path] = None,
        log_json_path: Optional[Path] = None,
        config: Optional[AppConfig] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.papers_json_path = papers_json_path or (base_dir / "data" / "metadata" / "papers.json")
        self.log_json_path = log_json_path or (base_dir / "data" / "metadata" / "collection_log.json")
        self.config = config or load_domain_config()

        # In-memory lookup indices
        self.accepted_papers: List[PaperMetadata] = []
        self.dois: Dict[str, str] = {}                 # norm_doi -> paper_id
        self.arxiv_ids: Dict[str, str] = {}             # arxiv_id -> paper_id
        self.source_ids: Dict[str, str] = {}            # source_id -> paper_id
        self.normalized_titles: Dict[str, str] = {}    # norm_title -> paper_id
        self.file_hashes: Dict[str, str] = {}          # file_hash -> paper_id

        self.reload()

    def reload(self) -> None:
        """Reload accepted papers from disk and build in-memory lookup sets."""
        self.accepted_papers.clear()
        self.dois.clear()
        self.arxiv_ids.clear()
        self.source_ids.clear()
        self.normalized_titles.clear()
        self.file_hashes.clear()

        if not self.papers_json_path.exists():
            return

        try:
            with open(self.papers_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                paper = PaperMetadata.model_validate(item)
                self._index_paper(paper)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load papers.json from {self.papers_json_path}: {e}")

    def _index_paper(self, paper: PaperMetadata) -> None:
        """Add paper metadata to in-memory lookup indices."""
        self.accepted_papers.append(paper)
        pid = paper.paper_id

        if paper.doi:
            self.dois[paper.doi.strip().lower()] = pid

        if paper.arxiv_id:
            self.arxiv_ids[paper.arxiv_id.strip()] = pid

        if paper.source_id:
            self.source_ids[paper.source_id.strip()] = pid

        # Title indexing using normalized string
        from src.utils.text import normalize_title
        norm_t = normalize_title(paper.title)
        if norm_t:
            self.normalized_titles[norm_t] = pid

        if paper.file_hash:
            self.file_hashes[paper.file_hash.strip().lower()] = pid

    def generate_next_paper_id(self, domain_name: str) -> str:
        """
        Generate next stable Paper ID for domain (e.g., AI001, AI002).
        Inspects existing records to prevent ID collision.
        """
        # Determine prefix
        prefix = DOMAIN_PREFIX_MAP.get(domain_name)
        if not prefix and domain_name in self.config.domains:
            prefix = self.config.domains[domain_name].id_prefix

        if not prefix:
            raise ValueError(f"No ID prefix configured for domain '{domain_name}'")

        pattern = rf"^{prefix}(\d+)$"
        max_num = 0

        for paper in self.accepted_papers:
            if paper.domain == domain_name or paper.paper_id.startswith(prefix):
                match = re.match(pattern, paper.paper_id)
                if match:
                    num = int(match.group(1))
                    if num > max_num:
                        max_num = num

        next_num = max_num + 1
        return f"{prefix}{next_num:03d}"

    def add_accepted_paper(self, paper: PaperMetadata) -> None:
        """
        Add new accepted paper record and atomically save papers.json.
        """
        # Check for paper_id conflict
        if any(p.paper_id == paper.paper_id for p in self.accepted_papers):
            raise ValueError(f"Paper ID '{paper.paper_id}' already exists in metadata store")

        self._index_paper(paper)
        self.save_papers_atomic()

    def save_papers_atomic(self) -> None:
        """Safely write papers.json using atomic temp file replacement."""
        self.papers_json_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.papers_json_path.with_suffix(".json.tmp")

        data = [p.model_dump() for p in self.accepted_papers]

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Atomic replace
        temp_path.replace(self.papers_json_path)

    def log_event(self, event: CollectionEvent) -> None:
        """Append audit event to collection_log.json."""
        self.log_json_path.parent.mkdir(parents=True, exist_ok=True)
        existing_log = []
        if self.log_json_path.exists():
            try:
                with open(self.log_json_path, "r", encoding="utf-8") as f:
                    existing_log = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing_log = []

        existing_log.append(event.model_dump())

        temp_log_path = self.log_json_path.with_suffix(".json.tmp")
        with open(temp_log_path, "w", encoding="utf-8") as f:
            json.dump(existing_log, f, indent=2, ensure_ascii=False)

        temp_log_path.replace(self.log_json_path)
