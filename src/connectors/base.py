from abc import ABC, abstractmethod
import logging
import time
from typing import List, Optional, Dict, Any
import requests
from src.models.candidate import PaperCandidate

logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """Abstract Base Class for scholarly API source connectors."""

    def __init__(self, timeout: int = 30, max_retries: int = 4, backoff_factor: float = 2.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Name of the scholarly source (e.g. arXiv, Europe PMC, Semantic Scholar)."""
        pass

    @abstractmethod
    def search_candidates(
        self,
        query: str,
        domain: str,
        subtopic: str,
        max_results: int = 10
    ) -> List[PaperCandidate]:
        """
        Search source API for candidate papers.

        Returns list of normalized PaperCandidate objects.
        """
        pass

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> requests.Response:
        """
        Execute GET request with retry backoff and timeout handling.
        """
        headers = headers or {}
        headers.setdefault("User-Agent", "ResearchMind/0.1.0 (Scientific Literature RAG System)")

        delay = 3.0
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                if response.status_code == 429:
                    logger.warning(f"Rate limited by {self.source_name} (HTTP 429). Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    continue

                response.raise_for_status()
                time.sleep(3) # Polite delay for all scholarly APIs
                return response
            except requests.RequestException as e:
                last_exception = e
                logger.warning(
                    f"[{self.source_name}] Request attempt {attempt}/{self.max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= self.backoff_factor

        raise RuntimeError(
            f"[{self.source_name}] API request failed after {self.max_retries} attempts: {last_exception}"
        )
