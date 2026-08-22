import logging
from pathlib import Path
import time
from uuid import uuid4
import requests

logger = logging.getLogger(__name__)


class PdfDownloader:
    """Robust streaming PDF downloader with temporary storage and rate-limiting support."""

    def __init__(self, timeout: int = 30, max_retries: int = 3, backoff_factor: float = 2.0):
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()

    def download_temp_pdf(self, pdf_url: str, temp_dir: Path) -> Path:
        """
        Download PDF from open-access URL into a temporary file.

        Returns path to the downloaded temporary file.
        Cleans up temporary file if download fails.
        """
        if not pdf_url or not pdf_url.strip():
            raise ValueError("Empty or invalid PDF URL")

        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file_path = temp_dir / f"download_{uuid4().hex}.pdf.tmp"

        headers = {
            "User-Agent": "ResearchMind/0.1.0 (Scientific Literature RAG System; open-access collector)",
            "Accept": "application/pdf,application/octet-stream,*/*",
        }

        delay = 1.0
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(pdf_url, headers=headers, stream=True, timeout=self.timeout)
                if response.status_code == 429:
                    logger.warning(f"Rate limited (429) downloading {pdf_url}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    continue

                response.raise_for_status()

                # Stream response bytes to temporary file
                with open(temp_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                return temp_file_path

            except Exception as e:
                last_exception = e
                logger.warning(f"Download attempt {attempt}/{self.max_retries} failed for {pdf_url}: {e}")
                self.cleanup_temp_file(temp_file_path)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= self.backoff_factor

        self.cleanup_temp_file(temp_file_path)
        raise RuntimeError(f"Failed to download PDF from {pdf_url} after {self.max_retries} attempts: {last_exception}")

    @staticmethod
    def cleanup_temp_file(file_path: Path) -> None:
        """Safely remove a temporary file if it exists."""
        try:
            if file_path and file_path.exists():
                file_path.unlink()
        except OSError as e:
            logger.warning(f"Failed to delete temp file {file_path}: {e}")
