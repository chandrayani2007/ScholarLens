from unittest.mock import MagicMock, patch
import pytest
import requests
from src.downloader.pdf_downloader import PdfDownloader


@patch("src.downloader.pdf_downloader.requests.Session.get")
def test_successful_pdf_download(mock_get, tmp_path):
    """Test successful streaming PDF download into temporary file."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"%PDF-1.4 sample content bytes"]
    mock_get.return_value = mock_response

    downloader = PdfDownloader()
    temp_file = downloader.download_temp_pdf("https://arxiv.org/pdf/2005.11401.pdf", tmp_path)

    assert temp_file.exists()
    assert temp_file.suffix == ".tmp"
    assert temp_file.read_bytes() == b"%PDF-1.4 sample content bytes"


@patch("src.downloader.pdf_downloader.requests.Session.get")
def test_failed_pdf_download_cleans_up_temp_file(mock_get, tmp_path):
    """Test that download failure raises RuntimeError and cleans up temporary file."""
    mock_get.side_effect = requests.RequestException("Connection error")

    downloader = PdfDownloader(max_retries=2, backoff_factor=0.01)

    with pytest.raises(RuntimeError) as exc_info:
        downloader.download_temp_pdf("https://invalid.url/paper.pdf", tmp_path)

    assert "Failed to download PDF" in str(exc_info.value)
    # Ensure no leftover .tmp files exist in temp_path
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0
