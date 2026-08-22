from pathlib import Path
import fitz
import pytest
from src.validator.pdf_validator import PdfValidator, ValidationResult


def create_sample_pdf(file_path: Path, text: str = "Sample extractable scientific research text for testing PDF validation."):
    """Helper function to create a valid minimal PDF file using PyMuPDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(str(file_path))
    doc.close()


def test_valid_pdf_validation(tmp_path):
    """Test PdfValidator on a valid generated PDF with sufficient text."""
    pdf_path = tmp_path / "valid.pdf"
    sample_text = "This is a comprehensive scientific research paper text containing over 100 characters of extractable text for testing purposes. " * 3
    create_sample_pdf(pdf_path, text=sample_text)

    validator = PdfValidator(min_text_length=50)
    res = validator.validate(pdf_path)

    assert res.is_valid is True
    assert res.page_count == 1
    assert res.text_length >= 50
    assert res.file_size_bytes > 0
    assert res.error_reason is None


def test_empty_file_validation(tmp_path):
    """Test PdfValidator on an empty 0-byte file."""
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.touch()

    validator = PdfValidator()
    res = validator.validate(pdf_path)

    assert res.is_valid is False
    assert "File is empty" in res.error_reason


def test_missing_pdf_header_validation(tmp_path):
    """Test PdfValidator on a non-PDF file (e.g. HTML error page saved as .pdf)."""
    pdf_path = tmp_path / "html_error.pdf"
    pdf_path.write_text("<html><body>404 Not Found</body></html>", encoding="utf-8")

    validator = PdfValidator()
    res = validator.validate(pdf_path)

    assert res.is_valid is False
    assert "missing '%PDF-' header" in res.error_reason


def test_insufficient_text_pdf_validation(tmp_path):
    """Test PdfValidator on a PDF with insufficient extractable text."""
    pdf_path = tmp_path / "short_text.pdf"
    create_sample_pdf(pdf_path, text="Short text")

    validator = PdfValidator(min_text_length=100)
    res = validator.validate(pdf_path)

    assert res.is_valid is False
    assert "Insufficient extractable text" in res.error_reason


def test_sha256_hash_computation(tmp_path):
    """Test SHA-256 hash calculation helper."""
    test_file = tmp_path / "test.txt"
    test_file.write_bytes(b"ResearchMind SHA256 Test Content")

    hash_str = PdfValidator.compute_sha256(test_file)
    assert isinstance(hash_str, str)
    assert len(hash_str) == 64
