import hashlib
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Structured validation output for a downloaded PDF file."""
    is_valid: bool = Field(..., description="True if PDF satisfies all validation criteria")
    error_reason: Optional[str] = Field(default=None, description="Detailed reason if validation failed")
    page_count: int = Field(default=0, description="Total number of pages in PDF")
    text_length: int = Field(default=0, description="Total extractable text length in characters")
    file_size_bytes: int = Field(default=0, description="File size in bytes")


class PdfValidator:
    """PDF file validator using PyMuPDF to enforce research PDF requirements."""

    def __init__(self, min_text_length: int = 100):
        self.min_text_length = min_text_length

    def validate(self, pdf_path: Path) -> ValidationResult:
        """
        Validate PDF file integrity, structure, page count, and extractable text.
        """
        if not pdf_path.exists():
            return ValidationResult(
                is_valid=False,
                error_reason="File does not exist",
            )

        file_size = pdf_path.stat().st_size
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                error_reason="File is empty (0 bytes)",
                file_size_bytes=0,
            )

        # Check PDF Magic Header (%PDF-)
        try:
            with open(pdf_path, "rb") as f:
                header = f.read(1024)
                if b"%PDF-" not in header:
                    return ValidationResult(
                        is_valid=False,
                        error_reason="Invalid PDF signature: missing '%PDF-' header",
                        file_size_bytes=file_size,
                    )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_reason=f"Failed to read file header: {e}",
                file_size_bytes=file_size,
            )

        # Open and inspect with PyMuPDF
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_reason=f"Corrupted PDF (failed to parse): {e}",
                file_size_bytes=file_size,
            )

        try:
            page_count = doc.page_count
            if page_count == 0:
                doc.close()
                return ValidationResult(
                    is_valid=False,
                    error_reason="PDF contains 0 pages",
                    file_size_bytes=file_size,
                    page_count=0,
                )

            # Extract text across pages
            total_text = ""
            for page in doc:
                text = page.get_text()
                if text:
                    total_text += text.strip() + " "

            doc.close()
            text_len = len(total_text.strip())

            if text_len < self.min_text_length:
                return ValidationResult(
                    is_valid=False,
                    error_reason=f"Insufficient extractable text ({text_len} chars < {self.min_text_length} required)",
                    file_size_bytes=file_size,
                    page_count=page_count,
                    text_length=text_len,
                )

            return ValidationResult(
                is_valid=True,
                file_size_bytes=file_size,
                page_count=page_count,
                text_length=text_len,
            )

        except Exception as e:
            try:
                doc.close()
            except Exception:
                pass
            return ValidationResult(
                is_valid=False,
                error_reason=f"Error inspecting PDF contents: {e}",
                file_size_bytes=file_size,
            )

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
