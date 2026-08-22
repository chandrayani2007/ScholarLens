from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
from typing import Optional, Tuple
from src.downloader.pdf_downloader import PdfDownloader
from src.models.candidate import PaperCandidate
from src.models.paper import PaperMetadata
from src.models.collection_event import CollectionEvent, EventAction
from src.storage.metadata_store import MetadataStore
from src.utils.dedup import evaluate_candidate_deduplication, DeduplicationSignalType
from src.validator.pdf_validator import PdfValidator

logger = logging.getLogger(__name__)


class PaperIngestionPipeline:
    """End-to-end paper acquisition, deduplication, validation, and metadata persistence orchestrator."""

    def __init__(
        self,
        metadata_store: Optional[MetadataStore] = None,
        downloader: Optional[PdfDownloader] = None,
        validator: Optional[PdfValidator] = None,
        base_dir: Optional[Path] = None,
    ):
        self.metadata_store = metadata_store or MetadataStore()
        self.downloader = downloader or PdfDownloader()
        self.validator = validator or PdfValidator()

        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent
        self.papers_dir = self.base_dir / "data" / "papers"
        self.temp_dir = self.base_dir / "data" / "temp"

    def process_candidate(
        self,
        candidate: PaperCandidate
    ) -> Tuple[bool, Optional[PaperMetadata], str]:
        """
        Process a single PaperCandidate through full ingestion workflow.

        Returns (accepted: bool, metadata: Optional[PaperMetadata], message: str).
        """
        now_str = datetime.now(timezone.utc).isoformat()

        # Step 1: Check Open Access PDF URL
        if not candidate.pdf_url or not candidate.pdf_url.strip():
            reason = "No open access PDF URL available"
            self.metadata_store.log_event(
                CollectionEvent(
                    timestamp=now_str,
                    domain=candidate.domain,
                    subtopic=candidate.subtopic,
                    source=candidate.source,
                    action=EventAction.REJECTED,
                    source_id=candidate.source_id,
                    doi=candidate.doi,
                    title=candidate.title,
                    reason=reason,
                )
            )
            return False, None, reason

        # Step 2: Pre-download Deduplication Check
        dedup_res = evaluate_candidate_deduplication(
            candidate_title=candidate.title,
            candidate_doi=candidate.doi,
            candidate_source_id=candidate.source_id or candidate.arxiv_id,
            existing_dois=self.metadata_store.dois,
            existing_source_ids=self.metadata_store.source_ids,
            existing_normalized_titles=self.metadata_store.normalized_titles,
            existing_file_hashes=self.metadata_store.file_hashes,
        )

        if dedup_res.is_duplicate:
            reason = dedup_res.reason
            self.metadata_store.log_event(
                CollectionEvent(
                    timestamp=now_str,
                    domain=candidate.domain,
                    subtopic=candidate.subtopic,
                    source=candidate.source,
                    action=EventAction.DUPLICATE_DETECTED,
                    source_id=candidate.source_id,
                    doi=candidate.doi,
                    title=candidate.title,
                    reason=reason,
                    paper_id=dedup_res.matching_paper_id,
                )
            )
            return False, None, f"Duplicate detected: {reason}"

        # Log download started
        self.metadata_store.log_event(
            CollectionEvent(
                timestamp=now_str,
                domain=candidate.domain,
                subtopic=candidate.subtopic,
                source=candidate.source,
                action=EventAction.DOWNLOAD_STARTED,
                source_id=candidate.source_id,
                doi=candidate.doi,
                title=candidate.title,
            )
        )

        # Step 3: Stream Download to Temp Location
        try:
            temp_pdf_path = self.downloader.download_temp_pdf(candidate.pdf_url, self.temp_dir)
        except Exception as e:
            reason = f"Download failed: {e}"
            self.metadata_store.log_event(
                CollectionEvent(
                    timestamp=now_str,
                    domain=candidate.domain,
                    subtopic=candidate.subtopic,
                    source=candidate.source,
                    action=EventAction.DOWNLOAD_FAILED,
                    source_id=candidate.source_id,
                    doi=candidate.doi,
                    title=candidate.title,
                    reason=reason,
                )
            )
            return False, None, reason

        self.metadata_store.log_event(
            CollectionEvent(
                timestamp=now_str,
                domain=candidate.domain,
                subtopic=candidate.subtopic,
                source=candidate.source,
                action=EventAction.DOWNLOAD_SUCCESS,
                source_id=candidate.source_id,
                doi=candidate.doi,
                title=candidate.title,
            )
        )

        # Step 4: PDF Validation
        val_result = self.validator.validate(temp_pdf_path)
        if not val_result.is_valid:
            self.downloader.cleanup_temp_file(temp_pdf_path)
            reason = f"Validation failed: {val_result.error_reason}"
            self.metadata_store.log_event(
                CollectionEvent(
                    timestamp=now_str,
                    domain=candidate.domain,
                    subtopic=candidate.subtopic,
                    source=candidate.source,
                    action=EventAction.VALIDATION_FAILED,
                    source_id=candidate.source_id,
                    doi=candidate.doi,
                    title=candidate.title,
                    reason=reason,
                )
            )
            return False, None, reason

        # Step 5: SHA-256 Calculation & Post-Download Hash Deduplication
        file_hash = self.validator.compute_sha256(temp_pdf_path)
        if file_hash in self.metadata_store.file_hashes:
            matching_id = self.metadata_store.file_hashes[file_hash]
            self.downloader.cleanup_temp_file(temp_pdf_path)
            reason = f"Identical file hash match with {matching_id}"
            self.metadata_store.log_event(
                CollectionEvent(
                    timestamp=now_str,
                    domain=candidate.domain,
                    subtopic=candidate.subtopic,
                    source=candidate.source,
                    action=EventAction.DUPLICATE_DETECTED,
                    source_id=candidate.source_id,
                    doi=candidate.doi,
                    title=candidate.title,
                    reason=reason,
                    paper_id=matching_id,
                )
            )
            return False, None, f"Duplicate detected: {reason}"

        # Step 6: Generate Stable Paper ID
        paper_id = self.metadata_store.generate_next_paper_id(candidate.domain)

        # Step 7: Relocate PDF to Final Destination
        domain_pdf_dir = self.papers_dir / candidate.domain
        domain_pdf_dir.mkdir(parents=True, exist_ok=True)
        final_pdf_path = domain_pdf_dir / f"{paper_id}.pdf"

        shutil.move(str(temp_pdf_path), str(final_pdf_path))

        # Relative path for metadata
        relative_path = f"data/papers/{candidate.domain}/{paper_id}.pdf"

        # Step 8: Build PaperMetadata object
        pub_year = candidate.publication_year or datetime.now(timezone.utc).year

        paper_metadata = PaperMetadata(
            paper_id=paper_id,
            domain=candidate.domain,
            subtopic=candidate.subtopic,
            title=candidate.title,
            authors=candidate.authors if candidate.authors else ["Unknown Author"],
            publication_year=pub_year,
            publication_date=candidate.publication_date,
            venue=candidate.venue,
            doi=candidate.doi,
            arxiv_id=candidate.arxiv_id,
            source_id=candidate.source_id,
            source=candidate.source,
            source_url=candidate.source_url,
            pdf_url=candidate.pdf_url,
            license=candidate.license,
            open_access=candidate.open_access if candidate.open_access is not None else True,
            abstract=candidate.abstract,
            local_path=relative_path,
            file_hash=file_hash,
            collection_date=now_str,
            status="downloaded",
        )

        # Step 9: Save PaperMetadata to papers.json & log ACCEPTED event
        self.metadata_store.add_accepted_paper(paper_metadata)

        self.metadata_store.log_event(
            CollectionEvent(
                timestamp=now_str,
                domain=candidate.domain,
                subtopic=candidate.subtopic,
                source=candidate.source,
                action=EventAction.ACCEPTED,
                source_id=candidate.source_id,
                doi=candidate.doi,
                title=candidate.title,
                paper_id=paper_id,
            )
        )

        logger.info(f"Successfully accepted candidate '{candidate.title}' as {paper_id}")
        return True, paper_metadata, f"Accepted as {paper_id}"
