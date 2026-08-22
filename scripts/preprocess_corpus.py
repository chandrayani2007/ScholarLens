import json
import logging
from pathlib import Path
from collections import Counter
from src.pipeline.preprocessing import PDFPreprocessingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    PAPERS_PATH = BASE_DIR / "data" / "metadata" / "papers.json"
    PROCESSED_DIR = BASE_DIR / "data" / "metadata" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SECTIONS_PATH = PROCESSED_DIR / "sections.jsonl"
    
    with open(PAPERS_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    pipeline = PDFPreprocessingPipeline(output_dir=PROCESSED_DIR)
    
    total_papers = len(papers)
    successful_papers = 0
    failed_papers = []
    
    total_sections = 0
    total_pages = 0
    total_text_len = 0
    
    section_name_counts = Counter()
    scanned_detections = 0
    
    # Open the output file for writing JSONL records
    with open(SECTIONS_PATH, "w", encoding="utf-8") as out_f:
        for idx, paper in enumerate(papers, 1):
            paper_id = paper["paper_id"]
            print(f"[{idx}/{total_papers}] Preprocessing {paper_id}: {paper['title'][:60]}...")
            
            try:
                sections = pipeline.segment_paper(paper)
                
                # Check text extraction density
                paper_text_len = sum(len(s["text"]) for s in sections)
                # Find max page end to estimate total pages
                paper_pages = max(s["page_end"] for s in sections) if sections else 0
                
                # Scanned / Image-only detection: average characters per page < 150
                avg_chars_per_page = (paper_text_len / paper_pages) if paper_pages > 0 else 0
                if paper_pages > 0 and avg_chars_per_page < 150:
                    scanned_detections += 1
                    logger.warning(f"Potential scanned/image-only PDF detected for {paper_id} (avg chars/page: {avg_chars_per_page:.1f})")
                
                for sec in sections:
                    out_f.write(json.dumps(sec, ensure_ascii=False) + "\n")
                    section_name_counts[sec["section_name"]] += 1
                    total_sections += 1
                    total_text_len += len(sec["text"])
                    
                total_pages += paper_pages
                successful_papers += 1
                
            except Exception as e:
                logger.error(f"Failed to preprocess {paper_id}: {str(e)}")
                failed_papers.append({
                    "paper_id": paper_id,
                    "title": paper["title"],
                    "error": str(e)
                })
                
    # Compile statistics
    avg_sections = total_sections / successful_papers if successful_papers > 0 else 0
    avg_pages = total_pages / successful_papers if successful_papers > 0 else 0
    avg_text_len = total_text_len / successful_papers if successful_papers > 0 else 0
    
    stats = {
        "total_papers_processed": total_papers,
        "successful_papers": successful_papers,
        "failed_papers_count": len(failed_papers),
        "failed_papers": failed_papers,
        "average_pages": round(avg_pages, 1),
        "average_sections": round(avg_sections, 1),
        "average_text_length": round(avg_text_len, 1),
        "total_sections_extracted": total_sections,
        "scanned_pdf_count": scanned_detections,
        "section_distribution": dict(section_name_counts.most_common(20))
    }
    
    # Save statistics file
    with open(BASE_DIR / "data" / "metadata" / "preprocessing_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    print("\n=== Preprocessing Execution Statistics ===")
    print(f"Total Papers Processed: {total_papers}")
    print(f"Success Count: {successful_papers}")
    print(f"Failure Count: {len(failed_papers)}")
    print(f"Average Pages: {avg_pages:.1f}")
    print(f"Average Sections: {avg_sections:.1f}")
    print(f"Average Text Length (chars): {avg_text_len:.1f}")
    print(f"Scanned / Low-text Detections: {scanned_detections}")
    print(f"Top 10 Sections: {section_name_counts.most_common(10)}")

if __name__ == "__main__":
    main()
