"""
Refined Ground Truth Builder for Phase 5 Retrieval Evaluation

Establishes independent ground truth mappings for all 50 evaluation queries
by analyzing corpus text, subtopics, and section content across 13,718 units.

Refinement Objective:
Focus ground truth on core substantive evidence passages (~10–25 relevant units per query),
eliminating broad peripheral noise that previously inflated ground-truth size to ~600 units per query.

Methodology:
1. Matches query domain, subtopics, and key conceptual terms against corpus unit metadata & text.
2. Excludes non-evidence section types (Header, References, Bibliography, Acknowledgements).
3. Requires multi-term semantic matching and subtopic alignment:
   - 3: Highly relevant (high semantic overlap >= 4 key terms, direct answer context)
   - 2: Relevant (strong subtopic match >= 3 key terms)
   - 1: Weakly relevant (clear keyword match >= 2 key terms)
4. Caps relevant evidence candidates per query to ~15-25 top substantive passages.
5. Retains complete provenance for every entry.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NON_EVIDENCE_SECTIONS = {"header", "references", "bibliography", "acknowledgements"}

# Key query-specific concepts to ensure precise ground-truth matching
QUERY_CONCEPTS = {
    "AI_Q001": ["retrieval", "augmented", "generation", "hallucination", "external", "knowledge"],
    "AI_Q002": ["rag", "dense", "vector", "search", "bm25", "hybrid", "retrieval"],
    "AI_Q003": ["large", "language", "model", "parameter", "scaling", "transformer"],
    "AI_Q004": ["hallucination", "mitigation", "llm", "factuality", "grounding"],
    "AI_Q005": ["multi-agent", "agent", "collaboration", "coordination", "swarm"],
    "AI_Q006": ["agent", "communication", "protocol", "negotiation", "task"],
    "AI_Q007": ["computer", "vision", "object", "detection", "segmentation", "cnn"],
    "AI_Q008": ["vision", "transformer", "attention", "image", "classification"],
    "AI_Q009": ["reinforcement", "learning", "reward", "policy", "q-learning", "agent"],
    "AI_Q10":  ["actor-critic", "ppo", "policy", "gradient", "rl"],
    "CY_Q001": ["network", "intrusion", "detection", "traffic", "anomaly", "packet"],
    "CY_Q002": ["zero-day", "exploit", "vulnerability", "patch", "defense"],
    "CY_Q003": ["threat", "intelligence", "ioc", "indicator", "compromise", "cyber"],
    "CY_Q004": ["threat", "hunting", "log", "siem", "forensics", "mitre"],
    "CY_Q005": ["zero", "trust", "architecture", "microsegmentation", "identity", "access"],
    "CY_Q006": ["iam", "authentication", "mfa", "rbac", "least", "privilege"],
    "CY_Q007": ["vulnerability", "scanner", "cve", "cvss", "remediation"],
    "CY_Q008": ["pen-testing", "penetration", "testing", "red", "team", "exploitation"],
    "CY_Q009": ["post-quantum", "cryptography", "lattice", "encryption", "rsa"],
    "CY_Q010": ["zero-knowledge", "proof", "zk-snark", "cryptographic", "privacy"],
    "AG_Q001": ["iot", "sensor", "smart", "farming", "monitoring", "telemetry"],
    "AG_Q002": ["drone", "uav", "remote", "sensing", "aerial", "imagery"],
    "AG_Q003": ["crop", "yield", "prediction", "machine", "learning", "harvest"],
    "AG_Q004": ["plant", "disease", "detection", "leaf", "cnn", "pathogen"],
    "AG_Q005": ["precision", "irrigation", "water", "soil", "moisture", "drip"],
    "AG_Q006": ["variable", "rate", "fertilizer", "application", "nitrogen"],
    "AG_Q007": ["soil", "organic", "matter", "carbon", "npk", "fertility"],
    "AG_Q008": ["soil", "erosion", "conservation", "tillage", "cover", "crop"],
    "AG_Q009": ["agroforestry", "biodiversity", "sustainable", "farming", "ecology"],
    "AG_Q010": ["climate-resilient", "crop", "drought", "salinity", "breeding"],
    "HC_Q001": ["mri", "ct", "scan", "medical", "imaging", "segmentation"],
    "HC_Q002": ["deep", "learning", "radiology", "x-ray", "tumor", "diagnosis"],
    "HC_Q003": ["clinical", "decision", "support", "ehr", "diagnosis", "recommendation"],
    "HC_Q004": ["alert", "fatigue", "clinical", "alarm", "ehr", "workflow"],
    "HC_Q005": ["genomic", "sequencing", "variant", "dna", "gene", "mutation"],
    "HC_Q006": ["crispr", "gene", "editing", "therapeutics", "off-target"],
    "HC_Q007": ["telemedicine", "remote", "patient", "monitoring", "wearable"],
    "HC_Q008": ["telehealth", "rural", "healthcare", "consultation", "access"],
    "HC_Q009": ["electronic", "health", "record", "interoperability", "fhir"],
    "HC_Q010": ["ehr", "privacy", "hipaa", "de-identification", "patient"],
    "CL_Q001": ["climate", "model", "simulation", "temperature", "forcing"],
    "CL_Q002": ["neural", "network", "climate", "downscaling", "weather"],
    "CL_Q003": ["heatwave", "hurricane", "flood", "extreme", "weather", "frequency"],
    "CL_Q004": ["tropical", "cyclone", "intensity", "sea", "surface", "temperature"],
    "CL_Q005": ["ocean", "carbon", "sink", "co2", "absorption", "flux"],
    "CL_Q006": ["deforestation", "forest", "carbon", "sequestration", "biomass"],
    "CL_Q007": ["ice", "sheet", "melting", "sea", "level", "rise", "glacier"],
    "CL_Q008": ["coastal", "inundation", "storm", "surge", "sea", "level"],
    "CL_Q009": ["solar", "wind", "grid", "renewable", "energy", "variability"],
    "CL_Q010": ["battery", "energy", "storage", "bess", "grid", "stability"],
}


def build_ground_truth():
    BASE_DIR = Path(__file__).resolve().parent.parent
    queries_path = BASE_DIR / "evaluation" / "test_queries.json"
    units_path = BASE_DIR / "data" / "indexes" / "chunk_embeddings_ids.json"
    out_path = BASE_DIR / "evaluation" / "ground_truth.json"

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    with open(units_path, "r", encoding="utf-8") as f:
        units = json.load(f)

    # Index units by domain and subtopic
    by_domain_subtopic: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for u in units:
        d = u.get("domain", "").lower()
        st = u.get("subtopic", "").lower()
        by_domain_subtopic.setdefault(d, {}).setdefault(st, []).append(u)

    ground_truth_map: Dict[str, List[Dict[str, Any]]] = {}
    total_entries = 0
    relevant_counts = []

    for q in queries:
        qid = q["query_id"]
        q_domain = q["domain"].lower()
        q_subtopic = q["subtopic"].lower()
        
        concepts = QUERY_CONCEPTS.get(qid, [w for w in q["question"].lower().replace("?", "").split() if len(w) > 3])

        candidate_units = by_domain_subtopic.get(q_domain, {}).get(q_subtopic, [])
        if not candidate_units:
            all_domain_units = []
            for st_list in by_domain_subtopic.get(q_domain, {}).values():
                all_domain_units.extend(st_list)
            candidate_units = all_domain_units

        unit_scores = []
        for u in candidate_units:
            sec_name = u.get("section_name", "").lower()
            if any(ex in sec_name for ex in NON_EVIDENCE_SECTIONS):
                continue

            u_text = u["text"].lower()
            match_count = sum(1 for kw in concepts if kw in u_text)

            if match_count >= 4:
                relevance = 3
            elif match_count == 3:
                relevance = 2
            elif match_count == 2:
                relevance = 1
            else:
                relevance = 0

            if relevance > 0:
                unit_scores.append((u, relevance, match_count))

        # Sort by relevance descending, then match_count descending
        unit_scores.sort(key=lambda x: (-x[1], -x[2]))

        # Limit to top 15-25 evidence passages per query to ensure high precision ground-truth
        selected = unit_scores[:22]

        entries = []
        for u, rel, _ in selected:
            unit_id = u.get("unit_id", u["chunk_id"])
            parent_chunk_id = u.get("parent_chunk_id", u["chunk_id"])
            entries.append({
                "query_id": qid,
                "unit_id": unit_id,
                "parent_chunk_id": parent_chunk_id,
                "chunk_id": u["chunk_id"],
                "paper_id": u["paper_id"],
                "section_id": u["section_id"],
                "page_start": u["page_start"],
                "page_end": u["page_end"],
                "relevance": rel,
            })

        ground_truth_map[qid] = entries
        total_entries += len(entries)
        relevant_counts.append(len(entries))
        logger.info(f"Query {qid} ({q_domain}/{q_subtopic}): {len(entries)} refined ground-truth entries established.")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth_map, f, indent=2)

    avg_cnt = sum(relevant_counts) / len(relevant_counts) if relevant_counts else 0
    logger.info(f"Refined ground truth saved to {out_path} ({total_entries} total mappings across {len(queries)} queries, avg {avg_cnt:.1f} units/query).")
    print(f"REFINED GROUND TRUTH BUILT: {total_entries} total mappings across 50 queries (Average: {avg_cnt:.1f} units/query).")


if __name__ == "__main__":
    build_ground_truth()
