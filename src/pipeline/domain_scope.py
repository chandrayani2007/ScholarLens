"""
Phase 27 Real-Source Provenance — Domain Scope Detector Module

Identifies whether a research query belongs to:
1. SINGLE_DOMAIN (Strict domain scope, e.g. cybersecurity)
2. MULTI_DOMAIN (Cross-domain query, e.g. AI + Healthcare)
3. ALL_DOMAINS (Broad or unconstrained search, or Out-of-Corpus Fallback)

Prevents domain leakage (e.g., retrieving cybersecurity papers for agriculture queries).
"""

import logging
import re
from enum import Enum
from typing import List, Set, Optional, Dict, Any
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ScopeType(str, Enum):
    SINGLE_DOMAIN = "SINGLE_DOMAIN"
    MULTI_DOMAIN = "MULTI_DOMAIN"
    ALL_DOMAINS = "ALL_DOMAINS"


DOMAIN_LABEL_MAP = {
    "artificial_intelligence": "Artificial Intelligence",
    "cybersecurity": "Cybersecurity",
    "agriculture": "Agriculture",
    "healthcare": "Healthcare",
    "climate": "Climate",
}

DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "artificial_intelligence": {
        "machine learning", "deep learning", "neural network", "transformer", "llm", "llms",
        "large language model", "large language models", "natural language processing", "nlp", "computer vision",
        "retrieval augmented generation", "rag", "ai agent", "explainable ai", "xai",
        "reinforcement learning", "generative ai", "diffusion model", "gan", "q-learning",
        "feature attribution", "self-attention", "lora", "fine-tuning", "vector retrieval",
        "agentic", "hallucination", "model drift", "supervised learning", "unsupervised learning",
        "federated learning", "robotic manipulation", "robotics", "graph neural network", "gnn"
    },
    "cybersecurity": {
        "cybersecurity", "network security", "intrusion detection", "intrusion prevention",
        "malware", "phishing", "ransomware", "authentication", "multi-factor authentication",
        "differential privacy", "homomorphic encryption", "cloud security", "container security",
        "iot security", "adversarial attack", "threat intelligence", "firewall", "zero trust",
        "zero-day", "credential stuffing", "sinkholing", "sql injection", "cyber attack", "c2",
        "security risks", "fraud detection", "fraud", "poisoning attack", "gradient leakage"
    },
    "agriculture": {
        "agriculture", "precision agriculture", "smart farming", "crop recommendation",
        "crop disease", "plant disease", "leaf disease", "soil analysis", "soil moisture",
        "soil nutrient", "smart irrigation", "yield prediction", "crop yield", "remote sensing",
        "ndvi", "digital agriculture", "greenhouse", "agricultural iot", "agricultural robotics",
        "farming", "crop", "orchard", "weed detection", "hydroponics", "aeroponics", "tillage"
    },
    "healthcare": {
        "healthcare", "medical image", "medical imaging", "disease prediction", "early disease",
        "clinical nlp", "clinical natural language processing", "healthcare ai", "drug discovery",
        "biomedical nlp", "clinical decision support", "cdss", "medical diagnosis",
        "electronic health records", "ehr", "explainable ai in healthcare", "radiology",
        "patient", "clinical", "tumor", "lesion", "sepsis", "diabetic retinopathy", "ecg", "mri", "ct scan"
    },
    "climate": {
        "climate", "climate change", "climate modeling", "earth system model", "weather prediction",
        "extreme weather", "flood forecasting", "drought prediction", "carbon emissions",
        "greenhouse gas", "decarbonization", "remote sensing climate", "environmental monitoring",
        "renewable energy", "climate risk", "global warming", "carbon flux", "cyclone",
        "ocean atmosphere", "sea level rise", "aerosol", "meteorological", "heatwave",
        "environmental impacts", "carbon footprint"
    },
}

CORE_AI_TERMS = {
    "artificial intelligence", "large language model", "large language models", "llm", "llms", "rag", "retrieval augmented generation",
    "explainable ai", "xai", "agentic ai", "reinforcement learning", "generative ai", "diffusion model", "federated learning", "robotic manipulation", "graph neural network", "gnn"
}
GENERIC_ML_TERMS = {"machine learning", "deep learning", "neural network", "convolutional"}


@dataclass
class DomainScopeResult:
    scope_type: ScopeType
    allowed_domains: List[str]
    scope_label: str
    detected_domains: List[str] = field(default_factory=list)


class DomainScopeDetector:
    """Detects domain scoping constraints for RAG retrieval."""

    @staticmethod
    def detect(question: str, user_domain: Optional[Any] = None) -> DomainScopeResult:
        if not question or not question.strip():
            return DomainScopeResult(
                scope_type=ScopeType.ALL_DOMAINS,
                allowed_domains=list(DOMAIN_LABEL_MAP.keys()),
                scope_label="All Domains",
            )

        q_lower = question.lower()

        # Step 1: Detect domain keyword matches
        detected = set()
        for domain, kws in DOMAIN_KEYWORDS.items():
            for kw in kws:
                if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                    detected.add(domain)
                    break

        has_core_ai = any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in CORE_AI_TERMS)
        has_generic_ml = any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in GENERIC_ML_TERMS)

        # Disambiguate generic ML terms in domain context (e.g. ML for crop prediction -> Agriculture)
        if "healthcare" in detected and (has_core_ai or ("ai" in q_lower or "medical imaging" in q_lower or "radiology" in q_lower)):
            detected.add("artificial_intelligence")
            detected.add("healthcare")

        if has_core_ai or has_generic_ml:
            detected.add("artificial_intelligence")

        # Step 2: Handle explicit user_domain selection
        user_dom_clean = None
        if isinstance(user_domain, list):
            valid_list = [d for d in user_domain if isinstance(d, str) and d in DOMAIN_LABEL_MAP]
            if len(valid_list) > 1:
                return DomainScopeResult(
                    scope_type=ScopeType.MULTI_DOMAIN,
                    allowed_domains=valid_list,
                    scope_label="Multi-Domain (" + ", ".join([DOMAIN_LABEL_MAP[d] for d in valid_list]) + ")",
                    detected_domains=valid_list,
                )
            elif len(valid_list) == 1:
                user_dom_clean = valid_list[0]
        elif isinstance(user_domain, str):
            user_dom_clean = user_domain.lower().strip() if user_domain.lower() not in ("none", "") else None

        if user_dom_clean in ("all", "all domains"):
            return DomainScopeResult(
                scope_type=ScopeType.ALL_DOMAINS,
                allowed_domains=list(DOMAIN_LABEL_MAP.keys()),
                scope_label="All Domains",
                detected_domains=sorted(list(detected)),
            )

        if user_dom_clean and user_dom_clean in DOMAIN_LABEL_MAP:
            if not has_core_ai and user_dom_clean != "artificial_intelligence":
                detected.discard("artificial_intelligence")
            detected.add(user_dom_clean)

        # Step 3: Determine Scope Type
        if len(detected) == 1:
            dom = list(detected)[0]
            return DomainScopeResult(
                scope_type=ScopeType.SINGLE_DOMAIN,
                allowed_domains=[dom],
                scope_label=f"SINGLE_DOMAIN ({DOMAIN_LABEL_MAP[dom]})",
                detected_domains=[dom],
            )

        if len(detected) > 1:
            sorted_doms = sorted(list(detected))
            labels = [DOMAIN_LABEL_MAP[d] for d in sorted_doms]
            return DomainScopeResult(
                scope_type=ScopeType.MULTI_DOMAIN,
                allowed_domains=sorted_doms,
                scope_label=f"MULTI_DOMAIN ({', '.join(labels)})",
                detected_domains=sorted_doms,
            )

        return DomainScopeResult(
            scope_type=ScopeType.ALL_DOMAINS,
            allowed_domains=list(DOMAIN_LABEL_MAP.keys()),
            scope_label="All Domains",
            detected_domains=[],
        )
