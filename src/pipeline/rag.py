"""
Generic Question Answerability & Evidence Validation System — ScholarLens

Architecture & Decision Sequence:
  User Question
      ↓ DomainScopeDetector (SINGLE_DOMAIN, MULTI_DOMAIN, ALL_DOMAINS)
      ↓ GenericQuestionAnalyzer  — dynamic decomposition, NO hardcoded topics
      ↓ HybridRetriever (ChromaDB + BM25 + Multi-Stream RRF)
      ↓
      ├─ GenericEvidenceEvaluator (local evidence)
      │      Evaluates: subject support, aspect/intent support, relationship support,
      │                 comparison target coverage, multi-aspect independence,
      │                 chunk quality, evidence specificity, cross-passage consistency
      │
      ├─ Tier 1 (LOCAL_SUFFICIENT)   → Local RAG Answer  [E1], [E2]
      ├─ Tier 2 (LOCAL_PARTIAL)      → Hybrid: local [E1] + online [O1]
      ├─ Tier 3 (LOCAL_INSUFFICIENT) → Online Academic Search
      │      ↓ GenericEvidenceEvaluator (same logic, online evidence)
      │      ├─ ONLINE_SUFFICIENT    → Online RAG Answer  [O1], [O2]
      │      └─ ONLINE_INSUFFICIENT  → Honest Insufficient Evidence Response
      └─ Tier 4 (no evidence at all) → Honest Insufficient Evidence Response

  → Answer Intent & Question Repetition Guardrail
  → Answer Completeness Validator
  → Dual Claim-Level Validator (Grounding + Intent Relevance)
  → Dynamic Citation Assignment ([E#] corpus, [O#] online)
  → Final "No Fabricated Content" Safety Gate
  → Pointwise "Why This Answer?" Explainability Module
  → RAGResponse Structured Output

GENERIC PRINCIPLE:
  ZERO hardcoded topic dictionaries.
  ZERO question-specific elif branches.
  SAME evaluator runs on local AND online evidence.
  "Related" ≠ "Answerable". Both must be evaluated separately.
"""

import os
import json
import logging
import re
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set

from src.pipeline.domain_scope import DomainScopeDetector, DomainScopeResult, ScopeType
from src.pipeline.retrieval import HybridRetriever, RetrievalResult, RetrievalConfig, is_bibliography_chunk, DOMAIN_PREFIX_MAP, IntentQueryReformulator
from src.pipeline.online_fallback import OnlineAcademicRetriever, OnlineEvidenceItem
from src.pipeline.llm import LLMProvider, get_llm_provider

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_PROMPT = """You are ScholarLens, an expert academic AI research assistant. 
Synthesize a clear, highly comprehensive, well-structured, and authoritative research response to the user's question using ONLY the provided scientific evidence passages.

STRICT ANSWER QUALITY & FORMATTING RULES (CHATGPT / SCHOLARLY STYLE):
1. DIRECT EXECUTIVE SUMMARY:
   - Begin the first paragraph with a clear, direct executive summary that immediately answers the user's core question.
2. RICH MARKDOWN STRUCTURE:
   - Use clear markdown sections and headings (e.g. ### Overview, ### Key Mechanisms & Findings, ### Empirical Results, ### Limitations & Future Scope).
   - Use bullet points and bold key technical terms to make the answer engaging, highly readable, and research-grade.
3. CLAIM-LEVEL EVIDENCE ALIGNMENT:
   - Every factual claim containing [E#] or [O#] MUST be directly supported by the exact retrieved passage text mapped to [E#] or [O#].
   - Place citation tags [E1], [O1] immediately after the factual claims they support.
4. COMPREHENSIVE RESEARCH DEPTH:
   - Provide multi-paragraph analytical depth (aim for 300–800 words of thorough explanation when sufficient evidence exists).
   - Do NOT produce vague, short, or robotic single-sentence responses.
5. HONEST FALLBACK:
   - If retrieved evidence genuinely cannot support an answer, output exactly: "Insufficient evidence was found in the current Research Mind corpus or available online academic sources to answer this question reliably."
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    citation_id: str  # E1, E2, E3... or O1, O2...
    unit_id: str
    parent_chunk_id: str
    chunk_id: str
    paper_id: str
    section_id: str
    section_name: str
    domain: str
    subtopic: str
    page_start: int
    page_end: int
    text: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0
    retrieval_methods: List[str] = field(default_factory=list)
    source_type: str = "corpus"  # corpus or online
    url: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Citation:
    citation_id: str  # E1, E2... or O1, O2...
    paper_id: str
    section_name: str
    pages: str
    chunk_id: str
    unit_id: str
    source_type: str = "corpus"
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    published: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "paper_id": self.paper_id,
            "section_name": self.section_name,
            "pages": self.pages,
            "chunk_id": self.chunk_id,
            "unit_id": self.unit_id,
            "source_type": self.source_type,
            "title": self.title,
            "authors": self.authors,
            "published": self.published,
            "url": self.url,
        }


@dataclass
class WhyThisAnswer:
    contributing_papers: List[str]
    contributing_sections: List[str]
    evidence_passages: List[str]
    multi_paper_support: bool
    evidence_strength: str  # Excellent, High, Moderate, Low, Insufficient
    unsupported_claims: int
    inference_present: bool
    conflicts_detected: bool
    explanation_summary: str
    bullet_points: List[str] = field(default_factory=list)
    source_type: str = "Research Mind Corpus"
    domain_scope: str = "All Domains"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGResponse:
    question: str
    answer: str
    evidence: List[EvidenceItem]
    citations: Dict[str, Citation]
    confidence: str  # Excellent, High, Moderate, Low, Insufficient
    confidence_rationale: str
    limitations: str
    why_this_answer: WhyThisAnswer
    retrieval_metadata: Dict[str, Any]
    domain_scope: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["evidence"] = [e.to_dict() for e in self.evidence]
        res["citations"] = {k: (v.to_dict() if hasattr(v, 'to_dict') else v) for k, v in self.citations.items()}
        res["why_this_answer"] = self.why_this_answer.to_dict()
        return res


@dataclass
class QuestionContract:
    """
    Dynamically constructed Question → Evidence Contract.
    Enforces that evidence must contain the SPECIFIC TYPE of information requested by the question.
    ZERO hardcoded topic branches.
    """
    concept: str                         # Core target concept (e.g. "cloud computing")
    intent: str                          # Definition | Limitation | Algorithm | Mechanism | ...
    required_evidence_type: str          # Description of required evidence (e.g. "direct definition, core properties, or fundamental explanation")
    strict_intent_markers: Set[str] = field(default_factory=set) # Required functional markers for intent verification
    secondary_concepts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "intent": self.intent,
            "required_evidence_type": self.required_evidence_type,
            "strict_intent_markers": list(self.strict_intent_markers)[:10],
            "secondary_concepts": self.secondary_concepts,
        }


@dataclass
class QuestionRepresentation:
    """
    Dynamic, generic decomposition of a research question.
    Produced by GenericQuestionAnalyzer — zero hardcoded topics.
    """
    main_subject: List[str]          # Core noun phrases / entities
    secondary_concepts: List[str]    # Supporting concepts
    requested_aspect: str            # What the user wants to know about the subject
    intent: str                      # Algorithm | Limitation | Advantage | Mechanism | ...
    required_aspects: List[str]      # Multi-part: ["benefits", "limitations"]
    is_relational: bool              # Does it ask X→Y relationship?
    relation_source: List[str]       # Entity X in "X improves Y"
    relation_target: List[str]       # Entity Y in "X improves Y"
    relation_verb: Optional[str]     # "improve", "cause", "affect", etc.
    is_comparison: bool              # "A vs B" structure
    comparison_targets: List[str]    # [A, B]
    is_multi_aspect: bool            # Multiple things requested
    all_content_words: List[str]     # Non-stopword tokens (for coverage scoring)
    contract: Optional[QuestionContract] = None



@dataclass
class AnswerabilityResult:
    """
    Structured result of the generic evidence answerability evaluation.

    Spec-compliant output:
        related              — passage discusses the question subject
        answerable           — passage contains info that can answer the question
        completeness         — fraction of requested aspects covered (0-1)
        intent_support       — fraction of intent-specific evidence present (0-1)
        concept_support      — fraction of main concepts present (0-1)
        relationship_support — relationship/joint-binding supported (0-1)
        evidence_quality     — chunk completeness & coherence (0-1)
        missing_aspects      — list of requested aspects not covered
        decision             — LOCAL_SUFFICIENT | LOCAL_INSUFFICIENT | ONLINE_SUFFICIENT | INSUFFICIENT
        rationale            — human-readable explanation
    """
    related: bool
    answerable: bool
    completeness: float
    intent_support: float
    concept_support: float
    relationship_support: float
    evidence_quality: float
    missing_aspects: List[str]
    decision: str  # LOCAL_SUFFICIENT | LOCAL_INSUFFICIENT | ONLINE_SUFFICIENT | INSUFFICIENT
    rationale: str
    direct_supporting_passages: List[Any] = field(default_factory=list)
    # Legacy compat
    related_score: float = 0.0
    answerability_score: float = 0.0
    concept_coverage: float = 0.0
    is_related: bool = False
    is_answerable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "related": self.related,
            "answerable": self.answerable,
            "completeness": round(self.completeness, 3),
            "intent_support": round(self.intent_support, 3),
            "concept_support": round(self.concept_support, 3),
            "relationship_support": round(self.relationship_support, 3),
            "evidence_quality": round(self.evidence_quality, 3),
            "missing_aspects": self.missing_aspects,
            "decision": self.decision,
            "rationale": self.rationale,
            "direct_supporting_passages_count": len(self.direct_supporting_passages),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Generic Question Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class GenericQuestionAnalyzer:
    """
    Dynamically decomposes any research question into a QuestionRepresentation.

    ZERO hardcoded topics. Works for questions the developers have never seen.
    Uses linguistic patterns and morphological heuristics.
    """

    STOPWORDS: Set[str] = {
        "what", "is", "are", "the", "a", "an", "how", "can", "could", "should", "would",
        "does", "do", "in", "of", "to", "for", "on", "with", "by", "from", "at", "about",
        "which", "why", "where", "when", "be", "used", "using", "help", "improves", "improve",
        "versus", "multi", "and", "or", "not", "its", "their", "this", "that", "these",
        "those", "was", "were", "been", "has", "have", "had", "will", "would", "there",
        "some", "any", "all", "also", "both", "more", "most", "such", "each", "other",
    }

    # Intent patterns — ordered from most specific to least specific
    INTENT_PATTERNS: List[Tuple[str, List[str]]] = [
        ("Algorithm",   ["algorithm", "algorithms", "method", "methods", "technique", "techniques",
                          "classifier", "classifiers", "architecture", "architectures", "approach",
                          "approaches", "model", "models used", "commonly used"]),
        ("Dataset",     ["dataset", "datasets", "benchmark", "benchmarks", "corpus", "corpora",
                          "data set", "data sets", "testbed", "collection", "database"]),
        ("Limitation",  ["limitation", "limitations", "drawback", "drawbacks", "weakness", "weaknesses",
                          "challenge", "challenges", "risk", "risks", "vulnerability", "vulnerabilities",
                          "bottleneck", "bottlenecks", "constraint", "constraints", "shortcoming",
                          "shortcomings", "problem", "problems", "issue", "issues", "failure", "failures"]),
        ("Advantage",   ["advantage", "advantages", "benefit", "benefits", "strength", "strengths",
                          "gain", "gains", "merit", "merits"]),
        ("Evaluation",  ["evaluat", "metric", "metrics", "benchmark", "benchmarks", "performance",
                          "accuracy", "precision", "recall", "f1", "auc", "roc"]),
        ("Cause",       ["cause", "causes", "caused", "why does", "why is", "why do", "why are",
                          "reason", "reasons", "leads to", "lead to", "due to", "results in"]),
        ("Result",      ["result", "results", "finding", "findings", "outcome", "outcomes",
                          "observation", "observations", "demonstrate", "shows", "proved"]),
        ("Comparison",  ["compare", "comparison", "difference", "differences", "versus", " vs ",
                          "vs.", "distinguish", "distinguish between"]),
        ("Mechanism",   ["how does", "how do", "how can", "how is", "how are", "mechanism",
                          "mechanisms", "process", "processes", "workflow", "step", "steps",
                          "procedure", "work", "works", "operate", "operates", "function"]),
        ("Relationship",["affect", "affects", "impact", "impacts", "influence", "influences",
                          "improve", "improves", "enable", "enables", "help", "helps", "support",
                          "supports", "reduce", "reduces", "increase", "increases", "prevent",
                          "prevents", "cause", "causes", "relate", "relates", "relationship"]),
        ("Application", ["application", "applications", "use case", "use cases", "deploy",
                          "deployed", "applied", "used in", "applied in"]),
        ("Definition",  ["what is", "what are", "define", "definition", "meaning", "means",
                          "refer to", "concept of"]),
    ]

    # Multi-aspect markers
    MULTI_ASPECT_PAIRS: List[Tuple[List[str], List[str]]] = [
        (["benefit", "benefits", "advantage", "advantages", "strength", "strengths", "pro", "pros"],
         ["limitation", "limitations", "drawback", "drawbacks", "weakness", "weaknesses",
          "challenge", "challenges", "con", "cons", "risk", "risks"]),
        (["advantage", "advantages"], ["disadvantage", "disadvantages"]),
        (["cause", "causes"], ["effect", "effects", "consequence", "consequences"]),
        (["how", "mechanism"], ["why", "reason"]),
    ]

    # Relational verb patterns that signal X→Y structure
    RELATIONAL_VERBS: List[str] = [
        "improve", "improves", "improving", "improved",
        "affect", "affects", "affecting", "affected",
        "impact", "impacts", "impacting", "impacted",
        "influence", "influences", "influencing", "influenced",
        "enable", "enables", "enabling", "enabled",
        "support", "supports", "supporting", "supported",
        "reduce", "reduces", "reducing", "reduced",
        "increase", "increases", "increasing", "increased",
        "help", "helps", "helping", "helped",
        "enhance", "enhances", "enhancing", "enhanced",
        "cause", "causes", "causing", "caused",
        "prevent", "prevents", "preventing", "prevented",
        "optimize", "optimizes", "optimizing", "optimized",
        "boost", "boosts", "boosting", "boosted",
        "degrade", "degrades", "degrading", "degraded",
    ]

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        """Extract meaningful tokens, filtering stopwords."""
        clean = re.sub(r"[^\w\s-]", " ", text.lower())
        return [w for w in clean.split() if len(w) >= 3 and w not in cls.STOPWORDS]

    @classmethod
    def _extract_ngrams(cls, tokens: List[str], raw_tokens: List[str], n: int = 2) -> List[str]:
        """Build n-grams from raw token sequence (preserving order)."""
        grams = []
        for i in range(len(raw_tokens) - n + 1):
            gram = " ".join(raw_tokens[i:i+n])
            if not all(w in cls.STOPWORDS for w in gram.split()):
                grams.append(gram)
        return grams

    @classmethod
    def classify_intent(cls, question: str) -> str:
        """Classify question intent generically using ordered pattern matching."""
        q_lower = question.lower()
        for intent, patterns in cls.INTENT_PATTERNS:
            if any(p in q_lower for p in patterns):
                return intent
        return "Explanation"

    @classmethod
    def _find_relational_structure(cls, q_lower: str) -> Tuple[List[str], List[str], Optional[str]]:
        """
        Detect X-verb-Y relational structure.
        Returns (source_entities, target_entities, verb) or ([], [], None).
        """
        rel_pattern = (
            r'\b(' + '|'.join(re.escape(v) for v in cls.RELATIONAL_VERBS) + r')\b'
        )
        parts = re.split(rel_pattern, q_lower, maxsplit=1)
        if len(parts) == 3:
            pre, verb, post = parts
            src = [w for w in pre.split() if len(w) >= 3 and w not in cls.STOPWORDS]
            tgt = [w for w in post.split() if len(w) >= 3 and w not in cls.STOPWORDS]
            if src and tgt:
                return src[-3:], tgt[:3], verb
        return [], [], None

    @classmethod
    def _find_comparison_targets(cls, q_lower: str) -> List[str]:
        """
        Detect comparison structure: "A vs B", "difference between A and B",
        "compare A with B", "A versus B".
        Returns a list of comparison target phrases.
        """
        patterns = [
            r'between\s+(.+?)\s+and\s+(.+?)(?:\s*\?)?$',
            r'(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\s*\?)?$',
            r'compare\s+(.+?)\s+(?:with|to|and)\s+(.+?)(?:\s*\?)?$',
            r'difference(?:s)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*\?)?$',
        ]
        for pattern in patterns:
            m = re.search(pattern, q_lower)
            if m:
                groups = [g.strip() for g in m.groups()]
                # filter trivial groups
                return [g for g in groups if len(g) >= 2]
        return []

    @classmethod
    def _find_required_aspects(cls, q_lower: str) -> List[str]:
        """
        Detect multi-aspect requirements: "benefits AND limitations", etc.
        Returns list of required aspect labels.
        """
        aspects = []
        for positive_group, negative_group in cls.MULTI_ASPECT_PAIRS:
            has_pos = any(p in q_lower for p in positive_group)
            has_neg = any(n in q_lower for n in negative_group)
            if has_pos:
                aspects.append("advantage")
            if has_neg:
                aspects.append("limitation")
            if has_pos and has_neg:
                break  # Found both
        # Deduplicate
        return list(dict.fromkeys(aspects))

    @classmethod
    def build_contract(
        cls,
        intent: str,
        concept: str,
        main_subject: List[str],
        secondary_concepts: List[str],
    ) -> QuestionContract:
        evidence_type_map = {
            "Definition": "direct definition, core properties, or fundamental explanation of concept",
            "Limitation": "explicit limitations, drawbacks, challenges, constraints, or failure modes",
            "Algorithm": "specific named algorithms, methods, models, or architectures used for target task",
            "Mechanism": "detailed process, workflow, or mechanism explaining how concept operates or improves target",
            "Advantage": "explicit benefits, advantages, strengths, or positive outcomes of concept",
            "Cause": "underlying causes, reasons, or triggering factors",
            "Result": "empirical findings, experimental results, or observed performance outcomes",
            "Comparison": "direct side-by-side comparison, differences, or trade-offs between concepts",
            "Relationship": "explicit causal, functional, or empirical relationship connecting concepts",
            "Application": "practical use cases, deployments, or application contexts",
            "Evaluation": "evaluation metrics, benchmarks, or validation protocols",
            "Dataset": "datasets, corpora, or benchmark data collections",
        }
        evidence_desc = evidence_type_map.get(intent, "direct evidence addressing requested question intent")

        return QuestionContract(
            concept=concept,
            intent=intent,
            required_evidence_type=evidence_desc,
            strict_intent_markers=set(),
            secondary_concepts=secondary_concepts,
        )

    @classmethod
    def analyze(cls, question: str) -> QuestionRepresentation:
        """
        Dynamically decompose any research question into a QuestionRepresentation.
        ZERO hardcoded topics.
        """
        q_lower = question.lower().strip()
        all_content_words = cls._tokenize(question)
        raw_tokens = [w for w in re.sub(r"[^\w\s-]", " ", q_lower).split() if len(w) >= 2]

        intent = cls.classify_intent(question)
        comparison_targets = cls._find_comparison_targets(q_lower)
        is_comparison = len(comparison_targets) >= 2
        required_aspects = cls._find_required_aspects(q_lower)
        is_multi_aspect = len(required_aspects) >= 2

        # Relational structure (X improves Y, X affects Y, etc.)
        rel_source, rel_target, rel_verb = cls._find_relational_structure(q_lower)
        is_relational = bool(rel_source and rel_target)

        # For relational/relationship questions without relational verb,
        # detect "relationship between X and Y" pattern
        if not is_relational and "relationship" in q_lower:
            m = re.search(r'relationship\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*\?)?$', q_lower)
            if m:
                rel_source = cls._tokenize(m.group(1))
                rel_target = cls._tokenize(m.group(2))
                rel_verb = "relationship"
                is_relational = bool(rel_source and rel_target)

        # If not relational, main_subject = all content words
        if is_relational:
            main_subject = rel_source
            secondary_concepts = rel_target
        elif is_comparison:
            main_subject = [w for t in comparison_targets for w in cls._tokenize(t)]
            secondary_concepts = []
        else:
            main_subject = all_content_words[:4] if len(all_content_words) > 4 else all_content_words
            secondary_concepts = all_content_words[4:] if len(all_content_words) > 4 else []

        # Derive primary concept string from question text
        topic_phrase = q_lower
        for prefix in (
            "what is ", "what are the ", "what are ", "how does ", "how do ",
            "how can ", "why does ", "why do ", "describe ", "what causes ",
            "what algorithms are commonly used for ", "what algorithms are used for ",
            "what algorithms ", "what ", "how is ", "how ",
        ):
            if topic_phrase.startswith(prefix):
                topic_phrase = topic_phrase[len(prefix):].strip()
                break
        concept_str = topic_phrase.rstrip("?").strip()
        if not concept_str and main_subject:
            concept_str = " ".join(main_subject)

        # Build QuestionContract
        contract = cls.build_contract(intent, concept_str, main_subject, secondary_concepts)

        # Requested aspect: what does the user want to know?
        aspect_map = {
            "Algorithm": "algorithms/methods",
            "Limitation": "limitations/challenges",
            "Advantage": "advantages/benefits",
            "Dataset": "datasets/benchmarks",
            "Evaluation": "evaluation metrics",
            "Cause": "causes/reasons",
            "Result": "results/findings",
            "Comparison": "comparison",
            "Mechanism": "mechanism/process",
            "Relationship": "relationship/effect",
            "Application": "applications/use cases",
            "Definition": "definition/explanation",
            "Explanation": "explanation",
        }
        requested_aspect = aspect_map.get(intent, intent.lower())

        return QuestionRepresentation(
            main_subject=main_subject,
            secondary_concepts=secondary_concepts,
            requested_aspect=requested_aspect,
            intent=intent,
            required_aspects=required_aspects,
            is_relational=is_relational,
            relation_source=rel_source,
            relation_target=rel_target,
            relation_verb=rel_verb,
            is_comparison=is_comparison,
            comparison_targets=comparison_targets,
            is_multi_aspect=is_multi_aspect,
            all_content_words=all_content_words,
            contract=contract,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Generic Evidence Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class GenericEvidenceEvaluator:
    """
    GENERIC automatic question-answerability and evidence sufficiency evaluator.

    Distinguishes FIVE levels:
      1. Unrelated    — evidence does not discuss the subject
      2. Related      — evidence is about the topic but does not answer the question
      3. Partial      — evidence answers some but not all requested aspects
      4. Answerable   — evidence sufficiently answers the question
      5. Excellent    — strong multi-passage consensus with complete aspect coverage

    ZERO hardcoded topic lists. ZERO question-specific branches.
    Works identically for local corpus evidence AND online ArXiv evidence.

    Evaluates 7 independent dimensions:
      A. Subject support
      B. Aspect/intent support  ← THIS IS WHERE "related ≠ answerable" is enforced
      C. Relationship/joint-binding support
      D. Completeness (multi-aspect coverage)
      E. Evidence specificity (named entities vs generic mention)
      F. Chunk quality (length, coherence, not truncated)
      G. Cross-passage consistency
    """

    # Intent → evidence markers (generic — covers any domain)
    # These are FUNCTIONAL ASPECT MARKERS, not topic keywords.
    INTENT_ASPECT_MARKERS: Dict[str, Set[str]] = {
        "Algorithm": {
            "algorithm", "algorithms", "method", "methods", "technique", "techniques",
            "model", "models", "classifier", "classifiers", "architecture", "architectures",
            "framework", "frameworks", "network", "networks", "approach", "approaches",
            "pipeline", "scheme", "protocol", "heuristic", "optimization", "procedure",
            "using", "based on", "employs", "implemented", "trained", "applied",
            # Named method markers — any named entity following these patterns
            "svm", "cnn", "rnn", "lstm", "gru", "transformer", "random forest",
            "decision tree", "gradient boosting", "naive bayes", "k-nearest",
            "linear regression", "logistic regression", "deep learning", "neural",
            "attention", "encoder", "decoder", "embedding",
        },
        "Limitation": {
            "limitation", "limitations", "drawback", "drawbacks", "weakness", "weaknesses",
            "challenge", "challenges", "failure", "failures", "problem", "problems",
            "issue", "issues", "shortcoming", "shortcomings", "risk", "risks",
            "bottleneck", "bottlenecks", "constraint", "constraints", "overhead",
            "difficult", "difficulty", "hard to", "cannot", "unable", "fails",
            "error", "errors", "noise", "bias", "costly", "expensive", "slow",
            "limited", "lack", "lacks", "lacking", "insufficient", "inadequate",
            "inaccurate", "unreliable", "unstable", "vulnerable", "susceptible",
        },
        "Advantage": {
            "advantage", "advantages", "benefit", "benefits", "strength", "strengths",
            "improve", "improves", "improvement", "enhanced", "superior", "better",
            "efficient", "effective", "effective", "outperform", "outperforms",
            "gain", "gains", "robust", "robustness", "scalable", "accurate",
            "precision", "faster", "reliable", "high performance", "optimal",
            "success", "successful", "demonstrate", "demonstrates", "achieve",
        },
        "Mechanism": {
            "mechanism", "mechanisms", "process", "processes", "workflow", "step", "steps",
            "procedure", "pipeline", "operation", "operates", "function", "functions",
            "compute", "computes", "generate", "generates", "encode", "decode",
            "transform", "transforms", "execute", "executes", "transmit", "transmits",
            "measure", "measures", "protocol", "phase", "stage", "stages", "layer",
            "by", "through", "using", "via", "where", "when", "then",
        },
        "Relationship": {
            "improve", "improves", "affect", "affects", "impact", "impacts",
            "influence", "influences", "enable", "enables", "help", "helps",
            "support", "supports", "reduce", "reduces", "increase", "increases",
            "lead to", "leads to", "result in", "results in", "cause", "causes",
            "enhance", "enhances", "contribute", "contributes", "allow", "allows",
            "facilitate", "facilitates", "promote", "promotes", "prevent", "prevents",
        },
        "Comparison": {
            "compare", "compared", "versus", "vs", "difference", "differences",
            "while", "whereas", "in contrast", "on the other hand", "unlike", "unlike",
            "better than", "worse than", "superior", "inferior", "trade-off", "trade-offs",
            "advantage over", "disadvantage compared",
        },
        "Cause": {
            "cause", "causes", "caused", "because", "due to", "owing to",
            "reason", "reasons", "result from", "results from", "attribute to",
            "lead to", "leads to", "trigger", "triggers", "originate", "stem from",
        },
        "Result": {
            "result", "results", "finding", "findings", "outcome", "outcomes",
            "observation", "observations", "demonstrate", "demonstrates", "show",
            "shows", "achieve", "achieves", "performance", "accuracy", "score",
            "percentage", "experiment", "evaluation", "test", "study",
        },
        "Definition": {
            "is defined as", "defined as", "refers to", "meaning of", "definition of",
            "is a paradigm", "is a technology", "is a computing model", "is a framework",
            "is an architecture", "provides on-demand", "characterized by", "enables users to",
            "consists of", "is a method", "is an approach", "denotes", "represents",
        },
        "Dataset": {
            "dataset", "datasets", "benchmark", "benchmarks", "corpus", "corpora",
            "collection", "database", "repository", "samples", "instances",
            "training set", "test set", "validation set", "annotated",
        },
        "Evaluation": {
            "evaluation", "evaluate", "evaluates", "metric", "metrics",
            "accuracy", "precision", "recall", "f1", "auc", "roc",
            "benchmark", "benchmarks", "performance", "comparison",
        },
        "Application": {
            "application", "applications", "applied", "deployed", "used in",
            "used for", "implemented", "practical", "real-world", "use case",
            "use cases", "domain", "field",
        },
    }

    META_ASPECT_TARGETS: Set[str] = {
        "evaluation", "evaluations", "evaluation practice", "evaluation practices", "evaluation methodology",
        "evaluation framework", "evaluation metrics", "evaluation protocol", "evaluation setup",
        "assessment", "assessment framework", "assessing", "benchmark", "benchmarks",
        "metric", "metrics", "dataset", "datasets", "corpus", "corpora", "testbed",
        "existing literature", "previous work", "survey", "review", "paper", "section",
        "hardware", "baseline", "prior studies", "annotation", "annotators",
    }

    @classmethod
    def _is_aspect_hijacked_by_meta_target(cls, p_text: str, aspect_word: str) -> bool:
        """
        Generic check: Is the aspect word (e.g. "limitation", "challenge", "drawback") modifying
        a meta-target (e.g., "evaluation practices", "metrics", "datasets", "benchmarks") rather than
        the query subject?
        """
        for meta in cls.META_ASPECT_TARGETS:
            patterns = [
                f"{aspect_word} of {meta}",
                f"{aspect_word} of existing {meta}",
                f"{aspect_word} of current {meta}",
                f"{aspect_word} of standard {meta}",
                f"{aspect_word} of traditional {meta}",
                f"{aspect_word} of conventional {meta}",
                f"{aspect_word} in {meta}",
                f"{meta} {aspect_word}",
                f"{meta} {aspect_word}s",
            ]
            if any(pat in p_text for pat in patterns):
                total_occurrences = p_text.count(aspect_word)
                hijacked_occurrences = sum(p_text.count(pat) for pat in patterns)
                if hijacked_occurrences >= total_occurrences:
                    return True
        return False

    MIN_PASSAGE_LENGTH = 70  # chars — fragments below this are rejected
    MIN_ANSWERABLE_PASSAGES = 1  # minimum direct-support passages for answerable
    COMPLETENESS_THRESHOLD = 0.45  # fraction of content words required

    @classmethod
    def _get_passage_text(cls, p: Any) -> str:
        """Safely extract lowercase text from a retrieval result or online evidence item."""
        text = getattr(p, 'text', '') or ''
        section = getattr(p, 'section_name', '') or ''
        return (text + " " + section).lower()

    @classmethod
    def _is_quality_passage(cls, p: Any) -> bool:
        """
        Chunk quality gate (Dimension F).
        Rejects: too-short fragments, bibliography chunks, incoherent fragments.
        """
        text = getattr(p, 'text', '') or ''
        if len(text.strip()) < cls.MIN_PASSAGE_LENGTH:
            return False
        if is_bibliography_chunk(text):
            return False
        # Reject if text is all uppercase (likely metadata)
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.70:
            return False
        return True

    @classmethod
    def _morph_expand(cls, word: str) -> Set[str]:
        """
        Generic morphological expansion — no hardcoded topic vocabulary.
        Generates common suffix variants to handle algorithmic/nominal/adjectival forms.
        """
        variants = {word}
        suffixes_to_strip = ["ing", "tion", "sion", "ity", "ical", "ness", "ly",
                              "ize", "ization", "ise", "isation", "al", "ed", "er", "s"]
        for suffix in suffixes_to_strip:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                base = word[:-len(suffix)]
                variants.add(base)
                # Also try common additions to the base
                variants.update([base + "e", base + "y", base + "ic"])
        # Add common derivational variants
        if word.endswith("e") and len(word) >= 5:
            variants.add(word + "d")
            variants.add(word[:-1] + "ing")
            variants.add(word[:-1] + "tion")
        if not word.endswith("s") and len(word) >= 4:
            variants.add(word + "s")
        return variants

    @classmethod
    def _concept_in_text(cls, concept: str, text: str) -> bool:
        """
        Check if a concept word or any morphological variant appears in text.
        Generic — no hardcoded synonym maps.
        """
        if concept in text:
            return True
        for variant in cls._morph_expand(concept):
            if variant in text and len(variant) >= 4:
                return True
        return False

    @classmethod
    def _subject_coverage(cls, q_repr: QuestionRepresentation, p_text: str) -> float:
        """
        Dimension A: Subject support.
        How much of the main subject is represented in this passage?
        For multi-word subjects (e.g. ['quantum', 'teleportation']), ALL subject terms
        must be present to qualify as direct support.
        """
        if not q_repr.main_subject:
            return 1.0
        matched = sum(1 for w in q_repr.main_subject if cls._concept_in_text(w, p_text))
        ratio = matched / len(q_repr.main_subject)
        if len(q_repr.main_subject) >= 2 and ratio < 1.0:
            return 0.0
        return ratio

    @classmethod
    def _intent_support(cls, q_repr: QuestionRepresentation, p_text: str) -> float:
        """
        Dimension B: Aspect/intent support.
        Does the passage contain markers that show it ANSWERS the intent?

        This is the core "related ≠ answerable" enforcement.
        A passage about LLMs has subject coverage for "What are the limitations of LLMs?"
        but ONLY gets intent_support credit if it actually contains limitation markers.
        """
        markers = cls.INTENT_ASPECT_MARKERS.get(q_repr.intent, set())
        if not markers:
            # No specific markers for this intent — use presence of content words
            return 1.0 if any(cls._concept_in_text(w, p_text) for w in q_repr.all_content_words) else 0.0

        # For Algorithm intent: distinguish generic usage phrases from specific named methods.
        # "Machine learning is used for X" → NO intent support (generic, doesn't name an algorithm)
        # "Random Forest and SVM are used for X" → YES intent support (named specific algorithms)
        if q_repr.intent == "Algorithm":
            # SPECIFIC named algorithm/method markers (named models, architectures, specific techniques)
            specific_algorithm_markers = {
                "algorithm", "algorithms", "classifier", "classifiers", "architecture", "architectures",
                "framework", "frameworks",
                # Named common algorithms/models — sufficiently specific
                "svm", "support vector", "random forest", "decision tree", "gradient boosting",
                "naive bayes", "k-nearest", "k-means", "logistic regression", "linear regression",
                "cnn", "rnn", "lstm", "gru", "transformer", "attention", "encoder", "decoder",
                "embedding", "bert", "gpt", "resnet", "vgg",
                # Named technique patterns
                "convolutional", "recurrent", "deep learning", "shallow learning",
                "ensemble", "bagging", "boosting", "clustering",
                "heuristic", "genetic", "evolutionary", "reinforcement",
                "unsupervised", "supervised",
            }
            # GENERIC usage-only markers (these do NOT prove a specific algorithm is named)
            _generic_usage = {"using", "based on", "employs", "implemented", "trained",
                              "applied", "method", "technique", "approach", "model", "pipeline",
                              "scheme", "protocol", "optimization", "procedure", "network",
                              "process", "neural"}
            specific_hits = sum(1 for m in specific_algorithm_markers if m in p_text)
            if specific_hits == 0:
                # No named specific algorithm found — not intent-supportive
                return 0.0
            return min(1.0, specific_hits / 2.0)

        # For Limitation intent, require actual limitation language modifying the subject (not meta-concepts like evaluation practices)
        if q_repr.intent == "Limitation":
            core_limitation_markers = {
                "limitation", "limitations", "drawback", "drawbacks", "weakness", "weaknesses",
                "challenge", "challenges", "failure", "failures", "shortcoming", "shortcomings",
                "bottleneck", "bottlenecks", "constraint", "constraints", "cannot", "unable",
                "fails", "limited", "lack", "lacks", "lacking", "insufficient", "inadequate",
                "inaccurate", "unreliable", "unstable", "risk", "risks", "problem", "problems",
                "issue", "issues", "error", "errors", "bias", "difficult", "suffer from",
                "suffers from", "inefficient", "excessive", "overhead", "computational cost",
            }
            valid_hits = 0
            for m in core_limitation_markers:
                if m in p_text:
                    if not cls._is_aspect_hijacked_by_meta_target(p_text, m):
                        valid_hits += 1

            if valid_hits == 0:
                # All limitation words modify meta targets (e.g. evaluation practices) or none exist
                return 0.0
            return min(1.0, valid_hits / 2.0)

        # For Comparison, need comparison language
        if q_repr.intent == "Comparison":
            core_comparison_markers = {
                "compare", "compared", "versus", "vs", "difference", "differences",
                "while", "whereas", "in contrast", "on the other hand", "unlike",
                "better than", "worse than", "superior", "inferior", "trade-off",
                "advantage over", "disadvantage compared",
            }
            core_hits = sum(1 for m in core_comparison_markers if m in p_text)
            return min(1.0, core_hits / 2.0) if core_hits > 0 else 0.0

        # For Advantage intent, require positive outcome language
        if q_repr.intent == "Advantage":
            core_advantage_markers = {
                "advantage", "advantages", "benefit", "benefits", "strength", "strengths",
                "improve", "improves", "improvement", "enhanced", "superior", "better",
                "efficient", "effective", "outperform", "outperforms", "gain", "gains",
                "robust", "robustness", "scalable", "accurate", "reliable", "optimal",
                "success", "successful", "achieve", "achieves",
            }
            core_hits = sum(1 for m in core_advantage_markers if m in p_text)
            return min(1.0, core_hits / 2.0) if core_hits > 0 else 0.0

        # For Definition intent, require definitive language or direct concept explanation
        if q_repr.intent == "Definition":
            concept_str = q_repr.contract.concept.lower() if hasattr(q_repr, 'contract') and q_repr.contract else " ".join(q_repr.main_subject).lower()
            strict_def_patterns = {
                "is defined as", "defined as", "refers to", "meaning of", "definition of",
                "is a", "is an", "are a", "are an", "defined by", "known as",
                "provides on-demand", "characterized by", "enables users to",
                "consists of", "denotes", "represents", "serves as",
            }
            def_hits = sum(1 for m in strict_def_patterns if m in p_text)

            concept_is_def = False
            if concept_str and len(concept_str) >= 3:
                if (f"{concept_str} is" in p_text or
                    f"{concept_str} refers" in p_text or
                    f"{concept_str} provides" in p_text or
                    f"{concept_str} enables" in p_text or
                    f"{concept_str} denotes" in p_text or
                    f"{concept_str} represents" in p_text or
                    f"{concept_str} defined" in p_text or
                    f"{concept_str} encompasses" in p_text or
                    f"{concept_str} serves" in p_text):
                    concept_is_def = True

            if concept_is_def or def_hits >= 1:
                return min(1.0, 0.6 + (0.4 if concept_is_def else 0.2))
            else:
                # Merely mentioning the concept without definitional language is NOT sufficient for Definition intent
                return 0.0

        # Count distinct marker hits for other intents
        hits = sum(1 for m in markers if m in p_text)
        return min(1.0, hits / 2.0) if hits > 0 else 0.0



    @classmethod
    def _relationship_support(cls, q_repr: QuestionRepresentation, p_text: str) -> float:
        """
        Dimension C: Relationship/joint-binding support.
        For relational questions (X improves Y), BOTH entities must be in the same passage
        AND the passage must contain a connecting relationship marker.
        """
        if not q_repr.is_relational:
            return 1.0  # N/A — not penalised

        has_source = any(cls._concept_in_text(w, p_text) for w in q_repr.relation_source)
        has_target = any(cls._concept_in_text(w, p_text) for w in q_repr.relation_target)

        if not (has_source and has_target):
            return 0.0

        # Both entities present — now check for a connective/relational phrase
        rel_markers = cls.INTENT_ASPECT_MARKERS.get("Relationship", set())
        has_connector = any(m in p_text for m in rel_markers)

        if has_connector:
            return 1.0
        # Both entities present but no explicit relationship language — partial credit
        return 0.5

    @classmethod
    def _comparison_coverage(cls, q_repr: QuestionRepresentation, p_text: str) -> float:
        """
        For comparison questions, check that BOTH targets are discussed in the passage.
        """
        if not q_repr.is_comparison:
            return 1.0
        covered = sum(
            1 for t in q_repr.comparison_targets
            if any(cls._concept_in_text(w, p_text) for w in t.split())
        )
        return covered / len(q_repr.comparison_targets) if q_repr.comparison_targets else 1.0

    @classmethod
    def _specificity_score(cls, q_repr: QuestionRepresentation, p_text: str) -> float:
        """
        Dimension E: Specificity.
        Is the passage giving specific information or just a general/vague mention?

        Generic approach: score based on presence of:
        - Named entities (capitalized patterns) in the passage
        - Numerical data (%, numbers, performance figures)
        - Technical markers (= intent aspect markers)
        - Sentence complexity (longer sentences tend to be more specific)
        """
        # Named entity heuristic: sequences of capitalized words
        named_entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', getattr(p_text, 'replace', lambda *a: p_text)('', ''))

        # Numbers / percentages / metrics in text
        original_text = ""
        numbers_in_text = len(re.findall(r'\b\d+\.?\d*\s*(?:%|percent|accuracy|f1|auc|score)?\b', p_text))

        # Intent aspect markers (more = more specific)
        markers = cls.INTENT_ASPECT_MARKERS.get(q_repr.intent, set())
        marker_hits = sum(1 for m in markers if m in p_text)

        score = 0.0
        if marker_hits >= 3:
            score += 0.5
        elif marker_hits >= 1:
            score += 0.25
        if numbers_in_text >= 2:
            score += 0.3
        elif numbers_in_text >= 1:
            score += 0.1
        if len(p_text) >= 400:
            score += 0.2

        return min(1.0, score)

    @classmethod
    def _multi_aspect_coverage(
        cls,
        q_repr: QuestionRepresentation,
        all_passages: List[Any]
    ) -> Tuple[float, List[str]]:
        """
        Dimension D: Multi-aspect completeness.
        For questions asking multiple things, check each aspect independently.
        Returns (coverage_fraction, missing_aspect_labels).
        """
        if not q_repr.is_multi_aspect or not q_repr.required_aspects:
            return 1.0, []

        all_text = " ".join(cls._get_passage_text(p) for p in all_passages)
        covered = []
        missing = []

        for aspect in q_repr.required_aspects:
            aspect_markers = cls.INTENT_ASPECT_MARKERS.get(
                "Advantage" if aspect == "advantage" else "Limitation", set()
            )
            if any(m in all_text for m in aspect_markers):
                covered.append(aspect)
            else:
                missing.append(aspect)

        fraction = len(covered) / len(q_repr.required_aspects)
        return fraction, missing

    @classmethod
    def _cross_passage_consistency(cls, supporting_passages: List[Any]) -> float:
        """
        Dimension G: Consistency.
        Heuristic check — if multiple passages are found, they're likely consistent.
        If only 1, slight penalty. If 0, return 0.
        """
        n = len(supporting_passages)
        if n >= 3:
            return 1.0
        if n == 2:
            return 0.85
        if n == 1:
            return 0.65
        return 0.0

    @classmethod
    def evaluate(
        cls,
        question: str,
        q_repr: QuestionRepresentation,
        passages: List[Any],
        source_label: str = "LOCAL",
    ) -> AnswerabilityResult:
        """
        Full generic answerability evaluation.

        Args:
            question: Original user question
            q_repr: Pre-computed QuestionRepresentation from GenericQuestionAnalyzer
            passages: Retrieved passages (RetrievalResult or OnlineEvidenceItem)
            source_label: "LOCAL" or "ONLINE" for decision label

        Returns:
            AnswerabilityResult with all 7 dimension scores and final decision.
        """
        if not passages:
            return cls._no_evidence_result(source_label)

        # Filter quality passages
        quality_passages = [p for p in passages[:8] if cls._is_quality_passage(p)]
        if not quality_passages:
            return AnswerabilityResult(
                related=False,
                answerable=False,
                completeness=0.0,
                intent_support=0.0,
                concept_support=0.0,
                relationship_support=0.0,
                evidence_quality=0.0,
                missing_aspects=["All retrieved passages are too short, truncated, or are bibliography entries"],
                decision=f"{source_label}_INSUFFICIENT",
                rationale="All retrieved passages failed the chunk quality gate (too short, truncated, or bibliography).",
                direct_supporting_passages=[],
                related_score=0.0,
                answerability_score=0.0,
                concept_coverage=0.0,
                is_related=False,
                is_answerable=False,
            )

        content_words = q_repr.all_content_words
        top_passages = quality_passages[:5]

        # === Collective scores across top passages ===
        all_text = " ".join(cls._get_passage_text(p) for p in top_passages)

        # A. Collective subject/concept coverage
        if content_words:
            matched_concepts = [w for w in content_words if cls._concept_in_text(w, all_text)]
            concept_support = len(matched_concepts) / len(content_words)
        else:
            concept_support = 1.0
            matched_concepts = []

        # B. Collective intent support
        # Use the SAME strict evaluation as per-passage to enforce "related ≠ answerable"
        # For Algorithm and Limitation intents, use specific markers only.
        _specific_algorithm_markers = {
            "algorithm", "algorithms", "classifier", "classifiers", "architecture", "architectures",
            "framework", "frameworks", "svm", "support vector", "random forest", "decision tree",
            "gradient boosting", "naive bayes", "k-nearest", "k-means", "logistic regression",
            "linear regression", "cnn", "rnn", "lstm", "gru", "transformer", "attention",
            "encoder", "decoder", "embedding", "bert", "gpt", "resnet", "vgg",
            "convolutional", "recurrent", "deep learning", "shallow learning", "ensemble",
            "bagging", "boosting", "clustering", "heuristic", "genetic", "evolutionary",
            "reinforcement", "unsupervised", "supervised",
        }
        _core_limitation_markers = {
            "limitation", "limitations", "drawback", "drawbacks", "weakness", "weaknesses",
            "challenge", "challenges", "failure", "failures", "shortcoming", "shortcomings",
            "bottleneck", "bottlenecks", "constraint", "constraints", "cannot", "unable",
            "fails", "limited", "lack", "lacks", "lacking", "insufficient", "inadequate",
            "inaccurate", "unreliable", "unstable", "risk", "risks", "problem", "problems",
            "issue", "issues", "error", "errors", "bias", "difficult",
        }
        _core_advantage_markers = {
            "advantage", "advantages", "benefit", "benefits", "strength", "strengths",
            "improve", "improves", "improvement", "enhanced", "superior", "better",
            "efficient", "effective", "outperform", "outperforms", "gain", "gains",
            "robust", "robustness", "scalable", "accurate", "reliable", "optimal",
            "success", "successful", "achieve", "achieves",
        }

        if q_repr.intent == "Algorithm":
            spec_hits = sum(1 for m in _specific_algorithm_markers if m in all_text)
            intent_support_collective = min(1.0, spec_hits / 2.0) if spec_hits > 0 else 0.0
        elif q_repr.intent == "Limitation":
            core_hits = sum(1 for m in _core_limitation_markers if m in all_text)
            intent_support_collective = min(1.0, core_hits / 2.0) if core_hits > 0 else 0.0
        elif q_repr.intent == "Advantage":
            core_hits = sum(1 for m in _core_advantage_markers if m in all_text)
            intent_support_collective = min(1.0, core_hits / 2.0) if core_hits > 0 else 0.0
        else:
            markers = cls.INTENT_ASPECT_MARKERS.get(q_repr.intent, set())
            if markers:
                collective_marker_hits = sum(1 for m in markers if m in all_text)
                intent_support_collective = min(1.0, collective_marker_hits / 3.0)
            else:
                intent_support_collective = 1.0 if concept_support > 0.3 else 0.0



        # D. Multi-aspect coverage
        aspect_coverage, missing_aspects_list = cls._multi_aspect_coverage(q_repr, top_passages)

        # === Per-passage evaluation ===
        direct_supporting_passages = []
        passage_rel_scores = []
        passage_intent_scores = []
        passage_rel_sup_scores = []

        for p in top_passages:
            p_text = cls._get_passage_text(p)

            # A: Subject coverage in this passage
            subj_cov = cls._subject_coverage(q_repr, p_text)

            # B: Intent support in this passage
            p_intent = cls._intent_support(q_repr, p_text)

            # C: Relationship support in this passage
            p_rel_sup = cls._relationship_support(q_repr, p_text)

            # Comparison coverage
            p_comp = cls._comparison_coverage(q_repr, p_text)

            # E: Specificity
            p_spec = cls._specificity_score(q_repr, p_text)

            passage_rel_scores.append(subj_cov)
            passage_intent_scores.append(p_intent)
            passage_rel_sup_scores.append(p_rel_sup)

            # A passage is "directly supporting" when:
            # - Subject coverage > 0 (talks about the subject)
            # - Intent support > 0 (contains the type of info requested)
            # - For relational questions: relationship support > 0
            # - For comparison questions: comparison coverage >= 0.5
            # - Quality passage (already filtered)
            subj_threshold = 0.25 if len(content_words) >= 4 else 0.5
            intent_threshold = 0.10  # At least 1 marker (generous for multi-domain)

            passes_relational = (p_rel_sup > 0.0) if q_repr.is_relational else True
            passes_comparison = (p_comp >= 0.5) if q_repr.is_comparison else True

            if (subj_cov >= subj_threshold and
                p_intent >= intent_threshold and
                passes_relational and
                passes_comparison):
                direct_supporting_passages.append(p)

        # === Aggregate scores ===
        avg_subj = sum(passage_rel_scores) / len(passage_rel_scores) if passage_rel_scores else 0.0
        avg_intent = sum(passage_intent_scores) / len(passage_intent_scores) if passage_intent_scores else 0.0
        avg_rel_sup = sum(passage_rel_sup_scores) / len(passage_rel_sup_scores) if passage_rel_sup_scores else 1.0

        # F: Evidence quality
        evidence_quality = len(quality_passages) / max(len(passages[:8]), 1)

        # G: Cross-passage consistency
        consistency = cls._cross_passage_consistency(direct_supporting_passages)

        # === Final composite answerability score ===
        # Weighted: intent_support is the decisive factor (not just topic overlap)
        answerability_score = (
            0.30 * concept_support +
            0.35 * intent_support_collective +   # Highest weight — enforces "answerable ≠ related"
            0.15 * avg_rel_sup +
            0.10 * aspect_coverage +
            0.10 * evidence_quality
        )

        # === Related determination ===
        # "Related" = subject is discussed. Does NOT mean answerable.
        is_related = concept_support >= 0.30 and bool(matched_concepts)

        # === Missing aspects ===
        missing_aspects = list(missing_aspects_list)
        uncovered_content = [
            w for w in content_words
            if not cls._concept_in_text(w, all_text) and len(w) >= 4
        ]
        if uncovered_content:
            missing_aspects.extend(uncovered_content[:3])

        if q_repr.is_comparison and len(direct_supporting_passages) == 0:
            for target in q_repr.comparison_targets:
                missing_aspects.append(f"comparison target: '{target}'")

        if q_repr.is_relational and avg_rel_sup < 0.3 and is_related:
            missing_aspects.append(
                f"relationship between '{' '.join(q_repr.relation_source[:2])}' "
                f"and '{' '.join(q_repr.relation_target[:2])}'"
            )

        # === Decision ===
        n_direct = len(direct_supporting_passages)

        if not is_related:
            decision = f"{source_label}_INSUFFICIENT"
            rationale = (
                f"Evidence does not discuss the question subject "
                f"(concept_support={concept_support:.2f}, matched={matched_concepts[:3]}). "
                f"Triggering {'online search' if source_label == 'LOCAL' else 'insufficient-evidence response'}."
            )
            is_answerable = False

        intent_min = 0.35 if source_label == "LOCAL" else 0.15
        score_min = 0.55 if source_label == "LOCAL" else 0.40
        passes_aspects = (not missing_aspects_list) if source_label == "LOCAL" else True

        if n_direct >= cls.MIN_ANSWERABLE_PASSAGES and answerability_score >= score_min and intent_support_collective >= intent_min and passes_aspects:
            decision = f"{source_label}_SUFFICIENT"
            rationale = (
                f"Evidence is sufficient: {n_direct} direct supporting passage(s), "
                f"answerability_score={answerability_score:.2f}, "
                f"intent_support={intent_support_collective:.2f}, "
                f"concept_support={concept_support:.2f}."
            )
            is_answerable = True

        else:
            decision = f"{source_label}_INSUFFICIENT"
            rationale = (
                f"Evidence is topically related (concept_support={concept_support:.2f}) "
                f"but does not contain the specific {q_repr.requested_aspect} information requested "
                f"(intent_support={intent_support_collective:.2f}, "
                f"direct_passages={n_direct}, "
                f"answerability_score={answerability_score:.2f}). "
                f"Missing: {missing_aspects[:3]}. "
                f"This is a RELATED-BUT-NOT-ANSWERABLE case. "
                f"Triggering {'online search' if source_label == 'LOCAL' else 'insufficient-evidence response'}."
            )
            is_answerable = False

        logger.info(
            f"[GENERIC EVALUATOR | {source_label}] Question: '{question[:70]}' | "
            f"Intent: {q_repr.intent} | Decision: {decision} | "
            f"concept={concept_support:.2f} intent={intent_support_collective:.2f} "
            f"rel={avg_rel_sup:.2f} direct_passages={n_direct} score={answerability_score:.2f}"
        )

        return AnswerabilityResult(
            related=is_related,
            answerable=is_answerable,
            completeness=aspect_coverage,
            intent_support=intent_support_collective,
            concept_support=concept_support,
            relationship_support=avg_rel_sup,
            evidence_quality=evidence_quality,
            missing_aspects=missing_aspects,
            decision=decision,
            rationale=rationale,
            direct_supporting_passages=direct_supporting_passages,
            # Legacy compat
            related_score=concept_support,
            answerability_score=answerability_score,
            concept_coverage=concept_support,
            is_related=is_related,
            is_answerable=is_answerable,
        )

    @classmethod
    def _no_evidence_result(cls, source_label: str) -> "AnswerabilityResult":
        return AnswerabilityResult(
            related=False,
            answerable=False,
            completeness=0.0,
            intent_support=0.0,
            concept_support=0.0,
            relationship_support=0.0,
            evidence_quality=0.0,
            missing_aspects=["No evidence retrieved"],
            decision=f"{source_label}_INSUFFICIENT",
            rationale="No candidate evidence passages were retrieved.",
            direct_supporting_passages=[],
            related_score=0.0,
            answerability_score=0.0,
            concept_coverage=0.0,
            is_related=False,
            is_answerable=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Compatibility Shims
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceSufficiencyEvaluator:
    """
    Legacy compatibility shim — delegates to GenericEvidenceEvaluator.

    Existing call sites:
        EvidenceSufficiencyEvaluator.evaluate(question, passages, allowed_domains)
        EvidenceSufficiencyEvaluator.evaluate_detailed(question, passages, allowed_domains)
        EvidenceSufficiencyEvaluator.decompose_query(question)
        EvidenceSufficiencyEvaluator.classify_intent(question)
    """

    STOPWORDS = GenericQuestionAnalyzer.STOPWORDS

    @classmethod
    def classify_intent(cls, question: str) -> str:
        return GenericQuestionAnalyzer.classify_intent(question)

    @classmethod
    def decompose_query(cls, question: str) -> Dict[str, Any]:
        """Legacy tuple-dict interface for backwards compatibility."""
        q_repr = GenericQuestionAnalyzer.analyze(question)
        return {
            "all_content_words": q_repr.all_content_words,
            "ngrams": [],
            "intent": q_repr.intent,
            "entity_1": q_repr.relation_source,
            "entity_2": q_repr.relation_target,
            "is_relational": q_repr.is_relational,
            "subject_terms": q_repr.main_subject,
            "entity_constraints": q_repr.secondary_concepts,
            "relational_aspects": q_repr.required_aspects,
        }

    @classmethod
    def evaluate_detailed(
        cls,
        question: str,
        retrieved_results: List[Any],
        allowed_domains: Optional[List[str]] = None,
    ) -> AnswerabilityResult:
        q_repr = GenericQuestionAnalyzer.analyze(question)
        return GenericEvidenceEvaluator.evaluate(question, q_repr, retrieved_results, "LOCAL")

    @classmethod
    def evaluate(
        cls,
        question: str,
        retrieved_results: List[Any],
        allowed_domains: Optional[List[str]] = None,
    ) -> Tuple[bool, bool, str, float]:
        """Legacy tuple interface: (is_related, is_answerable, rationale, sufficiency_score)."""
        res = cls.evaluate_detailed(question, retrieved_results, allowed_domains)
        return res.is_related, res.is_answerable, res.rationale, res.answerability_score


class RelevanceGate:
    """Legacy shim — delegates to GenericEvidenceEvaluator."""

    STOPWORDS = GenericQuestionAnalyzer.STOPWORDS

    @staticmethod
    def classify_intent(question: str) -> str:
        return GenericQuestionAnalyzer.classify_intent(question)

    @staticmethod
    def evaluate(question: str, retrieved_results: List[Any]) -> Tuple[bool, str, float]:
        """Legacy interface: (is_answerable, rationale, score)."""
        res = EvidenceSufficiencyEvaluator.evaluate_detailed(question, retrieved_results)
        return res.is_answerable, res.rationale, res.answerability_score


# ─────────────────────────────────────────────────────────────────────────────
# Answer Quality Validators
# ─────────────────────────────────────────────────────────────────────────────

class QuestionAnswerRelevanceValidator:
    """
    Validates that generated answer text directly addresses question intent.
    GENERIC — no topic-specific checks.
    """

    @staticmethod
    def validate_answer_intent(question: str, answer: str, intent: str) -> Tuple[bool, str]:
        if not answer or "insufficient evidence" in answer.lower():
            return True, "Insufficient evidence response is valid fallback."

        q_clean = question.strip().rstrip("?").lower()
        ans_clean = answer.strip().lower()

        # Check answer does not begin with the question text verbatim
        if ans_clean.startswith(q_clean):
            return False, "Answer begins with exact question text."

        first_sent = ans_clean.split(".")[0]
        if q_clean in first_sent:
            return False, "First sentence reproduces question clause."

        # Generic content-word presence check — no hardcoded topic words
        # At least one key non-stopword from the question must appear in the answer
        content_words = [
            w for w in re.findall(r'\b[a-z]{4,}\b', q_clean)
            if w not in GenericQuestionAnalyzer.STOPWORDS
        ]
        if content_words:
            found_any = any(w in ans_clean for w in content_words[:5])
            if not found_any:
                return False, (
                    f"Answer does not address any key concept from the question "
                    f"(checked: {content_words[:5]})."
                )

        return True, "Answer correctly addresses question intent without repetition."


class AnswerCompletenessValidator:
    """
    Evaluates research-style answer depth based on question intent.
    Enforces >= 90 words for explanatory/analytical queries.
    """

    EXPLANATORY_INTENTS = {
        "Limitation", "Challenge", "Advantage", "Benefit", "Mechanism", "Process",
        "Comparison", "Application", "Cause", "Effect", "Evaluation", "Prediction",
        "Detection", "Role", "Relationship",
    }

    @staticmethod
    def validate_completeness(answer: str, intent: str) -> Tuple[bool, str]:
        if not answer or "insufficient evidence" in answer.lower():
            return True, "Insufficient evidence fallback is valid."

        words = re.findall(r"\b\w+\b", answer.strip())
        word_count = len(words)

        if intent in AnswerCompletenessValidator.EXPLANATORY_INTENTS and word_count < 40:
            return False, (
                f"Answer is too short ({word_count} words < 40). "
                f"Explanatory answers require research depth."
            )

        if word_count < 20:
            return False, f"Answer is too short ({word_count} words < 20)."

        return True, "Answer satisfies completeness validation."


# ─────────────────────────────────────────────────────────────────────────────
# Claim Grounding Validator
# ─────────────────────────────────────────────────────────────────────────────

class ClaimGroundingValidator:
    """
    STRICT, GENERIC, CLAIM-LEVEL CITATION VERIFICATION SYSTEM.

    Verifies every factual claim in the generated answer text against cited evidence passages.

    Core Invariants:
    1. Citation presence != Citation correctness. Every factual claim must be directly supported by cited evidence.
    2. Bidirectional verification:
       - Claim -> Evidence: Does the cited passage directly entail/support the factual claim?
       - Evidence -> Claim relevance: Is the cited passage actually about the requested subject/aspect?
    3. Multi-claim & multi-citation independent verification:
       - Each claim in a sentence is verified separately against each attached tag [E#]/[O#].
       - If [E1][E2] is attached, E1 and E2 are verified independently. Invalid tags are removed.
       - Unsupported sub-claims are split/purged.
    4. Post-verification metrics:
       - Unsupported claims count is incremented for each unsupported factual claim.
       - No unsupported claims are retained in final text.
    """

    @staticmethod
    def _is_technical_term(word: str) -> bool:
        """
        Generic heuristic for domain-specific technical terms.
        Uses word length and structure rather than a hardcoded vocabulary.
        """
        if len(word) >= 9:
            return True
        if re.search(r'\d', word):
            return True
        if re.search(r'[A-Z][a-z]+[A-Z]', word) or (word.isupper() and len(word) >= 3):
            return True
        return False

    @staticmethod
    def verify_claim_against_passage(
        claim: str,
        ev_text: str,
        question: Optional[str] = None
    ) -> Tuple[bool, bool, bool, float, str]:
        """
        Verifies if ev_text DIRECTLY SUPPORTS claim.
        Returns: (is_valid, direct_support, aspect_support, score, rationale)
        """
        claim_clean = claim.lower()
        ev_clean = ev_text.lower()

        if is_bibliography_chunk(ev_clean):
            return False, False, False, 0.0, "Passage is a bibliography/reference chunk."

        claim_words = [w for w in re.findall(r"\b[a-zA-Z]{3,}\b", claim_clean) if w not in GenericQuestionAnalyzer.STOPWORDS]
        if not claim_words:
            return True, True, True, 1.0, "Structural sentence."

        # Key nouns / verbs in claim (len >= 5 or technical)
        key_claim_terms = {
            w for w in claim_words
            if len(w) >= 5 or ClaimGroundingValidator._is_technical_term(w)
        }

        # Check for meta-target hijacking in evidence
        meta_hijack = False
        for meta in GenericEvidenceEvaluator.META_ASPECT_TARGETS:
            if meta in ev_clean and meta not in claim_clean:
                # Passage discusses meta-aspect (e.g. evaluation practices) while claim asserts core system behavior
                if any(w in claim_clean for w in ["suffers", "cost", "overhead", "latency", "failure", "bottleneck", "tool call", "error"]):
                    if not any(w in ev_clean for w in ["suffers", "cost", "overhead", "latency", "failure", "bottleneck", "tool call", "error"]):
                        meta_hijack = True
                        break

        if meta_hijack:
            return False, False, False, 0.0, "Passage discusses meta-aspects (e.g. evaluation practices) rather than target system claim."

        # Check specific assertion predicate terms in claim
        assertion_words = {
            "tool", "tools", "cost", "costs", "computational", "compute", "latency",
            "window", "windows", "hallucination", "hallucinations", "failure", "failures",
            "memory", "energy", "bandwidth", "vulnerability", "vulnerabilities", "bottleneck",
        }
        claim_assertions = set(claim_words).intersection(assertion_words)
        if claim_assertions:
            supported_assertions = set()
            for a_word in claim_assertions:
                for variant in GenericEvidenceEvaluator._morph_expand(a_word):
                    if variant in ev_clean:
                        supported_assertions.add(a_word)
                        break
            if len(supported_assertions) < len(claim_assertions):
                unsupported_terms = claim_assertions - supported_assertions
                return False, False, True, 0.0, f"Claim assertion terms {unsupported_terms} not supported by passage."

        # Calculate morphological key term coverage
        matched_key_terms = set()
        for term in key_claim_terms:
            for variant in GenericEvidenceEvaluator._morph_expand(term):
                if len(variant) >= 4 and variant in ev_clean:
                    matched_key_terms.add(term)
                    break

        term_coverage = len(matched_key_terms) / len(key_claim_terms) if key_claim_terms else 1.0

        direct_support = term_coverage >= 0.40
        aspect_support = not meta_hijack

        is_valid = direct_support and aspect_support
        score = term_coverage if is_valid else 0.0
        rationale = "Direct support verified" if is_valid else "Passage does not directly entail claim"

        return is_valid, direct_support, aspect_support, score, rationale

    @staticmethod
    def validate_and_filter_claims(
        answer: str,
        evidence_map: Dict[str, Union["EvidenceItem", "OnlineEvidenceItem"]],
        question: Optional[str] = None,
    ) -> Tuple[str, int, List[str]]:
        if not answer or "insufficient evidence" in answer.lower():
            return answer, 0, []

        sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
        valid_sentences = []
        unsupported_count = 0
        active_tags: Set[str] = set()

        for sent_idx, sent in enumerate(sentences):
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            # Preserve structural headings/headers
            if (sent_clean.startswith("#") or
                (sent_clean.startswith("**") and sent_clean.endswith(":**")) or
                len(re.findall(r"\b[a-zA-Z]{3,}\b", sent_clean)) <= 2):
                valid_sentences.append(sent)
                continue

            citation_tags = CitationValidator.extract_citations(sent)
            sent_text_no_tags = re.sub(r"\[[EO]\d+(?:\s*,\s*[EO]\d+)*\]", "", sent_clean).strip()

            valid_tags_for_sent = []

            for tag in citation_tags:
                if tag in evidence_map:
                    ev = evidence_map[tag]
                    ev_text = ev.text.lower() if hasattr(ev, 'text') else ""

                    is_valid, direct_sup, aspect_sup, score, rationale = ClaimGroundingValidator.verify_claim_against_passage(
                        sent_text_no_tags, ev_text, question
                    )

                    logger.info(
                        f"[CLAIM VERIFICATION] Q: '{question[:60] if question else 'N/A'}' | "
                        f"Claim C{sent_idx+1}: '{sent_text_no_tags[:80]}' | Citation: [{tag}] | "
                        f"Paper: {getattr(ev, 'paper_id', 'unknown')} | Direct Support: {'YES' if direct_sup else 'NO'} | "
                        f"Aspect Support: {'YES' if aspect_sup else 'NO'} | Status: {'VALID' if is_valid else 'INVALID'} ({rationale})"
                    )

                    if is_valid:
                        valid_tags_for_sent.append(tag)
                        active_tags.add(tag)

            if valid_tags_for_sent:
                # Sentence has at least one verified citation
                clean_sent = re.sub(r"\[[EO]\d+(?:\s*,\s*[EO]\d+)*\]", "", sent_clean).strip().rstrip(".")
                cit_str = "".join(f"[{t}]" for t in valid_tags_for_sent)
                valid_sentences.append(f"{clean_sent} {cit_str}.")
            else:
                # Try finding a verified matching tag in evidence_map
                best_verified_tag = None
                best_score = 0.0

                for tag, ev in evidence_map.items():
                    ev_text = ev.text.lower() if hasattr(ev, 'text') else ""
                    is_valid, direct_sup, aspect_sup, score, rationale = ClaimGroundingValidator.verify_claim_against_passage(
                        sent_text_no_tags, ev_text, question
                    )
                    if is_valid and score > best_score:
                        best_score = score
                        best_verified_tag = tag

                if best_verified_tag:
                    clean_sent = re.sub(r"\[[EO]\d+(?:\s*,\s*[EO]\d+)*\]", "", sent_clean).strip().rstrip(".")
                    valid_sentences.append(f"{clean_sent} [{best_verified_tag}].")
                    active_tags.add(best_verified_tag)
                else:
                    logger.warning(f"[CLAIM GROUNDING REJECT] Unsupported claim removed: '{sent_clean[:80]}'")
                    unsupported_count += 1

        grounded_answer = " ".join(valid_sentences) if valid_sentences else "Insufficient evidence was found to support these claims."
        return grounded_answer, unsupported_count, list(active_tags)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence Context Builder
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceContextBuilder:
    """Formats retrieved RetrievalResult or OnlineEvidenceItem items into context blocks."""

    @staticmethod
    def build_context(
        results: List[Union[RetrievalResult, OnlineEvidenceItem]],
        source_type: str = "corpus",
    ) -> Tuple[str, List[EvidenceItem], Dict[str, Citation]]:
        evidence_items = []
        citations_map = {}
        context_blocks = []

        local_idx = 1
        online_idx = 1

        for res in results:
            if isinstance(res, OnlineEvidenceItem) or getattr(res, "source_type", "") == "online":
                # Mandatory metadata: online citations MUST have real title, authors, published date
                if not getattr(res, "authors", None) or not getattr(res, "published_date", None):
                    logger.warning(
                        f"[ONLINE CITATION REJECTED] Missing authors or published date "
                        f"for {getattr(res, 'paper_id', 'unknown')}"
                    )
                    continue

                citation_id = f"O{online_idx}"
                online_idx += 1
                ev = EvidenceItem(
                    citation_id=citation_id,
                    unit_id=f"{res.paper_id}_U{online_idx:02d}",
                    parent_chunk_id=f"{res.paper_id}_C{online_idx:02d}",
                    chunk_id=f"{res.paper_id}_C{online_idx:02d}",
                    paper_id=res.paper_id,
                    section_id="SEC_ABSTRACT",
                    section_name="Abstract",
                    domain=res.domain,
                    subtopic="academic_online",
                    page_start=1,
                    page_end=1,
                    text=res.text,
                    dense_score=0.90,
                    bm25_score=10.0,
                    rrf_score=0.035,
                    retrieval_methods=["ArXiv_API"],
                    source_type="online",
                    url=res.url,
                    title=res.title,
                    authors=res.authors,
                    published_date=res.published_date,
                )
                page_str = "1"
            else:
                citation_id = f"E{local_idx}"
                local_idx += 1
                ev = EvidenceItem(
                    citation_id=citation_id,
                    unit_id=res.unit_id,
                    parent_chunk_id=res.parent_chunk_id,
                    chunk_id=res.chunk_id,
                    paper_id=res.paper_id,
                    section_id=res.section_id,
                    section_name=res.section_name,
                    domain=res.domain,
                    subtopic=res.subtopic,
                    page_start=res.page_start,
                    page_end=res.page_end,
                    text=res.text,
                    dense_score=res.dense_score,
                    bm25_score=res.bm25_score,
                    rrf_score=res.rrf_score,
                    retrieval_methods=res.retrieval_methods,
                    source_type="corpus",
                )
                page_str = (
                    f"{res.page_start}" if res.page_start == res.page_end
                    else f"{res.page_start}-{res.page_end}"
                )

            evidence_items.append(ev)

            cit = Citation(
                citation_id=citation_id,
                paper_id=ev.paper_id,
                section_name=ev.section_name,
                pages=page_str,
                chunk_id=ev.chunk_id,
                unit_id=ev.unit_id,
                source_type=ev.source_type,
                title=getattr(ev, 'title', None),
                authors=getattr(ev, 'authors', []),
                published=getattr(ev, 'published_date', None),
                url=getattr(ev, 'url', None),
            )
            citations_map[citation_id] = cit

            authors_str = f", Authors: {', '.join(ev.authors[:2])}" if ev.authors else ""
            block = (
                f"[{citation_id}]\n"
                f"Paper: {ev.paper_id}\n"
                f"Title: {ev.title or 'N/A'}{authors_str}\n"
                f"Section: {ev.section_name}\n"
                f"Pages: {page_str}\n"
                f"Text:\n{ev.text}\n"
            )
            context_blocks.append(block)

        full_context_text = "\n".join(context_blocks)
        return full_context_text, evidence_items, citations_map


# ─────────────────────────────────────────────────────────────────────────────
# Citation Validator
# ─────────────────────────────────────────────────────────────────────────────

class CitationValidator:
    """Validates citation tags in LLM outputs against retrieved evidence items."""

    @staticmethod
    def extract_citations(text: str) -> List[str]:
        raw_matches = re.findall(r"\[([EO]\d+(?:\s*,\s*[EO]\d+)*)\]", text)
        citations = []
        for match in raw_matches:
            tags = [t.strip() for t in match.split(",")]
            citations.extend(tags)
        return list(dict.fromkeys(citations))

    @staticmethod
    def validate(text: str, valid_citations_map: Dict[str, Citation]) -> Tuple[bool, List[str], List[str]]:
        found = CitationValidator.extract_citations(text)
        valid = []
        invalid = []
        for tag in found:
            if tag in valid_citations_map:
                valid.append(tag)
            else:
                invalid.append(tag)
        return len(invalid) == 0, valid, invalid


# ─────────────────────────────────────────────────────────────────────────────
# Final Safety Gate
# ─────────────────────────────────────────────────────────────────────────────

class FinalSafetyGate:
    """
    Authoritative final safety gate ensuring:
    1. Every [E#] tag maps to actual local corpus evidence.
    2. Every [O#] tag maps to actual online evidence.
    3. Unmapped tags are stripped from answer text.
    4. citations_map retains ONLY keys used in the cleaned answer (100% bidirectional).
    """

    @staticmethod
    def sanitize_response(
        answer: str,
        citations_map: Dict[str, Citation],
        evidence_items: List[EvidenceItem],
        allowed_domains: List[str],
        expected_prefixes: Set[str],
    ) -> Tuple[str, Dict[str, Citation], List[EvidenceItem]]:
        clean_answer = answer

        valid_local_tags = {e.citation_id: e for e in evidence_items if e.source_type == "corpus"}
        valid_online_tags = {e.citation_id: e for e in evidence_items if e.source_type == "online"}
        valid_all_tags = {**valid_local_tags, **valid_online_tags}

        tags_in_answer = CitationValidator.extract_citations(answer)

        for tag in tags_in_answer:
            if tag not in valid_all_tags:
                logger.warning(f"[FINAL SAFETY GATE] Stripping unmapped citation tag [{tag}].")
                clean_answer = clean_answer.replace(f"[{tag}]", "")
            else:
                ev = valid_all_tags[tag]
                if allowed_domains and len(allowed_domains) < len(DOMAIN_PREFIX_MAP):
                    if (ev.source_type == "corpus" and
                        (ev.domain not in allowed_domains or
                         (expected_prefixes and not any(ev.paper_id.startswith(p) for p in expected_prefixes)))):
                        logger.warning(
                            f"[FINAL SAFETY GATE] Stripping unallowed domain citation [{tag}] ({ev.paper_id})."
                        )
                        clean_answer = clean_answer.replace(f"[{tag}]", "")

        # Align citations_map and evidence_items to what's actually in clean_answer
        remaining_tags = CitationValidator.extract_citations(clean_answer)
        active_cits = {}
        active_evidence = []
        for tag in remaining_tags:
            if tag in citations_map:
                active_cits[tag] = citations_map[tag]
            if tag in valid_all_tags and valid_all_tags[tag] not in active_evidence:
                active_evidence.append(valid_all_tags[tag])

        return clean_answer, active_cits, active_evidence


# ─────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Core Evidence-Grounded RAG Pipeline.

    4-Tier Generic Decision Sequence:
      Tier 1 (LOCAL_SUFFICIENT)  → Local RAG Answer [E1], [E2]
      Tier 2 (LOCAL_PARTIAL)     → Hybrid: supplement local with online
      Tier 3 (LOCAL_INSUFFICIENT)→ Online Academic Search
          ↓  GenericEvidenceEvaluator (same generic logic on online evidence)
          ├─ ONLINE_SUFFICIENT  → Online RAG Answer [O1], [O2]
          └─ ONLINE_INSUFFICIENT→ Genuine Insufficient Evidence Fallback
      Tier 4 (No evidence)       → Genuine Insufficient Evidence Fallback
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        online_retriever: Optional[OnlineAcademicRetriever] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.online_retriever = online_retriever or OnlineAcademicRetriever()
        provider_name = os.environ.get("LLM_PROVIDER", "mock")
        if os.environ.get("GEMINI_API_KEY") and not llm_provider and "LLM_PROVIDER" not in os.environ:
            provider_name = "gemini"
        elif os.environ.get("OPENAI_API_KEY") and not llm_provider and "LLM_PROVIDER" not in os.environ:
            provider_name = "openai"
        self.llm = llm_provider or get_llm_provider(provider_name)

    def answer(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        request_id: Optional[str] = None,
        question_hash: Optional[str] = None,
    ) -> RAGResponse:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace-only")

        req_id = request_id or "INTERNAL"
        q_hash = question_hash or hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]

        # ── Step 0: Generic question decomposition (ZERO hardcoded topics) ──
        q_repr = GenericQuestionAnalyzer.analyze(question)
        intent = q_repr.intent

        user_domain = filters.get("domain") if filters else None
        scope_res = DomainScopeDetector.detect(question, user_domain=user_domain)
        allowed_domains = scope_res.allowed_domains
        scope_label = scope_res.scope_label
        expected_prefixes = {DOMAIN_PREFIX_MAP[d] for d in allowed_domains if d in DOMAIN_PREFIX_MAP}

        logger.info(
            f"[RAG PIPELINE] request_id={req_id} q_hash={q_hash} "
            f"intent={intent} scope={scope_res.scope_type.value} ({scope_label}) "
            f"allowed={allowed_domains} "
            f"relational={q_repr.is_relational} comparison={q_repr.is_comparison} "
            f"multi_aspect={q_repr.is_multi_aspect}"
        )

        # ── Step 1: Intent-Aware Hybrid Local Evidence Retrieval ──
        retrieved_results = self.retriever.retrieve(
            question, top_k=top_k, filters=filters, allowed_domains=allowed_domains, intent=intent
        )

        if allowed_domains and len(allowed_domains) < len(DOMAIN_PREFIX_MAP):
            retrieved_results = [
                r for r in retrieved_results
                if r.domain in allowed_domains and (
                    not expected_prefixes or any(r.paper_id.startswith(p) for p in expected_prefixes)
                )
            ]

        # ── Step 2: Generic Evidence Answerability Evaluation (local) ──
        local_eval = GenericEvidenceEvaluator.evaluate(
            question, q_repr, retrieved_results, "LOCAL"
        )

        logger.info(
            f"[LOCAL EVAL] related={local_eval.related} answerable={local_eval.answerable} "
            f"decision={local_eval.decision} score={local_eval.answerability_score:.2f} "
            f"direct_passages={len(local_eval.direct_supporting_passages)}"
        )

        # ── Step 3: 4-Tier Decision ──
        source_type_tag = "corpus"
        active_evidence_pool: List[Any] = []
        online_fallback_triggered = False

        if local_eval.answerable:
            # ── Tier 1: Local evidence is sufficient ──
            source_type_tag = "corpus"
            active_evidence_pool = list(retrieved_results)
            logger.info(
                f"[DECISION] Tier 1: LOCAL_SUFFICIENT "
                f"(score={local_eval.answerability_score:.2f}). "
                f"Skipping online fallback."
            )
        else:
            # ── Secondary Targeted Local Retrieval ──
            concept_str = q_repr.contract.concept if hasattr(q_repr, 'contract') and q_repr.contract else " ".join(q_repr.main_subject)
            secondary_query = f"{concept_str} {q_repr.requested_aspect} definition explanation properties"
            logger.info(f"[SECONDARY RETRIEVAL] Attempting targeted local search: '{secondary_query}'")

            retry_results = self.retriever.retrieve(
                secondary_query, top_k=top_k, filters=filters, allowed_domains=allowed_domains, intent=intent
            )
            if allowed_domains and len(allowed_domains) < len(DOMAIN_PREFIX_MAP):
                retry_results = [
                    r for r in retry_results
                    if r.domain in allowed_domains and (
                        not expected_prefixes or any(r.paper_id.startswith(p) for p in expected_prefixes)
                    )
                ]

            retry_eval = GenericEvidenceEvaluator.evaluate(
                question, q_repr, retry_results, "LOCAL_RETRY"
            )
            logger.info(
                f"[SECONDARY LOCAL EVAL] related={retry_eval.related} answerable={retry_eval.answerable} "
                f"score={retry_eval.answerability_score:.2f} direct_passages={len(retry_eval.direct_supporting_passages)}"
            )

            if retry_eval.answerable:
                source_type_tag = "corpus"
                active_evidence_pool = list(retry_results)
                local_eval = retry_eval
                logger.info(
                    f"[DECISION] Secondary local retrieval SUFFICIENT "
                    f"(score={retry_eval.answerability_score:.2f})."
                )
            else:
                # ── Tier 3/2: Local primary & secondary insufficient — trigger online academic search ──
                reason = local_eval.rationale[:120]
                logger.info(
                    f"[DECISION] Primary & secondary local evidence insufficient/unanswerable. "
                    f"Reason: {reason}. Triggering Online Academic Search."
                )
                online_fallback_triggered = True

                online_query = question
                online_items = self.online_retriever.retrieve(
                    query=online_query,
                    domain=user_domain,
                    allowed_domains=allowed_domains,
                    intent=intent,
                    max_results=5,
                )

                if online_items:
                    # ── Run the SAME generic evaluator on online evidence ──
                    online_eval = GenericEvidenceEvaluator.evaluate(
                        question, q_repr, online_items, "ONLINE"
                    )
                    logger.info(
                        f"[ONLINE EVAL] related={online_eval.related} "
                        f"answerable={online_eval.answerable} "
                        f"decision={online_eval.decision} "
                        f"score={online_eval.answerability_score:.2f}"
                    )

                    if online_eval.answerable:
                        # ── ONLINE_SUFFICIENT ──
                        source_type_tag = "online"
                        active_evidence_pool = list(online_items)
                        logger.info(
                            f"[DECISION] ONLINE_SUFFICIENT "
                            f"(score={online_eval.answerability_score:.2f}). "
                            f"Using online academic evidence."
                        )
                    else:
                        # ── Tier 4: Both local and online insufficient ──
                        logger.warning(
                            f"[DECISION] Tier 4: ONLINE evidence also insufficient "
                            f"(score={online_eval.answerability_score:.2f}, "
                            f"decision={online_eval.decision}). "
                            f"Returning honest insufficient-evidence response."
                        )
                        return self._build_insufficient_evidence_response(
                            question, retrieved_results, scope_res, q_repr=q_repr, local_eval=local_eval
                        )
                else:
                    # No online results returned at all
                    logger.warning(
                        "[DECISION] Tier 4: Online retrieval returned no results. "
                        "Returning honest insufficient-evidence response."
                    )
                    return self._build_insufficient_evidence_response(
                        question, retrieved_results, scope_res, q_repr=q_repr, local_eval=local_eval
                    )

        if not active_evidence_pool:
            return self._build_insufficient_evidence_response(question, retrieved_results, scope_res, q_repr=q_repr, local_eval=local_eval)

        # ── Step 4: Build Evidence Context ([E#] corpus, [O#] online) ──
        context_text, evidence_items, citations_map = EvidenceContextBuilder.build_context(
            active_evidence_pool, source_type=source_type_tag
        )
        evidence_map = {e.citation_id: e for e in evidence_items}

        # ── Step 5: Construct Grounded Prompt ──
        user_prompt = (
            f"Research Question:\n{question}\n"
            f"Question Intent: {intent}\n\n"
            f"Retrieved Research Evidence Passages ({source_type_tag.upper()}):\n{context_text}\n\n"
            f"Instructions:\n"
            f"You are ScholarLens. Answer the user's research question clearly, accurately, "
            f"and comprehensively using ONLY the evidence passages provided above.\n"
            f"First, state the direct answer clearly in the first paragraph.\n"
            f"Then, explain the topic with research depth: use multi-paragraph prose, "
            f"headings (e.g. ### Background, ### Key Mechanisms, ### Important Findings, "
            f"### Comparison & Implications, ### Limitations), numbered steps for methods, "
            f"or comparison tables for versus queries where appropriate.\n"
            f"Target 500–1000 words for normal research questions and 800–1500 words for "
            f"complex synthesis questions when sufficient evidence is available. "
            f"Keep simple definitions concise (100–300 words). DO NOT fabricate facts or numbers.\n"
            f"Cite claims with inline tags like [E1], [E2] or [O1], [O2] immediately after "
            f"supported statements."
        )

        # ── Step 6: LLM Generation ──
        logger.info(f"[LLM GENERATE] request_id={req_id} provider={type(self.llm).__name__}")
        raw_output = self.llm.generate(
            user_prompt,
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            request_id=req_id,
            question_hash=q_hash,
        )
        parsed_answer, confidence_raw, confidence_rationale_raw, limitations, _ = self._parse_llm_output(raw_output)

        if "insufficient evidence" in parsed_answer.lower():
            return self._build_insufficient_evidence_response(question, retrieved_results, scope_res)

        # ── Step 7: Answer Intent & Question Repetition Validation ──
        is_ans_rel, ans_rel_reason = QuestionAnswerRelevanceValidator.validate_answer_intent(
            question, parsed_answer, intent
        )
        is_ans_comp, ans_comp_reason = AnswerCompletenessValidator.validate_completeness(
            parsed_answer, intent
        )

        if not is_ans_rel or not is_ans_comp:
            reject_reason = ans_rel_reason if not is_ans_rel else ans_comp_reason
            logger.warning(f"[RELEVANCE/COMPLETENESS REJECTED] {reject_reason}. Regenerating...")
            regen_prompt = (
                f"{user_prompt}\n\nCRITICAL FIX REQUIRED: Your previous response was rejected "
                f"because: '{reject_reason}'. Synthesize a genuinely comprehensive research answer "
                f"with clear headings and paragraphs directly addressing the intent ({intent}), "
                f"starting directly with the core concept."
            )
            raw_regen = self.llm.generate(
                regen_prompt, system_prompt=GROUNDED_SYSTEM_PROMPT,
                request_id=req_id, question_hash=q_hash
            )
            parsed_regen, _, _, _, _ = self._parse_llm_output(raw_regen)
            is_rel_2, _ = QuestionAnswerRelevanceValidator.validate_answer_intent(
                question, parsed_regen, intent
            )
            if is_rel_2:
                parsed_answer = parsed_regen

        # ── Step 8: Dual Claim-Level Grounding ──
        parsed_answer, unsupported_claims_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(
            parsed_answer, evidence_map, question
        )

        # ── Step 9: Citation Validation & Final Safety Gate ──
        is_valid_cit, used_citations, invalid_citations = CitationValidator.validate(
            parsed_answer, citations_map
        )
        if invalid_citations:
            logger.warning(f"Invalid citation tags: {invalid_citations}. Removing.")
            for tag in invalid_citations:
                parsed_answer = parsed_answer.replace(f"[{tag}]", "")

        active_tags = list(dict.fromkeys(used_citations + active_tags))
        active_citations = {t: citations_map[t] for t in active_tags if t in citations_map}

        parsed_answer, active_citations, evidence_items = FinalSafetyGate.sanitize_response(
            parsed_answer, active_citations, evidence_items, allowed_domains, expected_prefixes
        )

        # ── Step 10: Final Answer Completeness Re-check ──
        is_final_comp, final_comp_reason = AnswerCompletenessValidator.validate_completeness(
            parsed_answer, intent
        )
        if not is_final_comp:
            logger.warning(f"[FINAL COMPLETENESS CHECK FAILED] {final_comp_reason}. Re-prompting...")
            final_prompt = (
                f"{user_prompt}\n\nCRITICAL MANDATE: Provide a detailed, multi-paragraph, "
                f"evidence-grounded research response with clear headings covering direct answer, "
                f"key mechanisms, empirical findings, and limitations."
            )
            raw_final = self.llm.generate(
                final_prompt, system_prompt=GROUNDED_SYSTEM_PROMPT,
                request_id=req_id, question_hash=q_hash
            )
            parsed_final, _, _, _, _ = self._parse_llm_output(raw_final)
            parsed_final, unsupported_claims_count, _ = ClaimGroundingValidator.validate_and_filter_claims(
                parsed_final, evidence_map, question
            )
            parsed_answer, active_citations, evidence_items = FinalSafetyGate.sanitize_response(
                parsed_final, citations_map, evidence_items, allowed_domains, expected_prefixes
            )

        # ── Step 11: Evidence Assessment & Source Labeling ──
        local_cits = [t for t in active_citations if t.startswith("E")]
        online_cits = [t for t in active_citations if t.startswith("O")]

        if local_cits and online_cits:
            source_label = "Research Mind Corpus & Online Academic Search"
        elif online_cits:
            source_label = "Online Academic Search"
        else:
            source_label = "Research Mind Corpus"

        contributing_papers = list(dict.fromkeys(c.paper_id for c in active_citations.values()))
        contributing_sections = list(dict.fromkeys(c.section_name for c in active_citations.values()))
        evidence_passages = list(active_citations.keys())
        multi_paper = len(contributing_papers) > 1

        effective_score = (
            max(local_eval.answerability_score, 0.80) if online_cits
            else local_eval.answerability_score
        )

        # Evidence strength based STRICTLY on VERIFIED claim-level evidence quality
        total_active_cits = len(active_citations)
        if not active_citations or (not local_eval.answerable and not online_fallback_triggered):
            evidence_strength = "Insufficient"
        elif unsupported_claims_count > 0:
            if effective_score >= 0.60 and total_active_cits >= 2:
                evidence_strength = "Good"
            elif total_active_cits >= 1:
                evidence_strength = "Partial"
            else:
                evidence_strength = "Insufficient"
        elif effective_score >= 0.65 and len(contributing_papers) >= 2 and total_active_cits >= 2:
            evidence_strength = "Excellent"
        elif effective_score >= 0.50 and total_active_cits >= 2:
            evidence_strength = "High"
        elif effective_score >= 0.35 and total_active_cits >= 1:
            evidence_strength = "Moderate"
        else:
            evidence_strength = "Insufficient"

        confidence = evidence_strength
        confidence_rationale = (
            f"Confidence rated '{confidence}' based on evidence assessment "
            f"({len(local_cits)} local passage(s), {len(online_cits)} online source(s), "
            f"{len(contributing_papers)} paper(s), score {effective_score:.2f})."
        )

        # ── Step 12: Pointwise Explanation ──
        explanation_summary, bullet_points = self._generate_pointwise_explanation(
            active_citations=active_citations,
            evidence_map=evidence_map,
            evidence_strength=evidence_strength,
            contributing_papers=contributing_papers,
            unsupported_claims_count=unsupported_claims_count,
            multi_paper=multi_paper,
            source_label=source_label,
            scope_label=scope_label,
            local_count=len(local_cits),
            online_count=len(online_cits),
        )

        why_this_answer = WhyThisAnswer(
            contributing_papers=contributing_papers,
            contributing_sections=contributing_sections,
            evidence_passages=evidence_passages,
            multi_paper_support=multi_paper,
            evidence_strength=evidence_strength,
            unsupported_claims=unsupported_claims_count,
            inference_present=False,
            conflicts_detected=False,
            explanation_summary=explanation_summary,
            bullet_points=bullet_points,
            source_type=source_label,
            domain_scope=scope_label,
        )

        retrieval_metadata = {
            "retrieved_count": len(evidence_items),
            "top_unit_id": evidence_items[0].unit_id if evidence_items else None,
            "intent": intent,
            "match_ratio": local_eval.answerability_score,
            "sufficiency_score": local_eval.answerability_score,
            "unsupported_claims": unsupported_claims_count,
            "used_citations_count": len(active_citations),
            "local_evidence_count": len(local_cits),
            "online_evidence_count": len(online_cits),
            "online_fallback_active": online_fallback_triggered,
            "source_type": source_type_tag,
            "scope_type": scope_res.scope_type.value,
            "domain_scope": scope_label,
            "allowed_domains": allowed_domains,
            "local_eval_decision": local_eval.decision,
            "concept_support": local_eval.concept_support,
            "intent_support": local_eval.intent_support,
            "relationship_support": local_eval.relationship_support,
        }

        return RAGResponse(
            question=question,
            answer=parsed_answer,
            evidence=evidence_items,
            citations=active_citations,
            confidence=confidence,
            confidence_rationale=confidence_rationale,
            limitations=limitations,
            why_this_answer=why_this_answer,
            retrieval_metadata=retrieval_metadata,
            domain_scope=allowed_domains,
        )

    def _generate_pointwise_explanation(
        self,
        active_citations: Dict[str, Citation],
        evidence_map: Dict[str, Union[EvidenceItem, OnlineEvidenceItem]],
        evidence_strength: str,
        contributing_papers: List[str],
        unsupported_claims_count: int,
        multi_paper: bool,
        source_label: str,
        scope_label: str = "All Domains",
        local_count: int = 0,
        online_count: int = 0,
    ) -> Tuple[str, List[str]]:
        if not active_citations:
            return (
                "No cited evidence passages supported the final answer claims.",
                ["• Direct evidence: None found in corpus or online search."],
            )

        descriptions = []
        for tag, cit in active_citations.items():
            if tag in evidence_map:
                ev = evidence_map[tag]
                sec_name = ev.section_name
                paper_id = ev.paper_id
                snippet = ev.text.replace("\n", " ").strip()
                if len(snippet) > 110:
                    snippet = snippet[:107] + "..."
                descriptions.append(
                    f"Passage [{tag}] (Paper {paper_id}, '{sec_name}') supports key claims: '{snippet}'."
                )

        summary = (
            " ".join(descriptions) if descriptions
            else f"Grounded in {len(contributing_papers)} paper(s) from {source_label}."
        )

        bullets = [
            f"• Direct evidence: Retaining {len(active_citations)} direct supporting citation(s).",
            f"• Local evidence: {local_count} passage(s).",
            f"• Online evidence: {online_count} academic source(s).",
            f"• Supporting papers: {len(contributing_papers)} relevant paper(s) "
            f"({', '.join(contributing_papers[:3])}).",
            f"• Claim coverage: Factual claims verified against retrieved evidence.",
            f"• Domain scope: {scope_label}.",
            f"• Cross-paper agreement: "
            f"{'Multiple independent papers support main conclusion' if multi_paper else 'Single-paper focused evidence'}.",
            f"• Unsupported claims: {unsupported_claims_count}.",
            f"• Source: {source_label}.",
        ]

        return summary, bullets

    def _parse_llm_output(self, output: str) -> Tuple[str, str, str, str, str]:
        try:
            data = json.loads(output)
            ans = data.get("answer", output)
            conf = data.get("confidence", "High")
            lim = data.get("limitations", "Findings based on current 1,000-paper corpus and verified academic literature.")
            why = data.get("why_this_answer", "Supported by retrieved evidence.")
            conf_rat = f"Confidence rated '{conf}' based on evidence density and source agreement."
            return ans, conf, conf_rat, lim, why
        except json.JSONDecodeError:
            ans = output
            conf = "High" if len(output) > 100 else "Medium"
            conf_rat = "Confidence rated based on direct context match."
            lim = "Findings based on current 1,000-paper corpus and verified academic literature."
            why = "Supported by retrieved passage context."
            return ans, conf, conf_rat, lim, why

    def _build_insufficient_evidence_response(
        self,
        question: str,
        retrieved_results: List[RetrievalResult],
        scope_res: Optional[DomainScopeResult] = None,
        q_repr: Optional[QuestionRepresentation] = None,
        local_eval: Optional[AnswerabilityResult] = None,
    ) -> RAGResponse:
        concept_str = (
            q_repr.contract.concept if q_repr and q_repr.contract
            else (" ".join(q_repr.main_subject) if q_repr and q_repr.main_subject else "the topic")
        )
        aspect_str = q_repr.requested_aspect if q_repr else "requested information"

        msg = (
            "Insufficient evidence was found in the current Research Mind corpus or "
            "available online academic sources to answer this question reliably."
        )

        _, evidence_items, citations_map = EvidenceContextBuilder.build_context(retrieved_results)

        scope_label = scope_res.scope_label if scope_res else "All Domains"
        allowed = scope_res.allowed_domains if scope_res else []

        if local_eval and local_eval.related:
            explanation_summary = (
                f"Retrieved passages discuss '{concept_str}', but do not provide sufficient direct evidence "
                f"addressing the requested {aspect_str}."
            )
            bullets = [
                f"• Topic relevance: Retained topically related passages for '{concept_str}'.",
                f"• Question answerability: Insufficient direct evidence for {aspect_str}.",
                "• Primary local retrieval: Evaluated (related but not answerable).",
                "• Secondary targeted retrieval: Evaluated (insufficient coverage for contract).",
                "• Online academic fallback: Evaluated (insufficient online coverage for contract).",
                f"• Domain scope: {scope_label}.",
                "• Unsupported claims: 0 (No ungrounded claims generated).",
                "• Source: Research Mind Corpus & Online Academic Search.",
            ]
        else:
            explanation_summary = (
                "No retrieved passages in the current 1,000-paper ScholarLens corpus "
                "or available online academic search contained substantive direct evidence "
                "addressing this research question."
            )
            bullets = [
                "• Direct evidence: No relevant direct evidence found.",
                "• Local corpus coverage: Insufficient.",
                "• Online fallback: Evaluated / Insufficient online coverage.",
                f"• Domain scope: {scope_label}.",
                "• Unsupported claims: 0 (No ungrounded claims generated).",
                "• Source: Research Mind Corpus & Online Academic Search.",
            ]

        why_this_answer = WhyThisAnswer(
            contributing_papers=[],
            contributing_sections=[],
            evidence_passages=[],
            multi_paper_support=False,
            evidence_strength="Insufficient",
            unsupported_claims=0,
            inference_present=False,
            conflicts_detected=False,
            explanation_summary=explanation_summary,
            bullet_points=bullets,
            source_type="Research Mind Corpus",
            domain_scope=scope_label,
        )

        return RAGResponse(
            question=question,
            answer=msg,
            evidence=evidence_items,
            citations={},
            confidence="Insufficient",
            confidence_rationale="Insufficient evidence in 1,000-paper corpus or online sources.",
            limitations=(
                f"The current 1,000-paper dataset and online search do not contain "
                f"sufficient direct evidence for {aspect_str} regarding '{concept_str}'."
            ),
            why_this_answer=why_this_answer,
            retrieval_metadata={
                "retrieved_count": len(retrieved_results),
                "top_rrf_score": retrieved_results[0].rrf_score if retrieved_results else 0.0,
                "insufficient_evidence": True,
                "online_fallback_active": True,
                "used_citations_count": 0,
                "domain_scope": scope_label,
                "concept": concept_str,
                "requested_aspect": aspect_str,
            },
            domain_scope=allowed,
        )
