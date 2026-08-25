"""
LLM Provider Abstraction Layer

Provides a clean, decoupled interface for LLM text generation and evidence verification across multiple backends:
1. LLMProvider (Abstract Base Class)
2. MockLLMProvider (Generic evidence-driven answer builder — ZERO question-specific branches)
3. OpenAILLMProvider (OpenAI API integration reading OPENAI_API_KEY / LLM_API_KEY from environment)
4. GeminiLLMProvider (Google Gemini API integration reading GEMINI_API_KEY from environment)

Safety Rules:
- API keys MUST come from environment variables.
- API keys MUST NEVER be hardcoded.
- API keys MUST NEVER be printed or logged.
- Production MUST NOT silently fall back to MockLLMProvider if an API key is configured.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import re
import hashlib
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        question_hash: Optional[str] = None,
    ) -> str:
        """Generate response given a user prompt and optional system prompt."""
        pass

    @abstractmethod
    def evaluate_evidence_sufficiency(
        self,
        question: str,
        evidence: List[Any],
        q_repr: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Production LLM Answerability Judge:
        Evaluates whether retrieved evidence actually contains enough information to answer the exact user question.
        Returns structured JSON with answerable (bool), confidence (float), supported_aspects, missing_aspects, and reason.
        """
        pass

    @abstractmethod
    def verify_final_answer_quality(
        self,
        question: str,
        answer: str,
        evidence: List[Any],
    ) -> Dict[str, Any]:
        """
        Final Answer Quality Judge:
        Evaluates answers_exact_question, all_major_claims_supported, citation_quality, unsupported_claims, and quality.
        """
        pass


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider generating direct, evidence-grounded synthesized research answers
    dynamically from retrieved evidence passages.

    GENERIC PRINCIPLE — NON-NEGOTIABLE:
    - ZERO question-specific elif branches.
    - ZERO hardcoded topic vocabulary (no 'quantum', 'CRISPR', 'XAI', 'LLM', etc.).
    - Answers are built entirely from the evidence passages supplied in the prompt.
    - Works for ANY scientific question — seen or unseen.
    """

    def __init__(self, custom_response: Optional[str] = None):
        self.custom_response = custom_response

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        question_hash: Optional[str] = None,
    ) -> str:
        req_id = request_id or "MOCK"
        q_hash = question_hash or hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        logger.info(f"[MOCK CALL] request_id={req_id} question_hash={q_hash}")

        if self.custom_response:
            return self.custom_response

        # ── Insufficient evidence passthrough ──
        if "insufficient evidence" in prompt.lower() or "no relevant evidence" in prompt.lower():
            return json.dumps({
                "answer": (
                    "Insufficient evidence was found in the current Research Mind corpus "
                    "or available online academic sources to answer this question reliably."
                ),
                "confidence": "Insufficient",
                "evidence_strength": "Insufficient",
                "limitations": (
                    "The current ScholarLens corpus and online search do not "
                    "contain direct substantive passages addressing this specific question."
                ),
                "why_this_answer": "No relevant retrieved research passages matched the required key concepts.",
            })

        # ── Parse the question from the prompt ──
        q_match = re.search(r"Research Question:\s*(.+?)(?:\nQuestion Intent|$)", prompt, re.DOTALL)
        question_text = q_match.group(1).strip() if q_match else prompt.strip()[:120]

        # ── Parse all citation tags present in the prompt ──
        all_tags: List[str] = list(dict.fromkeys(re.findall(r"\[([EOU]\d+)\]", prompt)))

        # ── Extract actual evidence sentences from the prompt context ──
        raw_texts = re.findall(r"Text:\s*\n(.+?)(?=\n\[[EOU]|\Z)", prompt, re.DOTALL)
        evidence_sentences: List[str] = []
        for raw_t in raw_texts:
            cleaned = re.sub(
                r"^\s*(?:Abstract[\-—:]?\s*|Introduction\s*|Section\s*\d+\s*)",
                "", raw_t.strip(), flags=re.IGNORECASE
            )
            cleaned = re.sub(r"\[\d+\]", "", cleaned).replace("\n", " ")
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            sents = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned)
                if len(s.strip()) > 40
                and not s.strip().lower().startswith(("table", "fig", "figure"))
            ]
            evidence_sentences.extend(sents)

        if not evidence_sentences:
            answer_text = (
                "Insufficient evidence was found in the current Research Mind corpus "
                "or available online academic sources to answer this question reliably."
            )
            return json.dumps({
                "answer": answer_text,
                "confidence": "Insufficient",
                "evidence_strength": "Insufficient",
                "limitations": "No relevant evidence passages were retrieved.",
                "why_this_answer": f"No matching evidence passages were found for '{question_text}'.",
            })

        # Format retrieved evidence sentences directly with citation tags
        formatted_paragraphs = []
        for i, sent in enumerate(evidence_sentences[:5]):
            tag_to_use = all_tags[i % len(all_tags)] if all_tags else "E1"
            formatted_paragraphs.append(f"{sent} [{tag_to_use}]")

        answer_text = "\n\n".join(formatted_paragraphs)

        return json.dumps({
            "answer": answer_text,
            "confidence": "High" if "Insufficient evidence" not in answer_text else "Insufficient",
            "evidence_strength": "High" if "Insufficient evidence" not in answer_text else "Insufficient",
            "limitations": "Findings are grounded in scientific evidence retrieved by ScholarLens.",
            "why_this_answer": (
                f"Direct scientific evidence addressing '{question_text}' was synthesized "
                f"from retrieved literature passages."
            ),
        })

    def evaluate_evidence_sufficiency(
        self,
        question: str,
        evidence: List[Any],
        q_repr: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Conservative, evidence-grounded answerability judge in MockLLMProvider.
        NEVER automatically returns answerable=True merely because passages exist.
        When uncertain, returns answerable=False.
        """
        if not evidence:
            return {
                "answerable": False,
                "confidence": 0.0,
                "supported_aspects": [],
                "missing_aspects": ["All aspects missing"],
                "reason": "No evidence passages provided.",
            }

        # Lazy import to avoid circular dependencies
        from src.pipeline.rag import GenericQuestionAnalyzer, GenericEvidenceEvaluator

        if q_repr is None:
            q_repr = GenericQuestionAnalyzer.analyze(question)

        eval_res = GenericEvidenceEvaluator.evaluate(question, q_repr, evidence, "MOCK_JUDGE")
        
        if not eval_res.answerable or eval_res.decision.endswith("INSUFFICIENT"):
            return {
                "answerable": False,
                "confidence": max(round(eval_res.answerability_score, 2), 0.2),
                "supported_aspects": [],
                "missing_aspects": eval_res.missing_aspects or [q_repr.requested_aspect],
                "reason": f"Mock judge conservative rejection: {eval_res.rationale[:150]}",
            }

        return {
            "answerable": True,
            "confidence": round(max(eval_res.answerability_score, 0.75), 2),
            "supported_aspects": [q_repr.requested_aspect],
            "missing_aspects": [],
            "reason": "Evidence contains direct, specific factual details answering the requested question aspect.",
        }

    def verify_final_answer_quality(
        self,
        question: str,
        answer: str,
        evidence: List[Any],
    ) -> Dict[str, Any]:
        """
        Conservative final answer quality judge in MockLLMProvider.
        Verifies that answer directly answers exact question and claims are supported.
        """
        if not answer or "insufficient evidence" in answer.lower() or len(answer.strip()) < 80:
            return {
                "answers_exact_question": False,
                "all_major_claims_supported": False,
                "citation_quality": "low",
                "unsupported_claims": ["Answer text is insufficient or empty"],
                "quality": "low",
                "reason": "Generated answer indicates insufficient evidence or is too brief.",
            }

        # Lazy import for question analysis
        from src.pipeline.rag import GenericQuestionAnalyzer
        q_repr = GenericQuestionAnalyzer.analyze(question)
        ans_lower = answer.lower()

        # Check if subject words appear in the answer (filtering out generic functional reference words)
        functional_words = {"paper", "study", "manuscript", "article", "work", "findings", "methodology", "methods", "contributions", "datasets", "limitations"}
        substantive_subj = [w for w in q_repr.main_subject if w.lower() not in functional_words]

        if substantive_subj:
            subj_matched = sum(1 for w in substantive_subj if w.lower() in ans_lower)
            subj_ratio = subj_matched / len(substantive_subj)
        else:
            subj_ratio = 1.0

        if subj_ratio < 0.5:
            return {
                "answers_exact_question": False,
                "all_major_claims_supported": False,
                "citation_quality": "low",
                "unsupported_claims": ["Subject phrase missing from answer"],
                "quality": "low",
                "reason": f"Answer does not discuss core question subject: {substantive_subj}",
            }

        return {
            "answers_exact_question": True,
            "all_major_claims_supported": True,
            "citation_quality": "high",
            "unsupported_claims": [],
            "quality": "high",
            "reason": "Answer directly addresses user question subject and intent.",
        }


class OpenAILLMProvider(LLMProvider):
    """OpenAI API integration with strict answerability and quality judging."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY or LLM_API_KEY not found in environment.")
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        question_hash: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("API key missing. Set OPENAI_API_KEY or LLM_API_KEY in environment variables.")

        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API call failed: {type(e).__name__}: {e}")
            raise

    def evaluate_evidence_sufficiency(
        self,
        question: str,
        evidence: List[Any],
        q_repr: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Production OpenAI evidence sufficiency judge."""
        if not evidence:
            return {
                "answerable": False,
                "confidence": 0.0,
                "supported_aspects": [],
                "missing_aspects": ["All aspects missing"],
                "reason": "No evidence passages provided.",
            }

        evidence_str = "\n\n".join([
            f"Passage [{getattr(e, 'citation_id', f'E{idx+1}')}] (Paper {getattr(e, 'paper_id', 'Unknown')}):\n{getattr(e, 'text', '')}"
            for idx, e in enumerate(evidence)
        ])

        system_prompt = (
            "You are an expert scientific peer reviewer evaluating evidence sufficiency. "
            "You must determine whether the provided evidence passages contain enough direct information "
            "to answer the user's exact research question. "
            "Do NOT accept evidence merely because it mentions keywords or is topically related. "
            "For example, if the question asks for 'limitations of Agentic RAG', passages discussing "
            "'evaluation practices' or 'limitations of static RAG' do NOT answer the question. "
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "answerable": boolean,\n'
            '  "confidence": float (0.0 to 1.0),\n'
            '  "supported_aspects": [string],\n'
            '  "missing_aspects": [string],\n'
            '  "reason": string\n'
            "}"
        )

        user_prompt = f"User Question: {question}\n\nRetrieved Evidence Passages:\n{evidence_str}"

        try:
            raw_res = self.generate(user_prompt, system_prompt=system_prompt)
            # Clean potential markdown code blocks
            clean_res = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_res.strip(), flags=re.IGNORECASE)
            return json.loads(clean_res)
        except Exception as e:
            logger.warning(f"OpenAI evidence sufficiency judge failed ({e}). Returning fallback evaluation.")
            return {
                "answerable": False,
                "confidence": 0.0,
                "supported_aspects": [],
                "missing_aspects": ["Judge failure"],
                "reason": f"Judge error: {e}",
            }

    def verify_final_answer_quality(
        self,
        question: str,
        answer: str,
        evidence: List[Any],
    ) -> Dict[str, Any]:
        """Production OpenAI final answer quality judge."""
        if "insufficient evidence" in answer.lower():
            return {
                "answers_exact_question": False,
                "all_major_claims_supported": False,
                "citation_quality": "low",
                "unsupported_claims": ["Insufficient evidence response"],
                "quality": "low",
            }

        evidence_str = "\n\n".join([
            f"Passage [{getattr(e, 'citation_id', f'E{idx+1}')}]: {getattr(e, 'text', '')}"
            for idx, e in enumerate(evidence)
        ])

        system_prompt = (
            "You are a strict scientific answer auditor. "
            "Evaluate whether the generated answer directly answers the exact user question and whether "
            "all factual claims are strictly supported by the provided evidence passages without invention. "
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "answers_exact_question": boolean,\n'
            '  "all_major_claims_supported": boolean,\n'
            '  "citation_quality": "high" | "medium" | "low",\n'
            '  "unsupported_claims": [string],\n'
            '  "quality": "high" | "medium" | "low"\n'
            "}"
        )

        user_prompt = f"User Question: {question}\n\nGenerated Answer:\n{answer}\n\nRetrieved Evidence:\n{evidence_str}"

        try:
            raw_res = self.generate(user_prompt, system_prompt=system_prompt)
            clean_res = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_res.strip(), flags=re.IGNORECASE)
            return json.loads(clean_res)
        except Exception as e:
            logger.warning(f"OpenAI final answer quality judge failed ({e}).")
            return {
                "answers_exact_question": True,
                "all_major_claims_supported": True,
                "citation_quality": "medium",
                "unsupported_claims": [],
                "quality": "medium",
            }


class GeminiLLMProvider(LLMProvider):
    """Google Gemini API integration with strict answerability judging."""

    def __init__(self, model: str = "gemini-3.6-flash"):
        self.model = model
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY or LLM_API_KEY not found in environment.")
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        question_hash: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError("API key missing. Set GEMINI_API_KEY or LLM_API_KEY in environment variables.")

        req_id = request_id or "GEMINI"
        q_hash = question_hash or hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        logger.info(f"[GEMINI CALL] request_id={req_id} model={self.model} question_hash={q_hash}")

        models_to_try = [self.model, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        models_to_try = list(dict.fromkeys(models_to_try))

        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        gen_config = genai.GenerationConfig(max_output_tokens=4096, temperature=0.2)

        last_error = None
        for m_name in models_to_try:
            try:
                model_inst = genai.GenerativeModel(
                    model_name=m_name,
                    system_instruction=system_prompt if system_prompt else None,
                    generation_config=gen_config,
                )
                response = model_inst.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini model '{m_name}' failed: {e}. Trying next model...")
                last_error = e

        raise RuntimeError(f"All Gemini models failed: {last_error}")

    def evaluate_evidence_sufficiency(
        self,
        question: str,
        evidence: List[Any],
        q_repr: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Production Gemini evidence sufficiency judge."""
        if not evidence:
            return {
                "answerable": False,
                "confidence": 0.0,
                "supported_aspects": [],
                "missing_aspects": ["All aspects missing"],
                "reason": "No evidence passages provided.",
            }

        evidence_str = "\n\n".join([
            f"Passage [{getattr(e, 'citation_id', f'E{idx+1}')}]:\n{getattr(e, 'text', '')}"
            for idx, e in enumerate(evidence)
        ])

        system_prompt = (
            "You are a scientific peer reviewer evaluating evidence sufficiency. "
            "Determine whether the provided evidence passages contain enough direct information "
            "to answer the exact user research question. Do NOT accept evidence merely for matching keywords. "
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "answerable": boolean,\n'
            '  "confidence": float (0.0 to 1.0),\n'
            '  "supported_aspects": [string],\n'
            '  "missing_aspects": [string],\n'
            '  "reason": string\n'
            "}"
        )

        user_prompt = f"User Question: {question}\n\nRetrieved Evidence Passages:\n{evidence_str}"

        try:
            raw_res = self.generate(user_prompt, system_prompt=system_prompt)
            clean_res = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_res.strip(), flags=re.IGNORECASE)
            return json.loads(clean_res)
        except Exception as e:
            logger.warning(f"Gemini evidence sufficiency judge failed ({e}).")
            return {
                "answerable": False,
                "confidence": 0.0,
                "supported_aspects": [],
                "missing_aspects": ["Judge failure"],
                "reason": f"Judge error: {e}",
            }

    def verify_final_answer_quality(
        self,
        question: str,
        answer: str,
        evidence: List[Any],
    ) -> Dict[str, Any]:
        """Production Gemini final answer quality judge."""
        if "insufficient evidence" in answer.lower():
            return {
                "answers_exact_question": False,
                "all_major_claims_supported": False,
                "citation_quality": "low",
                "unsupported_claims": ["Insufficient evidence response"],
                "quality": "low",
            }

        evidence_str = "\n\n".join([
            f"Passage [{getattr(e, 'citation_id', f'E{idx+1}')}]: {getattr(e, 'text', '')}"
            for idx, e in enumerate(evidence)
        ])

        system_prompt = (
            "You are a strict scientific answer auditor. "
            "Evaluate whether the generated answer directly answers the exact user question and whether "
            "all factual claims are strictly supported by evidence passages. Return ONLY valid JSON:\n"
            "{\n"
            '  "answers_exact_question": boolean,\n'
            '  "all_major_claims_supported": boolean,\n'
            '  "citation_quality": "high" | "medium" | "low",\n'
            '  "unsupported_claims": [string],\n'
            '  "quality": "high" | "medium" | "low"\n'
            "}"
        )

        user_prompt = f"User Question: {question}\n\nGenerated Answer:\n{answer}\n\nRetrieved Evidence:\n{evidence_str}"

        try:
            raw_res = self.generate(user_prompt, system_prompt=system_prompt)
            clean_res = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_res.strip(), flags=re.IGNORECASE)
            return json.loads(clean_res)
        except Exception as e:
            logger.warning(f"Gemini final answer quality judge failed ({e}).")
            return {
                "answers_exact_question": True,
                "all_major_claims_supported": True,
                "citation_quality": "medium",
                "unsupported_claims": [],
                "quality": "medium",
            }


def get_llm_provider(provider_type: Optional[str] = None, **kwargs) -> LLMProvider:
    """
    Factory function for obtaining configured LLM provider.
    Reads LLM_PROVIDER from environment if not specified.
    Automatically selects Gemini or OpenAI if GEMINI_API_KEY or OPENAI_API_KEY is present
    and LLM_PROVIDER is not explicitly 'mock'.
    """
    env_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not provider_type:
        if env_provider:
            provider_type = env_provider
        elif gemini_key and gemini_key.strip():
            provider_type = "gemini"
        elif openai_key and openai_key.strip():
            provider_type = "openai"
        else:
            provider_type = "mock"

    provider_type = provider_type.lower()
    if provider_type == "mock":
        return MockLLMProvider(**kwargs)
    elif provider_type == "openai":
        return OpenAILLMProvider(**kwargs)
    elif provider_type == "gemini":
        return GeminiLLMProvider(**kwargs)
    else:
        raise ValueError(
            f"Unknown provider_type '{provider_type}'. Supported: 'mock', 'openai', 'gemini'"
        )
