"""
LLM Provider Abstraction Layer

Provides a clean, decoupled interface for LLM text generation across multiple backends:
1. LLMProvider (Abstract Base Class)
2. MockLLMProvider (Generic evidence-driven answer builder — ZERO question-specific branches)
3. OpenAILLMProvider (OpenAI API integration reading OPENAI_API_KEY / LLM_API_KEY from environment)
4. GeminiLLMProvider (Google Gemini API integration reading GEMINI_API_KEY from environment)

Safety Rules:
- API keys MUST come from environment variables.
- API keys MUST NEVER be hardcoded.
- API keys MUST NEVER be printed or logged.
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import re
import hashlib
from typing import Optional, List

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


class MockLLMProvider(LLMProvider):
    """
    Mock LLM provider generating direct, evidence-grounded synthesized research answers
    dynamically from retrieved evidence passages.

    GENERIC PRINCIPLE — NON-NEGOTIABLE:
    - ZERO question-specific elif branches.
    - ZERO hardcoded topic vocabulary (no 'quantum', 'CRISPR', 'XAI', 'LLM', etc.).
    - Answers are built entirely from the evidence passages supplied in the prompt.
    - Intent-aware structure is determined from the parsed question intent, not from
      matching specific question strings.
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
                    "The current 1,000-paper ScholarLens corpus and online search do not "
                    "contain direct substantive passages addressing this specific question."
                ),
                "why_this_answer": "No relevant retrieved research passages matched the required key concepts.",
            })

        # ── Parse the question from the prompt ──
        q_match = re.search(r"Research Question:\s*(.+?)(?:\nQuestion Intent|$)", prompt, re.DOTALL)
        question_text = q_match.group(1).strip() if q_match else prompt.strip()[:120]

        # ── Parse intent from prompt ──
        intent_match = re.search(r"Question Intent:\s*(\w+)", prompt)
        intent = intent_match.group(1).strip() if intent_match else "Explanation"

        # ── Parse all citation tags present in the prompt ──
        all_tags: List[str] = list(dict.fromkeys(re.findall(r"\[([EO]\d+)\]", prompt)))
        local_tags = [t for t in all_tags if t.startswith("E")]
        online_tags = [t for t in all_tags if t.startswith("O")]

        # Convenience references for the first few available tags
        c_1 = f"[{local_tags[0]}]" if local_tags else (f"[{online_tags[0]}]" if online_tags else "[E1]")
        c_2 = (
            f"[{local_tags[1]}]" if len(local_tags) > 1
            else (f"[{online_tags[0]}]" if online_tags else c_1)
        )
        c_o1 = f"[{online_tags[0]}]" if online_tags else c_1
        c_o2 = f"[{online_tags[1]}]" if len(online_tags) > 1 else c_o1

        # ── Extract actual evidence sentences from the prompt context ──
        # These are the ONLY source of factual content — no fabrication.
        raw_texts = re.findall(r"Text:\s*\n(.+?)(?=\n\[[EO]|\Z)", prompt, re.DOTALL)
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

        # ── Derive a concise topic phrase from the question ──
        topic_phrase = question_text.strip().rstrip("?")
        for prefix in (
            "what algorithms are commonly used for ", "what algorithms are used for ",
            "what are the limitations of ", "what are the benefits of ", "what causes ",
            "what is ", "what are the ", "what are ", "how does ", "how do ",
            "how can ", "how is ", "why does ", "why do ", "describe ", "what ", "how ",
        ):
            if topic_phrase.lower().startswith(prefix):
                topic_phrase = topic_phrase[len(prefix):]
                break

        # ── Build answer using ONLY actual evidence sentences — ZERO generic filler ──
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


class OpenAILLMProvider(LLMProvider):
    """OpenAI API integration."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY or LLM_API_KEY not found in environment. "
                "OpenAILLMProvider will fail if invoked."
            )
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        question_hash: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                "API key missing. Set OPENAI_API_KEY or LLM_API_KEY in environment variables."
            )

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
            logger.error(f"OpenAI API call failed: {type(e).__name__}")
            raise


class GeminiLLMProvider(LLMProvider):
    """Google Gemini API integration with multi-model fallback."""

    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY or LLM_API_KEY not found in environment. "
                "GeminiLLMProvider will fallback to MockLLMProvider if invoked."
            )
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        request_id: Optional[str] = None,
        question_hash: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            logger.warning("GEMINI_API_KEY missing. Falling back to MockLLMProvider.")
            return MockLLMProvider().generate(prompt, system_prompt, request_id, question_hash)

        req_id = request_id or "GEMINI"
        q_hash = question_hash or hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        q_snippet = prompt[:60].replace("\n", " ")
        logger.info(
            f"[GEMINI CALL] request_id={req_id} model={self.model} "
            f"question_hash={q_hash} called=true snippet='{q_snippet}'"
        )

        models_to_try = [self.model, "gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
        # Remove duplicates while preserving order
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
                raw_text = response.text.strip()
                res_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:8]
                logger.info(
                    f"[GEMINI RESPONSE] request_id={req_id} model={m_name} "
                    f"response_received=true response_hash={res_hash} response_length={len(raw_text)}"
                )
                return raw_text
            except Exception as e:
                logger.warning(f"Gemini model '{m_name}' failed: {e}. Trying next model...")
                last_error = e

        logger.error(f"All Gemini models failed ({last_error}). Falling back to MockLLMProvider.")
        return MockLLMProvider().generate(prompt, system_prompt, request_id, question_hash)


def get_llm_provider(provider_type: str = "mock", **kwargs) -> LLMProvider:
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
