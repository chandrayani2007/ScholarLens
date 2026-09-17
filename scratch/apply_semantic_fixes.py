"""
Applies all targeted semantic quality fixes to src/pipeline/rag.py.
Run from the project root: .venv\Scripts\python scratch/apply_semantic_fixes.py
"""
import re, pathlib, sys

RAG_PATH = pathlib.Path("src/pipeline/rag.py")
src = RAG_PATH.read_text(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 1 — Numbered list corruption in ClaimGroundingValidator.validate_and_filter_claims
# Replace the text normalisation + sentence-split block (lines ~2308–2318)
# so that numbered/bulleted list items are preserved as atomic units.
# ──────────────────────────────────────────────────────────────────────────────

OLD_NORM = '''        raw_lines = answer.strip().split("\\n")
        normalized_paragraphs = []
        for line in raw_lines:
            line_s = line.strip()
            if line_s.startswith("#"):
                normalized_paragraphs.append(line_s + "\\n")
            elif line_s:
                normalized_paragraphs.append(line_s)

        text_to_split = "\\n".join(normalized_paragraphs)
        sentences = re.split(r"(?<=[.!?])\\s+|\\n+", text_to_split)'''

NEW_NORM = '''        raw_lines = answer.strip().split("\\n")
        normalized_paragraphs = []
        for line in raw_lines:
            line_s = line.strip()
            if line_s.startswith("#"):
                normalized_paragraphs.append(line_s + "\\n")
            elif re.match(r"^\\d+\\.\\s+.+", line_s) or re.match(r"^[-*]\\s+.+", line_s):
                # Preserve numbered and bulleted list items as atomic units
                normalized_paragraphs.append("__LIST_ITEM__" + line_s + "\\n")
            elif line_s:
                normalized_paragraphs.append(line_s)

        text_to_split = "\\n".join(normalized_paragraphs)
        sentences = re.split(r"(?<=[.!?])\\s+|\\n+", text_to_split)'''

assert OLD_NORM in src, "FIX 1a (normalization) — TARGET NOT FOUND"
src = src.replace(OLD_NORM, NEW_NORM, 1)
print("✓ FIX 1a — Numbered list preservation in normalizer")

# Extend the per-sentence loop guard to recognise and pass-through list items
OLD_GUARD = '''            # Preserve structural headings/headers
            if (sent_clean.startswith("#") or
                (sent_clean.startswith("**") and sent_clean.endswith(":**")) or
                len(re.findall(r"\\b[a-zA-Z]{3,}\\b", sent_clean)) <= 2):
                valid_sentences.append(sent)
                continue'''

NEW_GUARD = '''            # Preserve structural headings/headers and list items
            if sent_clean.startswith("__LIST_ITEM__"):
                # Strip our internal marker, preserve list item verbatim
                list_item_text = sent_clean[len("__LIST_ITEM__"):]
                valid_sentences.append(list_item_text)
                continue
            if (sent_clean.startswith("#") or
                (sent_clean.startswith("**") and sent_clean.endswith(":**")) or
                len(re.findall(r"\\b[a-zA-Z]{3,}\\b", sent_clean)) <= 2):
                valid_sentences.append(sent)
                continue'''

assert OLD_GUARD in src, "FIX 1b (guard) — TARGET NOT FOUND"
src = src.replace(OLD_GUARD, NEW_GUARD, 1)
print("✓ FIX 1b — List item pass-through in per-sentence loop")

# Fix joined_parts assembly: list items use \n separator, not space
OLD_JOINED = '''        if cleaned_valid_sentences:
            joined_parts = []
            for vs in cleaned_valid_sentences:
                vs_stripped = vs.strip()
                if vs_stripped.startswith('#'):
                    if joined_parts:
                        joined_parts.append('\\n')
                    joined_parts.append(vs_stripped)
                    joined_parts.append('\\n')
                else:
                    if joined_parts and not joined_parts[-1].endswith('\\n'):
                        joined_parts.append(' ')
                    joined_parts.append(vs_stripped)
            grounded_answer = ''.join(joined_parts).strip()'''

NEW_JOINED = '''        if cleaned_valid_sentences:
            joined_parts = []
            for vs in cleaned_valid_sentences:
                vs_stripped = vs.strip()
                if vs_stripped.startswith('#'):
                    if joined_parts:
                        joined_parts.append('\\n')
                    joined_parts.append(vs_stripped)
                    joined_parts.append('\\n')
                elif re.match(r"^\\d+\\.\\s+", vs_stripped) or re.match(r"^[-*]\\s+", vs_stripped):
                    # Numbered/bulleted list items must be on their own line
                    if joined_parts and not joined_parts[-1].endswith('\\n'):
                        joined_parts.append('\\n')
                    joined_parts.append(vs_stripped)
                    joined_parts.append('\\n')
                else:
                    if joined_parts and not joined_parts[-1].endswith('\\n'):
                        joined_parts.append(' ')
                    joined_parts.append(vs_stripped)
            grounded_answer = ''.join(joined_parts).strip()'''

assert OLD_JOINED in src, "FIX 1c (joined_parts) — TARGET NOT FOUND"
src = src.replace(OLD_JOINED, NEW_JOINED, 1)
print("✓ FIX 1c — List items use \\n in joined_parts assembly")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 2 — Evidence strength must not be "High" when unsupported_claims > 0
# ──────────────────────────────────────────────────────────────────────────────

OLD_ES = '''        elif unsupported_claims_count > 0:
            if effective_score >= 0.60 and total_active_cits >= 2:
                evidence_strength = "High"
            elif total_active_cits >= 1:
                evidence_strength = "Moderate"
            else:
                evidence_strength = "Insufficient"'''

NEW_ES = '''        elif unsupported_claims_count > 0:
            # When unsupported claims remain, cap evidence strength at "Moderate"
            if total_active_cits >= 2:
                evidence_strength = "Moderate"
            elif total_active_cits >= 1:
                evidence_strength = "Low"
            else:
                evidence_strength = "Insufficient"'''

assert OLD_ES in src, "FIX 2 (evidence_strength) — TARGET NOT FOUND"
src = src.replace(OLD_ES, NEW_ES, 1)
print("✓ FIX 2 — Evidence strength capped at Moderate when unsupported_claims > 0")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 3 — Evidence-calibrated wording: add inflated assertion patterns
# ──────────────────────────────────────────────────────────────────────────────

OLD_INFLATED = '''    INFLATED_TERMS_MAP = {
        r"\\bsubstantial\\b": "reported",
        r"\\bsubstantially\\b": "measurably",
        r"\\bsignificant\\b": "measurable",
        r"\\bsignificantly\\b": "measurably",
        r"\\bremarkable\\b": "reported",
        r"\\bremarkably\\b": "notably",
        r"\\bstate-of-the-art\\b": "proposed",
        r"\\bsuperior\\b": "higher",
        r"\\bpowerful\\b": "specialized",
        r"\\bdramatic\\b": "reported",
        r"\\bdramatically\\b": "measurably",
        r"\\bhighly effective\\b": "effective",
        r"\\bmajor improvement\\b": "improvement",
        r"\\bperformance advantages\\b": "performance differences",
    }'''

NEW_INFLATED = '''    INFLATED_TERMS_MAP = {
        r"\\bsubstantial\\b": "reported",
        r"\\bsubstantially\\b": "measurably",
        r"\\bsignificant\\b": "measurable",
        r"\\bsignificantly\\b": "measurably",
        r"\\bremarkable\\b": "reported",
        r"\\bremarkably\\b": "notably",
        r"\\bstate-of-the-art\\b": "proposed",
        r"\\bsuperior\\b": "higher",
        r"\\bpowerful\\b": "specialized",
        r"\\bdramatic\\b": "reported",
        r"\\bdramatically\\b": "measurably",
        r"\\bhighly effective\\b": "effective",
        r"\\bmajor improvement\\b": "improvement",
        r"\\bperformance advantages\\b": "performance differences",
        # Absolute reduction/elimination claims — require evidence calibration
        r"\\bsignificantly reduces\\b": "can reduce",
        r"\\bgreatly reduces\\b": "may reduce",
        r"\\beliminate(?:s)?\\s+hallucination": "reduce hallucination",
        r"\\balways\\s+(?:prevents|eliminates|avoids)\\b": "can help avoid",
        r"\\bcompletely\\s+eliminates\\b": "reduces",
        r"\\bguarantee(?:s)?\\b": "aims to ensure",
    }'''

assert OLD_INFLATED in src, "FIX 3 (INFLATED_TERMS_MAP) — TARGET NOT FOUND"
src = src.replace(OLD_INFLATED, NEW_INFLATED, 1)
print("✓ FIX 3 — Added evidence-calibrated inflated assertion patterns")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 4, 5, 6 + OPEN QUESTION — Replace ALL hardcoded category_guidance blocks
# with paper-independent, evidence-driven guidance.
# ──────────────────────────────────────────────────────────────────────────────

OLD_CAT_BLOCK = '''        category_guidance = ""
        if intent in ["ResearchProblem", "Objective"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR RESEARCH PROBLEM:\\n"
                "- Provide a clear, detailed, comprehensive explanation synthesized from the introduction and motivation evidence.\\n"
                "- Clearly identify: (1) the core challenge of synthesizing scientific knowledge amidst rapid literature growth, "
                "(2) why staying informed is increasingly difficult, and (3) specific failure modes of general-purpose language models "
                "(hallucinations, outdated pre-training knowledge, lack of citation grounding).\\n"
                "- Do NOT provide a one-sentence answer; provide a well-rounded research-level explanation."
            )
            structure_hint = "### Research Problem\\nExplain the core scientific challenge, the impact of publication volume, and specific LLM limitations with citations."

        elif intent in ["Methodology", "Method", "Algorithm", "Mechanism"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR PROPOSED METHODOLOGY:\\n"
                "- Accurately distinguish and preserve exact paper terminology:\\n"
                "  a) OSDS / OpenScholar Data Store: the specialized scientific data store containing indexed literature passages.\\n"
                "  b) Scientific passage retriever: trained on scientific corpora to retrieve relevant context.\\n"
                "  c) Reranker & specialized generator language model (8B parameter model, etc.).\\n"
                "  d) Iterative self-feedback inference loop (termed OpenScholar Feedback Loop): dynamically critiques and refines draft responses to maximize factual accuracy and attribution.\\n"
                "  e) Additional retrieval / refinement and citation verification.\\n"
                "- Do NOT equate OSDS to only the feedback loop; OSDS is the scientific data store."
            )
            structure_hint = "### Proposed Methodology\\nDetail the OpenScholar system architecture, distinguishing the OSDS data store, scientific retriever, generator model, and iterative self-feedback critique loop with citations."

        elif intent in ["Dataset"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR DATASETS & BENCHMARKS:\\n"
                "- Accurately distinguish the benchmark suite and its subsets:\\n"
                "  - ScholarQABench: evaluation suite spanning four core disciplines (computer science, physics, biomedicine, neuroscience).\\n"
                "  - Scholar-CS subset: 100 questions formulated by PhD candidates.\\n"
                "  - Scholar-Multi subset: 108 challenging questions paired with 250 expert-written long-form answers.\\n"
                "- If subsets for specific domains (Scholar-Bio, Scholar-Neuro, etc.) are discussed, distinguish them clearly.\\n"
                "- Never combine numbers from different benchmark descriptions unless explicitly connected in the source."
            )
            structure_hint = "### Datasets & Benchmarks\\nDetail the ScholarQABench evaluation suite, distinguishing Scholar-CS (100 questions) and Scholar-Multi (108 questions + 250 expert answers across 4 disciplines) with citations."

        elif intent in ["DataPreparation", "Preprocessing"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR DATA PREPARATION / PREPROCESSING:\\n"
                "- State that benchmark queries and reference answers were formulated, prepared, and preprocessed by experienced PhD candidates and postdoctoral scholars to mirror authentic scientific workflows.\\n"
                "- If conventional computational data preprocessing (tokenization, filtering, normalization) is NOT detailed in the paper, explicitly state that the available evidence does not specify it.\\n"
                "- NEVER invent preprocessing steps.\\n"
                "- If asked about scientific data store (OSDS) construction, explain data store retrieval indexing."
            )
            structure_hint = "### Data Preparation & Preprocessing\\nState how queries and reference answers were prepared by PhD/postdoc scholars, and note if standard text preprocessing is not specified in the paper."

        elif intent in ["ExperimentalSetup"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR EXPERIMENTAL SETUP:\\n"
                "- Cover multiple relevant setup components:\\n"
                "  - Models: OpenScholar-8B parameter model and OpenScholar-GPT-4o paired with the scientific retriever.\\n"
                "  - Comparative Baselines: Unassisted GPT-4o, base language models, and expert-written human answers.\\n"
                "  - Datasets & Benchmarks: ScholarQABench (Scholar-CS with 100 questions, Scholar-Multi with 108 questions across 4 disciplines).\\n"
                "  - Evaluation Procedure: Double-blind expert evaluations conducted with 16 PhD researchers assessing correctness and preference win rates."
            )
            structure_hint = (
                "### Experimental Setup\\n"
                "Detail the models, comparative baselines, evaluation benchmarks, and expert evaluation procedure with citations."
            )

        elif intent in ["QuantitativeResults", "Finding", "Result", "Evaluation"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR QUANTITATIVE RESULTS:\\n"
                "- Detail exact empirical metrics:\\n"
                "  - OpenScholar-GPT-4o improves correctness by 12% over base GPT-4o.\\n"
                "  - In expert blind evaluations with 16 PhD researchers, OpenScholar-8B and OpenScholar-GPT-4o were preferred over expert human answers 51% and 70% of the time, respectively.\\n"
                "  - Baseline GPT-4o achieved only a 32% win rate.\\n"
                "- Distinguish percentage improvement from percentage points and preserve exact numerical precision from the source."
            )
            structure_hint = "### Quantitative Results\\nReport exact correctness gains (+12%) and expert preference win rates (51% for 8B, 70% for GPT-4o vs 32% for base GPT-4o) with citations."

        elif intent in ["BaselineComparison", "Comparison"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR BASELINE COMPARISON:\\n"
                "- Synthesize comparisons against all supported baselines (unassisted GPT-4o, base LLMs, expert human references).\\n"
                "- Highlight: OpenScholar-GPT-4o achieves 70% win rate vs 32% for unassisted GPT-4o, 12% correctness gain, and substantial reduction in citation fabrication compared to standard unaugmented models."
            )
            structure_hint = "### Baseline Comparison\\nCompare OpenScholar directly against unassisted GPT-4o and baseline models across correctness, human preference (70% vs 32%), and citation attribution with citations."

        elif intent in ["Contribution"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR CONTRIBUTIONS:\\n"
                "- Explicitly enumerate all 3 primary contributions from the paper:\\n"
                "  1. Creating OpenScholar, an open literature synthesis framework with iterative self-critique inference.\\n"
                "  2. Establishing ScholarQABench across four scientific disciplines with 250 expert-written answers.\\n"
                "  3. Demonstrating that retrieval-augmented feedback substantially boosts correctness (+12%) and human preference (up to 70%) over foundation baselines."
            )
            structure_hint = "### Scientific & Technical Contributions\\nList the three primary contributions (OpenScholar framework, ScholarQABench benchmark, and empirical retrieval-augmented feedback validation) with citations."

        elif intent in ["Limitation", "Challenge"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR LIMITATIONS:\\n"
                "- Group and explain the author-stated limitations:\\n"
                "  - Retrieval & Model Limitations: Standard models without retrieval frequently fabricate citations and struggle with multi-paper reference attribution.\\n"
                "  - Computational Latency: Executing iterative self-feedback critique loops introduces latency during long-form generation.\\n"
                "  - Future Work: Scaling datastore indexing and accelerating multi-step reasoning."
            )
            structure_hint = "### Author-Stated Limitations\\nCategorize the limitations into citation attribution difficulties, critique loop computational latency, and future datastore scaling with citations."

        elif intent in ["Comprehensive"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR COMPREHENSIVE SUMMARY:\\n"
                "- Synthesize all seven requested research components into coherent, well-organized markdown sections:\\n"
                "  ### Research Problem\\n"
                "  ### Proposed Methodology\\n"
                "  ### Datasets & Benchmarks\\n"
                "  ### Experimental Setup\\n"
                "  ### Quantitative Results & Baseline Comparison\\n"
                "  ### Scientific Contributions\\n"
                "  ### Author-Stated Limitations\\n"
                "- Provide substantive, citation-grounded prose for each of the seven sections.\\n"
                "- Do NOT omit any of the seven sections."
            )
            structure_hint = (
                "### Research Problem\\nState the core scientific challenge with citations.\\n\\n"
                "### Proposed Methodology\\nDetail the system architecture, retriever, OSDS, and generator with citations.\\n\\n"
                "### Datasets & Benchmarks\\nDescribe ScholarQABench (Scholar-CS and Scholar-Multi) with citations.\\n\\n"
                "### Experimental Setup\\nDetail models, baselines, and evaluation protocol with 16 PhD researchers with citations.\\n\\n"
                "### Quantitative Results & Baseline Comparison\\nReport the 12% correctness gain, 51% (8B), 70% (GPT-4o), and 32% (baseline) preference rates with citations.\\n\\n"
                "### Scientific Contributions\\nList the three primary contributions with citations.\\n\\n"
                "### Author-Stated Limitations\\nReport author-stated limitations regarding citation attribution and latency with citations."
            )'''

NEW_CAT_BLOCK = '''        category_guidance = ""
        if intent in ["ResearchProblem", "Objective"] or route_category == "RESEARCH_PROBLEM":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR RESEARCH PROBLEM:\\n"
                "- FIRST SENTENCE RULE: The very first sentence MUST name the specific concrete problem, gap, limitation, "
                "or unmet need stated in the retrieved paper evidence. NEVER begin with vague sentences such as:\\n"
                "  BAD: 'The paper addresses several important challenges.'\\n"
                "  BAD: 'The study investigates various problems in the field.'\\n"
                "  BAD: 'This paper explores an important topic.'\\n"
                "  GOOD: 'The paper identifies [specific problem from evidence] as a critical gap that [specific consequence].'\\n"
                "- Synthesize from the retrieved evidence: (1) the concrete research gap or challenge, "
                "(2) why existing approaches fail to solve it, and (3) what the paper aims to achieve.\\n"
                "- Do NOT produce a one-sentence answer. Provide a well-rounded, multi-sentence explanation grounded in evidence.\\n"
                "- Do NOT use general background statements as the research problem. Distinguish the specific problem from the general field context."
            )
            structure_hint = (
                "### Research Problem\\n"
                "[Sentence 1: State the specific problem/gap from the evidence]\\n"
                "[Sentence 2: Why existing approaches are insufficient]\\n"
                "[Sentence 3: What the paper proposes to address it]"
            )

        elif intent in ["Methodology", "Method", "Algorithm", "Mechanism"] or route_category == "METHODOLOGY":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR PROPOSED METHODOLOGY:\\n"
                "- Accurately describe the methodology as presented in the retrieved evidence.\\n"
                "- Preserve all exact technical terminology, component names, model names, and architecture labels from the evidence.\\n"
                "- Clearly distinguish between: the data source/datastore, the retrieval mechanism, the reranker (if present), and the generator language model (if distinct).\\n"
                "- If the paper describes an iterative feedback or critique loop, name it as the evidence does and explain what it does.\\n"
                "- Do NOT invent component names. Use only names found in the retrieved evidence.\\n"
                "- Do NOT equate separate components (e.g., a data store is NOT the same as the inference loop)."
            )
            structure_hint = (
                "### Proposed Methodology\\n"
                "[Sentence 1: High-level name and purpose of the proposed system]\\n"
                "[Sentence 2+: Describe each component with its role, citing evidence]"
            )

        elif intent in ["Dataset"] or route_category == "DATASET":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR DATASETS & BENCHMARKS:\\n"
                "- Report ALL datasets and benchmarks mentioned in the retrieved evidence for this paper.\\n"
                "- Preserve exact dataset names, sizes, domain labels, and split details exactly as stated in the evidence.\\n"
                "- Distinguish training sets, evaluation benchmarks, and held-out test sets if the evidence makes this distinction.\\n"
                "- If benchmark subsets are described (e.g. different domains or difficulty levels), list them separately.\\n"
                "- NEVER combine or merge numbers from different dataset descriptions unless the evidence explicitly connects them.\\n"
                "- If a dataset detail is not present in the evidence, state: 'The available evidence does not specify [detail].' Do NOT invent it."
            )
            structure_hint = (
                "### Datasets & Benchmarks\\n"
                "[List each dataset or benchmark with its size and domain from the evidence] [U#]"
            )

        elif intent in ["DataPreparation", "Preprocessing"] or route_category == "DATA_PREPARATION":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR DATA PREPARATION / PREPROCESSING:\\n"
                "- Describe only the data preparation steps explicitly stated in the retrieved evidence.\\n"
                "- If standard preprocessing (tokenization, filtering, normalization) is NOT detailed in the evidence, explicitly state: "
                "'The available evidence does not specify standard text preprocessing steps.'\\n"
                "- NEVER invent preprocessing steps.\\n"
                "- If evaluation queries or annotations were constructed by human experts, state this exactly as described in the evidence."
            )
            structure_hint = (
                "### Data Preparation & Preprocessing\\n"
                "[Describe each preprocessing or data construction step supported by evidence] [U#]\\n"
                "[If not specified: 'The available evidence does not specify [aspect].']"
            )

        elif intent in ["ExperimentalSetup"] or route_category in ["EXPERIMENTAL_SETUP"]:
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR EXPERIMENTAL SETUP:\\n"
                "- Cover all setup components mentioned in the retrieved evidence:\\n"
                "  - Model(s): names and sizes as stated in the evidence.\\n"
                "  - Comparative baselines: as named in the evidence.\\n"
                "  - Datasets/benchmarks used for evaluation: as named in the evidence.\\n"
                "  - Evaluation procedure/protocol: as described in the evidence (e.g. human evaluation, automatic metrics).\\n"
                "- Do NOT hardcode or invent model names, evaluator counts, or dataset details not present in the evidence.\\n"
                "- If a setup detail is not in the evidence, state: 'The available evidence does not describe [detail].'"
            )
            structure_hint = (
                "### Experimental Setup\\n"
                "[Detail models, baselines, benchmarks, and evaluation protocol from the evidence] [U#]"
            )

        elif intent in ["QuantitativeResults", "Finding", "Result", "Evaluation"] or route_category == "RESULTS":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR QUANTITATIVE RESULTS:\\n"
                "- FIRST SENTENCE RULE: Do NOT begin with vague phrases like 'The evaluation demonstrates measurable improvements.'\\n"
                "  GOOD: 'The proposed method achieves [exact metric value] on [benchmark] compared to [baseline value].'\\n"
                "- Report EXACT numerical metrics from the evidence: accuracy, F1, improvement %, win rates, latency, etc.\\n"
                "- Format as a bullet list where each bullet is: [Metric name] — [exact reported value] [citation].\\n"
                "- If evidence contains the exact value, use it. NEVER substitute 'measurable improvement' for an available number.\\n"
                "- If evidence does NOT contain a specific metric, write: 'The available evidence does not specify [metric name].'\\n"
                "- NEVER invent numbers, percentages, or ranks not found in the evidence.\\n"
                "- Distinguish what each metric measures (correctness gain vs. preference win rate vs. absolute accuracy, etc.)."
            )
            structure_hint = (
                "### Quantitative Results\\n"
                "- [Metric 1] — [exact value from evidence] [U#]\\n"
                "- [Metric 2] — [exact value from evidence] [U#]\\n"
                "- [Baseline comparison] — [exact values from evidence] [U#]"
            )

        elif intent in ["BaselineComparison", "Comparison"] or route_category == "BASELINE_COMPARISON":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR BASELINE COMPARISON:\\n"
                "- Compare the proposed method against each baseline explicitly named in the evidence.\\n"
                "- Report exact values for each comparison: correctness, preference rates, accuracy deltas, etc. as given in the evidence.\\n"
                "- Do NOT substitute vague phrases ('outperforms', 'substantially better') when exact values are available in the evidence.\\n"
                "- Use '[U#]' citations for values from the uploaded paper evidence.\\n"
                "- If a specific baseline comparison is not in the evidence, state: 'The available evidence does not specify comparison with [baseline].'"
            )
            structure_hint = (
                "### Baseline Comparison\\n"
                "[Proposed method] vs [Baseline 1]: [exact metric] [U#]\\n"
                "[Proposed method] vs [Baseline 2]: [exact metric] [U#]"
            )

        elif intent in ["Contribution"] or route_category == "CONTRIBUTION":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR CONTRIBUTIONS:\\n"
                "- Enumerate ALL primary contributions explicitly described in the retrieved evidence.\\n"
                "- Use only the contribution names and descriptions found in the evidence.\\n"
                "- Do NOT paraphrase contributions into generic statements. Preserve specificity.\\n"
                "- If the paper lists N contributions, enumerate exactly N from the evidence.\\n"
                "- Each contribution should reference the specific technical or empirical innovation described."
            )
            structure_hint = (
                "### Scientific & Technical Contributions\\n"
                "1. [Contribution 1 from evidence] [U#]\\n"
                "2. [Contribution 2 from evidence] [U#]\\n"
                "3. [Additional contributions if described in evidence] [U#]"
            )

        elif intent in ["Limitation", "Challenge"] or route_category == "LIMITATION":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR LIMITATIONS:\\n"
                "- Report ONLY author-stated limitations explicitly present in the retrieved evidence.\\n"
                "- Do NOT invent limitations based on general knowledge about the method or field.\\n"
                "- Do NOT classify future work as a limitation unless the authors themselves frame it as one.\\n"
                "- Group limitations by type (e.g. computational, data, scope) only if the evidence supports such grouping.\\n"
                "- If the evidence does not contain a limitations section, state: 'The available evidence does not describe explicit author-stated limitations.'"
            )
            structure_hint = (
                "### Author-Stated Limitations\\n"
                "[Limitation 1 from evidence] [U#]\\n"
                "[Limitation 2 from evidence] [U#]\\n"
                "[If not specified: 'The available evidence does not describe explicit limitations.']"
            )

        elif intent in ["Comprehensive"] or route_category == "COMPREHENSIVE":
            category_guidance = (
                "CATEGORY REQUIREMENTS FOR COMPREHENSIVE SUMMARY:\\n"
                "- Synthesize ALL seven requested research components using ONLY the retrieved evidence for the current paper.\\n"
                "- The section structure is fixed; the CONTENT must come exclusively from the retrieved evidence.\\n"
                "- Do NOT hardcode or assume any paper-specific names, numbers, datasets, or results.\\n"
                "- For each section, state concrete evidence-supported facts. If evidence is absent for a section, write:\\n"
                "  'The available paper evidence does not specify this aspect.'\\n"
                "- Do NOT omit any of the seven required sections."
            )
            structure_hint = (
                "### Research Problem\\n"
                "[Specific research gap or challenge from evidence] [U#]\\n\\n"
                "### Proposed Methodology\\n"
                "[System/method architecture and components from evidence] [U#]\\n\\n"
                "### Datasets & Benchmarks\\n"
                "[Dataset names, sizes, domains from evidence — or 'Not specified in evidence'] [U#]\\n\\n"
                "### Experimental Setup\\n"
                "[Models, baselines, evaluation protocol from evidence — or 'Not specified in evidence'] [U#]\\n\\n"
                "### Quantitative Results\\n"
                "[Exact numerical results from evidence — or 'Not specified in evidence'] [U#]\\n\\n"
                "### Scientific Contributions\\n"
                "[Enumerated contributions from evidence] [U#]\\n\\n"
                "### Author-Stated Limitations\\n"
                "[Author-stated limitations from evidence — or 'The available evidence does not describe explicit limitations.'] [U#]"
            )'''

assert OLD_CAT_BLOCK in src, "FIX 4/5/OPENQ (category_guidance block) — TARGET NOT FOUND"
src = src.replace(OLD_CAT_BLOCK, NEW_CAT_BLOCK, 1)
print("✓ FIX 4/5/OPENQ — Replaced ALL hardcoded category_guidance with paper-independent guidance")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 6 — Extend mandate point 1 with banned first-sentence patterns and
#          evidence-calibrated wording requirement
# ──────────────────────────────────────────────────────────────────────────────

OLD_MANDATE = '''            f"MANDATES:\\n"
            f"1. Start the first paragraph with a direct synthesized explanation answering the core research question. Do NOT include generic filler phrases like 'Experimental evaluations demonstrate reported improvements' or 'The authors identify several limitations'. State the concrete facts directly in sentence 1.\\n"
            f"2. Cite every factual claim using inline tags like [E1], [E2], [O1], [U1] immediately after supported statements.\\n"
            f"3. If evidence is missing for a specific sub-aspect, state explicitly: 'The available paper evidence does not provide enough information regarding [sub-aspect].'\\n"
            f"4. Do NOT fabricate numbers, statistics, datasets, methods, baseline comparisons, or author claims.\\n"
            f"5. Do NOT copy, paste, or quote contiguous phrases (6+ words) from the retrieved evidence. Paraphrase all facts into fresh prose.\\n"
            f"6. For quantitative results, report all major findings present in evidence (correctness gain, preference win rates for both proposed models, baseline win rate) and distinguish what each number measures.\\n\\n"'''

NEW_MANDATE = '''            f"MANDATES:\\n"
            f"1. ANSWER-FIRST REQUIREMENT: The VERY FIRST sentence MUST directly state the concrete answer. BANNED first-sentence patterns:\\n"
            f"   ❌ 'The paper addresses several important challenges...'\\n"
            f"   ❌ 'Experimental evaluations demonstrate reported improvements...'\\n"
            f"   ❌ 'The authors identify various limitations...'\\n"
            f"   ❌ 'This study investigates the relationship between...'\\n"
            f"   ❌ 'The paper explores an important problem...'\\n"
            f"   ✓ GOOD: 'The paper proposes [specific system] to address [specific problem], demonstrating [specific result].'\\n"
            f"   ✓ GOOD: 'RAG works by [specific mechanism] to [specific outcome].'\\n"
            f"2. EVIDENCE-CALIBRATED WORDING: For improvement or reduction claims, use qualified wording unless the evidence explicitly supports an absolute claim:\\n"
            f"   ❌ 'RAG significantly reduces hallucinations.' (absolute — requires direct evidence)\\n"
            f"   ✓ 'RAG can reduce hallucination frequency by grounding responses in retrieved context.' (qualified)\\n"
            f"   ✓ 'Evidence indicates RAG may reduce factual errors.' (evidence-calibrated)\\n"
            f"3. Cite every factual claim using inline tags like [E1], [E2], [O1], [U1] immediately after supported statements.\\n"
            f"4. If evidence is missing for a specific sub-aspect, state explicitly: 'The available paper evidence does not provide enough information regarding [sub-aspect].'\\n"
            f"5. Do NOT fabricate numbers, statistics, datasets, methods, baseline comparisons, or author claims.\\n"
            f"6. Do NOT copy, paste, or quote contiguous phrases (6+ words) from the retrieved evidence. Paraphrase all facts into fresh prose.\\n"
            f"7. For quantitative results, report ALL major findings present in evidence with exact values. NEVER substitute vague phrases like 'measurable improvements' when the evidence contains a specific number.\\n\\n"'''

assert OLD_MANDATE in src, "FIX 6 (mandates) — TARGET NOT FOUND"
src = src.replace(OLD_MANDATE, NEW_MANDATE, 1)
print("✓ FIX 6 — Answer-first requirement and evidence-calibrated wording mandate extended")

# ──────────────────────────────────────────────────────────────────────────────
# Write back
# ──────────────────────────────────────────────────────────────────────────────
RAG_PATH.write_text(src, encoding="utf-8")
print(f"\n✅ All fixes applied successfully to {RAG_PATH}")
print("   Run: .venv\\Scripts\\pytest tests/test_rag.py tests/test_relevance_gate.py -v")
