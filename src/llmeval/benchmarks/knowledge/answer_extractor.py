"""MCQ answer extraction for knowledge benchmarks.

Rules ported directly from existing implementations:
  - MMLU / MMLU-FLAN : new_pipeline/k2/mmlu.py, k2/mmlu_flan.py
  - MMLU-Pro         : new_pipeline/k2/mmlu_pro.py
  - BBH              : new_pipeline/k2/bbh.py  (free-text targets)
  - GPQA             : new_pipeline/evaluate_gpqa_new.py  (\\boxed{A} format)

Public API:
    extract_abcd(text)   — MMLU / MMLU-FLAN  (choices A–D, returns "(A)")
    extract_abcdj(text)  — MMLU-Pro  (choices A–J, returns "A")
    extract_bbh(text)    — BBH  (free-text targets: True/False, (A), phrases)
    extract_gpqa(text)   — GPQA  (\\boxed{A} → "answer is A" → (A), returns "A")
"""

import re


def extract_abcd(text: str) -> str:
    """Extract A–D MCQ answer from model output.

    Priority (mirrors k2/mmlu.py and k2/mmlu_flan.py):
      1. "answer is (X)" — case-insensitive
      2. First "(X)" anywhere in the text  (same regex as evaluate_mmlu.py)

    Returns the letter wrapped in parentheses, e.g. "(A)".
    Returns "" if nothing found.
    """
    # 1. "answer is (X)"
    m = re.search(r"answer is\s*\(([A-D])\)", text, re.IGNORECASE)
    if m:
        return f"({m.group(1).upper()})"

    # 2. First parenthesised choice anywhere (same as evaluate_mmlu.py)
    m = re.search(r"\(([A-D])\)", text)
    if m:
        return f"({m.group(1).upper()})"

    return ""


def extract_abcdj(text: str) -> str:
    """Extract A–J MCQ answer from MMLU-Pro model output.

    Priority (mirrors k2/mmlu_pro.py):
      1. "the answer is (X)" or "the answer is X"  for A–J
      2. Exact single capital letter A–J on its own line (last such line)

    Returns the letter only, e.g. "A" (no parentheses — matches mmlu_pro target format).
    Returns "" if nothing found.
    """
    # 1. "the answer is X" for A-J (with optional parens/spaces/punctuation)
    m = re.search(r"[Tt]he answer is[\s(]*([A-J])[\s).]*", text)
    if m:
        return m.group(1).upper()

    # 2. Exact single capital letter on its own line (last match)
    for line in reversed(text.strip().splitlines()):
        m = re.search(r"^([A-J])$", line.strip())
        if m:
            return m.group(1).upper()

    return ""


# ---------------------------------------------------------------------------
# BBH  (free-text targets: True/False, (A)/(B)/..., or short phrases)
# ---------------------------------------------------------------------------

def extract_bbh(text: str) -> str:
    """Extract BBH answer from model output (from k2/bbh.py).

    Priority:
      1. "the answer is <text>"  — captures everything until newline or period
      2. Last word of the last line that contains "answer"

    Returns a stripped string (no trailing punctuation).
    Returns "" if nothing found.
    """
    # 1. "the answer is X"
    m = re.search(r"the answer is ([^\n.]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(".")

    # 2. Last line containing "answer" — take last token
    for line in reversed(text.strip().splitlines()):
        if "answer" in line.lower():
            return line.split()[-1].strip(".").strip()

    return ""


# ---------------------------------------------------------------------------
# GPQA  (\boxed{A} format, then "answer is X", then (X))
# ---------------------------------------------------------------------------

def extract_gpqa(text: str) -> str:
    """Extract A–D answer from GPQA model output (from evaluate_gpqa_new.py).

    Priority:
      1. \\boxed{X}  — supports \\boxed{A}, \\\\boxed{A}, $\\boxed{A}$, etc.
      2. "answer is X"  — case-insensitive
      3. Last "(X)" anywhere in the text

    Returns the letter only, e.g. "A".
    Returns "" if nothing found.
    """
    # 1. \boxed{X}
    m = re.search(r"\\*boxed\s*[{[(]?\s*([A-Da-d])\s*[}\])]?", text)
    if m:
        return m.group(1).upper()

    # 2. "answer is X"
    m = re.search(r"(?:the\s+)?answer\s+is\s*:?\s*([A-Da-d])(?:\.|,|\s|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3. Last "(X)"
    matches = list(re.finditer(r"\(([A-Da-d])\)", text))
    if matches:
        return matches[-1].group(1).upper()

    return ""
