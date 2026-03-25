"""Math answer extraction and comparison utilities.

Ported from:
  - new_pipeline/evaluate_gsm8k.py  (parse_answer_with_verify, compare_answers)
  - new_pipeline/node/aime_verifier.py  (_find_last_boxed, score_answer)

Primary approach: math_verify.parse() + math_verify.verify()
Fallback:        regex patterns → string normalisation

Public API:
    extract_answer(text)                    — extract final answer string from model output
    compare_math_answers(gold, pred)        — True if pred matches gold mathematically
"""

import re
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Boxed extraction (nested-brace-aware)
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
    """Return content of the last \\boxed{...} in *text*, handling nested braces."""
    if "boxed" not in text:
        return None
    idx = text.rfind("boxed")
    after = text[idx + len("boxed"):]
    if not after or after[0] != "{":
        m = re.match(r"([^\s$]+)", after)
        return m.group(1) if m else None

    stack = 1
    result = []
    for ch in after[1:]:
        if ch == "{":
            stack += 1
            result.append(ch)
        elif ch == "}":
            stack -= 1
            if stack == 0:
                break
            result.append(ch)
        else:
            result.append(ch)
    return "".join(result).strip()


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# Regex fallback patterns (ordered by specificity, checked in order)
_ANSWER_PATTERNS = [
    r"The answer is:?\s*\$?([\-0-9\.,]+)",
    r"#### ?\$?([\-0-9\.,]+)",
    r"Therefore,? the answer is:?\s*\$?([\-0-9\.,]+)",
    r"So,? the answer is:?\s*\$?([\-0-9\.,]+)",
    r"Thus,? the answer is:?\s*\$?([\-0-9\.,]+)",
    r"Hence,? the answer is:?\s*\$?([\-0-9\.,]+)",
    r"Final answer:?\s*\$?([\-0-9\.,]+)",
    r"The final answer is:?\s*\$?([\-0-9\.,]+)",
]


def extract_answer(text: str) -> str:
    """Extract the final answer from model output.

    Priority:
      1. math_verify.parse() — handles \\boxed and inline math natively
      2. \\boxed{...} regex (nested-brace-aware)
      3. Common English answer patterns ("The answer is …", "####", etc.)
      4. Last number in the last non-empty sentence

    Returns an empty string if nothing is found.
    """
    text = text.strip()

    # 1. math_verify.parse — best effort; returns None if no extractable answer
    try:
        from math_verify import parse  # type: ignore[import]
        parsed = parse(text)
        if parsed is not None:
            return str(parsed)
    except Exception:
        pass

    # 2. \boxed{...}
    boxed = _extract_boxed(text)
    if boxed:
        return _clean(boxed)

    # 3. Regex patterns
    for pat in _ANSWER_PATTERNS:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            ans = matches[-1].replace(",", "").strip().rstrip(".")
            if ans:
                return ans

    # 4. Last number in last non-empty sentence
    sentences = [s for s in text.split(".") if s.strip()]
    for sent in reversed(sentences):
        if "Human:" in sent or "Assistant:" in sent:
            continue
        numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", sent)
        if numbers:
            return numbers[-1].lstrip("0") or "0"

    return ""


def _clean(s: str) -> str:
    s = re.sub(r"\n\s*", "", s.strip())
    return s.strip("./: ")


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def compare_math_answers(gold: str, pred: str) -> bool:
    """Return True if *pred* is mathematically equivalent to *gold*.

    Uses math_verify.parse() + math_verify.verify() when available;
    falls back to normalised string equality.
    """
    if not pred:
        return False
    if gold.strip().lower() == pred.strip().lower():
        return True

    try:
        from math_verify import parse, verify  # type: ignore[import]
        gold_parsed = parse(gold)
        pred_parsed = parse(pred)
        if gold_parsed is not None and pred_parsed is not None:
            return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        pass

    # String normalisation fallback
    def _norm(s: str) -> str:
        return s.replace(",", "").strip().lower().rstrip(".")

    return _norm(gold) == _norm(pred)
