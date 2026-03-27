"""Answer extraction and comparison from Qwen2.5-Math evaluation.

Ported from:
  - RL-eval/qwen2.5-math/evaluation/parser.py  (extract_answer, strip_string)
  - RL-eval/qwen2.5-math/evaluation/grader.py   (math_equal, symbolic_equal)

This is more stable than math_verify for MATH-500 style problems that involve
LaTeX expressions, fractions, symbolic answers, etc.

Public API:
    qwen_extract_answer(text, data_name="math500")  -- extract final answer string
    qwen_math_equal(prediction, reference)           -- True if equivalent
    qwen_strip_string(s)                             -- normalize answer string
"""

import re
import regex
from math import isclose
from typing import Union

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
from word2number import w2n


# ---------------------------------------------------------------------------
# String normalization (from parser.py)
# ---------------------------------------------------------------------------

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except:
        return string


def _fix_sqrt(string):
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except:
        pass
    return text


_UNIT_TEXTS = [
    "east", "degree", "mph", "kmph", "ft", "m sqaure", " m east", "sq m",
    "deg", "mile", "q .", "monkey", "prime", "ratio", "profit of rs", "rd",
    "o", "gm", "p . m", "lb", "tile", "per", "dm", "lt", "gain", "ab",
    "way", "west", "a .", "b .", "c .", "d .", "e .", "f .", "g .", "h .",
    "t", "a", "h", "no change", "men", "soldier", "pie", "bc", "excess",
    "st", "inches", "noon", "percent", "by", "gal", "kmh", "c", "acre",
    "rise", "a . m", "th", "\u03c0 r 2", "sq", "mark", "l", "toy", "coin",
    "sq . m", "gallon", "\u00b0 f", "profit", "minw", "yr", "women", "feet",
    "am", "pm", "hr", "cu cm", "square", "v \u00e2 \u20ac \u2122", "are",
    "rupee", "rounds", "cubic", "cc", "mtr", "s", "ohm", "number", "kmph",
    "day", "hour", "minute", "min", "second", "man", "woman", "sec", "cube",
    "mt", "sq inch", "mp", "\u220f cm \u00b3", "hectare", "more", "sec",
    "unit", "cu . m", "cm 2", "rs .", "rs", "kg", "g", "month", "km", "m",
    "cm", "mm", "apple", "liter", "loss", "yard", "pure", "year", "increase",
    "decrease", "d", "less", "Surface", "litre", "pi sq m", "s .", "metre",
    "meter", "inch",
]
_UNIT_TEXTS.extend([t + "s" for t in _UNIT_TEXTS])


def qwen_strip_string(string, skip_unit=False):
    """Normalize an answer string (from qwen2.5-math parser.py strip_string)."""
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")

    # matrix
    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    if not skip_unit:
        for _ in range(2):
            for unit_text in _UNIT_TEXTS:
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    string = _convert_word_number(string)
    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if (
        string.startswith("{") and string.endswith("}") and string.isalnum()
        or string.startswith("(") and string.endswith(")") and string.isalnum()
        or string.startswith("[") and string.endswith("]") and string.isalnum()
    ):
        string = string[1:-1]

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")
    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace('"', "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    return string


# ---------------------------------------------------------------------------
# Answer extraction (from parser.py)
# ---------------------------------------------------------------------------

def qwen_extract_answer(pred_str, data_name="math500", use_last_number=True):
    """Extract the final answer from model output (qwen2.5-math style)."""
    pred = ""
    pred_str = pred_str.replace("\u043a\u0438", "")

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        matches = list(re.finditer(r'he answer is[^0-9-]*(-?\d+)', pred_str))
        if matches:
            pred = matches[-1].group(1)
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    else:
        if use_last_number:
            pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
            found = re.findall(pattern, pred_str.replace(",", ""))
            numbers = [m[0] if m[0] else m[1] for m in found]
            if len(numbers) >= 1:
                pred = numbers[-1]
            else:
                pred = ""
        else:
            pred = ""

    pred = re.sub(r"\n\s*", "", str(pred) if pred is not None else "")
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = qwen_strip_string(pred)
    return pred


# ---------------------------------------------------------------------------
# Answer comparison (from grader.py)
# ---------------------------------------------------------------------------

def _parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except:
                pass
    return None


def _is_digit(num):
    return _parse_digits(num) is not None


def _numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)


def _str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []
    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)
    return ", ".join(pmatrix_list)


def _symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except:
                try:
                    return f(s)
                except:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except:
        pass

    try:
        if _numeric_equal(float(N(a)), float(N(b))):
            return True
    except:
        pass

    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except:
        pass

    return False


def _choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    pred = pred.rstrip(".").rstrip("/")
    return pred


def qwen_math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
) -> bool:
    """Check if prediction matches reference (from qwen2.5-math grader.py)."""
    if prediction is None or reference is None:
        return False
    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and _choice_answer_clean(prediction) == reference
    ):
        return True

    try:
        if _is_digit(prediction) and _is_digit(reference):
            prediction_f = _parse_digits(prediction)
            reference_f = _parse_digits(reference)
            if include_percentage:
                gt_result = [reference_f / 100, reference_f, reference_f * 100]
            else:
                gt_result = [reference_f]
            for item in gt_result:
                try:
                    if is_close:
                        if _numeric_equal(prediction_f, item):
                            return True
                    else:
                        if item == prediction_f:
                            return True
                except Exception:
                    continue
            return False
    except:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = _str_to_pmatrix(reference)

    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    ) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                qwen_math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                for i in range(len(pred_parts))
            ):
                return True

    if (
        (prediction.startswith("\\begin{pmatrix}") or prediction.startswith("\\begin{bmatrix}"))
        and (prediction.endswith("\\end{pmatrix}") or prediction.endswith("\\end{bmatrix}"))
        and (reference.startswith("\\begin{pmatrix}") or reference.startswith("\\begin{bmatrix}"))
        and (reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}"))
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}"):-len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        qwen_math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                        for i in range(len(pred_parts))
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if _symbolic_equal(pred, ref) or _symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if qwen_math_equal(prediction.split("=")[1], reference, include_percentage, is_close):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if qwen_math_equal(prediction, reference.split("=")[1], include_percentage, is_close):
            return True

    if _symbolic_equal(prediction, reference):
        return True

    return False
