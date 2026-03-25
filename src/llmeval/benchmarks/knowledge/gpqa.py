"""GPQA Diamond benchmark — 5-shot CoT, pass@k evaluation.

Dataset  : data/knowledge/gpqa/gpqa_diamond_test.jsonl  (local)
Columns  : id, question, answer  (answer is a plain letter "A"–"D")

Prompt format (from evaluate_gpqa_new.py):
  Q: <question with choices embedded + "Please reason step-by-step and put
     your choice letter without any other text with \\boxed{} in the end.">
  A: <CoT reasoning ending with \\boxed{X}>

  (repeated 5 times as few-shot prefix, then the test question)

Answer extraction: \\boxed{X} → "answer is X" → last (X)
"""

import json
from typing import Any, Dict, List

from llmeval.benchmarks.base import BaseBenchmark
from llmeval.benchmarks.knowledge.answer_extractor import extract_gpqa

# 5-shot examples (from evaluate_gpqa_new.py FEWSHOT_EXAMPLES)
_FEWSHOT_EXAMPLES = [
    {
        "question": (
            "A large gene has dozens of exons, of which the central ones code for folded triple "
            "helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. "
            "Each exon usually codes for one folded triple alpha helix. The most common mutations "
            "of the gene are central exon deletions that create out-of-frame peptides and progressive "
            "degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' "
            "end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome "
            "and creates exon skipping and in-frame joining. Several missing exons are well tolerated "
            "by an organism. Which structure below is not involved in the proposed therapy?\n\n"
            "A. R-loops.\nB. lariat.\nC. polyA tail.\nD. antisense.\n\n"
            "Please reason step-by-step and put your choice letter without any other text with "
            r"\boxed{} in the end."
        ),
        "answer": (
            "The text describes the dystrophin gene and the FDA-approved oligonucleotide therapy "
            "that causes exon skipping by creating a functional, albeit shorter, dystrophin protein. "
            "Morpholino is bound to the pre-mRNA in an antisense orientation. Every splicing mechanism "
            "creates the lariat molecule that is circular with a 3' tail and soon degraded. The spliced "
            "RNA is polyadenylated at the 3' end. R-loops are triple helix of DNA and the pre-mRNA and "
            r"a consequence of the RNA transcription, not splicing and RNA maturation. Therefore, the answer is \boxed{A}."
        ),
    },
    {
        "question": (
            "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, "
            "respectively. We want to clearly distinguish these two energy levels. Which one of the "
            "following options could be their energy difference so that they can be clearly resolved?\n\n"
            "A. 10^-9 eV\nB. 10^-8 eV\nC. 10^-11 eV\nD. 10^-4 eV\n\n"
            r"Please reason step-by-step and put your choice letter without any other text with \boxed{} in the end."
        ),
        "answer": (
            "According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime "
            "and Delta E is the width of the energy level. With Delta t=10^-9 s==> Delta E1= 3.3 10^-7 ev. "
            "And Delta t=10^-8 s gives Delta E2=3.3*10^-8 eV. Therefore, the energy difference between "
            r"the two states must be significantly greater than 10^-7 ev. So the answer is \boxed{D}."
        ),
    },
    {
        "question": (
            "How many of the following compounds exhibit optical activity?\n"
            "1-methyl-4-(prop-1-en-2-yl)cyclohex-1-ene\n2,3,3,3-tetrafluoroprop-1-ene\n"
            "di(cyclohex-2-en-1-ylidene)methane\n5-(5-methylhexan-2-ylidene)cyclopenta-1,3-diene\n"
            "3-(2-methylbut-1-en-1-ylidene)cyclohex-1-ene\n[1,1'-biphenyl]-3,3'-diol\n"
            "8,8-dichlorobicyclo[4.2.0]octan-7-one\ncyclopent-2-en-1-one\n\n"
            "A. 3\nB. 4\nC. 5\nD. 6\n\n"
            r"Please reason step-by-step and put your choice letter without any other text with \boxed{} in the end."
        ),
        "answer": (
            "The compounds 1-methyl-4-(prop-1-en-2-yl)cyclohex-1-ene, "
            "3-(2-methylbut-1-en-1-ylidene)cyclohex-1-ene, di(cyclohex-2-en-1-ylidene)methane, "
            "and 8,8-dichlorobicyclo[4.2.0]octan-7-one are chiral molecules, and thus will be "
            "optically active. All the others have a mirror plane of symmetry, and will be achiral. "
            r"Therefore, the answer is \boxed{B}."
        ),
    },
    {
        "question": (
            "A coating is applied to a substrate resulting in a perfectly smooth surface. The measured "
            "contact angles of this smooth coating are 132° and 102° for water and hexadecane respectively. "
            "The coating formulation is then modified and when now applied to the same type of substrate, "
            "a rough surface is produced. When a droplet of water or oil sits on the rough surface, the "
            "wettability of the surface can now be described by the Cassie-Baxter state. The water contact "
            "angle on the rough surface is now 148°. What would be the best estimate of the contact angle "
            "of a droplet of octane on the rough surface?\n\n"
            "A. 129°\nB. 134°\nC. 124°\nD. 139°\n\n"
            r"Please reason step-by-step and put your choice letter without any other text with \boxed{} in the end."
        ),
        "answer": (
            "In the Cassie-Baxter state, droplets are in contact with a non-uniform surface. "
            "Using the water data (θCB=148°, θ1=132°, θ2=180°) we get f1=0.46, f2=0.54. "
            "For hexadecane (θ1=102°): θCB=129°. Octane has lower surface tension than hexadecane, "
            r"so its contact angle is lower than 129°. Therefore, the answer is \boxed{C}."
        ),
    },
    {
        "question": (
            "In a parallel universe where a magnet can have an isolated North or South pole, "
            "Maxwell's equations look different. But, specifically, which of those equations are different?\n\n"
            "A. The ones related to the circulation of the electric field and the divergence of the magnetic field.\n"
            "B. The ones related to the divergence and the curl of the magnetic field.\n"
            "C. The one related to the divergence of the magnetic field.\n"
            "D. The one related to the circulation of the magnetic field and the flux of the electric field.\n\n"
            r"Please reason step-by-step and put your choice letter without any other text with \boxed{} in the end."
        ),
        "answer": (
            "Magnetic monopoles would modify: (1) div B = rho_m (magnetic charge density) and "
            "(2) curl E = -J_m (magnetic current). The other two equations (div E and curl B) "
            r"remain structurally unchanged. Therefore, the answer is \boxed{A}."
        ),
    },
]

_FEWSHOT_PREFIX = "".join(
    f"Q: {ex['question']}\nA: {ex['answer']}\n\n" for ex in _FEWSHOT_EXAMPLES
)


class GPQABenchmark(BaseBenchmark):
    """GPQA Diamond — 5-shot CoT, pass@k."""

    def load_dataset(self) -> List[Dict[str, Any]]:
        path = self._resolve_local_path(self.benchmark.dataset.path)
        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples

    def build_prompt(self, example: Dict[str, Any]) -> str:
        return _FEWSHOT_PREFIX + f"Q: {example['question']}\nA:"

    def check_answer(self, prediction: str, example: Dict[str, Any]) -> bool:
        gold = example["answer"].strip().upper()
        pred = extract_gpqa(prediction)
        return pred == gold
