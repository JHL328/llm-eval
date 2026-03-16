# Plan: Clean LLM Evaluation Framework (llm-eval)

## Context
The existing `RL-eval` repo is a functional but hard-to-maintain LLM evaluation framework.
Key pain points: 15+ near-duplicate `evaluate_*.py` scripts, hardcoded paths throughout,
scattered config, 58MB JSON files in git, mixed legacy+new architecture, and manual model registry updates.

This plan creates a **new repo from scratch** using Clean Architecture + DDD principles,
with zero changes to the original `RL-eval/` folder.

---

## Bounded Contexts (DDD)

| Context | Responsibility |
|---|---|
| **Model Registry** | What models exist, their paths, type (base/sft), GPU needs |
| **Benchmark Catalog** | Task definitions, datasets, prompts, answer extraction, metrics |
| **Job Orchestration** | SLURM job lifecycle: create → submit → monitor → complete/fail |
| **Inference** | vLLM backend abstraction |
| **Results** | Atomic storage, pass@k aggregation |

---

## Repository Structure

```
llm-eval/                                  ← new root (outside RL-eval/)
├── pyproject.toml
├── .gitignore
├── README.md
├── plan.md                                ← this file
│
├── config/                                ← all user-facing config, no Python
│   ├── cluster.yaml                       ← conda path, SLURM partition, output root
│   ├── models.yaml                        ← model registry (name, path, type, gpus)
│   └── tasks.yaml                         ← per-task: sampling params, dataset source, k_list
│
├── src/
│   └── llmeval/
│       ├── domain/                        ← pure Python, zero I/O, zero frameworks
│       │   ├── model.py                   ← Model entity, ModelType(base|sft) enum
│       │   ├── benchmark.py               ← Benchmark entity, BenchmarkCategory enum
│       │   ├── eval_job.py                ← EvalJob entity, JobStatus enum
│       │   ├── eval_result.py             ← EvalResult value object
│       │   └── sampling_config.py         ← SamplingConfig value object (temp, top_p, n)
│       │
│       ├── application/                   ← use cases, orchestrates domain objects
│       │   ├── submit_evaluation.py       ← SubmitEvaluationUseCase
│       │   ├── monitor_jobs.py            ← MonitorJobsUseCase (replaces ModelQueue)
│       │   └── aggregate_results.py       ← AggregateResultsUseCase (pass@k)
│       │
│       ├── benchmarks/                    ← one subpackage per benchmark category
│       │   ├── base.py                    ← abstract BaseBenchmark
│       │   ├── math/
│       │   │   ├── gsm8k.py
│       │   │   ├── math500.py
│       │   │   ├── aime.py
│       │   │   └── answer_extractor.py    ← extract_boxed(), extract_numeric()
│       │   ├── knowledge/
│       │   │   ├── mmlu.py                ← pass@1 only
│       │   │   ├── mmlu_pro.py            ← pass@1 only
│       │   │   ├── mmlu_flan.py           ← pass@1 only
│       │   │   ├── bbh.py                 ← pass@1 only
│       │   │   ├── gpqa.py
│       │   │   ├── ifeval.py              ← wraps lm-harness submodule
│       │   │   └── answer_extractor.py    ← extract_mcq_choice()
│       │   ├── code/
│       │   │   ├── humaneval.py
│       │   │   ├── mbpp.py
│       │   │   └── code_executor.py       ← wraps evalplus submodule
│       │   └── likelihood/
│       │       └── likelihood_tasks.py    ← ARC, HellaSwag, DROP, PIQA, etc.
│       │
│       ├── infrastructure/                ← I/O adapters
│       │   ├── slurm/
│       │   │   ├── job_submitter.py
│       │   │   ├── job_monitor.py         ← simple squeue polling (see note)
│       │   │   └── templates/
│       │   │       └── slurm_job.sh.j2    ← single Jinja2 SLURM template
│       │   ├── vllm/
│       │   │   └── inference_runner.py
│       │   ├── config_loader.py           ← YAML → domain objects
│       │   └── result_store.py            ← atomic JSON read/write (fcntl)
│       │
│       └── interfaces/
│           └── cli/
│               └── submit.py              ← unified CLI entry point
│
├── third_party/                           ← git submodules (version-pinned)
│   ├── lm-evaluation-harness              ← EleutherAI
│   ├── qwen2.5-math                       ← QwenLM
│   ├── evalplus                           ← evalplus team
│   └── lighteval                          ← HuggingFace
│
└── tests/
    ├── unit/
    │   ├── domain/
    │   └── benchmarks/
    └── integration/
        └── dry_run/
```

---

## Key Design Decisions

### 1. One `BaseBenchmark` class replaces 15 scripts
```python
class BaseBenchmark(ABC):
    def build_prompt(self, example: dict) -> str: ...
    def check_answer(self, prediction: str, ground_truth: str) -> bool: ...
    def get_sampling_config(self) -> SamplingConfig: ...
    def get_dataset(self) -> Dataset: ...
```

### 2. All paths in `config/cluster.yaml`, zero hardcoding
```yaml
conda_env: /mnt/weka/home/haolong.jia/miniconda3/envs/llmeval
output_root: /mnt/weka/shrd/k2m/haolong.jia/result
slurm_partition: gpu
max_concurrent_jobs: 10
```

### 3. Model registry as `config/models.yaml`
Adding a new model = edit YAML only, no Python changes.

### 4. Single Jinja2 SLURM template replaces all `.sh` scripts

### 5. Unified CLI
```bash
python -m llmeval submit --task gsm8k --type sft
python -m llmeval submit --task mmlu --dry-run
python -m llmeval results --task gsm8k
```

### 6. Library-first
| Old | New |
|---|---|
| 58MB JSON prompts in git | HuggingFace `datasets` |
| Custom pass@k in each script | `evalplus.eval.estimate_pass_at_k` |
| Inline fcntl in 3+ places | Centralized `result_store.py` |
| Custom retry logic | `tenacity` |
| Hardcoded `.sh` scripts | `jinja2` templates |

---

## Metrics

| Benchmark | Metric |
|---|---|
| MMLU, MMLU-Pro, MMLU-FLAN, BBH | **pass@1 only** (n=1, temp=0, greedy) |
| GSM8K, MATH500, AIME, GPQA | pass@k, k_list from `tasks.yaml` |
| HumanEval, MBPP | pass@k via `evalplus` |
| Likelihood tasks | accuracy |

---

## Job Monitoring
Simple `squeue` polling (from `evaluate_gsm8k.py` pattern) — no lock files, no state machine:
1. Submit all jobs, collect SLURM IDs
2. Poll `squeue -h -j <ids>` every 30s
3. Done when list is empty; completion verified by `result.json`

> The complicated monitoring in `evaluate.py` is **not** used — it was a workaround
> for SLURM jobs not exiting cleanly, which is no longer an issue.

---

## Third-Party Submodules

| Path | Remote | Used by |
|---|---|---|
| `third_party/lm-evaluation-harness` | github.com/EleutherAI/lm-evaluation-harness | IFEval, MMLU-Redux, ARC, HellaSwag |
| `third_party/qwen2.5-math` | github.com/QwenLM/Qwen2.5-Math | Math answer verification |
| `third_party/evalplus` | github.com/evalplus/evalplus | HumanEval+, MBPP+, pass@k |
| `third_party/lighteval` | github.com/huggingface/lighteval | Additional benchmarks |

---

## Learnings from xllm (reference repo at `../xllm/`)

The `xllm` repo implements 33+ benchmarks with a different engine (single-process inference,
no SLURM, no vLLM). We don't adopt its engine, but the following benchmark-specific details
inform our implementation:

### Code benchmarks (`benchmarks/code/`)
- **Execution isolation**: Use multiprocess-based timeout + `reliability_guard()` (disable
  `os.fork`, `os.kill`, resource limits). Do NOT use docker. This is sufficient for HumanEval/MBPP.
  Our `code_executor.py` should implement this pattern on top of evalplus test suites.
- **evalplus advantage**: evalplus provides extended test suites (HumanEval+ / MBPP+) which
  are more robust than xllm's original test cases — keep using evalplus.
- **HumanEval prompt**: `prompt` (function signature + docstring) + model prediction.
  Prediction must be indented (4 spaces). Full executable = `prompt + prediction`.
- **MBPP prompt format** (adopt from xllm):
  ```
  You are an expert Python programmer, and here is your task: {context}
  Your code should pass these tests:

  {tests}
  [BEGIN]
  ```
  Stop generation at `[DONE]` token.
- **Code sampling temperature**: xllm uses `temperature=1.0` for pass@k code tasks.
  We use `temperature=0.6` (from RL-eval, consistent with math tasks). Either is acceptable;
  0.6 is kept for consistency.
- **Pass@k calculation**: xllm uses exact "did any of k generations pass" (binary).
  We use `evalplus.eval.estimate_pass_at_k` (combinatorial estimator) — more robust for
  sparse k values and the standard in literature.

### BBH (`benchmarks/knowledge/bbh.py`)
- Few-shot examples should be **category-aware**: select from the same BBH subtask category.
  Each of the 23 subtasks has its own CoT prompt examples (from `bbh_cot_prompts.json`).
  Load category-specific examples, not a shared pool.

### MMLU-Pro (`benchmarks/knowledge/mmlu_pro.py`)
- Few-shot examples are also **category-aware**: select from the same subject category.

### Max tokens
- xllm uses very short max tokens (HumanEval: 512, MBPP: 256, BBH: 32).
- We keep **max_tokens=4096** for all tasks to allow full CoT reasoning chains.

---

## Current File Structure

```
llm-eval/
├── pyproject.toml                         ✅
├── .gitignore                             ✅
├── plan.md                                ✅
│
├── config/
│   ├── cluster.yaml                       ✅
│   ├── models.yaml                        ✅
│   └── tasks.yaml                         ✅
│
├── data/                                  ✅  ← local benchmark datasets (committed to git)
│   ├── math/
│   │   └── gsm8k_test.jsonl               ✅ (744KB, copied from RL-eval)
│   ├── knowledge/
│   │   ├── bbh/
│   │   │   └── bbh_cot_prompts.json       ✅ (74KB, category-aware CoT prompts)
│   │   ├── mmlu/
│   │   │   ├── mmlu_prompts.json          ✅ (206KB)
│   │   │   └── mmlu_cot_prompts.json      ✅ (214KB, FLAN CoT prompts)
│   │   └── gpqa/
│   │       └── gpqa_diamond_test.jsonl    ✅ (151KB)
│   ├── code/                              (empty — HumanEval/MBPP loaded via evalplus)
│   └── likelihood/                        (empty — loaded via lm-harness)
│
├── third_party/
│   ├── lm-evaluation-harness              ✅ (submodule)
│   ├── qwen2.5-math                       ✅ (submodule)
│   ├── evalplus                           ✅ (submodule)
│   └── lighteval                          ✅ (submodule)
│
└── src/                                   (Phase 2–5, not yet created)
    └── llmeval/
        ├── domain/
        ├── application/
        ├── benchmarks/
        ├── infrastructure/
        └── interfaces/
```

> **Data loading strategy per task:**
> | Task | Source |
> |---|---|
> | GSM8K | `data/math/gsm8k_test.jsonl` (local) |
> | MATH500, AIME | HuggingFace (no local copy) |
> | GPQA | `data/knowledge/gpqa/gpqa_diamond_test.jsonl` (local) |
> | MMLU | `data/knowledge/mmlu/mmlu_prompts.json` (local) |
> | MMLU-FLAN | `data/knowledge/mmlu/mmlu_cot_prompts.json` (local) |
> | MMLU-Pro | HuggingFace (56MB file excluded from git) |
> | BBH | `data/knowledge/bbh/bbh_cot_prompts.json` (local) |
> | HumanEval, MBPP | evalplus submodule |
> | Likelihood tasks | lm-evaluation-harness submodule |

---

## Implementation Progress

### Phase 0 — Repo Init & Submodules
- [x] `mkdir llm-eval && git init && git branch -m main`
- [x] `gh repo create llm-eval --private` (created manually via GitHub UI)
- [x] Add 4 submodules under `third_party/`
- [x] `plan.md` added to project root

### Phase 1 — Scaffold
- [x] `pyproject.toml`
- [x] `.gitignore`
- [x] `config/cluster.yaml`
- [x] `config/models.yaml`
- [x] `config/tasks.yaml`

### Phase 2 — Domain Layer
- [ ] `src/llmeval/domain/sampling_config.py`
- [ ] `src/llmeval/domain/model.py`
- [ ] `src/llmeval/domain/benchmark.py`
- [ ] `src/llmeval/domain/eval_job.py`
- [ ] `src/llmeval/domain/eval_result.py`

### Phase 3 — Infrastructure
- [ ] `src/llmeval/infrastructure/config_loader.py`
- [ ] `src/llmeval/infrastructure/result_store.py`
- [ ] `src/llmeval/infrastructure/slurm/templates/slurm_job.sh.j2`
- [ ] `src/llmeval/infrastructure/slurm/job_submitter.py`
- [ ] `src/llmeval/infrastructure/slurm/job_monitor.py`
- [ ] `src/llmeval/infrastructure/vllm/inference_runner.py`

### Phase 4 — Benchmarks
- [ ] `src/llmeval/benchmarks/base.py`
- [ ] `src/llmeval/benchmarks/math/answer_extractor.py`
- [ ] `src/llmeval/benchmarks/math/gsm8k.py`
- [ ] `src/llmeval/benchmarks/math/math500.py`
- [ ] `src/llmeval/benchmarks/math/aime.py`
- [ ] `src/llmeval/benchmarks/knowledge/answer_extractor.py`
- [ ] `src/llmeval/benchmarks/knowledge/mmlu.py`
- [ ] `src/llmeval/benchmarks/knowledge/mmlu_pro.py`
- [ ] `src/llmeval/benchmarks/knowledge/mmlu_flan.py`
- [ ] `src/llmeval/benchmarks/knowledge/bbh.py`
- [ ] `src/llmeval/benchmarks/knowledge/gpqa.py`
- [ ] `src/llmeval/benchmarks/knowledge/ifeval.py`
- [ ] `src/llmeval/benchmarks/code/code_executor.py`
- [ ] `src/llmeval/benchmarks/code/humaneval.py`
- [ ] `src/llmeval/benchmarks/code/mbpp.py`
- [ ] `src/llmeval/benchmarks/likelihood/likelihood_tasks.py`

### Phase 5 — Application + CLI
- [ ] `src/llmeval/application/monitor_jobs.py`
- [ ] `src/llmeval/application/aggregate_results.py`
- [ ] `src/llmeval/application/submit_evaluation.py`
- [ ] `src/llmeval/interfaces/cli/submit.py`

### Phase 6 — Tests
- [ ] `tests/unit/domain/test_sampling_config.py`
- [ ] `tests/unit/benchmarks/test_math_extractor.py`
- [ ] `tests/integration/dry_run/test_submit_dry_run.py`
