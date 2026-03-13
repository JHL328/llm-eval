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

## Implementation Progress

### Phase 0 — Repo Init & Submodules
- [x] `mkdir llm-eval && git init && git branch -m main`
- [ ] `gh repo create llm-eval --private`
- [ ] Add 4 submodules under `third_party/`
- [x] `plan.md` added to project root

### Phase 1 — Scaffold
- [ ] `pyproject.toml`
- [ ] `.gitignore`
- [ ] `config/cluster.yaml`
- [ ] `config/models.yaml`
- [ ] `config/tasks.yaml`

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
