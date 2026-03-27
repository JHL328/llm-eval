# llm-eval

A clean, config-driven LLM evaluation framework built with Clean Architecture + DDD principles. Replaces 15+ duplicate evaluation scripts with a unified system: one base class, one SLURM template, one CLI.

## Quick Start

```bash
# Submit a single task for all base models
llmeval submit --task gsm8k --type base

# Submit specific model + task
llmeval submit --task math500 --model Qwen3-0.6B

# Submit multiple tasks at once
llmeval submit --task gsm8k --task math500 --type base --wait

# Dry-run (prints SLURM script, no submission)
llmeval submit --task gsm8k --model Qwen3-0.6B --dry-run

# View results
llmeval results --task gsm8k

# Check running jobs
llmeval status
```

## Project Structure

```
llm-eval/
├── config/                           # All user-facing configuration (YAML)
│   ├── cluster.yaml                  # Conda envs, SLURM defaults, output root
│   ├── models.yaml                   # Model registry (name, path, type, gpus)
│   └── tasks.yaml                    # Per-task: sampling params, dataset, k_list
│
├── data/                             # Local benchmark datasets
│   ├── math/
│   │   ├── gsm8k_test.jsonl
│   │   └── math500_test.jsonl
│   └── knowledge/
│       ├── bbh/bbh_cot_prompts.json
│       ├── mmlu/mmlu_prompts.json
│       ├── mmlu/mmlu_cot_prompts.json
│       └── gpqa/gpqa_diamond_test.jsonl
│
├── out/                              # Evaluation outputs (git-ignored)
│   └── <task>/<model>/
│       ├── result.json               # Metrics (accuracy, pass@k)
│       ├── generations.jsonl         # All model generations per example
│       ├── correctness.csv           # Per-sample correctness matrix
│       ├── job.sh                    # Generated SLURM script
│       └── slurm.{out,err}          # Job logs
│
├── src/llmeval/                      # Main package
│   ├── domain/                       # Pure domain entities (no I/O)
│   ├── application/                  # Use cases (orchestration)
│   ├── benchmarks/                   # Benchmark implementations
│   ├── infrastructure/               # I/O adapters (SLURM, vLLM, config, storage)
│   └── interfaces/cli/              # CLI entry points
│
├── third_party/                      # Git submodules (version-pinned)
│   ├── lm-evaluation-harness/        # EleutherAI (IFEval, ARC, HellaSwag, etc.)
│   ├── qwen2.5-math/                 # QwenLM (math answer verification)
│   ├── evalplus/                     # HumanEval+, MBPP+, pass@k
│   └── lighteval/                    # HuggingFace (additional benchmarks)
│
└── tests/
```

## Architecture

The codebase follows **Clean Architecture** with four layers. Dependencies point inward only: interfaces -> application -> domain.

```
┌─────────────────────────────────────────────────┐
│  interfaces/cli/          (CLI entry points)     │
│    submit.py              llmeval submit/results │
│    run_job.py             GPU-side job runner    │
├─────────────────────────────────────────────────┤
│  application/             (use cases)            │
│    submit_evaluation.py   build & submit jobs    │
│    monitor_jobs.py        poll squeue            │
│    aggregate_results.py   collect & display      │
├─────────────────────────────────────────────────┤
│  infrastructure/          (I/O adapters)         │
│    config_loader.py       YAML -> domain objects │
│    result_store.py        atomic JSON read/write │
│    slurm/                 sbatch + templates     │
│    vllm/                  vLLM inference wrapper │
├─────────────────────────────────────────────────┤
│  domain/                  (pure entities, no I/O)│
│    model.py               sampling_config.py     │
│    benchmark.py           eval_job.py            │
│    eval_result.py                                │
└─────────────────────────────────────────────────┘
```

### `domain/` -- Pure Domain Entities

No I/O, no frameworks, no imports from other layers. These are the core data structures.

| File | Description |
|---|---|
| `model.py` | `Model` entity with `name`, `path`, `type` (BASE or SFT), `gpus`. `ModelType` enum controls whether chat template is applied. |
| `sampling_config.py` | `SamplingConfig` value object: `temperature`, `top_p`, `n_sampling`, `max_tokens`, `stop` (list of stop strings), `seed`, `k_list` (for pass@k), `max_model_len` (vLLM context window). Has `is_greedy` property (True when temperature=0). |
| `benchmark.py` | `Benchmark` entity with `name`, `category` (MATH/KNOWLEDGE/CODE/LIKELIHOOD), `sampling_config`, `dataset` config, `num_shots`, `prompt_type`, SLURM overrides. `is_pass_at_k` property checks if k_list has values > 1. |
| `eval_job.py` | `EvalJob` entity tracking a single (model, task) SLURM job. Fields: `model`, `benchmark`, `output_dir`, `slurm_job_id`, `status` (PENDING -> RUNNING -> COMPLETED/FAILED). |
| `eval_result.py` | `EvalResult` value object with `metrics` dict (accuracy, pass@k), `n_total`, `n_correct`, `per_category` breakdown. |

### `application/` -- Use Cases

Orchestration logic that coordinates domain objects and infrastructure.

| File | Description |
|---|---|
| `submit_evaluation.py` | `SubmitEvaluationUseCase` -- the main entry point. Given task names and model filters, it loads configs, creates `EvalJob` objects, renders SLURM scripts via Jinja2 template, and submits them via `sbatch`. Routes jobs to the correct conda env (primary for math/knowledge, code env for HumanEval/MBPP, harness env for likelihood tasks). |
| `monitor_jobs.py` | `MonitorJobsUseCase` -- polls `squeue -h -j <ids>` every 30 seconds until all SLURM jobs finish. Detects completion by the appearance of `result.json` or disappearance from squeue. |
| `aggregate_results.py` | `collect_results()` reads all `result.json` files under the output root. `print_results_table()` formats them as a markdown table. |

### `benchmarks/` -- Benchmark Implementations

Every benchmark inherits from `BaseBenchmark` and implements three methods:

```python
class BaseBenchmark(ABC):
    def load_dataset(self) -> List[Dict]:        # Load evaluation examples
    def build_prompt(self, example: dict) -> str: # Build input prompt for one example
    def check_answer(self, pred: str, ex: dict) -> bool:  # Is prediction correct?
```

Shared logic in `BaseBenchmark`:
- `build_result()` -- aggregates predictions into `EvalResult` with pass@k via unbiased combinatorial estimator
- `stop_tokens` property -- reads stop sequences from task config
- `extract_answer()` -- extracts answer from raw model output (for logging)

#### `benchmarks/math/`

| File | Description |
|---|---|
| `gsm8k.py` | **GSM8K** -- 8-shot chain-of-thought, pass@k. Few-shot examples hardcoded from the standard set. Supports 0-shot via `num_shots` config. Uses `math_verify` for answer comparison. |
| `math500.py` | **MATH-500** -- 4-shot CoT, pass@k. Uses the **qwen2.5-math grader** (`qwen_math_grader.py`) for answer extraction and comparison, which handles LaTeX expressions, fractions, and symbolic answers more robustly than math_verify. |
| `answer_extractor.py` | Shared math answer utilities using `math_verify` library. `extract_answer()` extracts from boxed/regex/last-number. `compare_math_answers()` uses `math_verify.parse()` + `math_verify.verify()` with string fallback. Used by GSM8K. |
| `qwen_math_grader.py` | Ported from **qwen2.5-math** (RL-eval). Three main functions: `qwen_extract_answer()` (boxed -> "the answer is" -> last number -> `strip_string` normalization), `qwen_strip_string()` (LaTeX/unit/formatting cleanup), `qwen_math_equal()` (numeric -> sympy symbolic -> string comparison). Used by MATH-500. |

#### `benchmarks/knowledge/`

| File | Description |
|---|---|
| `mmlu.py` | **MMLU** -- 5-shot multiple choice, pass@1 (greedy). Loads pre-built prompts from `data/knowledge/mmlu/mmlu_prompts.json`. |
| `mmlu_pro.py` | **MMLU-Pro** -- 5-shot, pass@1. Category-aware few-shot examples from HuggingFace dataset. |
| `mmlu_flan.py` | **MMLU-FLAN** -- 4-shot chain-of-thought with FLAN-style prompts. |
| `bbh.py` | **BIG-Bench Hard** -- 3-shot CoT, pass@1. 23 subtasks, each with its own few-shot examples. Per-subtask accuracy reported in `per_category`. |
| `gpqa.py` | **GPQA Diamond** -- 5-shot CoT, pass@k. Graduate-level science questions. |
| `ifeval.py` | **IFEval** -- delegates to `lm-evaluation-harness` submodule. |
| `answer_extractor.py` | `extract_mcq_choice()` for multiple-choice answer extraction (A/B/C/D/E). |

#### `benchmarks/code/`

| File | Description |
|---|---|
| `humaneval.py` | **HumanEval+** -- code generation, pass@k via `evalplus`. |
| `mbpp.py` | **MBPP+** -- code generation, pass@k via `evalplus`. |
| `code_executor.py` | Sandboxed code execution wrapper around evalplus. |

#### `benchmarks/likelihood/`

| File | Description |
|---|---|
| `likelihood_tasks.py` | **ARC-Easy, ARC-Challenge, HellaSwag, PIQA, Winogrande, TriviaQA, DROP, CommonsenseQA** -- all delegated to `lm-evaluation-harness` submodule. |

### `infrastructure/` -- I/O Adapters

| File | Description |
|---|---|
| `config_loader.py` | `ConfigLoader` -- parses `cluster.yaml`, `models.yaml`, `tasks.yaml` into domain objects. Handles task-to-category mapping, sampling config construction, dataset config resolution. |
| `result_store.py` | `ResultStore` -- atomic JSON read/write using temp file + `os.replace()`. Uses `fcntl` file locking for safe concurrent writes from multiple SLURM jobs. |
| `slurm/job_submitter.py` | `JobSubmitter` -- renders the Jinja2 SLURM template with job-specific parameters, writes `job.sh`, calls `sbatch`. Derives conda root from env path for proper activation. |
| `slurm/job_monitor.py` | `wait_for_jobs()` -- simple squeue polling loop. Checks every 30s until all job IDs disappear from the queue. |
| `slurm/templates/slurm_job.sh.j2` | Single Jinja2 template for ALL SLURM jobs. Sets `TRITON_CACHE_DIR`, activates conda, runs the eval command. |
| `vllm/inference_runner.py` | `VLLMRunner` -- thin wrapper around `vllm.LLM`. Converts `SamplingConfig` to vLLM `SamplingParams`. Supports `max_model_len` for controlling KV cache allocation and concurrency. Only imported on GPU nodes. |

### `interfaces/cli/` -- Entry Points

| File | Description |
|---|---|
| `submit.py` | Main CLI (`llmeval` command). Subcommands: `submit` (run evaluations), `results` (show results table), `status` (check running jobs). Supports `--task`, `--model`, `--type`, `--dry-run`, `--wait` flags. |
| `run_job.py` | **GPU-side runner** -- invoked by each SLURM job script. Loads the benchmark class from a registry, builds prompts, applies chat template for SFT models, runs vLLM inference, evaluates predictions, and saves outputs (`result.json`, `generations.jsonl`, `correctness.csv`). |
| `parse_lm_harness.py` | Parses lm-evaluation-harness JSON output into our `result.json` format. |

## Configuration

### Adding a new model

Edit `config/models.yaml`:

```yaml
base:
  - name: MyModel-7B
    path: /path/to/checkpoint
    gpus: 1

sft:
  - name: MyModel-7B-Instruct
    path: /path/to/instruct-checkpoint
    gpus: 2
```

- **base**: Raw pretrained model. Prompts are sent as-is (few-shot format).
- **sft**: Instruction-tuned model. `tokenizer.apply_chat_template()` is applied automatically, and EOS/pad tokens are added to stop sequences.

### Adding a new task

Edit `config/tasks.yaml`:

```yaml
my_task:
  sampling:
    temperature: 0.6
    top_p: 0.95
    n_sampling: 16
    max_tokens: 4096
    max_model_len: 8192
    stop: ["</s>", "<|im_end|>", "\n\nQuestion:", "Human:"]
  k_list: [1, 2, 4, 8, 16]
  num_shots: 4
  prompt_type: cot
  slurm:
    gpus: 1
    mem: "800G"
    time: "12:00:00"
  dataset:
    source: local
    path: data/my_task/test.jsonl
```

Then implement a benchmark class inheriting `BaseBenchmark` and register it in `run_job.py`'s `_REGISTRY`.

### Stop tokens

Stop tokens prevent base models from generating beyond the answer. Common patterns:

| Token | Purpose |
|---|---|
| `"</s>"`, `"<\|im_end\|>"`, `"<\|endoftext\|>"` | Model-specific EOS tokens |
| `"</think>"` | End of thinking (reasoning models) |
| `"\n\nQuestion:"` | Prevents generating next few-shot Q&A pair |
| `"Human:"` | Prevents generating next conversational turn |

For SFT models, the tokenizer's EOS and pad tokens are automatically added.

## Output Format

Each evaluation job produces the following files in `out/<task>/<model>/`:

| File | Description |
|---|---|
| `result.json` | Final metrics: `accuracy`, `pass@1`, `pass@2`, ..., `pass@k`, `n_total`, `n_correct` |
| `generations.jsonl` | One line per example: `{"idx": 0, "example": {...}, "generations": ["gen1", "gen2", ...]}` |
| `correctness.csv` | Matrix of 0/1 values. Rows = examples, columns = `sample_0` ... `sample_{n-1}` |
| `job.sh` | The generated SLURM script (for debugging) |
| `slurm.out` | Stdout log, including first example debug info (prompt, all generations, GT, parsed answers, exact match) |
| `slurm.err` | Stderr log (vLLM progress, speed stats) |
| `fail.flag` | Written on failure with traceback (for monitoring) |

## Supported Benchmarks

| Category | Task | Shots | Metric | Method |
|---|---|---|---|---|
| Math | gsm8k | 8-shot CoT | pass@k | vLLM + math_verify |
| Math | gsm8k_8shot_greedy | 8-shot CoT | pass@1 | vLLM + math_verify |
| Math | gsm8k_0shot | 0-shot CoT | pass@1 | vLLM + math_verify |
| Math | math500 | 4-shot CoT | pass@k | vLLM + qwen2.5-math grader |
| Knowledge | mmlu | 5-shot | pass@1 | vLLM greedy |
| Knowledge | mmlu_pro | 5-shot | pass@1 | vLLM greedy |
| Knowledge | mmlu_flan | 4-shot CoT | pass@1 | vLLM greedy |
| Knowledge | bbh | 3-shot CoT | pass@1 | vLLM greedy |
| Knowledge | gpqa | 5-shot CoT | pass@k | vLLM |
| Knowledge | ifeval | 0-shot | accuracy | lm-evaluation-harness |
| Code | humaneval | 0-shot | pass@k | vLLM + evalplus |
| Code | mbpp | 3-shot | pass@k | vLLM + evalplus |
| Likelihood | arc_easy, arc_challenge, hellaswag, piqa, winogrande, triviaqa, drop, commonsense_qa | varies | accuracy | lm-evaluation-harness |

## Three Conda Environments

Different benchmarks require different dependencies, so three conda environments are used:

| Env | Used for | Key packages |
|---|---|---|
| `qwen-eval` | Math, knowledge, GPQA | vLLM, transformers, math_verify, sympy, latex2sympy2 |
| `evalplus-eval` | HumanEval, MBPP | vLLM, transformers, evalplus |
| `harness-eval` | Likelihood tasks, IFEval | lm-eval (lm-evaluation-harness) |

The `SubmitEvaluationUseCase` automatically routes each task to the correct environment.

## Installation

```bash
pip install -e .                    # Core (CLI + config loading)
pip install -e ".[inference]"       # + vLLM for GPU inference
pip install -e ".[dev]"             # + pytest for testing

# Initialize submodules
git submodule update --init --recursive
```

## Design Decisions

1. **Config over code** -- All model paths, sampling parameters, and SLURM settings live in YAML. Adding a model or task requires zero Python changes.
2. **Single BaseBenchmark** -- One abstract base class replaces 15+ duplicate `evaluate_*.py` scripts. Each benchmark implements only `load_dataset`, `build_prompt`, `check_answer`.
3. **Single SLURM template** -- One Jinja2 template generates all job scripts, parameterized by task and model.
4. **Two-phase answer checking** -- GSM8K uses `math_verify` (fast, good for numeric). MATH-500 uses the qwen2.5-math grader (handles LaTeX, fractions, symbolic expressions).
5. **Atomic result storage** -- `ResultStore` uses `os.replace()` + `fcntl` locking for safe concurrent writes from parallel SLURM jobs.
6. **Editable install** -- `pip install -e .` means source changes take effect immediately without reinstalling.
