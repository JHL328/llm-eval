"""Main CLI entry point — `llmeval` command.

Sub-commands:
    submit   Submit evaluation jobs to SLURM.
    results  Display evaluation results in a summary table.
    status   Show per-task/model completion status.

Examples:
    llmeval submit --task gsm8k --task math500 --type sft --wait
    llmeval submit --task mmlu --model Qwen2.5-7B --dry-run
    llmeval results --task gsm8k
    llmeval status
"""

import argparse
import sys


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmeval",
        description="Submit and monitor LLM evaluation jobs on SLURM.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ submit
    p_sub = sub.add_parser("submit", help="Submit evaluation jobs to SLURM.")
    p_sub.add_argument(
        "--task", action="append", dest="tasks", metavar="TASK",
        help="Task to evaluate (repeat for multiple). Omit for all tasks.",
    )
    p_sub.add_argument(
        "--model", action="append", dest="models", metavar="MODEL",
        help="Model name to evaluate (repeat for multiple). Omit for all models.",
    )
    p_sub.add_argument(
        "--type", dest="model_type", choices=["base", "sft"],
        help="Filter models by type.",
    )
    p_sub.add_argument(
        "--dry-run", action="store_true",
        help="Write SLURM scripts without submitting.",
    )
    p_sub.add_argument(
        "--wait", action="store_true",
        help="After submitting, poll squeue until all jobs complete.",
    )

    # ----------------------------------------------------------------- results
    p_res = sub.add_parser("results", help="Display evaluation results.")
    p_res.add_argument("--task",  action="append", dest="tasks",  metavar="TASK")
    p_res.add_argument("--model", action="append", dest="models", metavar="MODEL")

    # ------------------------------------------------------------------ status
    p_sta = sub.add_parser("status", help="Show job completion status.")
    p_sta.add_argument("--task",  action="append", dest="tasks",  metavar="TASK")
    p_sta.add_argument("--model", action="append", dest="models", metavar="MODEL")

    return parser


def _cmd_submit(args) -> None:
    from llmeval.infrastructure.config_loader import ConfigLoader
    from llmeval.infrastructure.result_store import ResultStore
    from llmeval.application.submit_evaluation import SubmitEvaluationUseCase
    from llmeval.application.monitor_jobs import MonitorJobsUseCase

    cfg   = ConfigLoader()
    store = ResultStore(cfg.output_root())
    use_case = SubmitEvaluationUseCase(cfg, store, dry_run=args.dry_run)

    task_names = args.tasks or list(cfg.load_all_benchmarks().keys())
    jobs = use_case.execute(
        task_names=task_names,
        model_names=args.models,
        model_type=args.model_type,
    )

    if jobs and args.wait and not args.dry_run:
        MonitorJobsUseCase().execute(jobs)


def _cmd_results(args) -> None:
    from llmeval.infrastructure.config_loader import ConfigLoader
    from llmeval.infrastructure.result_store import ResultStore
    from llmeval.application.aggregate_results import collect_results, print_results_table

    cfg   = ConfigLoader()
    store = ResultStore(cfg.output_root())

    task_names  = args.tasks  or list(cfg.load_all_benchmarks().keys())
    model_names = args.models or [m.name for m in cfg.load_models()]

    results = collect_results(task_names, model_names, store)
    print_results_table(results)


def _cmd_status(args) -> None:
    from llmeval.infrastructure.config_loader import ConfigLoader
    from llmeval.infrastructure.result_store import ResultStore

    cfg   = ConfigLoader()
    store = ResultStore(cfg.output_root())

    task_names  = args.tasks  or list(cfg.load_all_benchmarks().keys())
    model_names = args.models or [m.name for m in cfg.load_models()]

    for task in task_names:
        for model in model_names:
            if store.is_completed(task, model):
                status = "DONE"
            elif store.is_failed(task, model):
                status = "FAILED"
            else:
                status = "PENDING"
            print(f"{task:<20} {model:<45} {status}")


def main() -> None:
    parser = _make_parser()
    args   = parser.parse_args()

    if args.command == "submit":
        _cmd_submit(args)
    elif args.command == "results":
        _cmd_results(args)
    elif args.command == "status":
        _cmd_status(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
