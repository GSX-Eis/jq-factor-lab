"""Single operator entrypoint; alignment implementations live in scripts/alignment."""

import fcntl
import json
from pathlib import Path


def run(args, repo_root):
    from scripts.alignment.alignment_cycle import run_cycle
    from scripts.alignment.strategy_cycle import run_strategy

    if args.fake_result:
        raise ValueError("alignment does not accept synthetic acceptance evidence")
    if args.iteration < 1 or args.iteration > args.max_iterations:
        raise ValueError("iteration must be between 1 and max-iterations")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".alignment_cycle.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("alignment_cycle_already_running") from exc
        if args.dry_run:
            return {
                "status": "PASS",
                "mode": "dry_run",
                "layers": {},
                "artifact_hashes": {},
            }
        if args.factor:
            if not args.factor_class:
                raise ValueError("--factor-class is required with --factor")
            return run_cycle(
                factor=args.factor,
                factor_class=args.factor_class,
                output_dir=output,
                real_cloud=args.real_cloud,
                resume=args.resume,
                iteration=args.iteration,
                max_iterations=args.max_iterations,
                result_file=Path(args.result_file) if args.result_file else None,
                work_log=Path(args.work_log) if args.work_log else None,
            )
        payload = run_strategy(
            Path(args.strategy_config).resolve(),
            output,
            real_cloud=args.real_cloud,
            resume=args.resume,
            timeout=args.timeout_sec,
        )
        if args.result_file and args.real_cloud:
            Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
            Path(args.result_file).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            )
        return payload
