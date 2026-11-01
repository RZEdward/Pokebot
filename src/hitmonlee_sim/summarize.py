"""Summarize comparable PPO run directories and their held-out evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Run directory containing run.json and best/best_model.evaluation.json; repeat per seed.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output summary JSON path")
    return parser.parse_args()


def _read_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = run_dir / "run.json"
    evaluation_path = run_dir / "best" / "best_model.evaluation.json"
    if not manifest_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError(
            f"Expected {manifest_path} and {evaluation_path}; evaluate the best checkpoint first."
        )
    return (
        json.loads(manifest_path.read_text(encoding="utf-8")),
        json.loads(evaluation_path.read_text(encoding="utf-8")),
    )


def summarize(run_dirs: list[Path]) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("Provide at least two run directories for a multi-seed summary.")

    runs: list[dict[str, Any]] = []
    episodes_by_seed: dict[int, list[dict[str, Any]]] = {}
    training_seeds: set[int] = set()
    heldout_seeds: set[int] | None = None
    initial_states: dict[int, dict[str, Any] | None] = {}
    matching_initial_states = True
    reference_signature: tuple[Any, ...] | None = None
    reference_manifest: dict[str, Any] | None = None
    model_hashes: set[str] = set()

    for run_dir in run_dirs:
        manifest, evaluation = _read_run(run_dir)
        training_seed = int(manifest["seed"])
        if training_seed in training_seeds:
            raise ValueError(f"Training seed {training_seed} appears more than once.")
        training_seeds.add(training_seed)
        environment_config = {
            key: value
            for key, value in manifest.get("environment", {}).items()
            if key != "evaluation_seed"
        }
        signature = (
            manifest.get("task"),
            manifest.get("requested_steps"),
            manifest.get("model_sha256"),
            json.dumps(environment_config, sort_keys=True),
            json.dumps(manifest.get("ppo_config", {}), sort_keys=True),
            manifest.get("evaluation", {}).get("frequency_steps"),
        )
        if reference_signature is None:
            reference_signature = signature
            reference_manifest = manifest
        elif signature != reference_signature:
            raise ValueError(
                "Run manifests must have the same task, requested steps, environment, "
                "robot-model hash, PPO settings, and evaluation cadence."
            )
        if manifest.get("model_sha256"):
            model_hashes.add(str(manifest["model_sha256"]))
        evaluation_model_hash = evaluation.get("robot_model_sha256")
        if evaluation_model_hash and evaluation_model_hash != manifest.get("model_sha256"):
            raise ValueError(
                f"{run_dir} was evaluated with a different robot MJCF than it was trained with."
            )

        episodes = evaluation["episodes"]
        current_seeds = {int(episode["seed"]) for episode in episodes}
        if heldout_seeds is None:
            heldout_seeds = current_seeds
        elif current_seeds != heldout_seeds:
            raise ValueError("All run evaluations must use the same held-out seed set.")

        for episode in episodes:
            eval_seed = int(episode["seed"])
            state = episode.get("initial_state")
            if eval_seed in initial_states and initial_states[eval_seed] != state:
                matching_initial_states = False
            else:
                initial_states[eval_seed] = state
            episodes_by_seed.setdefault(eval_seed, []).append(
                {"training_seed": training_seed, "fell": bool(episode["fell"])}
            )

        evaluation_config = manifest.get("evaluation", {})
        runs.append(
            {
                "training_seed": training_seed,
                "run_dir": run_dir.as_posix(),
                "algorithm": manifest.get("algorithm"),
                "model_sha256": manifest.get("model_sha256"),
                "requested_steps": manifest.get("requested_steps"),
                "actual_timesteps": manifest.get("actual_timesteps"),
                "evaluation_frequency_steps": evaluation_config.get("frequency_steps"),
                "best_checkpoint_timestep": evaluation_config.get("best_checkpoint_timestep"),
                "best_checkpoint_mean_reward_on_callback_episodes": evaluation_config.get(
                    "best_mean_reward"
                ),
                "heldout_episodes": len(episodes),
                "successes": round(evaluation["success_rate_no_fall"] * len(episodes)),
                "success_rate_no_fall": float(evaluation["success_rate_no_fall"]),
                "success_rate_wilson_95_interval": evaluation.get(
                    "success_rate_wilson_95_interval"
                ),
                "mean_episode_steps": float(evaluation["mean_steps"]),
                "failure_seeds": sorted(
                    int(episode["seed"]) for episode in episodes if episode["fell"]
                ),
                "evaluation_path": (run_dir / "best" / "best_model.evaluation.json").as_posix(),
            }
        )

    runs.sort(key=lambda row: row["training_seed"])
    rates = [run["success_rate_no_fall"] for run in runs]
    failure_counts = Counter(
        sum(item["fell"] for item in episodes) for episodes in episodes_by_seed.values()
    )
    cross_run = {
        "number_of_policy_failures_by_evaluation_seed": {
            str(count): failure_counts.get(count, 0) for count in range(len(runs) + 1)
        },
        "failed_by_all_policies": sorted(
            seed
            for seed, episodes in episodes_by_seed.items()
            if len(episodes) == len(runs) and all(item["fell"] for item in episodes)
        ),
        "survived_all_policies": sorted(
            seed
            for seed, episodes in episodes_by_seed.items()
            if all(not item["fell"] for item in episodes)
        ),
        "per_evaluation_seed": [
            {
                "seed": seed,
                "failed_training_seeds": sorted(
                    item["training_seed"] for item in episodes if item["fell"]
                ),
                "initial_state": initial_states[seed],
            }
            for seed, episodes in sorted(episodes_by_seed.items())
        ],
        "initial_states_match_across_runs": matching_initial_states,
    }
    return {
        "purpose": "Controlled descriptive comparison across PPO training seeds; not a formal robustness estimate.",
        "configuration": {
            "algorithm": runs[0].get("algorithm"),
            "requested_steps": runs[0]["requested_steps"],
            "heldout_seed_range_inclusive": {
                "start": min(heldout_seeds or {0}),
                "end": max(heldout_seeds or {0}),
            },
            "heldout_episodes_per_run": len(heldout_seeds or set()),
            "deterministic_policy": True,
            "task": reference_signature[0] if reference_signature else None,
            "environment": {
                key: value
                for key, value in (reference_manifest or {}).get("environment", {}).items()
                if key != "evaluation_seed"
            },
            "ppo_config": (reference_manifest or {}).get("ppo_config"),
            "model_file_hashes_match": len(model_hashes) == 1
            and all(run["model_sha256"] for run in runs),
            "model_file_hashes": sorted(model_hashes),
        },
        "results": runs,
        "across_training_seed_descriptives": {
            "training_seed_count": len(runs),
            "mean_success_rate": statistics.mean(rates),
            "median_success_rate": statistics.median(rates),
            "minimum_success_rate": min(rates),
            "maximum_success_rate": max(rates),
            "mean_successful_episodes_per_run": statistics.mean(run["successes"] for run in runs),
        },
        "cross_run_failure_diagnostics": cross_run,
        "interpretation": (
            "Between-run results are descriptive. They do not establish generalization "
            "beyond the tested starts, this simulator model, or hardware."
        ),
    }


def main() -> None:
    args = _parse_args()
    output_path = args.output.resolve()
    for run_dir in args.run_dir:
        resolved_run_dir = run_dir.resolve()
        if output_path == resolved_run_dir or resolved_run_dir in output_path.parents:
            raise ValueError(
                f"Summary output cannot be written inside an input run directory: {run_dir}"
            )
    summary = summarize(args.run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written: {output_path}")


if __name__ == "__main__":
    main()
