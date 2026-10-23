#!/usr/bin/env python3
"""Run the frozen EVOP-PCE calibration -> selection -> transfer track.

Single VM:
    python run_main.py --stage all --workers 30

Several VMs:
    python run_main.py --stage calibrate --shard-index 0 --shard-count 4 --workers 30
    ... repeat shard-index 1,2,3 ...
    # bring the shard CSVs into one output directory
    python run_main.py --stage select
    python run_main.py --stage transfer --shard-index 0 --shard-count 4 --workers 30
    ... repeat shard-index 1,2,3 ...
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np

from evop_model import build_instance, solve_exact_baseline
from instances import (
    development_key_for,
    load_instance_bank,
    transfer_keys_for,
)
from optimisers import OptimisationJob, execute_optimisation_job
from pce_model import representation_spec


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolved_path(config_path: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else config_path.parent / path


def stable_hash(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def calibration_design_payload(config: dict) -> dict:
    """Scientific choices that determine the expensive calibration evidence."""
    return {
        "experiment_sizes": config["experiment"]["sizes"],
        "instance_file": config["experiment"]["instance_file"],
        "development_instances": config["development_instances"],
        "pce": config["pce"],
        "penalty_shapes": config["penalty_shapes"],
        "controllers": config["controllers"],
        "calibration": config["calibration"],
    }


def preserve_calibration_design(config: dict, output_directory: Path) -> None:
    """Freeze calibration inputs while allowing selection/transfer rules to change."""
    snapshot_path = output_directory / "calibration_design_snapshot.json"
    design = calibration_design_payload(config)
    payload = {"design_hash": stable_hash(design), "design": design}

    if snapshot_path.exists():
        with snapshot_path.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("design_hash") != payload["design_hash"]:
            raise RuntimeError(
                "Calibration-driving settings changed, but this output directory "
                "already contains a different calibration design. Use a new output "
                "directory rather than mixing evidence."
            )
        return

    with snapshot_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def calibration_fingerprint(
    calibration_rows: list[dict],
    selected_sizes: list[int],
) -> str:
    """Fingerprint the exact raw calibration rows used for selection."""
    relevant = [
        row for row in calibration_rows
        if int(row["N"]) in selected_sizes
    ]
    relevant.sort(key=run_key_from_row)
    canonical = [
        {key: row[key] for key in sorted(row)}
        for row in relevant
    ]
    return stable_hash(canonical)


def selection_id_for(
    config: dict,
    calibration_rows: list[dict],
    selected_sizes: list[int],
) -> tuple[str, str]:
    fingerprint = calibration_fingerprint(calibration_rows, selected_sizes)
    payload = {
        "selection": config["selection"],
        "sizes": sorted(selected_sizes),
        "calibration_fingerprint": fingerprint,
    }
    return f"sel_{stable_hash(payload)[:12]}", fingerprint


def transfer_id_for(
    config: dict,
    selection_id: str,
    selected_sizes: list[int],
) -> str:
    pce = config["pce"]
    payload = {
        "selection_id": selection_id,
        "sizes": sorted(selected_sizes),
        "transfer": config["transfer"],
        "controllers": config["controllers"],
        "penalty_shapes": config["penalty_shapes"],
        "pce_execution": {
            "regularisation_scale": pce["regularisation_scale"],
            "pad_gap_weight": pce["pad_gap_weight"],
            "robust_decoding_margin": pce["robust_decoding_margin"],
        },
    }
    return f"xfer_{stable_hash(payload)[:12]}"


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def run_key_from_row(row: dict) -> tuple:
    return (
        row["stage"],
        row["instance_key"],
        int(row["compression_order"]),
        row["controller"],
        int(row["seed"]),
        round(float(row["alpha"]), 12),
        round(float(row["beta"]), 12),
        round(float(row["gamma"]), 12),
        row["penalty_shape"],
        int(row["evaluation_budget"]),
        row.get("selection_id", ""),
        row.get("transfer_id", ""),
    )


def completed_keys(path: Path) -> set[tuple]:
    return {
        run_key_from_row(row)
        for row in read_csv_rows(path)
    }


def belongs_to_shard(key: tuple, shard_index: int, shard_count: int) -> bool:
    encoded = json.dumps(key, separators=(",", ":"), default=str).encode("utf-8")
    number = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return number % shard_count == shard_index


def shuffled_jobs(jobs: list[OptimisationJob], seed: int) -> list[OptimisationJob]:
    jobs = list(jobs)
    random.Random(seed).shuffle(jobs)
    return jobs


def run_jobs(
    jobs: list[OptimisationJob],
    output_path: Path,
    workers: int,
    shard_index: int,
    shard_count: int,
    order_seed: int,
) -> None:
    jobs = [
        job
        for job in jobs
        if belongs_to_shard(job.key, shard_index, shard_count)
    ]
    completed = completed_keys(output_path)
    pending = [job for job in jobs if job.key not in completed]
    pending = shuffled_jobs(pending, order_seed + shard_index)

    print(
        f"{output_path.name}: {len(pending)} pending / {len(jobs)} assigned | "
        f"workers={min(workers, max(1, len(pending)))}"
    )
    if not pending:
        return

    started = time.perf_counter()
    finished = 0

    with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        future_to_job = {
            executor.submit(execute_optimisation_job, job): job
            for job in pending
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception:
                print(f"Worker failed: {job.key}")
                raise

            append_csv_row(output_path, result["summary"])
            finished += 1
            elapsed = time.perf_counter() - started
            jobs_per_hour = 3600.0 * finished / elapsed
            summary = result["summary"]
            print(
                f"[{finished:>5}/{len(pending)}] "
                f"N={summary['N']} k={summary['compression_order']} "
                f"{summary['controller']} seed={summary['seed']} | "
                f"feasible={summary['feasible_found']} | "
                f"{jobs_per_hour:.1f} jobs/hour",
                flush=True,
            )


def load_exact_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def exact_baseline_for(instance, cache: dict, cache_path: Path) -> dict:
    if instance.key in cache:
        return cache[instance.key]

    print(f"Exact baseline: {instance.key}", flush=True)
    result = solve_exact_baseline(instance)
    cache[instance.key] = result
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2)
    return result


def calibration_jobs_for_size(
    size: int,
    config: dict,
    model: dict,
    raw_instances: dict,
    exact_cache: dict,
    exact_cache_path: Path,
) -> list[OptimisationJob]:
    development_key = development_key_for(
        size,
        raw_instances,
        config["development_instances"],
    )
    development = build_instance(
        development_key,
        raw_instances[development_key],
        model,
    )
    exact = exact_baseline_for(
        development,
        exact_cache,
        exact_cache_path,
    )

    pce = config["pce"]
    budget_multiplier = int(config["calibration"]["budget_multiplier_p"])
    jobs = []

    for compression_order in pce["compression_orders"]:
        spec = representation_spec(
            development.logical_variables,
            int(compression_order),
        )
        budget = budget_multiplier * spec["parameter_count"]

        for controller, controller_settings in config["controllers"].items():
            for penalty_shape, penalty_weights in config["penalty_shapes"].items():
                for gamma in pce["gamma"]:
                    for alpha in pce["alpha"]:
                        for beta in pce["beta"]:
                            for seed in config["calibration"]["seeds"]:
                                jobs.append(OptimisationJob(
                                    stage="CALIBRATION",
                                    instance=development,
                                    compression_order=int(compression_order),
                                    controller=controller,
                                    controller_settings=controller_settings,
                                    seed=int(seed),
                                    alpha=float(alpha),
                                    beta=float(beta),
                                    gamma=float(gamma),
                                    penalty_shape=penalty_shape,
                                    penalty_weights=penalty_weights,
                                    regularisation_scale=float(pce["regularisation_scale"]),
                                    pad_gap_weight=float(pce["pad_gap_weight"]),
                                    robust_decoding_margin=float(pce["robust_decoding_margin"]),
                                    evaluation_budget=int(budget),
                                    exact_baseline=exact,
                                ))
    return jobs


def _as_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _median_absolute_deviation(values: list[float]) -> float:
    if not values:
        return float("inf")
    median = statistics.median(values)
    return float(statistics.median(abs(value - median) for value in values))


def _percentile_scores(values: list[float], larger_is_better: bool) -> list[float]:
    if len(values) == 1:
        return [1.0]

    if not larger_is_better:
        values = [-value for value in values]

    order = sorted(range(len(values)), key=lambda index: values[index])
    scores = [0.0] * len(values)
    position = 0

    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        score = average_rank / (len(values) - 1)
        for offset in range(position, end):
            scores[order[offset]] = score
        position = end

    return scores


def select_calibration_winners(
    config: dict,
    calibration_rows: list[dict],
    selected_sizes: list[int],
    output_directory: Path,
) -> tuple[dict, str]:
    expected_seeds = sorted(int(seed) for seed in config["calibration"]["seeds"])
    selection_id, source_fingerprint = selection_id_for(
        config,
        calibration_rows,
        selected_sizes,
    )
    quality_weight = float(config["selection"]["quality_weight"])
    stability_weight = float(config["selection"]["stability_weight"])

    grouped: dict[tuple, list[dict]] = {}
    for row in calibration_rows:
        if int(row["N"]) not in selected_sizes:
            continue
        key = (
            int(row["N"]),
            int(row["compression_order"]),
            row["controller"],
            row["penalty_shape"],
            float(row["gamma"]),
            float(row["alpha"]),
            float(row["beta"]),
        )
        grouped.setdefault(key, []).append(row)

    candidates_by_group: dict[tuple, list[dict]] = {}

    for key, rows in grouped.items():
        seeds = sorted(int(row["seed"]) for row in rows)
        if seeds != expected_seeds:
            raise RuntimeError(
                f"Incomplete calibration evidence for {key}: "
                f"expected seeds {expected_seeds}, found {seeds}."
            )

        ratios = [
            float(row["optimum_ratio"])
            for row in rows
            if _as_bool(row["feasible_found"])
            and np.isfinite(float(row["optimum_ratio"]))
        ]
        feasible_seed_count = sum(
            _as_bool(row["feasible_found"])
            for row in rows
        )

        candidate = {
            "N": key[0],
            "compression_order": key[1],
            "controller": key[2],
            "penalty_shape": key[3],
            "gamma": key[4],
            "alpha": key[5],
            "beta": key[6],
            "feasible_seed_count": feasible_seed_count,
            "seed_count": len(rows),
            "median_optimum_ratio": (
                float(statistics.median(ratios))
                if ratios
                else -float("inf")
            ),
            "optimum_ratio_mad": _median_absolute_deviation(ratios),
            "median_feasible_decode_fraction": float(statistics.median(
                float(row["feasible_decode_fraction"])
                for row in rows
            )),
        }
        group = (key[0], key[1], key[2])
        candidates_by_group.setdefault(group, []).append(candidate)

    rankings = []
    selections = {}

    for group, candidates in sorted(candidates_by_group.items()):
        maximum_feasible = max(
            candidate["feasible_seed_count"]
            for candidate in candidates
        )
        gated = [
            candidate
            for candidate in candidates
            if candidate["feasible_seed_count"] == maximum_feasible
        ]

        selection_key = f"N{group[0]}_k{group[1]}_{group[2]}"

        if maximum_feasible == 0:
            # No package produced an exactly feasible decode on any calibration
            # seed. Record this as a calibration failure rather than inventing a
            # winner from soft-objective behaviour. Transfer for this group is
            # therefore skipped.
            selections[selection_key] = {
                "N": group[0],
                "compression_order": group[1],
                "controller": group[2],
                "status": "no_feasible_calibration_configuration"
            }
            for candidate in candidates:
                candidate["quality_score"] = float("nan")
                candidate["stability_score"] = float("nan")
                candidate["selection_score"] = float("nan")
        else:
            quality_scores = _percentile_scores(
                [candidate["median_optimum_ratio"] for candidate in gated],
                larger_is_better=True,
            )
            stability_scores = _percentile_scores(
                [candidate["optimum_ratio_mad"] for candidate in gated],
                larger_is_better=False,
            )

            for candidate, quality_score, stability_score in zip(
                gated,
                quality_scores,
                stability_scores,
            ):
                candidate["quality_score"] = quality_score
                candidate["stability_score"] = stability_score
                candidate["selection_score"] = (
                    quality_weight * quality_score
                    + stability_weight * stability_score
                )

            gated.sort(
                key=lambda candidate: (
                    candidate["selection_score"],
                    candidate["median_optimum_ratio"],
                    -candidate["optimum_ratio_mad"],
                    candidate["median_feasible_decode_fraction"],
                    -candidate["gamma"],
                    -candidate["alpha"],
                    -candidate["beta"],
                    candidate["penalty_shape"],
                ),
                reverse=True,
            )

            winner = gated[0]
            selections[selection_key] = {
                "N": group[0],
                "compression_order": group[1],
                "controller": group[2],
                "status": "selected",
                "penalty_shape": winner["penalty_shape"],
                "gamma": winner["gamma"],
                "alpha": winner["alpha"],
                "beta": winner["beta"],
                "feasible_seed_count": winner["feasible_seed_count"],
                "median_optimum_ratio": winner["median_optimum_ratio"],
                "optimum_ratio_mad": winner["optimum_ratio_mad"],
                "selection_score": winner["selection_score"],
            }

        for candidate in candidates:
            candidate["passed_feasibility_gate"] = (
                candidate["feasible_seed_count"] == maximum_feasible
            )
            if not candidate["passed_feasibility_gate"]:
                candidate["quality_score"] = float("nan")
                candidate["stability_score"] = float("nan")
                candidate["selection_score"] = float("nan")
            rankings.append(candidate)

    selection_directory = output_directory / "selections" / selection_id
    selection_directory.mkdir(parents=True, exist_ok=True)

    rankings_path = selection_directory / "calibration_rankings.csv"
    if rankings:
        with rankings_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rankings[0].keys()))
            writer.writeheader()
            writer.writerows(rankings)

    selections_path = selection_directory / "selected_configurations.json"
    with selections_path.open("w", encoding="utf-8") as file:
        json.dump(selections, file, indent=2, allow_nan=True)

    snapshot = {
        "selection_id": selection_id,
        "selected_sizes": sorted(selected_sizes),
        "selection_rule": config["selection"],
        "source_calibration_fingerprint": source_fingerprint,
        "source_calibration_rows": sum(
            int(row["N"]) in selected_sizes
            for row in calibration_rows
        ),
        "selected_configurations_file": selections_path.name,
        "rankings_file": rankings_path.name,
    }
    with (selection_directory / "selection_snapshot.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(snapshot, file, indent=2)

    with (output_directory / "latest_selection.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump({"selection_id": selection_id}, file, indent=2)

    print(
        f"Selected {len(selections)} frozen (N,k,controller) packages | "
        f"selection_id={selection_id}"
    )
    print(f"Selection snapshot: {selection_directory}")
    return selections, selection_id


def transfer_jobs_for_size(
    size: int,
    config: dict,
    selections: dict,
    selection_id: str,
    transfer_id: str,
    model: dict,
    raw_instances: dict,
    exact_cache: dict,
    exact_cache_path: Path,
) -> list[OptimisationJob]:
    development_key = development_key_for(
        size,
        raw_instances,
        config["development_instances"],
    )
    transfer_keys = transfer_keys_for(
        size,
        development_key,
        raw_instances,
    )
    pce = config["pce"]
    budget_multiplier = int(config["transfer"]["budget_multiplier_p"])
    jobs = []

    for instance_key in transfer_keys:
        instance = build_instance(
            instance_key,
            raw_instances[instance_key],
            model,
        )
        exact = exact_baseline_for(instance, exact_cache, exact_cache_path)

        for compression_order in pce["compression_orders"]:
            spec = representation_spec(
                instance.logical_variables,
                int(compression_order),
            )
            budget = budget_multiplier * spec["parameter_count"]

            for controller, controller_settings in config["controllers"].items():
                selection_key = f"N{size}_k{compression_order}_{controller}"
                if selection_key not in selections:
                    raise KeyError(f"Missing frozen selection {selection_key}.")
                selected = selections[selection_key]
                if selected.get("status") != "selected":
                    print(
                        f"Skipping transfer for {selection_key}: "
                        f"{selected.get('status', 'not selected')}"
                    )
                    continue
                penalty_shape = selected["penalty_shape"]

                for seed in config["transfer"]["seeds"]:
                    jobs.append(OptimisationJob(
                        stage="TRANSFER",
                        instance=instance,
                        compression_order=int(compression_order),
                        controller=controller,
                        controller_settings=controller_settings,
                        seed=int(seed),
                        alpha=float(selected["alpha"]),
                        beta=float(selected["beta"]),
                        gamma=float(selected["gamma"]),
                        penalty_shape=penalty_shape,
                        penalty_weights=config["penalty_shapes"][penalty_shape],
                        regularisation_scale=float(pce["regularisation_scale"]),
                        pad_gap_weight=float(pce["pad_gap_weight"]),
                        robust_decoding_margin=float(pce["robust_decoding_margin"]),
                        evaluation_budget=int(budget),
                        exact_baseline=exact,
                        selection_id=selection_id,
                        transfer_id=transfer_id,
                    ))
    return jobs


def all_shard_rows(output_directory: Path, prefix: str) -> list[dict]:
    rows = []
    for path in sorted(output_directory.glob(f"{prefix}_shard_*.csv")):
        rows.extend(read_csv_rows(path))
    if not rows:
        single = output_directory / f"{prefix}_shard_00-of-01.csv"
        rows.extend(read_csv_rows(single))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("experiment_config.json"),
    )
    parser.add_argument(
        "--stage",
        choices=("calibrate", "select", "transfer", "all"),
        default="all",
    )
    parser.add_argument("--n", dest="sizes", type=int, nargs="*")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--selection-id",
        help=(
            "Use a specific saved selection for transfer. If omitted, the most "
            "recent selection snapshot is used."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise ValueError("Require 0 <= shard-index < shard-count.")
    if args.stage == "all" and args.shard_count != 1:
        raise ValueError(
            "Use explicit calibrate -> select -> transfer stages when sharding across VMs."
        )

    instance_path = resolved_path(
        config_path,
        config["experiment"]["instance_file"],
    )
    output_directory = resolved_path(
        config_path,
        config["experiment"]["output_directory"],
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    model, raw_instances = load_instance_bank(instance_path)
    selected_sizes = (
        args.sizes
        if args.sizes
        else [int(size) for size in config["experiment"]["sizes"]]
    )
    workers = (
        args.workers
        if args.workers is not None
        else int(config["parallel"]["default_workers"])
    )

    resolved_snapshot = dict(config)
    resolved_snapshot["runtime"] = {
        "selected_sizes": selected_sizes,
        "workers": workers,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
    }
    with (output_directory / "resolved_experiment.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(resolved_snapshot, file, indent=2)

    if args.dry_run:
        calibration_per_n = (
            len(config["controllers"])
            * len(config["pce"]["compression_orders"])
            * len(config["pce"]["alpha"])
            * len(config["pce"]["beta"])
            * len(config["pce"]["gamma"])
            * len(config["penalty_shapes"])
            * len(config["calibration"]["seeds"])
        )
        print(f"Calibration runs per N: {calibration_per_n:,}")
        for size in selected_sizes:
            dev = development_key_for(size, raw_instances, config["development_instances"])
            held_out = len(transfer_keys_for(size, dev, raw_instances))
            transfer_runs = (
                held_out
                * len(config["controllers"])
                * len(config["pce"]["compression_orders"])
                * len(config["transfer"]["seeds"])
            )
            print(
                f"N={size}: development={dev}, held-out={held_out}, "
                f"calibration={calibration_per_n:,}, transfer={transfer_runs:,}"
            )
        return

    exact_cache_path = output_directory / "exact_baselines.json"
    exact_cache = load_exact_cache(exact_cache_path)
    order_seed = int(config["parallel"]["job_order_seed"])

    if args.stage in {"calibrate", "all"}:
        preserve_calibration_design(config, output_directory)
        jobs = []
        for size in selected_sizes:
            jobs.extend(calibration_jobs_for_size(
                size,
                config,
                model,
                raw_instances,
                exact_cache,
                exact_cache_path,
            ))

        calibration_path = output_directory / (
            f"calibration_runs_shard_{args.shard_index:02d}-of-{args.shard_count:02d}.csv"
        )
        run_jobs(
            jobs,
            calibration_path,
            workers,
            args.shard_index,
            args.shard_count,
            order_seed,
        )

    selections = None
    selection_id = None

    if args.stage in {"select", "all"}:
        rows = all_shard_rows(output_directory, "calibration_runs")
        selections, selection_id = select_calibration_winners(
            config,
            rows,
            selected_sizes,
            output_directory,
        )

    if args.stage in {"transfer", "all"}:
        if selections is None:
            selection_id = args.selection_id
            if selection_id is None:
                latest_path = output_directory / "latest_selection.json"
                if not latest_path.exists():
                    raise FileNotFoundError(
                        "Run the selection stage before transfer, or pass "
                        "--selection-id for a saved selection."
                    )
                with latest_path.open("r", encoding="utf-8") as file:
                    selection_id = json.load(file)["selection_id"]

            selections_path = (
                output_directory
                / "selections"
                / selection_id
                / "selected_configurations.json"
            )
            if not selections_path.exists():
                raise FileNotFoundError(
                    f"Saved selection not found: {selections_path}"
                )
            with selections_path.open("r", encoding="utf-8") as file:
                selections = json.load(file)

        transfer_id = transfer_id_for(config, selection_id, selected_sizes)
        transfer_directory = output_directory / "transfers" / transfer_id
        transfer_directory.mkdir(parents=True, exist_ok=True)

        transfer_snapshot = {
            "transfer_id": transfer_id,
            "selection_id": selection_id,
            "selected_sizes": sorted(selected_sizes),
            "transfer_policy": config["transfer"],
            "selection_source": str(
                output_directory / "selections" / selection_id
            ),
        }
        with (transfer_directory / "transfer_snapshot.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(transfer_snapshot, file, indent=2)

        jobs = []
        for size in selected_sizes:
            jobs.extend(transfer_jobs_for_size(
                size,
                config,
                selections,
                selection_id,
                transfer_id,
                model,
                raw_instances,
                exact_cache,
                exact_cache_path,
            ))

        transfer_path = transfer_directory / (
            f"transfer_runs_shard_{args.shard_index:02d}-of-{args.shard_count:02d}.csv"
        )
        print(
            f"Transfer policy: selection_id={selection_id} | "
            f"transfer_id={transfer_id}"
        )
        run_jobs(
            jobs,
            transfer_path,
            workers,
            args.shard_index,
            args.shard_count,
            order_seed + 1000,
        )


if __name__ == "__main__":
    freeze_support()
    main()
