"""DE/SPSA execution under one objective-evaluation budget.

The same module is used by the main experiment and the runtime benchmark so the
benchmark measures the actual scientific execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import time

# Whole optimiser runs are the unit of parallelism. Prevent nested BLAS/OpenMP
# thread pools from multiplying the requested worker count.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import differential_evolution
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals

from evop_model import EVOPInstance, route_text
from pce_model import PCEObjectiveSettings, QuantumLayer, evaluate_parameters


class EvaluationBudgetReached(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimisationJob:
    stage: str
    instance: EVOPInstance
    compression_order: int
    controller: str
    controller_settings: dict
    seed: int
    alpha: float
    beta: float
    gamma: float
    penalty_shape: str
    penalty_weights: dict[str, float]
    regularisation_scale: float
    pad_gap_weight: float
    robust_decoding_margin: float
    evaluation_budget: int
    exact_baseline: dict | None = None
    selection_id: str = ""
    transfer_id: str = ""

    @property
    def key(self) -> tuple:
        return (
            self.stage,
            self.instance.key,
            self.compression_order,
            self.controller,
            self.seed,
            round(self.alpha, 12),
            round(self.beta, 12),
            round(self.gamma, 12),
            self.penalty_shape,
            self.evaluation_budget,
            self.selection_id,
            self.transfer_id,
        )

    @property
    def objective_settings(self) -> PCEObjectiveSettings:
        return PCEObjectiveSettings(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            penalty_shape=self.penalty_shape,
            penalty_weights=self.penalty_weights,
            regularisation_scale=self.regularisation_scale,
            pad_gap_weight=self.pad_gap_weight,
            robust_decoding_margin=self.robust_decoding_margin,
        )


class EvaluationTracker:
    def __init__(self, job: OptimisationJob, quantum_layer: QuantumLayer):
        self.job = job
        self.quantum_layer = quantum_layer
        self.started_at = time.perf_counter()
        self.evaluations = 0
        self.feasible_evaluations = 0
        self.failure_counts = {
            "repeated customer": 0,
            "station visit limit": 0,
            "battery infeasible": 0,
        }

        self.best_objective = float("inf")
        self.best_objective_parts: dict | None = None
        self.best_objective_decoded: dict | None = None
        self.best_feasible: dict | None = None
        self.best_feasible_objective = float("nan")
        self.first_feasible_evaluation: int | None = None
        self.best_feasible_evaluation: int | None = None

    def objective(self, parameters: np.ndarray) -> float:
        if self.evaluations >= self.job.evaluation_budget:
            raise EvaluationBudgetReached

        objective, parts, decoded = evaluate_parameters(
            self.job.instance,
            self.quantum_layer,
            parameters,
            self.job.objective_settings,
        )
        self.evaluations += 1

        if objective < self.best_objective - 1.0e-12:
            self.best_objective = objective
            self.best_objective_parts = parts
            self.best_objective_decoded = decoded

        record = decoded["record"]
        if record is None:
            if decoded["problem"] in self.failure_counts:
                self.failure_counts[decoded["problem"]] += 1
        else:
            self.feasible_evaluations += 1
            if self.first_feasible_evaluation is None:
                self.first_feasible_evaluation = self.evaluations

            is_better_profit = (
                self.best_feasible is None
                or record["profit"]
                > self.best_feasible["record"]["profit"] + 1.0e-12
            )
            is_more_robust_tie = (
                self.best_feasible is not None
                and abs(
                    record["profit"]
                    - self.best_feasible["record"]["profit"]
                ) <= 1.0e-12
                and decoded["min_gap"] > self.best_feasible["min_gap"]
            )

            if is_better_profit or is_more_robust_tie:
                self.best_feasible = decoded
                self.best_feasible_objective = objective
                self.best_feasible_evaluation = self.evaluations

        return objective

    def summary(self, termination: str) -> dict:
        instance = self.job.instance
        best_record = (
            self.best_feasible["record"]
            if self.best_feasible is not None
            else None
        )
        exact_profit = (
            float(self.job.exact_baseline["profit"])
            if self.job.exact_baseline is not None
            else float("nan")
        )
        best_profit = (
            float(best_record["profit"])
            if best_record is not None
            else float("nan")
        )
        optimum_ratio = (
            best_profit / exact_profit
            if np.isfinite(best_profit)
            and np.isfinite(exact_profit)
            and exact_profit > 0.0
            else float("nan")
        )
        denominator = max(1, self.evaluations)

        return {
            "stage": self.job.stage,
            "selection_id": self.job.selection_id,
            "transfer_id": self.job.transfer_id,
            "instance_key": instance.key,
            "profile": instance.profile,
            "customers": instance.customer_count,
            "stations": instance.station_count,
            "N": instance.size,
            "station_visit_limit": instance.station_visit_limit,
            "route_positions": instance.route_positions,
            "logical_variables": instance.logical_variables,
            "compression_order": self.job.compression_order,
            "qubits": self.quantum_layer.qubits,
            "brick_layers": self.quantum_layer.brick_layers,
            "circuit_parameters": self.quantum_layer.parameter_count,
            "controller": self.job.controller,
            "controller_settings": json.dumps(
                self.job.controller_settings,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "seed": self.job.seed,
            "alpha": self.job.alpha,
            "beta": self.job.beta,
            "gamma": self.job.gamma,
            "penalty_shape": self.job.penalty_shape,
            "repeat_penalty_weight": self.job.penalty_weights["customer_repeat"],
            "battery_penalty_weight": self.job.penalty_weights["battery"],
            "station_penalty_weight": self.job.penalty_weights["station_count"],
            "evaluation_budget": self.job.evaluation_budget,
            "objective_evaluations": self.evaluations,
            "termination": termination,
            "runtime_seconds": time.perf_counter() - self.started_at,
            "exact_best_profit": exact_profit,
            "feasible_found": best_record is not None,
            "best_feasible_profit": best_profit,
            "optimum_ratio": optimum_ratio,
            "exact_hit": bool(
                np.isfinite(optimum_ratio)
                and abs(optimum_ratio - 1.0) <= 1.0e-7
            ),
            "first_feasible_evaluation": self.first_feasible_evaluation,
            "best_feasible_evaluation": self.best_feasible_evaluation,
            "best_feasible_soft_objective": self.best_feasible_objective,
            "best_feasible_min_gap": (
                self.best_feasible["min_gap"]
                if self.best_feasible is not None
                else float("nan")
            ),
            "best_feasible_robust": (
                self.best_feasible["robust"]
                if self.best_feasible is not None
                else False
            ),
            "best_feasible_route": (
                route_text(instance, best_record["route"])
                if best_record is not None
                else ""
            ),
            "best_feasible_station_visits": (
                best_record["station_visits"]
                if best_record is not None
                else ""
            ),
            "feasible_decode_fraction": self.feasible_evaluations / denominator,
            "repeated_customer_fraction": (
                self.failure_counts["repeated customer"] / denominator
            ),
            "station_limit_fraction": (
                self.failure_counts["station visit limit"] / denominator
            ),
            "battery_infeasible_fraction": (
                self.failure_counts["battery infeasible"] / denominator
            ),
            "best_soft_objective": self.best_objective,
            "soft_winner_problem": (
                self.best_objective_decoded["problem"]
                if self.best_objective_decoded is not None
                else ""
            ),
            "soft_expected_profit": (
                self.best_objective_parts or {}
            ).get("expected_profit", float("nan")),
            "soft_normalised_profit": (
                self.best_objective_parts or {}
            ).get("normalised_profit", float("nan")),
            "soft_repeat_penalty": (
                self.best_objective_parts or {}
            ).get("repeat_penalty", float("nan")),
            "soft_battery_penalty": (
                self.best_objective_parts or {}
            ).get("battery_penalty", float("nan")),
            "soft_station_penalty": (
                self.best_objective_parts or {}
            ).get("station_penalty", float("nan")),
            "soft_pad_gap_penalty": (
                self.best_objective_parts or {}
            ).get("pad_gap_penalty", float("nan")),
            "soft_regularisation": (
                self.best_objective_parts or {}
            ).get("regularisation", float("nan")),
        }


_WORKER_QUANTUM_LAYERS: dict[tuple[int, int], QuantumLayer] = {}


def _quantum_layer_for(job: OptimisationJob) -> QuantumLayer:
    key = (job.instance.logical_variables, job.compression_order)
    if key not in _WORKER_QUANTUM_LAYERS:
        _WORKER_QUANTUM_LAYERS[key] = QuantumLayer(*key)
    return _WORKER_QUANTUM_LAYERS[key]


def _initial_parameters(seed: int, parameter_count: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * math.pi, size=parameter_count)


def _run_differential_evolution(
    tracker: EvaluationTracker,
    initial_parameters: np.ndarray,
) -> str:
    settings = tracker.job.controller_settings
    parameter_count = tracker.quantum_layer.parameter_count
    population_evaluations = settings["popsize_multiplier"] * parameter_count
    maximum_generations = max(
        1,
        math.ceil(tracker.job.evaluation_budget / population_evaluations) + 1,
    )
    bounds = [(0.0, 2.0 * math.pi)] * parameter_count

    try:
        result = differential_evolution(
            tracker.objective,
            bounds,
            strategy=settings["strategy"],
            maxiter=maximum_generations,
            popsize=settings["popsize_multiplier"],
            tol=settings["tolerance"],
            mutation=tuple(settings["mutation"]),
            recombination=settings["recombination"],
            seed=tracker.job.seed,
            polish=settings["polish"],
            init=settings["initial_population"],
            x0=initial_parameters,
            updating="immediate",
            workers=1,
        )
        return str(result.message)
    except EvaluationBudgetReached:
        return "common objective-evaluation budget reached"


def _run_spsa(
    tracker: EvaluationTracker,
    initial_parameters: np.ndarray,
) -> str:
    settings = tracker.job.controller_settings
    algorithm_globals.random_seed = tracker.job.seed

    try:
        learning_rate, perturbation = SPSA.calibrate(
            tracker.objective,
            initial_parameters,
            c=settings["calibration_c"],
            target_magnitude=settings["target_magnitude"],
        )

        optimizer = SPSA(
            maxiter=tracker.job.evaluation_budget,
            blocking=settings["blocking"],
            trust_region=settings["trust_region"],
            learning_rate=learning_rate,
            perturbation=perturbation,
            resamplings=settings["resamplings"],
            second_order=settings["second_order"],
        )
        optimizer.minimize(
            tracker.objective,
            x0=initial_parameters,
        )
        return "SPSA completed"
    except EvaluationBudgetReached:
        return "common objective-evaluation budget reached"


def execute_optimisation_job(job: OptimisationJob) -> dict:
    quantum_layer = _quantum_layer_for(job)
    tracker = EvaluationTracker(job, quantum_layer)
    initial_parameters = _initial_parameters(
        job.seed,
        quantum_layer.parameter_count,
    )

    if job.controller == "differential_evolution":
        termination = _run_differential_evolution(
            tracker,
            initial_parameters,
        )
    elif job.controller == "spsa":
        termination = _run_spsa(
            tracker,
            initial_parameters,
        )
    else:
        raise ValueError(f"Unsupported controller: {job.controller}")

    return {
        "key": job.key,
        "summary": tracker.summary(termination),
    }
