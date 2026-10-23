"""PCE representation, relaxed objective, and exact decoding."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from evop_model import EVOPInstance, simulate_exact_route


STATION_TARGET_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
ROW_MASS_WARNING_THRESHOLD = 0.25


@dataclass(frozen=True)
class PCEObjectiveSettings:
    alpha: float
    beta: float
    gamma: float
    penalty_shape: str
    penalty_weights: dict[str, float]
    regularisation_scale: float
    pad_gap_weight: float
    robust_decoding_margin: float


def minimum_pce_qubits(logical_variables: int, compression_order: int) -> int:
    qubits = compression_order
    while 3 * math.comb(qubits, compression_order) < logical_variables:
        qubits += 1
    return qubits


def same_axis_k_labels(
    number_of_qubits: int,
    compression_order: int,
) -> list[str]:
    labels = []

    for pauli in ("X", "Y", "Z"):
        for support in combinations(range(number_of_qubits), compression_order):
            label = ["I"] * number_of_qubits
            for qubit in support:
                label[number_of_qubits - 1 - qubit] = pauli
            labels.append("".join(label))

    return labels


def paper_brick_circuit(
    number_of_qubits: int,
    number_of_layers: int,
) -> tuple[QuantumCircuit, tuple]:
    parameter_count = 0

    for layer in range(number_of_layers):
        if layer % 2 == 0:
            parameter_count += number_of_qubits
        else:
            entangling_layer = layer // 2
            offset = entangling_layer % 2
            parameter_count += len(
                range(offset, number_of_qubits - 1, 2)
            )

    theta = ParameterVector("theta", parameter_count)
    circuit = QuantumCircuit(
        number_of_qubits,
        name=f"pce_{number_of_qubits}q_{number_of_layers}l",
    )

    cursor = 0
    single_layer = 0
    entangling_layer = 0

    for layer in range(number_of_layers):
        if layer % 2 == 0:
            gate_name = ("rx", "ry", "rz")[single_layer % 3]
            for qubit in range(number_of_qubits):
                getattr(circuit, gate_name)(theta[cursor], qubit)
                cursor += 1
            single_layer += 1
        else:
            gate_name = ("rxx", "ryy", "rzz")[entangling_layer % 3]
            offset = entangling_layer % 2
            for left_qubit in range(offset, number_of_qubits - 1, 2):
                getattr(circuit, gate_name)(
                    theta[cursor],
                    left_qubit,
                    left_qubit + 1,
                )
                cursor += 1
            entangling_layer += 1

    return circuit, tuple(theta)


class QuantumLayer:
    def __init__(self, logical_variables: int, compression_order: int):
        self.logical_variables = logical_variables
        self.compression_order = compression_order
        self.qubits = minimum_pce_qubits(
            logical_variables,
            compression_order,
        )
        self.brick_layers = math.ceil(
            logical_variables ** (1.0 - 1.0 / compression_order)
        )

        available_labels = same_axis_k_labels(
            self.qubits,
            compression_order,
        )
        if len(available_labels) < logical_variables:
            raise RuntimeError("PCE label capacity is smaller than the logical problem.")

        self.labels = tuple(available_labels[:logical_variables])
        self.circuit, self.parameters = paper_brick_circuit(
            self.qubits,
            self.brick_layers,
        )
        self.observables = tuple(
            SparsePauliOp.from_list([(label, 1.0)])
            for label in self.labels
        )

    @property
    def parameter_count(self) -> int:
        return len(self.parameters)

    @property
    def label_capacity(self) -> int:
        return 3 * math.comb(self.qubits, self.compression_order)

    def expectations(self, parameter_values: np.ndarray) -> np.ndarray:
        parameter_values = np.asarray(parameter_values, dtype=float) % (2.0 * math.pi)
        bindings = {
            parameter: value
            for parameter, value in zip(self.parameters, parameter_values)
        }
        bound_circuit = self.circuit.assign_parameters(bindings, inplace=False)
        state = Statevector.from_instruction(bound_circuit)

        return np.asarray(
            [
                float(np.real(state.expectation_value(observable)))
                for observable in self.observables
            ],
            dtype=float,
        )


def representation_spec(logical_variables: int, compression_order: int) -> dict:
    layer = QuantumLayer(logical_variables, compression_order)
    return {
        "compression_order": compression_order,
        "qubits": layer.qubits,
        "label_capacity": layer.label_capacity,
        "brick_layers": layer.brick_layers,
        "parameter_count": layer.parameter_count,
    }


def _action_bit_indices(instance: EVOPInstance, position: int) -> tuple[int, int]:
    first = instance.route_variables + 2 * position
    return first, first + 1


def _soft_values_from_expectations(
    expectations: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    soft_signs = np.tanh(alpha * np.asarray(expectations, dtype=float))
    soft_binary = 0.5 * (1.0 + soft_signs)
    return soft_signs, soft_binary


def _route_rows_from_soft_binary(
    instance: EVOPInstance,
    soft_binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_rows = np.asarray(
        soft_binary[:instance.route_variables],
        dtype=float,
    ).reshape(
        instance.route_positions,
        len(instance.route_choices),
    )

    positive_rows = np.clip(raw_rows, 0.0, 1.0) + 1.0e-12
    row_probabilities = positive_rows / np.sum(
        positive_rows,
        axis=1,
        keepdims=True,
    )
    return raw_rows, row_probabilities


def _soft_target_fraction(bit_zero: float, bit_one: float) -> float:
    return 0.25 + 0.25 * bit_zero + 0.50 * bit_one


def _physical_profit_ceiling(instance: EVOPInstance) -> float:
    customer_bound = sum(
        max(0.0, reward)
        for reward in instance.rewards.values()
    )
    maximum_discharge_per_visit = (
        (1.0 - min(STATION_TARGET_FRACTIONS))
        * instance.battery_capacity_kwh
    )
    maximum_discharge_price = max(
        instance.discharge_prices.values(),
        default=0.0,
    )
    discharge_bound = (
        instance.station_visit_limit
        * maximum_discharge_per_visit
        * maximum_discharge_price
    )
    ceiling = customer_bound + discharge_bound
    return float(ceiling if ceiling > 0.0 else 1.0)


def _normalised_battery_shortfall(
    instance: EVOPInstance,
    battery_kwh: float,
) -> float:
    shortfall = max(
        0.0,
        instance.minimum_reserve_kwh - battery_kwh,
    )
    return (shortfall / instance.battery_capacity_kwh) ** 2


def _expected_ev_terms(
    instance: EVOPInstance,
    row_probabilities: np.ndarray,
    soft_binary: np.ndarray,
) -> dict:
    choice_index = {
        node: index
        for index, node in enumerate(instance.route_choices)
    }

    location_distribution = {
        instance.start_node: 1.0,
        **{node: 0.0 for node in instance.real_nodes},
    }

    battery_kwh = instance.initial_battery_kwh
    expected_customer_reward = 0.0
    expected_v2g_profit = 0.0
    expected_station_visits = 0.0
    battery_violation_terms = []

    for position in range(instance.route_positions):
        row = row_probabilities[position]
        probabilities = {
            node: float(row[choice_index[node]])
            for node in instance.route_choices
        }

        for customer in instance.customer_nodes:
            expected_customer_reward += (
                probabilities[customer] * instance.rewards[customer]
            )

        expected_energy_kwh = 0.0
        for target_node in instance.real_nodes:
            expected_energy_from_sources = sum(
                source_probability
                * instance.energy_kwh[source_node, target_node]
                for source_node, source_probability
                in location_distribution.items()
            )
            expected_energy_kwh += (
                probabilities[target_node]
                * expected_energy_from_sources
            )

        arrival_battery_kwh = battery_kwh - expected_energy_kwh
        battery_violation_terms.append(
            _normalised_battery_shortfall(instance, arrival_battery_kwh)
        )

        bit_zero_index, bit_one_index = _action_bit_indices(instance, position)
        target_fraction = _soft_target_fraction(
            float(soft_binary[bit_zero_index]),
            float(soft_binary[bit_one_index]),
        )
        target_battery_kwh = target_fraction * instance.battery_capacity_kwh

        station_probability = 0.0
        station_profit_at_position = 0.0

        for station in instance.station_nodes:
            probability = probabilities[station]
            station_probability += probability

            charged_kwh = max(
                0.0,
                target_battery_kwh - arrival_battery_kwh,
            )
            discharged_kwh = max(
                0.0,
                arrival_battery_kwh - target_battery_kwh,
            )
            station_profit = (
                discharged_kwh * instance.discharge_prices[station]
                - charged_kwh * instance.charge_prices[station]
            )
            station_profit_at_position += probability * station_profit

        expected_station_visits += station_probability
        expected_v2g_profit += station_profit_at_position

        battery_kwh = (
            arrival_battery_kwh
            + station_probability
            * (target_battery_kwh - arrival_battery_kwh)
        )
        battery_violation_terms.append(
            _normalised_battery_shortfall(instance, battery_kwh)
        )

        pad_probability = probabilities[instance.pad_node]
        next_distribution = {
            node: pad_probability * location_distribution[node]
            for node in (instance.start_node,) + instance.real_nodes
        }
        for node in instance.real_nodes:
            next_distribution[node] += probabilities[node]
        location_distribution = next_distribution

    end_energy_kwh = sum(
        source_probability
        * instance.energy_kwh[source_node, instance.end_node]
        for source_node, source_probability in location_distribution.items()
    )
    final_battery_kwh = battery_kwh - end_energy_kwh
    battery_violation_terms.append(
        _normalised_battery_shortfall(instance, final_battery_kwh)
    )

    return {
        "expected_profit": float(expected_customer_reward + expected_v2g_profit),
        "expected_customer_reward": float(expected_customer_reward),
        "expected_v2g_profit": float(expected_v2g_profit),
        "expected_station_visits": float(expected_station_visits),
        "expected_final_battery_kwh": float(final_battery_kwh),
        "battery_penalty": float(np.mean(battery_violation_terms)),
    }


def _normalised_customer_repeat_penalty(
    instance: EVOPInstance,
    row_probabilities: np.ndarray,
) -> float:
    pair_count = math.comb(instance.route_positions, 2)
    if pair_count == 0:
        return 0.0

    choice_index = {
        node: index
        for index, node in enumerate(instance.route_choices)
    }
    penalty = 0.0

    for customer in instance.customer_nodes:
        column = row_probabilities[:, choice_index[customer]]
        for left in range(instance.route_positions):
            for right in range(left + 1, instance.route_positions):
                penalty += float(column[left] * column[right])

    return float(penalty / pair_count)


def _normalised_station_excess_penalty(
    instance: EVOPInstance,
    expected_station_visits: float,
) -> float:
    maximum_excess = max(1, instance.customer_count)
    excess = max(
        0.0,
        expected_station_visits - instance.station_visit_limit,
    )
    return float((excess / maximum_excess) ** 2)


def _pad_gap_penalty(
    instance: EVOPInstance,
    row_probabilities: np.ndarray,
) -> float:
    pair_count = math.comb(instance.route_positions, 2)
    if pair_count == 0:
        return 0.0

    pad_index = instance.route_choices.index(instance.pad_node)
    penalty = 0.0

    for left in range(instance.route_positions):
        pad_mass = row_probabilities[left, pad_index]
        for right in range(left + 1, instance.route_positions):
            later_real_mass = 1.0 - row_probabilities[right, pad_index]
            penalty += float(pad_mass * later_real_mass)

    return float(penalty / pair_count)


def _objective_from_soft_values(
    instance: EVOPInstance,
    soft_signs: np.ndarray,
    soft_binary: np.ndarray,
    settings: PCEObjectiveSettings,
) -> tuple[float, dict]:
    raw_rows, row_probabilities = _route_rows_from_soft_binary(
        instance,
        soft_binary,
    )
    ev = _expected_ev_terms(instance, row_probabilities, soft_binary)

    repeat_penalty = _normalised_customer_repeat_penalty(
        instance,
        row_probabilities,
    )
    station_penalty = _normalised_station_excess_penalty(
        instance,
        ev["expected_station_visits"],
    )
    pad_penalty = _pad_gap_penalty(instance, row_probabilities)

    ceiling = _physical_profit_ceiling(instance)
    capped_expected_profit = min(ev["expected_profit"], ceiling)
    normalised_profit = capped_expected_profit / ceiling

    feasibility_penalty = (
        settings.penalty_weights["customer_repeat"] * repeat_penalty
        + settings.penalty_weights["battery"] * ev["battery_penalty"]
        + settings.penalty_weights["station_count"] * station_penalty
    )

    regularisation = (
        settings.beta
        * settings.regularisation_scale
        * float(np.mean(np.asarray(soft_signs, dtype=float) ** 2) ** 2)
    )

    total = (
        -normalised_profit
        + settings.gamma * feasibility_penalty
        + settings.pad_gap_weight * pad_penalty
        + regularisation
    )

    row_masses = np.sum(raw_rows, axis=1)

    return float(total), {
        **ev,
        "profit_ceiling": float(ceiling),
        "capped_expected_profit": float(capped_expected_profit),
        "normalised_profit": float(normalised_profit),
        "repeat_penalty": float(repeat_penalty),
        "station_penalty": float(station_penalty),
        "pad_gap_penalty": float(pad_penalty),
        "feasibility_penalty": float(feasibility_penalty),
        "weighted_feasibility_penalty": float(settings.gamma * feasibility_penalty),
        "regularisation": float(regularisation),
        "min_raw_row_mass": float(np.min(row_masses)),
        "median_raw_row_mass": float(np.median(row_masses)),
        "low_raw_row_count": int(np.sum(row_masses < ROW_MASS_WARNING_THRESHOLD)),
    }


def _decode_soft_binary(
    instance: EVOPInstance,
    soft_binary: np.ndarray,
    robust_decoding_margin: float,
) -> dict:
    _, row_probabilities = _route_rows_from_soft_binary(instance, soft_binary)

    route_slots = []
    row_gaps = []

    for row in row_probabilities:
        order = np.argsort(row)[::-1]
        route_slots.append(instance.route_choices[int(order[0])])
        row_gaps.append(float(row[order[0]] - row[order[1]]))

    action_codes = []
    for position in range(instance.route_positions):
        bit_zero_index, bit_one_index = _action_bit_indices(instance, position)
        bit_zero = int(soft_binary[bit_zero_index] >= 0.5)
        bit_one = int(soft_binary[bit_one_index] >= 0.5)
        action_codes.append(bit_zero + 2 * bit_one)

    record, problem = simulate_exact_route(
        instance,
        tuple(route_slots),
        tuple(action_codes),
    )

    return {
        "route_slots": tuple(route_slots),
        "action_codes": tuple(action_codes),
        "record": record,
        "problem": problem,
        "mean_gap": float(np.mean(row_gaps)),
        "min_gap": float(np.min(row_gaps)),
        "robust": bool(np.min(row_gaps) >= robust_decoding_margin),
    }


def evaluate_parameters(
    instance: EVOPInstance,
    quantum_layer: QuantumLayer,
    parameters: np.ndarray,
    settings: PCEObjectiveSettings,
) -> tuple[float, dict, dict]:
    expectations = quantum_layer.expectations(parameters)
    soft_signs, soft_binary = _soft_values_from_expectations(
        expectations,
        settings.alpha,
    )
    objective, parts = _objective_from_soft_values(
        instance,
        soft_signs,
        soft_binary,
        settings,
    )
    decoded = _decode_soft_binary(
        instance,
        soft_binary,
        settings.robust_decoding_margin,
    )
    return objective, parts, decoded
