"""Canonical EVOP-V2G formulation and exact classical validation.

This module owns the physical problem definition. Experiment runners should not
redefine route validity, battery propagation, charging/discharging profit, or the
exact baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np


STATION_TARGET_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
BATTERY_SCALE = 100  # 0.01 kWh exact DP grid
PROFIT_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class EVOPInstance:
    key: str
    profile: str
    customer_count: int
    station_count: int
    station_visit_limit: int
    route_positions: int

    battery_capacity_kwh: float
    initial_battery_kwh: float
    minimum_reserve_kwh: float

    start_node: int
    customer_nodes: tuple[int, ...]
    station_nodes: tuple[int, ...]
    end_node: int
    pad_node: int

    real_nodes: tuple[int, ...]
    route_choices: tuple[int, ...]

    names: dict[int, str]
    rewards: dict[int, float]
    charge_prices: dict[int, float]
    discharge_prices: dict[int, float]
    energy_kwh: np.ndarray

    route_variables: int
    action_bits: int
    logical_variables: int

    @property
    def size(self) -> int:
        return self.customer_count + self.station_count


def build_instance(key: str, raw: dict, model: dict) -> EVOPInstance:
    customers = raw["customers"]
    stations = raw["stations"]

    customer_count = len(customers)
    station_count = len(stations)

    # Current station-revisit formulation: customers are visitable once, while
    # the route permits one more station event than the number of physical stations.
    station_visit_limit = station_count + 1
    route_positions = customer_count + station_visit_limit

    start_node = 0
    customer_nodes = tuple(range(1, customer_count + 1))
    station_nodes = tuple(
        range(customer_count + 1, customer_count + station_count + 1)
    )
    end_node = customer_count + station_count + 1
    pad_node = end_node + 1

    real_nodes = customer_nodes + station_nodes
    route_choices = real_nodes + (pad_node,)

    coordinates = {
        start_node: tuple(raw["start"]),
        end_node: tuple(raw["end"]),
    }
    names = {
        start_node: "START",
        end_node: "END",
        pad_node: "PAD",
    }
    rewards: dict[int, float] = {}
    charge_prices: dict[int, float] = {}
    discharge_prices: dict[int, float] = {}

    for index, node in enumerate(customer_nodes):
        customer = customers[index]
        coordinates[node] = tuple(customer["xy"])
        names[node] = f"C{index + 1:02d}"
        rewards[node] = float(customer["reward"])

    for index, node in enumerate(station_nodes):
        station = stations[index]
        coordinates[node] = tuple(station["xy"])
        names[node] = f"S{index + 1}"
        charge_prices[node] = float(station["charge_price"])
        discharge_prices[node] = float(station["discharge_price"])

        if discharge_prices[node] >= charge_prices[node] - 1.0e-12:
            raise ValueError(
                f"{key} {names[node]} violates the tariff rule: "
                f"discharge={discharge_prices[node]} >= charge={charge_prices[node]}."
            )

    energy_per_coordinate_unit = (
        float(model["km_per_coordinate_unit"])
        * float(model["energy_kwh_per_km"])
    )

    physical_nodes = (start_node,) + real_nodes + (end_node,)
    energy_kwh = np.zeros((end_node + 1, end_node + 1), dtype=float)

    for source in physical_nodes:
        source_x, source_y = coordinates[source]
        for destination in physical_nodes:
            destination_x, destination_y = coordinates[destination]
            manhattan_distance = (
                abs(source_x - destination_x)
                + abs(source_y - destination_y)
            )
            energy_kwh[source, destination] = (
                energy_per_coordinate_unit * manhattan_distance
            )

    route_variables = route_positions * len(route_choices)
    action_bits = 2 * route_positions
    logical_variables = route_variables + action_bits

    # With N=C+S, R=N+1 route positions and N+1 route choices (real nodes + PAD):
    # m = (N+1)^2 + 2(N+1) = (N+1)(N+3).
    expected_logical_variables = (
        (customer_count + station_count + 1)
        * (customer_count + station_count + 3)
    )
    if logical_variables != expected_logical_variables:
        raise RuntimeError("Logical-variable sizing invariant failed.")

    return EVOPInstance(
        key=key,
        profile=str(raw.get("meta", {}).get("profile", "")),
        customer_count=customer_count,
        station_count=station_count,
        station_visit_limit=station_visit_limit,
        route_positions=route_positions,
        battery_capacity_kwh=float(model["battery_capacity_kwh"]),
        initial_battery_kwh=float(raw["initial_battery_kwh"]),
        minimum_reserve_kwh=float(model["minimum_reserve_kwh"]),
        start_node=start_node,
        customer_nodes=customer_nodes,
        station_nodes=station_nodes,
        end_node=end_node,
        pad_node=pad_node,
        real_nodes=real_nodes,
        route_choices=route_choices,
        names=names,
        rewards=rewards,
        charge_prices=charge_prices,
        discharge_prices=discharge_prices,
        energy_kwh=energy_kwh,
        route_variables=route_variables,
        action_bits=action_bits,
        logical_variables=logical_variables,
    )


def route_text(instance: EVOPInstance, route: tuple[int, ...]) -> str:
    return " -> ".join(instance.names[node] for node in route)


def target_battery_for_action(instance: EVOPInstance, action_code: int) -> float:
    return STATION_TARGET_FRACTIONS[action_code] * instance.battery_capacity_kwh


def simulate_exact_route(
    instance: EVOPInstance,
    route_slots: tuple[int, ...],
    action_codes: tuple[int, ...],
) -> tuple[dict | None, str]:
    customers = [
        node
        for node in route_slots
        if node in instance.customer_nodes
    ]
    if len(customers) != len(set(customers)):
        return None, "repeated customer"

    station_visits = sum(
        node in instance.station_nodes
        for node in route_slots
    )
    if station_visits > instance.station_visit_limit:
        return None, "station visit limit"

    events = tuple(
        (node, int(action_codes[position]))
        for position, node in enumerate(route_slots)
        if node != instance.pad_node
    )

    battery_kwh = instance.initial_battery_kwh
    customer_reward = 0.0
    v2g_profit = 0.0
    travel_energy_kwh = 0.0
    station_actions = []
    source = instance.start_node

    for destination, action_code in events:
        travel_energy = instance.energy_kwh[source, destination]
        travel_energy_kwh += travel_energy
        battery_kwh -= travel_energy

        if battery_kwh < instance.minimum_reserve_kwh - 1.0e-9:
            return None, "battery infeasible"

        if destination in instance.customer_nodes:
            customer_reward += instance.rewards[destination]

        if destination in instance.station_nodes:
            target_battery_kwh = target_battery_for_action(instance, action_code)
            charged_kwh = max(0.0, target_battery_kwh - battery_kwh)
            discharged_kwh = max(0.0, battery_kwh - target_battery_kwh)

            station_profit = (
                discharged_kwh * instance.discharge_prices[destination]
                - charged_kwh * instance.charge_prices[destination]
            )

            station_actions.append({
                "station": instance.names[destination],
                "action_code": int(action_code),
                "target_fraction": STATION_TARGET_FRACTIONS[action_code],
                "arrival_battery_kwh": float(battery_kwh),
                "target_battery_kwh": float(target_battery_kwh),
                "charged_kwh": float(charged_kwh),
                "discharged_kwh": float(discharged_kwh),
                "net_v2g_profit": float(station_profit),
            })

            battery_kwh = target_battery_kwh
            v2g_profit += station_profit

        source = destination

    travel_energy = instance.energy_kwh[source, instance.end_node]
    travel_energy_kwh += travel_energy
    battery_kwh -= travel_energy

    if battery_kwh < instance.minimum_reserve_kwh - 1.0e-9:
        return None, "battery infeasible"

    route = (
        instance.start_node,
        *(node for node, _ in events),
        instance.end_node,
    )

    pad_seen = False
    had_pad_gap = False
    for node in route_slots:
        if node == instance.pad_node:
            pad_seen = True
        elif pad_seen:
            had_pad_gap = True
            break

    return {
        "route": route,
        "route_slots": route_slots,
        "action_codes": action_codes,
        "customer_reward": float(customer_reward),
        "v2g_profit": float(v2g_profit),
        "profit": float(customer_reward + v2g_profit),
        "travel_energy_kwh": float(travel_energy_kwh),
        "final_battery_kwh": float(battery_kwh),
        "station_visits": int(station_visits),
        "station_actions": station_actions,
        "had_pad_gap": had_pad_gap,
    }, ""


def _prune_dominated_exact_states(
    states: dict,
    parents: dict,
) -> tuple[dict, dict]:
    groups: dict[tuple, list] = {}

    for state, profit in states.items():
        customer_mask, current_node, battery_units, station_visits = state
        structural_key = (customer_mask, current_node, station_visits)
        groups.setdefault(structural_key, []).append(
            (battery_units, profit, state)
        )

    kept_states = {}
    kept_parents = {}

    for labels in groups.values():
        labels.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_profit_so_far = -float("inf")

        for _, profit, state in labels:
            if profit <= best_profit_so_far + PROFIT_TOLERANCE:
                continue
            kept_states[state] = profit
            kept_parents[state] = parents[state]
            best_profit_so_far = profit

    return kept_states, kept_parents


def solve_exact_baseline(instance: EVOPInstance) -> dict:
    """Exact finite-horizon DP for the customer-once/station-revisit model."""
    started_at = time.perf_counter()

    scaled_energy = instance.energy_kwh * BATTERY_SCALE
    if not np.allclose(
        scaled_energy,
        np.rint(scaled_energy),
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{instance.key}: travel energy is not exact on the 0.01-kWh DP grid."
        )

    energy_units = np.rint(scaled_energy).astype(int)
    reserve_units = int(round(instance.minimum_reserve_kwh * BATTERY_SCALE))
    initial_battery_units = int(round(instance.initial_battery_kwh * BATTERY_SCALE))
    target_units = tuple(
        int(round(
            fraction * instance.battery_capacity_kwh * BATTERY_SCALE
        ))
        for fraction in STATION_TARGET_FRACTIONS
    )

    start_state = (
        0,
        instance.start_node,
        initial_battery_units,
        0,
    )

    current_layer = {start_state: 0.0}
    parent_by_layer: list[dict] = [{}]

    best_profit = -float("inf")
    best_terminal: tuple[int, tuple, int] | None = None
    states_expanded = 0
    transitions_considered = 0

    for event_count in range(instance.route_positions + 1):
        next_layer = {}
        next_parents = {}

        for state, accumulated_profit in current_layer.items():
            states_expanded += 1
            customer_mask, current_node, battery_units, station_visits = state

            final_battery_units = (
                battery_units
                - energy_units[current_node, instance.end_node]
            )

            if final_battery_units >= reserve_units:
                better_profit = accumulated_profit > best_profit + PROFIT_TOLERANCE
                equal_profit_fewer_visits = (
                    abs(accumulated_profit - best_profit) <= PROFIT_TOLERANCE
                    and best_terminal is not None
                    and station_visits < best_terminal[1][3]
                )

                if better_profit or equal_profit_fewer_visits:
                    best_profit = accumulated_profit
                    best_terminal = (
                        event_count,
                        state,
                        final_battery_units,
                    )

            if event_count == instance.route_positions:
                continue

            for customer_index, customer in enumerate(instance.customer_nodes):
                if customer_mask & (1 << customer_index):
                    continue

                transitions_considered += 1
                arrival_battery = (
                    battery_units
                    - energy_units[current_node, customer]
                )
                if arrival_battery < reserve_units:
                    continue

                next_state = (
                    customer_mask | (1 << customer_index),
                    customer,
                    arrival_battery,
                    station_visits,
                )
                next_profit = accumulated_profit + instance.rewards[customer]

                if next_profit > next_layer.get(
                    next_state,
                    -float("inf"),
                ) + PROFIT_TOLERANCE:
                    next_layer[next_state] = next_profit
                    next_parents[next_state] = (
                        state,
                        customer,
                        0,
                    )

            if station_visits < instance.station_visit_limit:
                for station in instance.station_nodes:
                    arrival_battery = (
                        battery_units
                        - energy_units[current_node, station]
                    )
                    if arrival_battery < reserve_units:
                        continue

                    arrival_kwh = arrival_battery / BATTERY_SCALE

                    for action_code, target_battery in enumerate(target_units):
                        transitions_considered += 1
                        target_kwh = target_battery / BATTERY_SCALE

                        charged_kwh = max(0.0, target_kwh - arrival_kwh)
                        discharged_kwh = max(0.0, arrival_kwh - target_kwh)
                        station_profit = (
                            discharged_kwh * instance.discharge_prices[station]
                            - charged_kwh * instance.charge_prices[station]
                        )

                        next_state = (
                            customer_mask,
                            station,
                            target_battery,
                            station_visits + 1,
                        )
                        next_profit = accumulated_profit + station_profit

                        if next_profit > next_layer.get(
                            next_state,
                            -float("inf"),
                        ) + PROFIT_TOLERANCE:
                            next_layer[next_state] = next_profit
                            next_parents[next_state] = (
                                state,
                                station,
                                action_code,
                            )

        if event_count == instance.route_positions or not next_layer:
            break

        next_layer, next_parents = _prune_dominated_exact_states(
            next_layer,
            next_parents,
        )
        current_layer = next_layer
        parent_by_layer.append(next_parents)

    if best_terminal is None:
        raise RuntimeError(f"{instance.key}: exact DP found no feasible route.")

    terminal_layer, terminal_state, final_battery_units = best_terminal
    events_reversed = []
    state = terminal_state

    for layer in range(terminal_layer, 0, -1):
        previous_state, node, action_code = parent_by_layer[layer][state]
        events_reversed.append((node, action_code))
        state = previous_state

    events = list(reversed(events_reversed))
    route_slots = [node for node, _ in events]
    action_codes = [action for _, action in events]

    route_slots.extend(
        [instance.pad_node] * (instance.route_positions - len(route_slots))
    )
    action_codes.extend(
        [0] * (instance.route_positions - len(action_codes))
    )

    record, problem = simulate_exact_route(
        instance,
        tuple(route_slots),
        tuple(action_codes),
    )
    if record is None:
        raise RuntimeError(
            f"{instance.key}: exact DP reconstruction failed: {problem}"
        )
    if abs(record["profit"] - best_profit) > 1.0e-7:
        raise RuntimeError(
            f"{instance.key}: exact DP profit and reconstruction disagree."
        )

    return {
        "instance_key": instance.key,
        "profit": float(record["profit"]),
        "customer_reward": float(record["customer_reward"]),
        "v2g_profit": float(record["v2g_profit"]),
        "route": route_text(instance, record["route"]),
        "station_visits": int(record["station_visits"]),
        "station_actions": record["station_actions"],
        "final_battery_kwh": float(final_battery_units / BATTERY_SCALE),
        "states_expanded": int(states_expanded),
        "transitions_considered": int(transitions_considered),
        "runtime_seconds": float(time.perf_counter() - started_at),
    }
