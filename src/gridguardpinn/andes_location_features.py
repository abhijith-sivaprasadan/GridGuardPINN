"""Static network-aware fault-location descriptors for ANDES IEEE-14."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .andes_surrogate import N_BUSES, N_GENERATORS


FEATURE_NAMES = (
    "zdist_gen_1",
    "zdist_gen_2",
    "zdist_gen_3",
    "zdist_gen_4",
    "zdist_gen_5",
    "hop_gen_1",
    "hop_gen_2",
    "hop_gen_3",
    "hop_gen_4",
    "hop_gen_5",
    "prefault_voltage_dev_0p1pu",
    "prefault_angle_sin",
    "prefault_angle_cos",
    "normalized_degree",
)


@dataclass(frozen=True)
class FaultLocationFeatureSet:
    """Fourteen-dimensional static descriptors keyed by bus index."""

    descriptors: dict[int, np.ndarray]
    generator_buses: tuple[int, ...]
    feature_names: tuple[str, ...]
    rows: tuple[dict[str, object], ...]

    def metadata(self) -> dict[str, object]:
        return {
            "generator_buses": list(self.generator_buses),
            "feature_names": list(self.feature_names),
            "rows": [dict(row) for row in self.rows],
        }


def _dijkstra(adjacency, source: int) -> dict[int, float]:
    distances = {node: math.inf for node in adjacency}
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbor, weight in adjacency[node].items():
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _hop_distances(adjacency, source: int) -> dict[int, int]:
    distances = {node: N_BUSES + 1 for node in adjacency}
    distances[source] = 0
    queue = [source]
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for neighbor in adjacency[node]:
            if distances[neighbor] > distances[node] + 1:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    return distances


def _build_adjacency(system):
    buses = tuple(int(value) for value in system.Bus.idx.v)
    if len(buses) != N_BUSES:
        raise RuntimeError(
            f"Expected IEEE-14 to expose {N_BUSES} buses, got {len(buses)}."
        )

    weighted = {bus: {} for bus in buses}
    unweighted = {bus: {} for bus in buses}

    statuses = np.asarray(system.Line.u.v, dtype=float)
    bus1 = system.Line.bus1.v
    bus2 = system.Line.bus2.v
    resistance = np.asarray(system.Line.r.v, dtype=float)
    reactance = np.asarray(system.Line.x.v, dtype=float)

    for enabled, left, right, r, x in zip(
        statuses,
        bus1,
        bus2,
        resistance,
        reactance,
        strict=True,
    ):
        if enabled <= 0:
            continue
        left = int(left)
        right = int(right)
        impedance = max(float(math.hypot(r, x)), 1e-12)
        previous = weighted[left].get(right, math.inf)
        weight = min(previous, impedance)
        weighted[left][right] = weight
        weighted[right][left] = weight
        unweighted[left][right] = 1.0
        unweighted[right][left] = 1.0

    if any(len(neighbors) == 0 for neighbors in weighted.values()):
        raise RuntimeError("IEEE-14 feature graph contains an isolated bus.")
    return buses, weighted, unweighted


def build_ieee14_fault_location_features() -> FaultLocationFeatureSet:
    """Build static descriptors using topology and one pre-fault power flow."""
    from .andes_reference import _import_andes

    andes = _import_andes()
    system = andes.load(andes.get_case("ieee14/ieee14.json"))
    system.PFlow.run()
    if system.exit_code != 0:
        raise RuntimeError(
            f"ANDES power flow failed with exit_code={system.exit_code}."
        )

    buses, weighted, unweighted = _build_adjacency(system)
    generator_buses = tuple(int(value) for value in system.GENROU.bus.v)
    if len(generator_buses) != N_GENERATORS:
        raise RuntimeError(
            "Expected five GENROU generator buses in dynamic IEEE-14."
        )

    voltage = np.asarray(
        system.dae.y[system.Bus.v.a],
        dtype=float,
    )
    angle = np.asarray(
        system.dae.y[system.Bus.a.a],
        dtype=float,
    )
    bus_position = {bus: index for index, bus in enumerate(buses)}

    z_matrix = np.zeros((N_BUSES, N_GENERATORS), dtype=float)
    hop_matrix = np.zeros((N_BUSES, N_GENERATORS), dtype=float)
    for row, bus in enumerate(buses):
        z = _dijkstra(weighted, bus)
        hops = _hop_distances(unweighted, bus)
        z_matrix[row] = [z[target] for target in generator_buses]
        hop_matrix[row] = [hops[target] for target in generator_buses]

    if not np.all(np.isfinite(z_matrix)):
        raise RuntimeError("Non-finite impedance distance in IEEE-14 graph.")
    if not np.all(np.isfinite(hop_matrix)):
        raise RuntimeError("Non-finite hop distance in IEEE-14 graph.")

    z_scale = np.maximum(np.max(z_matrix, axis=0), 1e-12)
    hop_scale = np.maximum(np.max(hop_matrix, axis=0), 1.0)
    degree_scale = max(len(weighted[bus]) for bus in buses)

    descriptors: dict[int, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for row, bus in enumerate(buses):
        position = bus_position[bus]
        descriptor = np.concatenate(
            [
                z_matrix[row] / z_scale,
                hop_matrix[row] / hop_scale,
                np.asarray(
                    [
                        (voltage[position] - 1.0) / 0.1,
                        math.sin(angle[position]),
                        math.cos(angle[position]),
                        len(weighted[bus]) / degree_scale,
                    ],
                    dtype=float,
                ),
            ]
        )
        if descriptor.shape != (N_BUSES,):
            raise RuntimeError(
                f"Fault descriptor must have {N_BUSES} features."
            )
        if not np.all(np.isfinite(descriptor)):
            raise RuntimeError("Fault descriptor contains non-finite values.")
        descriptors[bus] = descriptor
        rows.append(
            {
                "bus": bus,
                **{
                    name: float(value)
                    for name, value in zip(
                        FEATURE_NAMES,
                        descriptor,
                        strict=True,
                    )
                },
            }
        )

    return FaultLocationFeatureSet(
        descriptors=descriptors,
        generator_buses=generator_buses,
        feature_names=FEATURE_NAMES,
        rows=tuple(rows),
    )
