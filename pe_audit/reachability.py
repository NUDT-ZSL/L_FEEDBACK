"""Name-based call-graph reachability with deterministic paths and cycles."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from .callgraph import CallAnalysis


@dataclass(frozen=True)
class GraphEdge:
    caller: str
    callee: str
    offset: int
    instruction_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller": self.caller,
            "callee": self.callee,
            "offset": self.offset,
            "instruction_index": self.instruction_index,
        }


def build_edges(analysis: CallAnalysis) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for function in sorted(analysis.functions):
        for call in analysis.functions[function].calls:
            if call.kind == "local" and call.callee is not None:
                edges.append(
                    GraphEdge(
                        caller=function,
                        callee=call.callee,
                        offset=call.offset,
                        instruction_index=call.instruction_index,
                    )
                )
    edges.sort(
        key=lambda edge: (
            edge.caller,
            edge.callee,
            edge.offset,
            edge.instruction_index,
        )
    )
    return edges


def _broken_edges(analysis: CallAnalysis) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for function in sorted(analysis.functions):
        for broken in analysis.functions[function].broken_calls:
            row = dict(broken)
            row["caller"] = function
            result.append(row)
    result.sort(
        key=lambda item: (
            str(item.get("caller", "")),
            int(item.get("offset", 0)),
            str(item.get("reason", "")),
        )
    )
    return result


def _shortest_paths(
    nodes: Iterable[str],
    edges: list[GraphEdge],
    entries: list[str],
) -> tuple[dict[str, list[GraphEdge]], list[str]]:
    node_set = set(nodes)
    outgoing: dict[str, GraphEdge] = {}
    for edge in edges:
        key = (edge.caller, edge.callee)
        old = outgoing.get(key)
        if old is None or (edge.offset, edge.instruction_index) < (
            old.offset,
            old.instruction_index,
        ):
            outgoing[key] = edge

    adjacency: dict[str, list[str]] = {node: [] for node in node_set}
    for (caller, callee) in outgoing:
        adjacency[caller].append(callee)
    for caller in adjacency:
        adjacency[caller].sort()

    missing_entries = sorted(entry for entry in entries if entry not in node_set)
    paths: dict[str, list[GraphEdge]] = {}
    queue: deque[str] = deque(sorted(entry for entry in entries if entry in node_set))
    for entry in queue:
        paths[entry] = []
    while queue:
        caller = queue.popleft()
        for callee in adjacency.get(caller, ()):
            if callee in paths:
                continue
            paths[callee] = paths[caller] + [outgoing[(caller, callee)]]
            queue.append(callee)
    return paths, missing_entries


def _cycles(
    nodes: Iterable[str],
    edges: list[GraphEdge],
) -> list[list[str]]:
    """Return deterministic strongly-connected components that contain cycles."""

    node_set = set(nodes)
    adjacency: dict[str, list[str]] = {node: [] for node in node_set}
    for edge in edges:
        adjacency[edge.caller].append(edge.callee)
    for caller in adjacency:
        adjacency[caller] = sorted(set(adjacency[caller]))

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbour in adjacency[node]:
            if neighbour not in indices:
                visit(neighbour)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbour])
            elif neighbour in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbour])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(sorted(component))

    for node in sorted(node_set):
        if node not in indices:
            visit(node)

    cycles: list[list[str]] = []
    for component in components:
        if len(component) > 1:
            cycles.append(component)
        elif component and component[0] in adjacency[component[0]]:
            cycles.append(component)
    return sorted(cycles)


def reachability(analysis: CallAnalysis, entries: Iterable[str]) -> dict[str, Any]:
    node_names = sorted(analysis.functions)
    entry_list = sorted(set(entries))
    edges = build_edges(analysis)
    paths, missing_entries = _shortest_paths(node_names, edges, entry_list)
    broken = _broken_edges(analysis)
    unreachable = sorted(set(node_names) - set(paths))

    reachable_rows: list[dict[str, Any]] = []
    for function in sorted(paths):
        edge_list = paths[function]
        if edge_list:
            path = [edge_list[0].caller]
            for edge in edge_list:
                path.append(edge.callee)
        else:
            path = [function]
        reachable_rows.append(
            {
                "function": function,
                "path": path,
                "edges": [edge.to_dict() for edge in edge_list],
            }
        )

    reachable_incomplete = sorted(
        name
        for name in paths
        if not analysis.functions[name].complete
    )
    broken_callers = sorted({str(item["caller"]) for item in broken})
    return {
        "entries": entry_list,
        "missing_entries": missing_entries,
        "edges": [edge.to_dict() for edge in edges],
        "cycles": _cycles(node_names, edges),
        "reachable": reachable_rows,
        "unreachable": unreachable,
        "broken_edges": broken,
        "reachable_broken_edges": [
            item for item in broken if str(item["caller"]) in paths
        ],
        "reachable_incomplete_functions": reachable_incomplete,
        "graph_incomplete": bool(
            missing_entries
            or broken
            or reachable_incomplete
            or broken_callers
        ),
    }


def path_for_function(result: dict[str, Any], function: str) -> dict[str, Any] | None:
    for item in result["reachable"]:
        if item["function"] == function:
            return item
    return None
