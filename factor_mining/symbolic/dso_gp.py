"""DEAP-based GP meld for DSO hall-of-fame expressions."""

from __future__ import annotations

import copy
import random
from typing import Dict, List

from .dso_tokens import _constant_token_name
from .expression import ExpressionNode
from .feature_space import FeatureSpace
from .search_base import SearchCandidate


def _build_pset(feature_space: FeatureSpace, search_config: dict):
    from deap import gp

    pset = gp.PrimitiveSet("MAIN", 0)
    terminal_map: Dict[str, ExpressionNode] = {}

    for operator in sorted(feature_space.operators.values(), key=lambda item: item.name):
        if not operator.exportable or operator.arity <= 0:
            continue

        def primitive(*args):
            return args[0] if args else None

        pset.addPrimitive(primitive, operator.arity, name=operator.name)

    for idx, feature_name in enumerate(sorted(feature_space.features)):
        name = f"F_{idx}"
        pset.addTerminal(feature_name, name=name)
        terminal_map[name] = ExpressionNode(kind="feature", value=feature_name)

    constants = search_config.get("constants", {})
    if constants.get("enabled", False):
        for idx, raw_value in enumerate(constants.get("values", [])):
            value = float(raw_value)
            name = f"C_{idx}"
            pset.addTerminal(_constant_token_name(value), name=name)
            terminal_map[name] = ExpressionNode(kind="constant", value=value)

    for idx, window in enumerate(feature_space.windows):
        name = f"W_{idx}"
        pset.addTerminal(f"window:{int(window)}", name=name)
        terminal_map[name] = ExpressionNode(kind="window", value=int(window))

    return pset, terminal_map


def _expr_to_primitives(expr: ExpressionNode, pset) -> list:
    if expr.kind == "operator":
        nodes = [pset.mapping[str(expr.value)]]
        for child in expr.children:
            nodes.extend(_expr_to_primitives(child, pset))
        return nodes
    for name, node in pset.mapping.items():
        if getattr(node, "arity", None) != 0:
            continue
        if expr.kind == "feature" and str(expr.value) == str(getattr(node, "value", "")):
            return [node]
        if expr.kind == "constant" and str(getattr(node, "value", "")) == _constant_token_name(float(expr.value)):
            return [node]
        if expr.kind == "window" and str(getattr(node, "value", "")) == f"window:{int(expr.value)}":
            return [node]
    raise ValueError(f"Cannot convert expression terminal to GP node: {expr.to_string()}")


def _tree_to_expr(tree, terminal_map: Dict[str, ExpressionNode]) -> ExpressionNode:
    def parse_at(index: int) -> tuple[ExpressionNode, int]:
        node = tree[index]
        if node.arity == 0:
            terminal = terminal_map.get(node.name)
            if terminal is None:
                raise ValueError(f"Unknown GP terminal: {node.name}")
            return terminal, index + 1

        children = []
        cursor = index + 1
        for _ in range(node.arity):
            child, cursor = parse_at(cursor)
            children.append(child)
        return ExpressionNode(kind="operator", value=node.name, children=tuple(children)), cursor

    expr, consumed = parse_at(0)
    if consumed != len(tree):
        raise ValueError("GP tree has trailing nodes")
    return expr


def generate_gp_meld_expressions(
    hall: List[SearchCandidate],
    feature_space: FeatureSpace,
    search_config: dict,
    rng: random.Random,
) -> List[ExpressionNode]:
    gp_config = search_config.get("gp_meld", {})
    if not gp_config.get("enabled", False) or not hall:
        return []

    from deap import gp, tools

    pset, terminal_map = _build_pset(feature_space, search_config)
    population_size = int(gp_config.get("population_size", min(20, len(hall))))
    generations = int(gp_config.get("generations", 1))
    crossover_rate = float(gp_config.get("crossover_rate", 0.5))
    mutation_rate = float(gp_config.get("mutation_rate", 0.3))
    max_depth = int(search_config.get("priors", {}).get("max_depth", search_config.get("max_length", 20)))
    grow_max_depth = max(1, min(max_depth, 4))

    def mutation_expr(pset, type_=None):
        return gp.genGrow(pset=pset, min_=0, max_=grow_max_depth, type_=type_)

    population = []
    for candidate in hall[:population_size]:
        try:
            population.append(gp.PrimitiveTree(_expr_to_primitives(candidate.expression, pset)))
        except Exception:
            continue

    while len(population) < population_size:
        population.append(gp.PrimitiveTree(gp.genHalfAndHalf(pset=pset, min_=0, max_=grow_max_depth)))

    offspring = [copy.deepcopy(item) for item in population]
    for _ in range(generations):
        rng.shuffle(offspring)
        for left_idx in range(1, len(offspring), 2):
            if rng.random() <= crossover_rate:
                offspring[left_idx - 1], offspring[left_idx] = gp.cxOnePoint(offspring[left_idx - 1], offspring[left_idx])
        for idx, individual in enumerate(list(offspring)):
            if rng.random() <= mutation_rate:
                mutant, = gp.mutUniform(
                    copy.deepcopy(individual),
                    expr=mutation_expr,
                    pset=pset,
                )
                offspring[idx] = mutant

    expressions = []
    seen = set()
    for tree in offspring:
        try:
            expr = _tree_to_expr(tree, terminal_map)
        except Exception:
            continue
        key = expr.to_string()
        if key not in seen:
            seen.add(key)
            expressions.append(expr)
    return expressions[: int(gp_config.get("inject_top_k", len(expressions)))]
