"""PyTorch DSO-style symbolic search backend."""

from __future__ import annotations

import math
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .dso_gp import generate_gp_meld_expressions
from .dso_policy import (
    DSORNNPolicy,
    evaluate_program_log_probs,
    resolve_torch_device,
    sample_programs,
)
from .dso_priors import build_dso_prior_config, expression_prior_error
from .dso_program import parse_dso_program
from .dso_tokens import build_dso_vocab
from .elite_pool import load_elite_expressions
from .expression import (
    ExpressionNode,
    canonical_expression_hash,
    evaluate_expression_cached,
    expression_hash,
    iter_constants,
    replace_constants,
)
from .feature_space import FeatureSpace
from .objective import anchor_matrix_from_panel, score_expression
from .search_base import SearchCandidate, SearchResult
from .search_random import (
    _label_data,
    maintain_hall_of_fame,
    score_candidate,
    score_candidate_for_hall,
)


INVALID_REWARD = -1.0


def _duration(started_at: float) -> float:
    return round(time.perf_counter() - started_at, 6)


def _emit_progress(config: dict, event: str, **payload) -> None:
    progress = config.get("_progress_writer")
    if progress is not None:
        progress.emit(event, **payload)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _score_expression_candidate(
    expr: ExpressionNode,
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    config: dict,
) -> SearchCandidate | None:
    candidate = score_candidate(expr, train_panel, valid_panel, config)
    if candidate is None:
        return None
    train_score = float(candidate.train_metrics.get("score", INVALID_REWARD))
    valid_score = float(candidate.valid_metrics.get("score", INVALID_REWARD))
    if not np.isfinite(train_score) or not np.isfinite(valid_score):
        return None
    return candidate


def _expression_eval_cache(config: dict) -> dict | None:
    if bool(config.get("_disable_expression_cache", False)):
        return None
    return config.setdefault("_expression_eval_cache", {})


def _evaluate_expression_for_search(
    expr: ExpressionNode, panel: pd.DataFrame, config: dict
) -> pd.Series:
    return evaluate_expression_cached(expr, panel, _expression_eval_cache(config))


def _expression_digest(expr: ExpressionNode, canonicalize: bool) -> str:
    return canonical_expression_hash(expr) if canonicalize else expression_hash(expr)


def _constant_optimized_expression(
    expr: ExpressionNode,
    train_panel: pd.DataFrame,
    config: dict,
) -> ExpressionNode:
    search_config = config.get("search", {})
    constants_config = search_config.get("constants", {})
    optimize_config = constants_config.get("optimize", {})
    if not optimize_config.get("enabled", False):
        return expr

    values = list(iter_constants(expr))
    if not values:
        return expr
    if len(values) > int(optimize_config.get("max_constants", 4)):
        return expr

    try:
        from scipy.optimize import minimize
    except Exception:
        return expr

    best_values = np.asarray(values, dtype=float)
    best_score = -math.inf
    train_anchors = anchor_matrix_from_panel(train_panel)

    def train_score(raw_values: np.ndarray) -> float:
        nonlocal best_score, best_values
        try:
            candidate_expr = replace_constants(
                expr, [float(value) for value in raw_values]
            )
            pred = _evaluate_expression_for_search(candidate_expr, train_panel, config)
            metrics = score_expression(
                pred,
                _label_data(train_panel),
                train_panel["date"],
                candidate_expr,
                config,
                anchors=train_anchors,
            )
            score = float(metrics.get("score", INVALID_REWARD))
        except Exception:
            return -math.inf
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_values = np.asarray(raw_values, dtype=float)
        return score if np.isfinite(score) else -math.inf

    train_score(best_values)
    bounds = None
    if "min_value" in optimize_config and "max_value" in optimize_config:
        bounds = [
            (float(optimize_config["min_value"]), float(optimize_config["max_value"]))
            for _ in range(len(best_values))
        ]

    try:
        minimize(
            lambda raw_values: -train_score(np.asarray(raw_values, dtype=float)),
            best_values,
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(optimize_config.get("maxiter", 20)), "disp": False},
        )
    except Exception:
        return expr

    if not np.isfinite(best_score):
        return expr
    return replace_constants(expr, [float(value) for value in best_values])


def _fallback_feature_candidate(
    train_panel: pd.DataFrame,
    valid_panel: pd.DataFrame,
    feature_space: FeatureSpace,
    config: dict,
) -> SearchCandidate | None:
    best: SearchCandidate | None = None
    for feature_name in sorted(feature_space.features):
        expr = ExpressionNode(kind="feature", value=feature_name)
        candidate = _score_expression_candidate(expr, train_panel, valid_panel, config)
        if candidate is not None and (best is None or candidate.score > best.score):
            best = candidate
    return best


def _progress_enabled(search_config: dict) -> bool:
    if "show_progress" in search_config:
        return bool(search_config["show_progress"])
    return bool(sys.stderr.isatty())


def _epoch_iterator(epochs: int, enabled: bool):
    if not enabled:
        return range(epochs)
    from tqdm.auto import tqdm

    return tqdm(range(epochs), desc="dso_pytorch", file=sys.stderr, leave=False)


def _risk_pg_update(
    policy: DSORNNPolicy,
    optimizer: torch.optim.Optimizer,
    sample,
    reward_tensor: torch.Tensor,
    elite_indices: torch.Tensor,
    advantages: torch.Tensor,
    entropy_weight: float,
    grad_clip: float,
) -> torch.Tensor:
    loss = -(sample.log_probs[elite_indices] * advantages).mean()
    if entropy_weight:
        loss = loss - entropy_weight * sample.entropies[elite_indices].mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()
    return loss.detach()


def _ppo_update(
    policy: DSORNNPolicy,
    optimizer: torch.optim.Optimizer,
    vocab,
    sample,
    reward_tensor: torch.Tensor,
    elite_indices: torch.Tensor,
    baseline: torch.Tensor,
    entropy_weight: float,
    grad_clip: float,
    max_length: int,
    device: torch.device,
    prior_config,
    search_config: dict,
) -> torch.Tensor:
    ppo_clip = float(search_config.get("ppo_clip", 0.2))
    ppo_epochs = int(search_config.get("ppo_epochs", 2))
    minibatch_size = int(
        search_config.get("ppo_minibatch_size", max(1, len(elite_indices)))
    )
    last_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

    for _ in range(ppo_epochs):
        order = torch.randperm(len(elite_indices), device=device)
        for start in range(0, len(elite_indices), minibatch_size):
            batch_elite = elite_indices[order[start : start + minibatch_size]]
            batch_rows = [int(idx) for idx in batch_elite.detach().cpu().tolist()]
            # Replay must use the same prior masks as sampling, otherwise the
            # recomputed log-probs diverge from the sampled ones.
            new_sample = evaluate_program_log_probs(
                policy,
                vocab,
                [sample.token_ids[idx] for idx in batch_rows],
                max_length,
                device,
                priors=prior_config,
            )
            old_log_probs = sample.log_probs[batch_elite].detach()
            advantages = (reward_tensor[batch_elite] - baseline).detach()
            ratio = torch.exp(new_sample.log_probs - old_log_probs)
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * advantages
            loss = -torch.min(unclipped, clipped).mean()
            if entropy_weight:
                loss = loss - entropy_weight * new_sample.entropies.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            last_loss = loss.detach()
    return last_loss


class DSOPyTorchSearchBackend:
    def search(
        self,
        train_panel: pd.DataFrame,
        valid_panel: pd.DataFrame,
        feature_space: FeatureSpace,
        config: dict,
    ) -> SearchResult:
        search_config = config.get("search", {})
        seed = int(search_config.get("seed", 42))
        _set_seeds(seed)
        rng = random.Random(seed)

        device = resolve_torch_device(str(search_config.get("device", "auto")))
        vocab = build_dso_vocab(feature_space, search_config)
        prior_config = build_dso_prior_config(search_config)
        policy = DSORNNPolicy(
            vocab_size=vocab.size,
            embedding_dim=int(search_config.get("embedding_dim", 64)),
            hidden_dim=int(search_config.get("hidden_dim", 128)),
            num_layers=int(search_config.get("num_layers", 1)),
        ).to(device)
        optimizer = torch.optim.Adam(
            policy.parameters(), lr=float(search_config.get("learning_rate", 0.001))
        )

        epochs = int(search_config.get("epochs", 20))
        batch_size = int(search_config.get("batch_size", 256))
        max_length = int(search_config.get("max_length", 20))
        epsilon = max(0.0, min(1.0, float(search_config.get("epsilon", 0.2))))
        entropy_weight = float(search_config.get("entropy_weight", 0.01))
        hall_top_k = int(search_config.get("hall_top_k", 50))
        elite_count = max(1, int(math.ceil(batch_size * epsilon)))
        optimizer_type = str(search_config.get("optimizer", "risk_pg"))
        grad_clip = float(search_config.get("grad_clip", 5.0))
        canonicalize = bool(search_config.get("canonicalize", {}).get("enabled", True))
        progress_enabled = _progress_enabled(search_config)

        _emit_progress(
            config,
            "dso_search_initialized",
            device=str(device),
            vocab_size=int(vocab.size),
            epochs=int(epochs),
            batch_size=int(batch_size),
            max_length=int(max_length),
            optimizer=optimizer_type,
            canonicalize=bool(canonicalize),
        )

        seen_hashes: set[str] = set()
        cached_candidates: Dict[str, SearchCandidate] = {}
        hall: List[SearchCandidate] = []
        attempted = 0
        parsed = 0
        scored = 0
        duplicate = 0
        invalid = 0
        invalid_by_reason: Dict[str, int] = defaultdict(int)
        finished_count = 0
        gp_generated = 0
        gp_scored = 0
        full_robustness_rescored = 0
        epoch_summaries = []

        def mark_invalid(reason: str) -> None:
            invalid_by_reason[reason or "invalid"] += 1

        def prepare_expression(
            expr: ExpressionNode,
        ) -> tuple[ExpressionNode | None, str]:
            reason = expression_prior_error(expr, feature_space, prior_config)
            if reason:
                return None, reason
            optimized = _constant_optimized_expression(expr, train_panel, config)
            reason = expression_prior_error(optimized, feature_space, prior_config)
            if reason:
                return None, reason
            return optimized, ""

        def score_prepared_expression(
            expr: ExpressionNode,
        ) -> tuple[SearchCandidate | None, str]:
            nonlocal full_robustness_rescored
            digest = _expression_digest(expr, canonicalize)
            if digest in cached_candidates:
                return cached_candidates[digest], "cached"
            if digest in seen_hashes:
                return None, "duplicate"
            seen_hashes.add(digest)
            candidate, rescored = score_candidate_for_hall(
                expr,
                train_panel,
                valid_panel,
                config,
                hall=hall,
                top_k=hall_top_k,
            )
            if candidate is None:
                return None, "score_failed"
            if rescored:
                full_robustness_rescored += 1
            cached_candidates[digest] = candidate
            return candidate, ""

        # Warm start: seed the hall with proven expressions from prior runs so
        # the GP-meld recombines from known good structures instead of starting
        # cold (warm-start GP lifts OOS on weak signals, arXiv 2412.00896). The
        # pool is empty on the first run (no-op); seeds are prior-validated,
        # feature-filtered, and scored/deduped through the same path as sampled
        # programs.
        warm_start_config = search_config.get("warm_start", {})
        if isinstance(warm_start_config, dict) and warm_start_config.get("enabled"):
            available_features = set(feature_space.features.keys())
            seed_expressions = load_elite_expressions(
                warm_start_config.get("pool_path") or "state/symbolic_elite_pool.json",
                max_seeds=int(warm_start_config.get("max_seeds", 24) or 0),
                available_features=available_features,
            )
            warm_started = 0
            for seed_expr in seed_expressions:
                prepared_seed, prepare_error = prepare_expression(seed_expr)
                if prepared_seed is None:
                    mark_invalid(prepare_error or "warm_start_prior_error")
                    continue
                seed_candidate, status = score_prepared_expression(prepared_seed)
                if seed_candidate is None:
                    continue
                if status != "cached":
                    scored += 1
                hall.append(seed_candidate)
                warm_started += 1
            if warm_started:
                hall = maintain_hall_of_fame(hall, hall_top_k)
            _emit_progress(
                config,
                "dso_warm_start_seeded",
                requested=int(len(seed_expressions)),
                seeded=int(warm_started),
                pool_path=str(
                    warm_start_config.get("pool_path")
                    or "state/symbolic_elite_pool.json"
                ),
            )

        for epoch in _epoch_iterator(epochs, progress_enabled):
            epoch_started_at = time.perf_counter()
            epoch_score_seconds = 0.0
            epoch_max_score_seconds = 0.0
            epoch_scored_before = scored
            _emit_progress(
                config,
                "dso_epoch_start",
                epoch=int(epoch + 1),
                attempted=int(attempted),
                scored=int(scored),
                invalid=int(invalid),
                duplicates=int(duplicate),
            )
            policy.train()
            sample_started_at = time.perf_counter()
            sample = sample_programs(
                policy,
                vocab,
                batch_size,
                max_length,
                device,
                priors=prior_config,
            )
            _emit_progress(
                config,
                "dso_sample_done",
                epoch=int(epoch + 1),
                duration_sec=_duration(sample_started_at),
                finished=int(sample.finished_mask.sum().detach().cpu()),
            )
            rewards: List[float] = []
            epoch_valid = 0
            finished_count += int(sample.finished_mask.sum().detach().cpu())

            for token_ids in sample.token_ids:
                attempted += 1
                candidate_started_at = time.perf_counter()
                candidate_context = {
                    "backend": "dso_pytorch",
                    "epoch": int(epoch + 1),
                    "attempted": int(attempted),
                    "scored": int(scored),
                }
                config["_progress_context"] = candidate_context
                parse_started_at = time.perf_counter()
                parsed_program = parse_dso_program(
                    token_ids, vocab, feature_space, max_length
                )
                if not parsed_program.ok or parsed_program.expression is None:
                    invalid += 1
                    mark_invalid(parsed_program.error)
                    rewards.append(INVALID_REWARD)
                    _emit_progress(
                        config,
                        "dso_candidate_done",
                        **candidate_context,
                        status="parse_failed",
                        reason=parsed_program.error,
                        parse_duration_sec=_duration(parse_started_at),
                        duration_sec=_duration(candidate_started_at),
                    )
                    config.pop("_progress_context", None)
                    continue

                parsed += 1
                prepare_started_at = time.perf_counter()
                prepared_expr, error = prepare_expression(parsed_program.expression)
                if prepared_expr is None:
                    invalid += 1
                    mark_invalid(error)
                    rewards.append(INVALID_REWARD)
                    _emit_progress(
                        config,
                        "dso_candidate_done",
                        **candidate_context,
                        status="prepare_failed",
                        reason=error,
                        parse_duration_sec=_duration(parse_started_at),
                        prepare_duration_sec=_duration(prepare_started_at),
                        duration_sec=_duration(candidate_started_at),
                    )
                    config.pop("_progress_context", None)
                    continue

                score_started_at = time.perf_counter()
                candidate, status = score_prepared_expression(prepared_expr)
                score_duration = _duration(score_started_at)
                epoch_score_seconds += score_duration
                epoch_max_score_seconds = max(epoch_max_score_seconds, score_duration)
                if status == "cached" and candidate is not None:
                    duplicate += 1
                    rewards.append(
                        float(candidate.train_metrics.get("score", INVALID_REWARD))
                    )
                    _emit_progress(
                        config,
                        "dso_candidate_done",
                        **candidate_context,
                        status="cached",
                        expression=prepared_expr.to_string(),
                        score_duration_sec=score_duration,
                        duration_sec=_duration(candidate_started_at),
                    )
                    config.pop("_progress_context", None)
                    continue
                if status == "duplicate":
                    duplicate += 1
                    rewards.append(INVALID_REWARD)
                    _emit_progress(
                        config,
                        "dso_candidate_done",
                        **candidate_context,
                        status="duplicate",
                        expression=prepared_expr.to_string(),
                        score_duration_sec=score_duration,
                        duration_sec=_duration(candidate_started_at),
                    )
                    config.pop("_progress_context", None)
                    continue
                if candidate is None:
                    invalid += 1
                    mark_invalid(status)
                    rewards.append(INVALID_REWARD)
                    _emit_progress(
                        config,
                        "dso_candidate_done",
                        **candidate_context,
                        status="score_failed",
                        expression=prepared_expr.to_string(),
                        score_duration_sec=score_duration,
                        duration_sec=_duration(candidate_started_at),
                    )
                    config.pop("_progress_context", None)
                    continue

                scored += 1
                epoch_valid += 1
                rewards.append(
                    float(candidate.train_metrics.get("score", INVALID_REWARD))
                )
                hall.append(candidate)
                _emit_progress(
                    config,
                    "dso_candidate_done",
                    **candidate_context,
                    status="scored",
                    expression=prepared_expr.to_string(),
                    score=float(candidate.score),
                    train_score=candidate.train_metrics.get("score"),
                    valid_score=candidate.valid_metrics.get("score"),
                    score_duration_sec=score_duration,
                    duration_sec=_duration(candidate_started_at),
                )
                config.pop("_progress_context", None)

            hall = maintain_hall_of_fame(hall, hall_top_k)
            reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            topk = min(elite_count, reward_tensor.numel())
            elite_indices = torch.topk(reward_tensor, k=topk).indices
            elite_rewards = reward_tensor[elite_indices]
            baseline = reward_tensor.mean()
            advantages = (elite_rewards - baseline).detach()
            update_started_at = time.perf_counter()
            if optimizer_type == "ppo":
                loss = _ppo_update(
                    policy,
                    optimizer,
                    vocab,
                    sample,
                    reward_tensor,
                    elite_indices,
                    baseline,
                    entropy_weight,
                    grad_clip,
                    max_length,
                    device,
                    prior_config,
                    search_config,
                )
            else:
                loss = _risk_pg_update(
                    policy,
                    optimizer,
                    sample,
                    reward_tensor,
                    elite_indices,
                    advantages,
                    entropy_weight,
                    grad_clip,
                )
            update_duration = _duration(update_started_at)

            gp_epoch_scored = 0
            gp_config = search_config.get("gp_meld", {})
            gp_started_at = time.perf_counter()
            if (
                gp_config.get("enabled", False)
                and hall
                and (epoch + 1) % int(gp_config.get("every_epochs", 1)) == 0
            ):
                gp_exprs = generate_gp_meld_expressions(
                    hall, feature_space, search_config, rng
                )
                gp_generated += len(gp_exprs)
                for expr in gp_exprs:
                    prepared_expr, error = prepare_expression(expr)
                    if prepared_expr is None:
                        mark_invalid(f"gp_{error}")
                        continue
                    candidate, status = score_prepared_expression(prepared_expr)
                    if status in {"cached", "duplicate"}:
                        continue
                    if candidate is None:
                        mark_invalid(f"gp_{status}")
                        continue
                    hall.append(candidate)
                    scored += 1
                    gp_scored += 1
                    gp_epoch_scored += 1
                hall = maintain_hall_of_fame(hall, hall_top_k)
            gp_duration = _duration(gp_started_at)

            epoch_summaries.append(
                {
                    "epoch": epoch + 1,
                    "valid": epoch_valid,
                    "finished": int(sample.finished_mask.sum().detach().cpu()),
                    "reward_mean": float(reward_tensor.mean().detach().cpu()),
                    "reward_max": float(reward_tensor.max().detach().cpu()),
                    "loss": float(loss.detach().cpu()),
                    "gp_scored": gp_epoch_scored,
                    "duration_sec": _duration(epoch_started_at),
                    "score_duration_sec": round(epoch_score_seconds, 6),
                    "max_candidate_score_duration_sec": round(
                        epoch_max_score_seconds, 6
                    ),
                    "optimizer_update_duration_sec": update_duration,
                    "gp_duration_sec": gp_duration,
                }
            )
            _emit_progress(
                config,
                "dso_epoch_done",
                epoch=int(epoch + 1),
                duration_sec=_duration(epoch_started_at),
                epoch_scored=int(scored - epoch_scored_before),
                valid=int(epoch_valid),
                finished=int(sample.finished_mask.sum().detach().cpu()),
                reward_mean=float(reward_tensor.mean().detach().cpu()),
                reward_max=float(reward_tensor.max().detach().cpu()),
                loss=float(loss.detach().cpu()),
                score_duration_sec=round(epoch_score_seconds, 6),
                max_candidate_score_duration_sec=round(epoch_max_score_seconds, 6),
                optimizer_update_duration_sec=update_duration,
                gp_duration_sec=gp_duration,
            )

        fallback_used = False
        if not hall:
            fallback = _fallback_feature_candidate(
                train_panel, valid_panel, feature_space, config
            )
            if fallback is None:
                raise RuntimeError("dso_pytorch produced no valid candidates")
            hall = [fallback]
            fallback_used = True

        return SearchResult(
            best=hall[0],
            hall_of_fame=hall,
            metadata={
                "backend": "dso_pytorch",
                "attempted": attempted,
                "parsed": parsed,
                "scored": scored,
                "duplicates": duplicate,
                "invalid": invalid,
                "invalid_by_reason": dict(sorted(invalid_by_reason.items())),
                "unique_expressions": len(seen_hashes),
                "epochs": epochs,
                "batch_size": batch_size,
                "max_length": max_length,
                "epsilon": epsilon,
                "elite_count": elite_count,
                "optimizer": optimizer_type,
                "device": str(device),
                "seed": seed,
                "vocab_size": vocab.size,
                "finished_count": finished_count,
                "progress_enabled": progress_enabled,
                "canonicalize": canonicalize,
                "gp_generated": gp_generated,
                "gp_scored": gp_scored,
                "full_robustness_rescored": full_robustness_rescored,
                "fallback_used": fallback_used,
                "epoch_summaries": epoch_summaries,
            },
        )
