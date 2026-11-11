"""PyTorch controller policy for DSO-style expression sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import nn

from .dso_priors import DSOPriorConfig
from .dso_tokens import DSOTokenVocab


def resolve_torch_device(requested: str) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class SampleBatch:
    token_ids: List[List[int]]
    log_probs: torch.Tensor
    entropies: torch.Tensor
    finished_mask: torch.Tensor
    lengths: torch.Tensor


class DSORNNPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def step(
        self,
        previous_token: torch.Tensor,
        hidden: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(previous_token).unsqueeze(1)
        output, next_hidden = self.rnn(embedded, hidden)
        return self.output(output[:, -1, :]), next_hidden


def _valid_token_mask(
    vocab: DSOTokenVocab,
    slots: int,
    remaining_positions: int,
    device: torch.device,
    expected_type: str = "expr",
    current_depth: int | None = None,
    max_depth: int | None = None,
    current_complexity: int = 0,
    max_complexity: int | None = None,
    used_feature_ids: frozenset[int] = frozenset(),
    min_feature_refs: int = 0,
    forbid_repeated_unary: bool = False,
    parent_unary: bool = False,
    requires_binary_op: bool = False,
    has_binary: bool = False,
    open_expr_slots: int = 0,
    open_nonwindow_slots: int = 0,
    rolling_depth: int = 0,
    max_rolling_depth: int | None = None,
    binary_rolling_count: int = 0,
    max_binary_rolling_ops: int | None = None,
    max_window: int | None = None,
) -> torch.Tensor:
    mask = torch.zeros(vocab.size, dtype=torch.bool, device=device)
    if slots <= 0:
        mask[vocab.end_id] = True
        return mask

    if expected_type == "window":
        mask[vocab.window_ids] = True
    elif expected_type == "scalar":
        mask[vocab.scalar_ids] = True
    elif remaining_positions <= slots:
        mask[vocab.terminal_ids] = True
    else:
        mask[vocab.expression_ids] = True

    # Cost of the current slot in the non-window close-budget accounting.
    slot_nonwindow_cost = 0 if expected_type in {"window", "scalar"} else 1

    constrained = mask.clone()
    for token_id, token in enumerate(vocab.tokens):
        if not bool(constrained[token_id]):
            continue
        if token.kind == "operator" and expected_type != "expr":
            constrained[token_id] = False
            continue
        if token.kind == "window":
            if expected_type != "window":
                constrained[token_id] = False
                continue
            # Window tokens above the prior cap make the whole program invalid
            # post-hoc; mask them at the slot instead of wasting the sample.
            if max_window is not None and float(token.value or 0) > float(max_window):
                constrained[token_id] = False
                continue
        # Constants are only meaningful as explicit scalar arguments (power /
        # clip / winsorize thresholds). Allowing them as standalone expr/terminal
        # leaves lets the policy emit constant-only, feature-free programs that
        # are rank-invariant noise and get rejected post-hoc as
        # ``no_feature_reference`` (~90% of generated programs). Restrict
        # constants to scalar slots so every expr/terminal leaf is a feature.
        if token.kind == "constant" and expected_type != "scalar":
            constrained[token_id] = False
            continue
        if (
            max_depth is not None
            and token.kind == "operator"
            and current_depth is not None
            and current_depth >= max_depth
        ):
            constrained[token_id] = False
            continue
        if token.kind == "operator":
            # Auxiliary generators such as sequence(window) produce an
            # expression without consuming a feature.  Do not let one close
            # the final feature-capable slot while the minimum feature prior
            # is still unmet.
            future_expr_slots = open_expr_slots - 1 + token.expr_arity
            if len(used_feature_ids) < int(min_feature_refs) and future_expr_slots <= 0:
                constrained[token_id] = False
                continue
            # Closability by length: placing an operator of arity a costs one
            # position and opens a slots, so the headroom D = remaining - slots
            # shrinks by a. D < a would leave more open slots than positions
            # and the program could never terminate (post-hoc
            # ``unterminated_expression``).
            if remaining_positions - slots < token.arity:
                constrained[token_id] = False
                continue
        if max_complexity is not None and token.kind in {
            "feature",
            "constant",
            "operator",
            "window",
        }:
            # Reserve close-out complexity: after placing this token, every
            # remaining non-window slot still needs at least a complexity-1
            # leaf (feature/constant). Without the reserve the budget can
            # exhaust mid-program, emptying the mask and cascading into
            # arbitrary fallback tokens (depth/complexity blowups).
            nonwindow_after = (
                open_nonwindow_slots
                - slot_nonwindow_cost
                + sum(1 for arg in token.arg_types if arg != "window")
            )
            if (
                current_complexity + token.complexity + max(0, nonwindow_after)
                > max_complexity
            ):
                constrained[token_id] = False
                continue
        if token.kind == "operator":
            # unary(unary(x)) chains are rejected post-hoc; mask the inner
            # unary at the slot whose parent is already a unary operator.
            if forbid_repeated_unary and parent_unary and token.arity == 1:
                constrained[token_id] = False
                continue
            if token.category == "time_series":
                if max_rolling_depth is not None and rolling_depth >= max_rolling_depth:
                    constrained[token_id] = False
                    continue
                if (
                    max_binary_rolling_ops is not None
                    and token.expr_arity >= 2
                    and binary_rolling_count >= max_binary_rolling_ops
                ):
                    constrained[token_id] = False
                    continue

    # Enforce the requires_binary_op prior during sampling. Capacity headroom
    # D = remaining_positions - slots is non-increasing (terminals keep it,
    # an operator of arity a costs a), and placing the cheapest binary needs
    # D >= 2, so while no binary-expr operator has been placed:
    #   * a binary op is allowed only when it still fits (D >= arity),
    #   * a non-binary operator is allowed only if a binary still fits after
    #     it, both by length (D - arity >= 2) and by complexity budget,
    #   * a terminal may not close the LAST open expr slot, and is pointless
    #     once a binary can never fit again.
    if (
        requires_binary_op
        and not has_binary
        and expected_type == "expr"
        and remaining_positions > slots
    ):
        headroom = remaining_positions - slots
        # Cheapest binary by complexity: its own complexity plus close-out
        # leaves for the non-window slots it opens, minus the close-leaf it
        # replaces (it occupies an expr slot already accounted for).
        min_binary_cost = min(
            (
                token.complexity
                + sum(1 for arg in token.arg_types if arg != "window")
                - 1
                for token in vocab.tokens
                if token.kind == "operator" and token.expr_arity >= 2
            ),
            default=2,
        )
        forced = constrained.clone()
        for token_id, token in enumerate(vocab.tokens):
            if not bool(forced[token_id]):
                continue
            if token.kind == "operator" and token.expr_arity >= 2:
                if headroom < token.arity:
                    forced[token_id] = False
                continue
            if token.kind == "operator":
                if headroom - token.arity < 2:
                    forced[token_id] = False
                    continue
                # While no binary has been placed, non-binary operators keep
                # at most one expr slot open (expr_arity <= 1), so the binary
                # must eventually fit on this single chain. A non-binary op at
                # depth d puts its expr child at d+1; if operators are blocked
                # there (d+1 >= max_depth) the binary can never be placed.
                if (
                    max_depth is not None
                    and current_depth is not None
                    and current_depth + 1 >= max_depth
                ):
                    forced[token_id] = False
                    continue
            elif token.kind in {"feature", "constant"}:
                if open_expr_slots <= 1 or headroom < 2:
                    forced[token_id] = False
                    continue
            if max_complexity is not None:
                nonwindow_after = (
                    open_nonwindow_slots
                    - slot_nonwindow_cost
                    + sum(1 for arg in token.arg_types if arg != "window")
                )
                if (
                    current_complexity
                    + token.complexity
                    + max(0, nonwindow_after)
                    + min_binary_cost
                    > max_complexity
                ):
                    forced[token_id] = False
        if bool(forced.any()):
            constrained = forced

    # Enforce >= min_feature_refs DISTINCT feature references during sampling so
    # such programs are valid by construction instead of being rejected post-hoc
    # (``min_feature_refs`` is the dominant waste once constant-only programs are
    # eliminated; referenced_features() counts distinct names, so mul(x, x) only
    # counts as one). Two rules, both guarded by spare length budget so they
    # never fight the terminal-closing logic:
    #   * slots < deficit  -> force a slot-growing (binary+) operator so more
    #     distinct feature leaves can fit (e.g. forces a binary root when 2 are
    #     required).
    #   * slots == deficit -> the budget is exactly tight, so drop already-used
    #     features: each remaining slot must net a NEW distinct feature (or an
    #     operator that grows the budget further).
    deficit = int(min_feature_refs) - len(used_feature_ids)
    if (
        deficit > 0
        and slots > 0
        and expected_type == "expr"
        and remaining_positions > slots
    ):
        if slots < deficit:
            # Only operators with >=2 expr arguments add feature-capable slots
            # (e.g. add/mul/ts_corr); ts_mean(expr, window) has arity 2 but one
            # window arg and would not make room for a second feature.
            grow = torch.zeros(vocab.size, dtype=torch.bool, device=device)
            for token_id, token in enumerate(vocab.tokens):
                if (
                    bool(constrained[token_id])
                    and token.kind == "operator"
                    and token.expr_arity >= 2
                ):
                    grow[token_id] = True
            if bool(grow.any()):
                return grow
        elif slots == deficit and used_feature_ids:
            tight = constrained.clone()
            for token_id in used_feature_ids:
                tight[token_id] = False
            if bool(tight.any()):
                return tight

    if bool(constrained.any()):
        return constrained
    return mask


@dataclass
class _RowState:
    """Mutable per-program prefix state threaded through autoregressive steps."""

    slots: int = 1
    pending_types: List[str] = field(default_factory=lambda: ["expr"])
    pending_depths: List[int] = field(default_factory=lambda: [1])
    pending_parent_unary: List[bool] = field(default_factory=lambda: [False])
    pending_rolling: List[int] = field(default_factory=lambda: [0])
    complexity: int = 0
    used_features: set[int] = field(default_factory=set)
    has_binary: bool = False
    binary_rolling_ops: int = 0
    finished: bool = False

    @property
    def expected_type(self) -> str:
        return self.pending_types[0] if self.pending_types else "expr"

    @property
    def current_depth(self) -> int | None:
        return self.pending_depths[0] if self.pending_depths else None

    @property
    def parent_unary(self) -> bool:
        return self.pending_parent_unary[0] if self.pending_parent_unary else False

    @property
    def rolling_depth(self) -> int:
        return self.pending_rolling[0] if self.pending_rolling else 0

    @property
    def open_expr_slots(self) -> int:
        return sum(1 for kind in self.pending_types if kind == "expr")

    @property
    def open_nonwindow_slots(self) -> int:
        return sum(1 for kind in self.pending_types if kind != "window")


def _update_prefix_state(
    vocab: DSOTokenVocab,
    token_id: int,
    state: _RowState,
) -> None:
    token = vocab.tokens[token_id]
    if state.pending_types:
        state.pending_types.pop(0)
    current_depth = state.pending_depths.pop(0) if state.pending_depths else 1
    parent_rolling = state.pending_rolling.pop(0) if state.pending_rolling else 0
    if state.pending_parent_unary:
        state.pending_parent_unary.pop(0)
    if token.kind == "end":
        state.slots = 0
        state.pending_types = []
        state.pending_depths = []
        state.pending_parent_unary = []
        state.pending_rolling = []
        state.finished = True
        return

    if token.kind == "feature":
        state.used_features.add(token_id)
    state.slots = state.slots - 1 + token.arity
    state.complexity += token.complexity

    is_operator = token.kind == "operator"
    is_unary = is_operator and token.arity == 1
    is_time_series = is_operator and token.category == "time_series"
    if is_operator and token.expr_arity >= 2:
        state.has_binary = True
        if is_time_series:
            state.binary_rolling_ops += 1
    child_rolling = parent_rolling + (1 if is_time_series else 0)

    if token.arity:
        arg_types = list(token.arg_types or ("expr",) * token.arity)
        state.pending_types = arg_types + state.pending_types
        state.pending_depths = [
            current_depth + 1 for _ in range(token.arity)
        ] + state.pending_depths
        state.pending_parent_unary = [
            is_unary for _ in range(token.arity)
        ] + state.pending_parent_unary
        state.pending_rolling = [
            child_rolling for _ in range(token.arity)
        ] + state.pending_rolling
    state.finished = state.slots <= 0
    if state.finished:
        state.pending_types = []
        state.pending_depths = []
        state.pending_parent_unary = []
        state.pending_rolling = []


def _resolve_priors(
    priors: DSOPriorConfig | None,
    max_depth: int | None,
    max_complexity: int | None,
    min_feature_refs: int,
) -> DSOPriorConfig:
    """Resolve the effective prior set for in-situ masking.

    Legacy callers pass the three structural limits as bare kwargs; they get a
    permissive prior config (no binary/unary/rolling/window masking) so replay
    parity with previously sampled programs is preserved.
    """
    if priors is not None:
        return priors
    return DSOPriorConfig(
        max_depth=max_depth,
        max_complexity=max_complexity,
        forbid_repeated_unary=False,
        min_feature_refs=min_feature_refs,
        requires_binary_op=False,
        max_rolling_depth=None,
        max_binary_rolling_ops=None,
        max_window=None,
    )


def _row_mask(
    vocab: DSOTokenVocab,
    state: _RowState,
    remaining_positions: int,
    device: torch.device,
    priors: DSOPriorConfig,
) -> torch.Tensor:
    return _valid_token_mask(
        vocab,
        0 if state.finished else state.slots,
        remaining_positions,
        device,
        expected_type=state.expected_type,
        current_depth=state.current_depth,
        max_depth=priors.max_depth,
        current_complexity=state.complexity,
        max_complexity=priors.max_complexity,
        used_feature_ids=frozenset(state.used_features),
        min_feature_refs=priors.min_feature_refs,
        forbid_repeated_unary=priors.forbid_repeated_unary,
        parent_unary=state.parent_unary,
        requires_binary_op=priors.requires_binary_op,
        has_binary=state.has_binary,
        open_expr_slots=state.open_expr_slots,
        open_nonwindow_slots=state.open_nonwindow_slots,
        rolling_depth=state.rolling_depth,
        max_rolling_depth=priors.max_rolling_depth,
        binary_rolling_count=state.binary_rolling_ops,
        max_binary_rolling_ops=priors.max_binary_rolling_ops,
        max_window=priors.max_window,
    )


def sample_programs(
    policy: DSORNNPolicy,
    vocab: DSOTokenVocab,
    batch_size: int,
    max_length: int,
    device: torch.device,
    max_depth: int | None = None,
    max_complexity: int | None = None,
    min_feature_refs: int = 0,
    priors: DSOPriorConfig | None = None,
) -> SampleBatch:
    effective_priors = _resolve_priors(
        priors, max_depth, max_complexity, min_feature_refs
    )
    previous = torch.full(
        (batch_size,), vocab.start_id, dtype=torch.long, device=device
    )
    hidden = None
    states = [_RowState() for _ in range(batch_size)]
    token_ids: List[List[int]] = [[] for _ in range(batch_size)]
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []

    for step_idx in range(max_length):
        logits, hidden = policy.step(previous, hidden)
        masks = torch.stack(
            [
                _row_mask(
                    vocab, states[row], max_length - step_idx, device, effective_priors
                )
                for row in range(batch_size)
            ]
        )
        masked_logits = logits.masked_fill(~masks, -1e9)
        distribution = torch.distributions.Categorical(logits=masked_logits)
        sampled = distribution.sample()
        log_prob = distribution.log_prob(sampled)
        entropy = distribution.entropy()

        for row, token_id in enumerate(sampled.tolist()):
            if states[row].finished:
                continue
            token_ids[row].append(token_id)
            _update_prefix_state(vocab, token_id, states[row])

        log_probs.append(log_prob)
        entropies.append(entropy)
        previous = sampled
        if all(state.finished for state in states):
            break

    finished = [state.finished for state in states]
    return SampleBatch(
        token_ids=token_ids,
        log_probs=torch.stack(log_probs, dim=1).sum(dim=1),
        entropies=torch.stack(entropies, dim=1).sum(dim=1),
        finished_mask=torch.tensor(finished, dtype=torch.bool, device=device),
        lengths=torch.tensor(
            [len(row) for row in token_ids], dtype=torch.long, device=device
        ),
    )


def evaluate_program_log_probs(
    policy: DSORNNPolicy,
    vocab: DSOTokenVocab,
    token_ids: List[List[int]],
    max_length: int,
    device: torch.device,
    max_depth: int | None = None,
    max_complexity: int | None = None,
    min_feature_refs: int = 0,
    priors: DSOPriorConfig | None = None,
) -> SampleBatch:
    effective_priors = _resolve_priors(
        priors, max_depth, max_complexity, min_feature_refs
    )
    batch_size = len(token_ids)
    previous = torch.full(
        (batch_size,), vocab.start_id, dtype=torch.long, device=device
    )
    hidden = None
    states = [_RowState() for _ in range(batch_size)]
    max_steps = min(max((len(row) for row in token_ids), default=0), max_length)
    log_prob_sums = torch.zeros(batch_size, dtype=torch.float32, device=device)
    entropy_sums = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for step_idx in range(max_steps):
        logits, hidden = policy.step(previous, hidden)
        masks = torch.stack(
            [
                _row_mask(
                    vocab, states[row], max_length - step_idx, device, effective_priors
                )
                for row in range(batch_size)
            ]
        )
        distribution = torch.distributions.Categorical(
            logits=logits.masked_fill(~masks, -1e9)
        )
        targets = []
        active = []
        for row in range(batch_size):
            has_token = step_idx < len(token_ids[row])
            active.append(has_token and not states[row].finished)
            targets.append(token_ids[row][step_idx] if has_token else vocab.end_id)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
        active_tensor = torch.tensor(active, dtype=torch.bool, device=device)
        log_prob_sums = (
            log_prob_sums + distribution.log_prob(target_tensor) * active_tensor.float()
        )
        entropy_sums = entropy_sums + distribution.entropy() * active_tensor.float()

        for row, token_id in enumerate(targets):
            if not active[row]:
                continue
            _update_prefix_state(vocab, token_id, states[row])
        previous = target_tensor

    finished = [state.finished for state in states]
    return SampleBatch(
        token_ids=token_ids,
        log_probs=log_prob_sums,
        entropies=entropy_sums,
        finished_mask=torch.tensor(finished, dtype=torch.bool, device=device),
        lengths=torch.tensor(
            [len(row) for row in token_ids], dtype=torch.long, device=device
        ),
    )
