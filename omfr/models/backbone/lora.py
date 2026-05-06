"""
lora.py — Task-conditional DoRA + AdaLoRA-flavor adapters for OMFR.

DoRA (Liu et al. 2024) decomposes a frozen pretrained linear weight W_0 into
its column magnitude m and direction V/||V||, then re-parameterizes the
trainable update as a magnitude vector m and a direction perturbation
delta = scaling * P @ diag(E) @ Q (AdaLoRA's SVD form, Zhang et al. 2023).

    W' = m * (W_0 + delta) / ||W_0 + delta||_col

We expose a ``TaskConditionalAdaDoRALinear`` that holds independent
(P, E, Q, m) per task id and is dispatched at forward time via ``set_task``.
This lets a shared frozen backbone serve identity and PAD with disjoint
adapters — the requirement for joint OMFR LoRA training.
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskConditionalAdaDoRALinear(nn.Module):
    """DoRA-decomposed nn.Linear with AdaLoRA SVD update, multi-task."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 16,
        alpha: float = 32.0,
        tasks: Sequence[str] = ("identity", "pad"),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("tasks must be non-empty")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.tasks: List[str] = list(tasks)
        self.current_task: str = self.tasks[0]
        self._dropout_p = float(dropout)

        in_features = base.in_features
        out_features = base.out_features

        self.lora_P = nn.ParameterDict()
        self.lora_E = nn.ParameterDict()
        self.lora_Q = nn.ParameterDict()
        self.magnitude = nn.ParameterDict()
        for t in self.tasks:
            P = torch.empty(out_features, self.rank)
            Q = torch.empty(self.rank, in_features)
            nn.init.kaiming_uniform_(P, a=math.sqrt(5))
            nn.init.kaiming_uniform_(Q, a=math.sqrt(5))
            self.lora_P[t] = nn.Parameter(P)
            self.lora_Q[t] = nn.Parameter(Q)
            # E init to 0 → adapter is identity at start (only m is non-trivial).
            self.lora_E[t] = nn.Parameter(torch.zeros(self.rank))
            with torch.no_grad():
                col_norm = base.weight.detach().norm(p=2, dim=1).clamp_min(1e-8)
            self.magnitude[t] = nn.Parameter(col_norm.clone())
            mask = torch.ones(self.rank)
            self.register_buffer(f"rank_mask_{t}", mask, persistent=True)

    def set_task(self, task: str) -> None:
        if task not in self.tasks:
            raise KeyError(f"task '{task}' not registered. Have: {self.tasks}")
        self.current_task = task

    def _delta(self, t: str) -> torch.Tensor:
        mask = getattr(self, f"rank_mask_{t}")
        E = self.lora_E[t] * mask
        delta = (self.lora_P[t] * E.unsqueeze(0)) @ self.lora_Q[t]
        return delta * self.scaling

    def _col_norm_no_grad(self, t: str, dtype: torch.dtype) -> torch.Tensor:
        """Detached ||W0 + Δ||_row, computed without retaining a graph.

        DoRA's "stable magnitude" trick already detaches the norm, so the
        full (W0 + Δ) only needs to exist transiently in no-grad context.
        We materialize Δ once, add to W0, take the row norm, then drop both.
        """
        with torch.no_grad():
            delta = self._delta(t).to(dtype)
            W = self.base.weight.to(dtype) + delta
            col_norm = W.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        return col_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """DoRA forward via the split path:

            y = (m / col_norm) * (W0 @ x + Δ @ x) + bias
              = (m / col_norm) * (F.linear(x, W0) + scaling * P (E (Q x))) + bias

        We never materialize ``W = W0 + Δ`` inside the autograd graph; only
        ``col_norm`` (detached, used as a scalar gain per output row) needs
        the full matrix, and that lives in a no-grad context.

        Numerically equivalent to the old ``F.linear(x, m*(W0+Δ)/||W0+Δ||)``
        path up to floating-point ordering — see scripts/ab_test_lora_cache_norm.py.

        Memory effect: the autograd graph no longer carries an ``(out, in)``
        tensor per Linear; only small ``(..., rank)`` intermediates from the
        ``Q -> E -> P`` projection are kept for backward.
        """
        t = self.current_task
        if self._dropout_p > 0 and self.training:
            x = F.dropout(x, p=self._dropout_p)

        W0 = self.base.weight                                  # (out, in), frozen
        bias = self.base.bias                                  # (out,) or None

        # 1) Base path — frozen weight, no graph through W0.
        y_base = F.linear(x, W0)                               # (..., out)

        # 2) LoRA delta path — small intermediates only.
        mask = getattr(self, f"rank_mask_{t}")
        E_eff = self.lora_E[t] * mask                          # (rank,)
        x_q = F.linear(x, self.lora_Q[t])                      # (..., rank)
        x_q = x_q * E_eff                                      # (..., rank)
        y_delta = F.linear(x_q, self.lora_P[t]) * self.scaling  # (..., out)

        # 3) DoRA scale via detached column norm (no graph through W).
        col_norm = self._col_norm_no_grad(t, W0.dtype)         # (out, 1), detached
        m = self.magnitude[t].to(W0.dtype)                     # (out,)
        scale = (m / col_norm.squeeze(-1))                     # (out,)

        y = (y_base + y_delta) * scale
        if bias is not None:
            y = y + bias
        return y

    def orthogonality_loss(self, task: Optional[str] = None) -> torch.Tensor:
        """AdaLoRA ||P^T P - I||_F^2 + ||Q Q^T - I||_F^2.

        ``task=None`` averages over every registered task so the regularizer
        keeps unused adapters numerically tame as well.
        """
        targets = [task] if task is not None else self.tasks
        I_r = torch.eye(
            self.rank,
            device=self.lora_P[self.tasks[0]].device,
            dtype=self.lora_P[self.tasks[0]].dtype,
        )
        accum = I_r.new_zeros(())
        for t in targets:
            P = self.lora_P[t]
            Q = self.lora_Q[t]
            accum = accum + (P.transpose(0, 1) @ P - I_r).pow(2).sum()
            accum = accum + (Q @ Q.transpose(0, 1) - I_r).pow(2).sum()
        return accum / max(len(targets), 1)


def inject_adadora(
    root: nn.Module,
    target_leaf_names: Sequence[str],
    rank: int,
    alpha: float,
    tasks: Sequence[str],
    dropout: float = 0.0,
) -> int:
    """Replace every nn.Linear whose attribute name is in target_leaf_names
    with a TaskConditionalAdaDoRALinear. Returns the number of replacements.
    """
    target_set = set(target_leaf_names)
    count = 0
    for module in list(root.modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target_set and isinstance(child, nn.Linear):
                wrapped = TaskConditionalAdaDoRALinear(
                    child, rank=rank, alpha=alpha, tasks=tasks, dropout=dropout
                )
                setattr(module, child_name, wrapped)
                count += 1
    return count


def set_lora_task(root: nn.Module, task: str) -> None:
    for m in root.modules():
        if isinstance(m, TaskConditionalAdaDoRALinear):
            m.set_task(task)


def freeze_non_lora(root: nn.Module) -> None:
    """Freeze every parameter under root that is not a LoRA adapter param."""
    lora_param_ids: set[int] = set()
    for m in root.modules():
        if isinstance(m, TaskConditionalAdaDoRALinear):
            for pdict in (m.lora_P, m.lora_E, m.lora_Q, m.magnitude):
                for p in pdict.parameters():
                    lora_param_ids.add(id(p))
    for p in root.parameters():
        if id(p) not in lora_param_ids:
            p.requires_grad = False


def lora_parameters(
    root: nn.Module,
    task: Optional[str] = None,
    include_magnitude: bool = True,
) -> Iterator[nn.Parameter]:
    """Yield (P, E, Q) and optionally DoRA magnitude params for one or all tasks."""
    for m in root.modules():
        if isinstance(m, TaskConditionalAdaDoRALinear):
            tasks = m.tasks if task is None else [task]
            for t in tasks:
                if t not in m.tasks:
                    continue
                yield m.lora_P[t]
                yield m.lora_E[t]
                yield m.lora_Q[t]
                if include_magnitude:
                    yield m.magnitude[t]


def collect_orthogonality_loss(
    root: nn.Module, task: Optional[str] = None
) -> torch.Tensor:
    """Sum the per-layer AdaLoRA orthogonality regularizer.

    ``task=None`` averages across all tasks per layer so the regularizer
    constrains every registered adapter even if only one was forwarded.
    """
    accum: Optional[torch.Tensor] = None
    n = 0
    for m in root.modules():
        if isinstance(m, TaskConditionalAdaDoRALinear):
            l = m.orthogonality_loss(task=task)
            accum = l if accum is None else accum + l
            n += 1
    if accum is None:
        return torch.zeros(())
    return accum / max(n, 1)


def count_trainable(root: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in root.parameters() if p.requires_grad)
    total = sum(p.numel() for p in root.parameters())
    return trainable, total
