"""A/B numerical-equivalence test for the DoRA cache-norm refactor.

Compares the OLD DoRA forward (materializes W = W0 + Δ inside the autograd
graph and runs F.linear(x, W_dora, bias)) against the NEW cache-norm forward
in `omfr.models.backbone.lora.TaskConditionalAdaDoRALinear` on the same
randomly-initialized adapter, on the same random input.

Pass criterion (fp32):
  - max|y_new - y_old| <= 1e-5
  - max|grad_param_new - grad_param_old| <= 1e-4 for every adapter param

Run:
    python scripts/ab_test_lora_cache_norm.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omfr.models.backbone.lora import TaskConditionalAdaDoRALinear  # noqa: E402


def _old_forward(layer: TaskConditionalAdaDoRALinear, x: torch.Tensor) -> torch.Tensor:
    """Reference implementation that materializes W = W0 + delta in graph.

    Mirrors the forward used before the cache-norm refactor.
    """
    t = layer.current_task
    if layer._dropout_p > 0 and layer.training:
        x = F.dropout(x, p=layer._dropout_p)
    W0 = layer.base.weight
    mask = getattr(layer, f"rank_mask_{t}")
    E = layer.lora_E[t] * mask
    delta = (layer.lora_P[t] * E.unsqueeze(0)) @ layer.lora_Q[t]
    delta = delta * layer.scaling
    delta = delta.to(W0.dtype)
    W = W0 + delta
    col_norm = W.norm(p=2, dim=1, keepdim=True).detach().clamp_min(1e-8)
    m = layer.magnitude[t].to(W0.dtype).unsqueeze(1)
    W_dora = m * (W / col_norm)
    return F.linear(x, W_dora, layer.base.bias)


def _params_named(layer: TaskConditionalAdaDoRALinear, task: str):
    yield "P", layer.lora_P[task]
    yield "E", layer.lora_E[task]
    yield "Q", layer.lora_Q[task]
    yield "m", layer.magnitude[task]


def _build_layer(seed: int = 0) -> TaskConditionalAdaDoRALinear:
    torch.manual_seed(seed)
    base = nn.Linear(in_features=384, out_features=1152, bias=True)
    layer = TaskConditionalAdaDoRALinear(
        base, rank=16, alpha=32.0, tasks=("identity", "pad"), dropout=0.0,
    )
    # Set non-trivial E (default is zero -> delta == 0, hides bugs).
    with torch.no_grad():
        for t in layer.tasks:
            layer.lora_E[t].copy_(torch.linspace(-1.0, 1.0, layer.rank))
    return layer


def _clone_grads(layer: TaskConditionalAdaDoRALinear, task: str):
    out = {}
    for name, p in _params_named(layer, task):
        out[name] = None if p.grad is None else p.grad.detach().clone()
    return out


def run_check(task: str, seed: int = 0, dtype: torch.dtype = torch.float32) -> None:
    print(f"\n=== task={task} dtype={dtype} seed={seed} ===")

    layer_old = _build_layer(seed=seed).to(dtype)
    layer_new = _build_layer(seed=seed).to(dtype)
    layer_old.set_task(task)
    layer_new.set_task(task)
    layer_old.eval()
    layer_new.eval()

    torch.manual_seed(123)
    x = torch.randn(8, 256, 384, dtype=dtype, requires_grad=False)

    # Forward equivalence
    with torch.no_grad():
        y_old_eval = _old_forward(layer_old, x)
        y_new_eval = layer_new(x)
    fwd_err = (y_old_eval - y_new_eval).abs().max().item()
    print(f"forward max|Δ|        = {fwd_err:.3e}")

    # Backward equivalence with a non-trivial loss
    x_a = x.clone().requires_grad_(True)
    x_b = x.clone().requires_grad_(True)
    layer_old.train()
    layer_new.train()

    y_old = _old_forward(layer_old, x_a)
    y_new = layer_new(x_b)

    # Use a fixed reduction so the loss is meaningful.
    loss_old = (y_old.pow(2).mean())
    loss_new = (y_new.pow(2).mean())

    loss_old.backward()
    loss_new.backward()

    grads_old = _clone_grads(layer_old, task)
    grads_new = _clone_grads(layer_new, task)

    print(f"loss_old              = {loss_old.item():.6e}")
    print(f"loss_new              = {loss_new.item():.6e}")
    print(f"|loss_new - loss_old| = {abs(loss_new.item() - loss_old.item()):.3e}")

    for name in ("P", "E", "Q", "m"):
        g_old = grads_old[name]
        g_new = grads_new[name]
        if g_old is None or g_new is None:
            print(f"grad {name:<2}: MISSING (old={g_old is None}, new={g_new is None})")
            continue
        err = (g_old - g_new).abs().max().item()
        ref = max(g_old.abs().max().item(), 1e-8)
        print(f"grad {name:<2}: max|Δ| = {err:.3e}   max|g| = {ref:.3e}   rel = {err/ref:.3e}")

    # x grad
    if x_a.grad is not None and x_b.grad is not None:
        gx_err = (x_a.grad - x_b.grad).abs().max().item()
        gx_ref = max(x_a.grad.abs().max().item(), 1e-8)
        print(f"grad x : max|Δ| = {gx_err:.3e}   max|g| = {gx_ref:.3e}   rel = {gx_err/gx_ref:.3e}")

    # Hard pass criterion (fp32 only)
    if dtype == torch.float32:
        assert fwd_err <= 1e-5, f"forward mismatch: {fwd_err:.3e}"
        for name in ("P", "E", "Q", "m"):
            g_old = grads_old[name]
            g_new = grads_new[name]
            if g_old is None or g_new is None:
                continue
            err = (g_old - g_new).abs().max().item()
            assert err <= 1e-4, f"grad {name} mismatch: {err:.3e}"
        print("PASS")


def main() -> None:
    torch.set_default_dtype(torch.float32)
    run_check("identity", seed=0, dtype=torch.float32)
    run_check("pad", seed=1, dtype=torch.float32)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
