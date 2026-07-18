"""CPU checks for the k-offset smear math in train_gpt.py.

Mirrors the forward-pass implementation (train_gpt.py needs GPU + torch.distributed
to import, so the smear block is replicated here verbatim modulo variable names).
Run with any torch install: pytest experiments/multi-token-smear/smear_math_test.py
"""

import pytest
import torch
import torch.nn.functional as F

GATE_DIM = 4


def smear(x: torch.Tensor, weight: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    k = weight.size(0)
    gates = lambdas.bfloat16() * torch.sigmoid(x[:, : weight.size(-1)] @ weight.T)
    xp = F.pad(x, (0, 0, k, 0))
    shifts = xp.as_strided((k, x.size(0), x.size(1)), (x.size(1), x.size(1), 1))
    return x + torch.einsum("etc,te->tc", shifts, gates)


def master_smear_1pos(x: torch.Tensor, weight: torch.Tensor, smear_lambda: torch.Tensor) -> torch.Tensor:
    gate = smear_lambda * torch.sigmoid(x[1:, : weight.size(-1)] @ weight.T)
    return torch.cat([x[:1], x[1:] + gate * x[:-1]])


def rand_inputs(k: int, seed: int = 0, t: int = 33, d: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(t, d, generator=g).bfloat16()
    weight = (0.3 * torch.randn(k, GATE_DIM, generator=g)).bfloat16()
    lambdas = torch.randn(k, generator=g)
    return x, weight, lambdas


@pytest.mark.parametrize("k", [1, 2, 3, 5], ids=lambda k: f"k{k}")
def test_each_position_receives_gated_sum_of_k_predecessors(k: int) -> None:
    x, weight, lambdas = rand_inputs(k)
    out = smear(x, weight, lambdas)
    gates = lambdas.bfloat16() * torch.sigmoid(x[:, :GATE_DIM] @ weight.T)
    ref = x.float().clone()
    for t in range(x.size(0)):
        for e in range(k):
            d = k - e
            if t - d >= 0:
                ref[t] += gates[t, e].float() * x[t - d].float()
    assert torch.allclose(out.float(), ref, atol=0.05), f"{(out.float() - ref).abs().max()=}"


def test_k1_reproduces_master_smear_bitwise() -> None:
    x, weight, _ = rand_inputs(1)
    smear_lambda = torch.tensor(0.375)
    out = smear(x, weight, smear_lambda.expand(1))
    ref = master_smear_1pos(x, weight, smear_lambda)
    assert out.dtype == ref.dtype == torch.bfloat16
    assert torch.equal(out, ref), f"{(out.float() - ref.float()).abs().max()=}"


@pytest.mark.parametrize("k", [1, 2, 3, 5], ids=lambda k: f"k{k}")
def test_zero_init_is_exact_identity(k: int) -> None:
    x, _, _ = rand_inputs(k)
    out = smear(x, torch.zeros(k, GATE_DIM).bfloat16(), torch.zeros(k))
    assert torch.equal(out, x)


def test_gradient_reaches_fp32_lambdas_through_bf16_cast() -> None:
    x, weight, _ = rand_inputs(3)
    lambdas = torch.zeros(3, requires_grad=True)
    smear(x, weight, lambdas).sum().backward()
    assert lambdas.grad is not None
    assert (lambdas.grad != 0).all(), f"{lambdas.grad=}"


def test_compiles_fullgraph_without_graph_breaks() -> None:
    x, weight, lambdas = rand_inputs(3)
    compiled = torch.compile(smear, dynamic=False, fullgraph=True)
    out, ref = compiled(x, weight, lambdas), smear(x, weight, lambdas)
    assert torch.allclose(out.float(), ref.float(), atol=0.05), f"{(out.float() - ref.float()).abs().max()=}"
