# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("humming.schema")

from humming import dtypes, ops
from humming.config import GemmType as HummingGemmType
from humming.layer import HummingLayer, HummingMethod
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.utils.test import skip_if_unsupported
from humming.utils.weight import quantize_weight

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.humming import (
    HummingMoeQuantInfo,
    HummingRunnerCore,
    HummingRunnerInput,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.cuda,
    reason="Humming MoE runner tests require CUDA.",
)

PRINT_ACCURACY_STATS = os.getenv("HUMMING_MOE_TEST_PRINT_STATS") == "1"


@dataclass(frozen=True)
class HummingMoETestCase:
    name: str
    activation_dtype: str
    weight_dtype: str
    weight_scale_dtype: str
    weight_scale_group_size: int
    swiglu_limit: float | None
    rtol: float
    atol: float
    normalized_diff_tol: float


# Tolerance sources:
# - Humming's own low-bit GEMM/MoE tests use loose kernel-level bounds:
#   humming/tests/test_datatype.py and humming/tests/test_pad.py use
#   rtol=0.03, atol=0.1; humming/tests/test_moe.py uses rtol=0.05, atol=0.1;
#   and humming/tests/test_scale.py::test_fused_e8m0_weight_scale uses
#   rtol=0.05, atol=0.5.
# - This runner test is intentionally tighter because it compares two complete
#   MoE runner paths against the same dequantized reference. The thresholds below
#   were set from the deterministic stat sweep (seed=0, shape_m in {1, 128},
#   (tp_size, ep_size) in {(1, 1), (8, 1), (1, 8)}): worst observed max_abs was
#   <= 2.5e-4 and worst observed normalized_diff was <= 1.3e-4. Thus atol=1e-3
#   gives about 4x absolute-error headroom, while normalized_diff_tol keeps a
#   whole-output guardrail. rtol is secondary for these cases because atol
#   dominates near-zero outputs, but still catches scale drift on larger values.
HUMMING_MOE_TEST_CASES = [
    HummingMoETestCase(
        name="w4a8_mxfp4",
        activation_dtype="float8e4m3",
        weight_dtype="float4e2m1",
        weight_scale_dtype="float8e8m0",
        weight_scale_group_size=32,
        swiglu_limit=10.0,
        rtol=0.03,
        atol=1e-3,
        normalized_diff_tol=1e-3,
    ),
    HummingMoETestCase(
        name="w4a16_int4",
        activation_dtype="bfloat16",
        weight_dtype="int4",
        weight_scale_dtype="bfloat16",
        weight_scale_group_size=128,
        swiglu_limit=None,
        rtol=0.02,
        atol=1e-3,
        normalized_diff_tol=5e-4,
    ),
]


class BenchMoELayer(torch.nn.Module):
    pass


@dataclass(frozen=True)
class AccuracyStats:
    max_abs: float
    mean_abs: float
    max_rel: float
    rel_l2: float
    normalized_diff: float


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denominator = (x * x + y * y).sum()
    if denominator.item() == 0:
        return 0.0
    return (1 - 2 * (x * y).sum() / denominator).item()


def calc_accuracy_stats(x: torch.Tensor, y: torch.Tensor) -> AccuracyStats:
    x = x.float()
    y = y.float()
    abs_diff = (x - y).abs()
    abs_ref = y.abs()
    nonzero_ref = abs_ref > 0
    if bool(nonzero_ref.any()):
        max_rel = (abs_diff[nonzero_ref] / abs_ref[nonzero_ref]).max().item()
    else:
        max_rel = math.inf if bool((abs_diff > 0).any()) else 0.0
    diff_norm = torch.linalg.vector_norm((x - y).double())
    ref_norm = torch.linalg.vector_norm(y.double())
    rel_l2 = 0.0 if ref_norm.item() == 0 else (diff_norm / ref_norm).item()
    return AccuracyStats(
        max_abs=abs_diff.max().item(),
        mean_abs=abs_diff.mean().item(),
        max_rel=max_rel,
        rel_l2=rel_l2,
        normalized_diff=calc_diff(x, y),
    )


def calc_required_rtol(actual: torch.Tensor, expected: torch.Tensor, atol: float) -> float:
    actual = actual.float()
    expected = expected.float()
    residual = (actual - expected).abs() - atol
    residual = residual.clamp_min(0)
    abs_expected = expected.abs()
    if bool(((residual > 0) & (abs_expected == 0)).any()):
        return math.inf
    denominator = abs_expected.clamp_min(torch.finfo(torch.float32).tiny)
    return (residual / denominator).max().item()


def format_accuracy_stats(stats: AccuracyStats) -> str:
    return (
        f"max_abs={stats.max_abs:.6g}, mean_abs={stats.mean_abs:.6g}, "
        f"max_rel={stats.max_rel:.6g}, rel_l2={stats.rel_l2:.6g}, "
        f"normalized_diff={stats.normalized_diff:.6g}"
    )


def assert_close_with_stats(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    normalized_diff_tol: float,
    label: str,
) -> None:
    stats = calc_accuracy_stats(actual, expected)
    required_rtol = calc_required_rtol(actual, expected, atol)
    if PRINT_ACCURACY_STATS:
        print(
            f"[humming-moe-accuracy] {label}: "
            f"configured_rtol={rtol:.6g}, configured_atol={atol:.6g}, "
            f"normalized_diff_tol={normalized_diff_tol:.6g}; "
            f"{format_accuracy_stats(stats)}, "
            f"min_atol_if_rtol0={stats.max_abs:.6g}, "
            f"min_rtol_with_configured_atol={required_rtol:.6g}"
        )
    assert stats.normalized_diff < normalized_diff_tol, (
        f"{label} normalized_diff {stats.normalized_diff:.6g} exceeds "
        f"{normalized_diff_tol:.6g}; {format_accuracy_stats(stats)}"
    )
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        raise AssertionError(
            f"{label} failed assert_close with rtol={rtol}, atol={atol}; "
            f"{format_accuracy_stats(stats)}"
        ) from exc


def make_topk(
    shape_m: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    router_logits = torch.randn(
        shape_m,
        num_experts,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    topk_ids = router_logits.topk(top_k, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(
            shape_m,
            top_k,
            device=device,
            dtype=torch.float32,
            generator=generator,
        ),
        dim=-1,
    )
    return topk_ids.contiguous(), topk_weights.contiguous()


def mask_topk_for_ep(
    topk_ids: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    num_local_routed_experts: int,
) -> torch.Tensor:
    if ep_size == 1:
        return topk_ids.contiguous()
    expert_start = ep_rank * num_local_routed_experts
    expert_end = expert_start + num_local_routed_experts
    local_topk_ids = topk_ids - expert_start
    is_local = (topk_ids >= expert_start) & (topk_ids < expert_end)
    return torch.where(
        is_local,
        local_topk_ids,
        torch.full_like(local_topk_ids, -1),
    ).contiguous()


def quant_dequant_activation(
    inputs: torch.Tensor,
    input_schema: HummingInputSchema,
) -> torch.Tensor:
    a_dtype = input_schema.a_dtype
    assert a_dtype is not None
    if a_dtype.num_bits == 16:
        return inputs

    quanted, scale = ops.quant_input(
        inputs=inputs,
        dtype=str(a_dtype),
        group_size=None,
    )
    if a_dtype == dtypes.float8e4m3:
        dequanted = quanted.to(torch.float32)
    elif isinstance(a_dtype, dtypes.FloatingPointType):
        dequanted = ops.dequant_weight(
            quanted,
            exponent_bits=a_dtype.exponent_bits,
            mantissa_bits=a_dtype.mantissa_bits,
            is_signed=a_dtype.is_signed,
        )
    else:
        dequanted = quanted.float()

    assert scale is not None
    return (dequanted * scale.repeat_interleave(inputs.size(-1), dim=-1)).to(
        inputs.dtype
    )


def silu_and_mul_ref(
    gate_up: torch.Tensor,
    swiglu_limit: float | None,
) -> torch.Tensor:
    if swiglu_limit is not None:
        from sglang.srt.layers.moe.moe_runner.deep_gemm import _apply_swiglu_limit

        gate_up = _apply_swiglu_limit(gate_up, swiglu_limit=swiglu_limit)

    gate, up = gate_up.chunk(2, dim=-1)
    return (F.silu(gate.float()) * up.float()).to(gate_up.dtype)


def attach_humming_sublayer(
    layer: torch.nn.Module,
    sublayer_name: str,
    unquant_weight: torch.Tensor,
    weight_schema: HummingWeightSchema,
    input_schema: HummingInputSchema,
    params_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    shape_n = unquant_weight.size(-2)
    shape_k = unquant_weight.size(-1)
    num_experts = unquant_weight.size(0)

    scale_dtype = weight_schema.bs_dtype or dtypes.DataType.from_torch_dtype(
        params_dtype
    )
    quanted_weight, weight_scale, zero_point, global_scale = quantize_weight(
        weight=unquant_weight,
        dtype=weight_schema.b_dtype,
        scale_dtype=scale_dtype,
        group_size=weight_schema.weight_scale_group_size,
        group_size_n=(
            None
            if weight_schema.weight_scale_group_size_n <= 0
            else weight_schema.weight_scale_group_size_n
        ),
        has_zero_point=weight_schema.has_zero_point,
        has_global_scale="TENSOR" in str(weight_schema.weight_scale_type),
        is_fp_zero_point=weight_schema.is_fp_zero_point,
        pack=True,
    )
    quant_tensors = {"weight": quanted_weight}
    if weight_scale is not None:
        quant_tensors["weight_scale"] = weight_scale
    if zero_point is not None:
        quant_tensors["zero_point"] = zero_point
    if global_scale is not None:
        quant_tensors["global_scale"] = global_scale

    dequant_weight = weight_schema.dequant_tensors(quant_tensors).to(params_dtype)

    sublayer = HummingLayer(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_config=weight_schema,
        input_config=input_schema,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        num_experts=num_experts,
        has_bias=False,
        torch_dtype=params_dtype,
    ).to(device)
    sublayer.load_from_tensors(quant_tensors)
    sublayer.transform()

    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=shape_n,
        shape_k=shape_k,
        weight_schema=sublayer.weight_schema,
        input_schema=sublayer.input_schema,
        num_experts=num_experts,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=False,
        torch_dtype=params_dtype,
        sublayer_name=sublayer_name,
    )

    for name, param in list(sublayer.named_parameters()):
        tensor = param.detach()
        setattr(
            layer,
            f"{sublayer_name}_{name}",
            torch.nn.Parameter(tensor, requires_grad=False),
        )

    return dequant_weight


def make_humming_moe_layer(
    case: HummingMoETestCase,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    params_dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor, HummingInputSchema]:
    weight_schema = HummingWeightSchema(
        b_dtype=case.weight_dtype,
        bs_dtype=case.weight_scale_dtype,
        weight_scale_group_size=case.weight_scale_group_size,
    )
    input_schema = HummingInputSchema(a_dtype=case.activation_dtype)

    layer = BenchMoELayer().to(device)
    layer.hidden_size = hidden_size
    layer.intermediate_size = intermediate_size
    layer.intermediate_size_per_partition = intermediate_size
    layer.num_experts = num_experts
    layer.params_dtype = params_dtype
    layer.param_dtype = params_dtype
    layer.with_bias = False
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=device))

    w13 = (
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(hidden_size)
    )
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        / math.sqrt(intermediate_size)
    )

    w13_dequant = attach_humming_sublayer(
        layer=layer,
        sublayer_name="w13",
        unquant_weight=w13,
        weight_schema=weight_schema,
        input_schema=input_schema,
        params_dtype=params_dtype,
        device=device,
    )
    w2_dequant = attach_humming_sublayer(
        layer=layer,
        sublayer_name="w2",
        unquant_weight=w2,
        weight_schema=weight_schema,
        input_schema=input_schema,
        params_dtype=params_dtype,
        device=device,
    )

    return layer, w13_dequant, w2_dequant, input_schema


def run_humming_runner(
    runner: HummingRunnerCore,
    gemm_type: HummingGemmType,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    runner_input = HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        gemm_type=gemm_type,
    )
    output = runner.run(runner_input, HummingMoeQuantInfo(), {})
    return output.hidden_states


def run_dequant_reference(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_dequant: torch.Tensor,
    w2_dequant: torch.Tensor,
    input_schema: HummingInputSchema,
    swiglu_limit: float | None,
    routed_scaling_factor: float,
) -> torch.Tensor:
    shape_m, hidden_size = hidden_states.shape
    top_k = topk_ids.size(1)
    routed_rows = shape_m * top_k
    routed_inputs = (
        hidden_states[:, None, :]
        .expand(shape_m, top_k, hidden_size)
        .reshape(routed_rows, hidden_size)
        .contiguous()
    )
    routed_experts = topk_ids.view(-1)
    intermediate_size = w2_dequant.size(-1)

    # zeros (not empty) so EP -1 rows stay 0 and contribute 0 to weighted reduce.
    gate_up = torch.zeros(
        routed_rows,
        intermediate_size * 2,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for expert_id in range(w13_dequant.size(0)):
        rows = torch.where(routed_experts == expert_id)[0]
        if rows.numel() == 0:
            continue
        inputs = quant_dequant_activation(routed_inputs[rows], input_schema)
        gate_up[rows] = (inputs.float() @ w13_dequant[expert_id].float().T).to(
            hidden_states.dtype
        )

    activation = silu_and_mul_ref(gate_up, swiglu_limit=swiglu_limit)
    down = torch.zeros(
        routed_rows,
        hidden_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for expert_id in range(w2_dequant.size(0)):
        rows = torch.where(routed_experts == expert_id)[0]
        if rows.numel() == 0:
            continue
        inputs = quant_dequant_activation(activation[rows], input_schema)
        down[rows] = (inputs.float() @ w2_dequant[expert_id].float().T).to(
            hidden_states.dtype
        )

    output = (
        down.view(shape_m, top_k, hidden_size).float()
        * topk_weights.float().unsqueeze(-1)
    ).sum(dim=1)
    output = output * routed_scaling_factor
    return output.to(hidden_states.dtype)


@pytest.mark.parametrize(
    "case",
    HUMMING_MOE_TEST_CASES,
    ids=[case.name for case in HUMMING_MOE_TEST_CASES],
)
@pytest.mark.parametrize("shape_m", [1, 128])
@pytest.mark.parametrize(
    "tp_size, ep_size",
    [
        (1, 1),   # baseline, no shard
        (8, 1),   # TP-only: shard intermediate
        (1, 8),   # EP-only: shard experts, exercise ignore_invalid_expert + is_ep
    ],
    ids=["tp1ep1", "tp8ep1", "tp1ep8"],
)
@torch.inference_mode()
def test_humming_moe_runner_w4_accuracy(
    case: HummingMoETestCase,
    shape_m: int,
    tp_size: int,
    ep_size: int,
):
    if case.weight_scale_dtype == "float8e8m0" and not hasattr(
        torch, "float8_e8m0fnu"
    ):
        pytest.skip("torch.float8_e8m0fnu is required for MXFP4 scales.")

    skip_if_unsupported(a_dtype=case.activation_dtype)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(20260519 + shape_m + 31 * tp_size + 1009 * ep_size)

    hidden_size = 512
    intermediate_global = 1024            # min that survives TP=8 + pad_n_to_multiple=256
    num_experts_global = 64               # large enough that EP=8 gives 8 local experts
    top_k = 6
    ep_rank = 0
    params_dtype = torch.bfloat16
    routed_scaling_factor = 1.5

    if intermediate_global % tp_size != 0:
        pytest.skip(
            f"intermediate_global={intermediate_global} not divisible by tp_size={tp_size}"
        )
    if num_experts_global % ep_size != 0:
        pytest.skip(
            f"num_experts_global={num_experts_global} not divisible by ep_size={ep_size}"
        )

    intermediate_size = intermediate_global // tp_size
    num_local_routed_experts = num_experts_global // ep_size
    num_local_experts = num_local_routed_experts

    layer, w13_dequant, w2_dequant, input_schema = make_humming_moe_layer(
        case=case,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_local_experts,
        params_dtype=params_dtype,
        device=device,
        generator=generator,
    )
    runner = HummingRunnerCore(
        MoeRunnerConfig(
            num_experts=num_experts_global,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            top_k=top_k,
            params_dtype=params_dtype,
            activation="silu",
            swiglu_limit=case.swiglu_limit,
            routed_scaling_factor=routed_scaling_factor,
            layer=layer,
        )
    )

    hidden_states = (
        torch.randn(
            shape_m,
            hidden_size,
            device=device,
            dtype=params_dtype,
            generator=generator,
        )
        * 0.1
    )
    topk_ids, topk_weights = make_topk(
        shape_m=shape_m,
        top_k=top_k,
        num_experts=num_experts_global,
        device=device,
        generator=generator,
    )
    if ep_size > 1:
        topk_ids = mask_topk_for_ep(
            topk_ids,
            ep_rank=ep_rank,
            ep_size=ep_size,
            num_local_routed_experts=num_local_routed_experts,
        )
        if int((topk_ids >= 0).sum()) == 0:
            pytest.skip(
                f"No local routed rows for shape_m={shape_m} ep_size={ep_size}; "
                "degenerate case for this rank."
            )

    ref = run_dequant_reference(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        w13_dequant=w13_dequant,
        w2_dequant=w2_dequant,
        input_schema=input_schema,
        swiglu_limit=case.swiglu_limit,
        routed_scaling_factor=routed_scaling_factor,
    )

    outputs = {}
    for gemm_type in [
        HummingGemmType.INDEXED,
        HummingGemmType.GROUPED_CONTIGUOUS,
    ]:
        output = run_humming_runner(
            runner=runner,
            gemm_type=gemm_type,
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        torch.cuda.synchronize()
        assert_close_with_stats(
            output,
            ref,
            rtol=case.rtol,
            atol=case.atol,
            normalized_diff_tol=case.normalized_diff_tol,
            label=f"{case.name} {gemm_type.value} vs reference",
        )
        outputs[gemm_type] = output

    # INDEXED vs GROUPED_CONTIGUOUS should be near-equivalent (same math, different
    # scheduling/accumulation order); tighten from case.rtol/atol used vs ref.
    assert_close_with_stats(
        outputs[HummingGemmType.INDEXED],
        outputs[HummingGemmType.GROUPED_CONTIGUOUS],
        rtol=case.rtol * 0.5,
        atol=case.atol * 0.4,
        normalized_diff_tol=case.normalized_diff_tol * 0.5,
        label=f"{case.name} indexed vs grouped_contiguous",
    )
