"""Benchmark Humming MoE runner dataflow choices.

This benchmark measures the rank-local SGLang Humming MoE runner core. Unlike a
single GEMM microbenchmark, it includes the route-specific overhead:

* indexed: moe_align_block_size + indexed w13/w2 + moe_fused_mul_sum
* grouped: moe_permute + grouped-contiguous w13/w2 + moe_unpermute

It does not include outer dispatcher communication or the final all-reduce in
FusedMoE.forward_impl when reduce_results=True and TP/EP is active.

Use it to find the crossover point for SGLANG_HUMMING_MOE_GEMM_TYPE=indexed
vs grouped before turning that choice into a runtime heuristic.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Iterable

import torch
import triton

from humming.config import GemmType as HummingGemmType
from humming.layer import HummingLayer, HummingMethod
from humming.schema import HummingInputSchema, HummingWeightSchema
from humming.utils.test import random_fill_tensor
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.humming import (
    HummingMoeQuantInfo,
    HummingRunnerCore,
    HummingRunnerInput,
)


TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

GEMM_TYPES = {
    "indexed": HummingGemmType.INDEXED,
    "grouped": HummingGemmType.GROUPED_CONTIGUOUS,
}

CSV_FIELDS = [
    "global_shape_m",
    "local_shape_m",
    "router_top_k",
    "seed",
    "input_seed",
    "swiglu_limit",
    "local_routed_rows",
    "routed_rows_per_local_expert",
    "tp_size",
    "ep_size",
    "ep_rank",
    "ep_input_mode",
    "global_num_routed_experts",
    "num_local_routed_experts",
    "num_fused_shared_experts",
    "num_local_experts",
    "global_intermediate_size",
    "local_intermediate_size",
    "bench_backend",
    "return_mode",
    "gemm_type",
    "latency_us",
    "effective_tops",
    "peak_memory_delta_mb",
    "grouped_over_indexed",
    "faster",
    "max_abs_diff",
]


class BenchMoELayer(torch.nn.Module):
    pass


def parse_shape_m_list(value: str) -> list[int]:
    if "," in value:
        return [int(x) for x in value.split(",") if x]
    return [2**i for i in range(int(value) + 1)]


def make_topk(
    shape_m: int,
    top_k: int,
    num_experts: int,
    device: torch.device,
    balanced: bool,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if balanced:
        token_offsets = torch.arange(shape_m, device=device, dtype=torch.int64)[:, None]
        rank_offsets = torch.arange(top_k, device=device, dtype=torch.int64)[None, :]
        topk_ids = (token_offsets * top_k + rank_offsets) % num_experts
    else:
        # rand(...).topk gives unique experts per token and roughly uniform load.
        topk_ids = torch.rand(
            shape_m,
            num_experts,
            device=device,
            generator=generator,
        ).topk(top_k, dim=-1).indices

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
    return topk_ids.to(torch.int32), topk_weights


def get_num_local_experts(args: argparse.Namespace) -> int:
    if args.ep_size < 1:
        raise ValueError("--ep-size must be >= 1.")
    if args.ep_rank < 0 or args.ep_rank >= args.ep_size:
        raise ValueError("--ep-rank must be in [0, ep_size).")
    if args.num_fused_shared_experts < 0:
        raise ValueError("--num-fused-shared-experts must be >= 0.")
    if args.num_experts % args.ep_size != 0:
        raise ValueError("--num-experts must be divisible by --ep-size.")
    return get_num_local_routed_experts(args) + args.num_fused_shared_experts


def get_num_local_routed_experts(args: argparse.Namespace) -> int:
    if args.num_experts % args.ep_size != 0:
        raise ValueError("--num-experts must be divisible by --ep-size.")
    return args.num_experts // args.ep_size


def get_global_num_expert_slots(args: argparse.Namespace) -> int:
    return args.num_experts + args.num_fused_shared_experts * args.ep_size


def get_local_intermediate_size(args: argparse.Namespace) -> int:
    if args.tp_size < 1:
        raise ValueError("--tp-size must be >= 1.")
    if args.intermediate_size % args.tp_size != 0:
        raise ValueError("--intermediate-size must be divisible by --tp-size.")
    return args.intermediate_size // args.tp_size


def map_topk_to_local_experts(
    topk_ids: torch.Tensor,
    num_local_routed_experts: int,
    ep_rank: int,
    ep_size: int,
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


def make_rank_local_inputs(
    shape_m: int,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    hidden_states = torch.randn(
        shape_m,
        args.hidden_size,
        dtype=TORCH_DTYPES[args.params_dtype],
        device=device,
        generator=generator,
    )
    topk_ids, topk_weights = make_topk(
        shape_m=shape_m,
        top_k=args.top_k,
        num_experts=args.num_experts,
        device=device,
        balanced=args.balanced,
        generator=generator,
    )
    num_local_routed_experts = get_num_local_routed_experts(args)
    topk_ids = map_topk_to_local_experts(
        topk_ids=topk_ids,
        num_local_routed_experts=num_local_routed_experts,
        ep_rank=args.ep_rank,
        ep_size=args.ep_size,
    )

    if args.ep_size > 1 and args.ep_input_mode == "deepep-normal":
        # DeepEP normal dispatch sends this rank only the token rows that have
        # at least one local expert. Non-local top-k slots stay as -1.
        local_token_mask = (topk_ids >= 0).any(dim=1)
        hidden_states = hidden_states[local_token_mask].contiguous()
        topk_ids = topk_ids[local_token_mask].contiguous()
        topk_weights = topk_weights[local_token_mask].contiguous()

    local_routed_rows = int((topk_ids >= 0).sum().item())
    return hidden_states, topk_ids, topk_weights.contiguous(), local_routed_rows


def attach_humming_sublayer(
    layer: torch.nn.Module,
    sublayer_name: str,
    shape_n: int,
    shape_k: int,
    num_experts: int,
    params_dtype: torch.dtype,
    weight_schema: HummingWeightSchema,
    input_schema: HummingInputSchema,
    device: torch.device,
) -> None:
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

    with torch.no_grad():
        for param in sublayer.parameters():
            random_fill_tensor(param)
    sublayer.transform()

    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=shape_n,
        shape_k=shape_k,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        input_schema=input_schema,
        weight_schema=weight_schema,
        has_bias=False,
        num_experts=num_experts,
        torch_dtype=params_dtype,
        sublayer_name=sublayer_name,
    )

    for name, param in list(sublayer.named_parameters()):
        tensor = param.detach()
        setattr(layer, f"{sublayer_name}_{name}", torch.nn.Parameter(tensor, requires_grad=False))

    del sublayer
    gc.collect()
    torch.cuda.empty_cache()


def make_layer(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    params_dtype = TORCH_DTYPES[args.params_dtype]
    num_local_experts = get_num_local_experts(args)
    local_intermediate_size = get_local_intermediate_size(args)
    weight_schema = HummingWeightSchema(
        b_dtype=args.weight_dtype,
        bs_dtype=args.weight_scale_dtype,
        weight_scale_group_size=args.weight_scale_group_size,
        has_zero_point=args.has_zero_point,
        is_fp_zero_point=args.is_fp_zero_point,
    )
    input_schema = HummingInputSchema(a_dtype=args.activation_dtype)

    layer = BenchMoELayer().to(device)
    layer.hidden_size = args.hidden_size
    layer.intermediate_size = local_intermediate_size
    layer.intermediate_size_per_partition = local_intermediate_size
    layer.num_experts = num_local_experts
    layer.params_dtype = params_dtype
    layer.param_dtype = params_dtype
    layer.with_bias = False
    layer.register_buffer("locks", torch.zeros(1024, dtype=torch.int32, device=device))

    attach_humming_sublayer(
        layer=layer,
        sublayer_name="w13",
        shape_n=local_intermediate_size * 2,
        shape_k=args.hidden_size,
        num_experts=num_local_experts,
        params_dtype=params_dtype,
        weight_schema=weight_schema,
        input_schema=input_schema,
        device=device,
    )
    attach_humming_sublayer(
        layer=layer,
        sublayer_name="w2",
        shape_n=args.hidden_size,
        shape_k=local_intermediate_size,
        num_experts=num_local_experts,
        params_dtype=params_dtype,
        weight_schema=weight_schema,
        input_schema=input_schema,
        device=device,
    )
    return layer


def effective_moe_ops(
    local_routed_rows: int,
    hidden_size: int,
    local_intermediate_size: int,
) -> int:
    gate_up_ops = 2 * local_routed_rows * hidden_size * (2 * local_intermediate_size)
    down_ops = 2 * local_routed_rows * local_intermediate_size * hidden_size
    return gate_up_ops + down_ops


def bench_one(
    runner: HummingRunnerCore,
    gemm_type: HummingGemmType,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    warmup: int,
    rep: int,
    bench_backend: str,
    return_mode: str,
) -> tuple[float, torch.Tensor, int]:
    runner_input = HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        gemm_type=gemm_type,
    )
    quant_info = HummingMoeQuantInfo()

    def run() -> torch.Tensor:
        return runner.run(runner_input, quant_info, {}).hidden_states

    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = run()
    torch.cuda.synchronize()
    if bench_backend == "eager":
        ms = triton.testing.do_bench(
            run,
            warmup=warmup,
            rep=rep,
            return_mode=return_mode,
        )
    elif bench_backend == "cudagraph":
        ms = triton.testing.do_bench_cudagraph(
            run,
            rep=rep,
            return_mode=return_mode,
        )
    else:
        raise ValueError(f"Unsupported benchmark backend: {bench_backend}")
    torch.cuda.synchronize()
    peak_memory_delta = max(0, torch.cuda.max_memory_allocated() - baseline_memory)
    return ms, output, peak_memory_delta


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_csv(rows: Iterable[dict[str, object]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-m-list", default="14", help="Comma list, or max power for 2**i.")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--intermediate-size", type=int, default=2048, help="Global expert intermediate size before TP sharding.")
    parser.add_argument("--num-experts", type=int, default=256, help="Global number of routed experts before EP sharding.")
    parser.add_argument("--num-fused-shared-experts", type=int, default=0)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--ep-rank", type=int, default=0)
    parser.add_argument(
        "--ep-input-mode",
        choices=["standard", "deepep-normal"],
        default="standard",
        help=(
            "standard keeps all token rows and masks non-local experts as -1; "
            "deepep-normal keeps only token rows received by the local EP rank."
        ),
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--swiglu-limit",
        type=float,
        default=None,
        help="Pre-silu clamp on gate/up halves. DSV4 production uses 10.0; default None matches plain silu_and_mul.",
    )
    parser.add_argument("--params-dtype", choices=TORCH_DTYPES.keys(), default="bfloat16")
    parser.add_argument("--activation-dtype", default="float8e4m3")
    parser.add_argument("--weight-dtype", default="int4")
    parser.add_argument("--weight-scale-dtype", default="bfloat16")
    parser.add_argument("--weight-scale-group-size", type=int, default=128)
    parser.add_argument("--has-zero-point", action="store_true")
    parser.add_argument("--is-fp-zero-point", action="store_true")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--gemm-types", nargs="+", choices=GEMM_TYPES.keys(), default=["indexed", "grouped"])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument(
        "--bench-backend",
        choices=["eager", "cudagraph"],
        default="eager",
        help=(
            "eager measures normal runner calls including launch overhead; "
            "cudagraph uses CUDA graph replay to approximate decode graph mode."
        ),
    )
    parser.add_argument(
        "--return-mode",
        choices=["mean", "median", "min", "max"],
        default="median",
        help="Statistic returned by triton.testing benchmark helpers.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-seed",
        type=int,
        default=None,
        help="Seed for synthetic hidden/top-k inputs. Defaults to --seed.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.swiglu_limit is not None:
        if args.swiglu_limit != 10.0:
            raise ValueError("--swiglu-limit currently supports only 10.0.")
        if args.params_dtype != "bfloat16":
            raise ValueError("--swiglu-limit requires --params-dtype bfloat16.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Humming MoE runner benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    input_seed = args.seed if args.input_seed is None else args.input_seed
    num_local_experts = get_num_local_experts(args)
    num_local_routed_experts = get_num_local_routed_experts(args)
    global_num_expert_slots = get_global_num_expert_slots(args)
    local_intermediate_size = get_local_intermediate_size(args)
    layer = make_layer(args, device)
    runner = HummingRunnerCore(
        MoeRunnerConfig(
            num_experts=global_num_expert_slots,
            num_local_experts=num_local_experts,
            hidden_size=args.hidden_size,
            intermediate_size_per_partition=local_intermediate_size,
            top_k=args.top_k,
            params_dtype=TORCH_DTYPES[args.params_dtype],
            activation="silu",
            swiglu_limit=args.swiglu_limit,
            routed_scaling_factor=1.0,
            layer=layer,
        )
    )

    rows: list[dict[str, object]] = []
    shape_m_list = parse_shape_m_list(args.shape_m_list)
    swiglu_limit = "" if args.swiglu_limit is None else f"{args.swiglu_limit:g}"
    for shape_m in shape_m_list:
        # Keep synthetic inputs independent from weight RNG and shape-list order.
        input_generator = torch.Generator(device=device)
        input_generator.manual_seed(input_seed + shape_m)
        hidden_states, topk_ids, topk_weights, local_routed_rows = make_rank_local_inputs(
            shape_m,
            args,
            device,
            input_generator,
        )
        ops = effective_moe_ops(
            local_routed_rows=local_routed_rows,
            hidden_size=args.hidden_size,
            local_intermediate_size=local_intermediate_size,
        )
        local_shape_m = hidden_states.size(0)
        routed_rows_per_local_expert = local_routed_rows / num_local_experts

        outputs: dict[str, torch.Tensor] = {}
        timings: dict[str, float] = {}
        for gemm_name in args.gemm_types:
            ms, output, peak_memory_delta = bench_one(
                runner=runner,
                gemm_type=GEMM_TYPES[gemm_name],
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                warmup=args.warmup,
                rep=args.rep,
                bench_backend=args.bench_backend,
                return_mode=args.return_mode,
            )
            outputs[gemm_name] = output
            timings[gemm_name] = ms

            rows.append(
                {
                    "global_shape_m": shape_m,
                    "local_shape_m": local_shape_m,
                    "router_top_k": args.top_k,
                    "seed": args.seed,
                    "input_seed": input_seed,
                    "swiglu_limit": swiglu_limit,
                    "local_routed_rows": local_routed_rows,
                    "routed_rows_per_local_expert": f"{routed_rows_per_local_expert:.3f}",
                    "tp_size": args.tp_size,
                    "ep_size": args.ep_size,
                    "ep_rank": args.ep_rank,
                    "ep_input_mode": args.ep_input_mode,
                    "global_num_routed_experts": args.num_experts,
                    "num_local_routed_experts": num_local_routed_experts,
                    "num_fused_shared_experts": args.num_fused_shared_experts,
                    "num_local_experts": num_local_experts,
                    "global_intermediate_size": args.intermediate_size,
                    "local_intermediate_size": local_intermediate_size,
                    "bench_backend": args.bench_backend,
                    "return_mode": args.return_mode,
                    "gemm_type": gemm_name,
                    "latency_us": f"{ms * 1000:.4f}",
                    "effective_tops": f"{ops / (ms * 1e-3) / 1e12:.3f}",
                    "peak_memory_delta_mb": f"{peak_memory_delta / 1024 / 1024:.3f}",
                    "grouped_over_indexed": "",
                    "faster": "",
                    "max_abs_diff": "",
                }
            )

        if "indexed" in timings and "grouped" in timings:
            indexed_ms = timings["indexed"]
            grouped_ms = timings["grouped"]
            faster = "indexed" if indexed_ms <= grouped_ms else "grouped"
            if outputs["indexed"].numel() == 0:
                max_abs_diff = 0.0
            else:
                max_abs_diff = (
                    outputs["indexed"] - outputs["grouped"]
                ).abs().max().item()
            rows.append(
                {
                    "global_shape_m": shape_m,
                    "local_shape_m": local_shape_m,
                    "router_top_k": args.top_k,
                    "seed": args.seed,
                    "input_seed": input_seed,
                    "swiglu_limit": swiglu_limit,
                    "local_routed_rows": local_routed_rows,
                    "routed_rows_per_local_expert": f"{routed_rows_per_local_expert:.3f}",
                    "tp_size": args.tp_size,
                    "ep_size": args.ep_size,
                    "ep_rank": args.ep_rank,
                    "ep_input_mode": args.ep_input_mode,
                    "global_num_routed_experts": args.num_experts,
                    "num_local_routed_experts": num_local_routed_experts,
                    "num_fused_shared_experts": args.num_fused_shared_experts,
                    "num_local_experts": num_local_experts,
                    "global_intermediate_size": args.intermediate_size,
                    "local_intermediate_size": local_intermediate_size,
                    "bench_backend": args.bench_backend,
                    "return_mode": args.return_mode,
                    "gemm_type": "compare",
                    "latency_us": "",
                    "effective_tops": "",
                    "peak_memory_delta_mb": "",
                    "grouped_over_indexed": f"{grouped_ms / indexed_ms:.3f}",
                    "faster": faster,
                    "max_abs_diff": f"{max_abs_diff:.6g}",
                }
            )

        del hidden_states, topk_ids, topk_weights, outputs
        torch.cuda.empty_cache()

    print_csv(rows)
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)


if __name__ == "__main__":
    main()
