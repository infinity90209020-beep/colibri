#!/usr/bin/env python3
"""Colibri Nex backend: validated SGLang launcher and throughput gate for Nex-N2-Pro.

This module intentionally does not pretend that Colibri's GLM C kernel can execute
Qwen3.5 MoE weights. It adds a first-class, reproducible backend around Nex's
supported SGLang runtime while keeping model validation, hardware planning, launch
configuration, and benchmark enforcement inside the Colibri repository.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

MODEL_BF16 = "nex-agi/Nex-N2-Pro"
MODEL_FP8 = "nex-agi/Nex-N2-Pro-fp8"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"

EXPECTED_TEXT_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 4096,
    "num_hidden_layers": 60,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 1024,
    "shared_expert_intermediate_size": 1024,
    "mtp_num_hidden_layers": 1,
    "vocab_size": 248320,
}


@dataclass(frozen=True)
class Profile:
    name: str
    model: str
    description: str
    min_gpus: int
    min_vram_gib_per_gpu: int
    tp: int
    nnodes: int
    context_length: int
    mem_fraction_static: float
    checkpoint_gib: int
    target_tps: float
    enable_nextn: bool
    extra_args: tuple[str, ...] = ()


PROFILES: dict[str, Profile] = {
    "reference-h100x16-bf16": Profile(
        name="reference-h100x16-bf16",
        model=MODEL_BF16,
        description="Exact Nex-N2-Pro BF16 reference deployment across two 8xH100 nodes.",
        min_gpus=16,
        min_vram_gib_per_gpu=80,
        tp=16,
        nnodes=2,
        context_length=32768,
        mem_fraction_static=0.88,
        checkpoint_gib=794,
        target_tps=20.0,
        enable_nextn=False,
        extra_args=("--mamba-scheduler-strategy", "extra_buffer"),
    ),
    "fast-h200x8-fp8": Profile(
        name="fast-h200x8-fp8",
        model=MODEL_FP8,
        description="Single HGX-class 8xH200/H100 FP8 profile with MTP/NEXTN enabled.",
        min_gpus=8,
        min_vram_gib_per_gpu=80,
        tp=8,
        nnodes=1,
        context_length=65536,
        mem_fraction_static=0.86,
        checkpoint_gib=400,
        target_tps=20.0,
        enable_nextn=True,
        extra_args=(
            "--mamba-scheduler-strategy", "extra_buffer",
            "--enable-flashinfer-allreduce-fusion",
            "--enable-tokenizer-batch-encode",
            "--enable-mixed-chunk",
            "--kv-cache-dtype", "bfloat16",
        ),
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_model_config(model: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate a local Nex-N2-Pro compatible config and return a compact report."""
    root = Path(model)
    config = _read_json(root / "config.json")
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError(
            f"unsupported model_type={config.get('model_type')!r}; expected 'qwen3_5_moe'"
        )
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise ValueError("config.json is missing the composite text_config object")
    mismatches: list[str] = []
    for key, expected in EXPECTED_TEXT_CONFIG.items():
        actual = text.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list) or len(layer_types) != 60:
        mismatches.append("layer_types: expected 60 hybrid attention entries")
    else:
        full = sum(1 for item in layer_types if item == "full_attention")
        linear = sum(1 for item in layer_types if item == "linear_attention")
        if (full, linear) != (15, 45):
            mismatches.append(
                f"layer_types: expected 15 full + 45 linear attention layers, got {full} + {linear}"
            )
    if mismatches:
        raise ValueError("model architecture mismatch:\n  - " + "\n  - ".join(mismatches))
    return {
        "model_type": config["model_type"],
        "architecture": (config.get("architectures") or [None])[0],
        "hidden_size": text["hidden_size"],
        "layers": text["num_hidden_layers"],
        "experts": text["num_experts"],
        "active_experts": text["num_experts_per_tok"],
        "mtp_layers": text["mtp_num_hidden_layers"],
        "max_context": text.get("max_position_embeddings"),
        "full_attention_layers": 15,
        "linear_attention_layers": 45,
    }


def detect_gpus() -> list[dict[str, Any]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    command = [
        nvidia_smi,
        "--query-gpu=index,name,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            memory_gib = round(float(parts[2]) / 1024, 2)
        except ValueError:
            memory_gib = 0.0
        gpus.append({"index": int(parts[0]), "name": parts[1], "vram_gib": memory_gib,
                     "compute_capability": parts[3]})
    return gpus


def check_hardware(profile: Profile, gpus: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    if len(gpus) < profile.min_gpus:
        issues.append(f"needs at least {profile.min_gpus} GPUs; detected {len(gpus)}")
    undersized = [g for g in gpus[: profile.min_gpus]
                 if g.get("vram_gib", 0) < profile.min_vram_gib_per_gpu]
    if undersized:
        issues.append(
            f"needs >= {profile.min_vram_gib_per_gpu} GiB per GPU; undersized devices: "
            + ", ".join(str(g["index"]) for g in undersized)
        )
    return issues


def build_launch_command(
    profile: Profile,
    *,
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    node_rank: int = 0,
    dist_init_addr: str | None = None,
    context_length: int | None = None,
    mem_fraction_static: float | None = None,
    enable_nextn: bool | None = None,
    loader_threads: int = 8,
    extra_args: Iterable[str] = (),
) -> list[str]:
    if not 0 <= node_rank < profile.nnodes:
        raise ValueError(f"node_rank must be in [0, {profile.nnodes - 1}]")
    if profile.nnodes > 1 and not dist_init_addr:
        raise ValueError("multi-node profiles require --dist-init-addr HOST:PORT")
    use_nextn = profile.enable_nextn if enable_nextn is None else enable_nextn
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model or profile.model,
        "--served-model-name",
        MODEL_BF16,
        "--host",
        host,
        "--port",
        str(port),
        "--tp",
        str(profile.tp),
        "--nnodes",
        str(profile.nnodes),
        "--node-rank",
        str(node_rank),
        "--context-length",
        str(context_length or profile.context_length),
        "--mem-fraction-static",
        str(mem_fraction_static or profile.mem_fraction_static),
        "--chunked-prefill-size",
        "8192",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--model-loader-extra-config",
        json.dumps({"enable_multithread_load": True, "num_threads": loader_threads}, separators=(",", ":")),
    ]
    if dist_init_addr:
        command.extend(("--dist-init-addr", dist_init_addr))
    command.extend(profile.extra_args)
    if use_nextn:
        command.extend((
            "--speculative-algorithm", "NEXTN",
            "--speculative-num-steps", "3",
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", "4",
        ))
    command.extend(extra_args)
    return command


def shell_join(command: Iterable[str]) -> str:
    import shlex
    return shlex.join(list(command))


def _post_json(url: str, payload: dict[str, Any], timeout: float, api_key: str | None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc}") from exc
    value = json.loads(body)
    if not isinstance(value, dict):
        raise RuntimeError("server returned a non-object response")
    return value


def _one_benchmark_request(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    api_key: str | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = _post_json(endpoint, payload, timeout, api_key)
    elapsed = time.perf_counter() - started
    usage = response.get("usage") or {}
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or tokens < 0:
        raise RuntimeError("response did not include integer usage.completion_tokens")
    return {"seconds": elapsed, "completion_tokens": tokens,
            "request_tps": tokens / elapsed if elapsed > 0 else float("inf")}


def benchmark_endpoint(
    endpoint: str,
    *,
    model: str = MODEL_BF16,
    prompt: str,
    requests: int = 3,
    concurrency: int = 1,
    max_tokens: int = 256,
    timeout: float = 900,
    warmup: int = 1,
    api_key: str | None = None,
) -> dict[str, Any]:
    if requests < 1 or concurrency < 1 or max_tokens < 1 or warmup < 0:
        raise ValueError("requests, concurrency, and max_tokens must be positive; warmup cannot be negative")
    for _ in range(warmup):
        _one_benchmark_request(endpoint, model, prompt, min(max_tokens, 32), timeout, api_key)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_one_benchmark_request, endpoint, model, prompt,
                               max_tokens, timeout, api_key) for _ in range(requests)]
        for future in as_completed(futures):
            results.append(future.result())
    wall = time.perf_counter() - started
    total_tokens = sum(item["completion_tokens"] for item in results)
    return {
        "requests": requests,
        "concurrency": concurrency,
        "completion_tokens": total_tokens,
        "wall_seconds": wall,
        "aggregate_tps": total_tokens / wall if wall > 0 else float("inf"),
        "median_request_tps": statistics.median(item["request_tps"] for item in results),
        "min_request_tps": min(item["request_tps"] for item in results),
        "samples": sorted(results, key=lambda item: item["seconds"]),
    }


def _profile(name: str) -> Profile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILES)}") from exc


def command_plan(args: argparse.Namespace) -> int:
    profile = _profile(args.profile)
    command = build_launch_command(
        profile,
        model=args.model,
        host=args.host,
        port=args.port,
        node_rank=args.node_rank,
        dist_init_addr=args.dist_init_addr,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        enable_nextn=args.nextn,
        loader_threads=args.loader_threads,
        extra_args=args.sglang_arg,
    )
    gpus = detect_gpus()
    report = {
        "profile": asdict(profile),
        "detected_gpus": gpus,
        "hardware_issues": check_hardware(profile, gpus),
        "launch_command": command,
        "launch_command_shell": shell_join(command),
        "throughput_contract": {
            "target_tps": args.target_tps or profile.target_tps,
            "meaning": "Measured aggregate decode throughput gate; not a hardware-independent guarantee.",
        },
    }
    print(json.dumps(report, indent=2))
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    profile = _profile(args.profile)
    gpus = detect_gpus()
    issues = check_hardware(profile, gpus)
    checks: dict[str, Any] = {
        "python": sys.version.split()[0],
        "sglang_importable": False,
        "nvidia_smi": bool(shutil.which("nvidia-smi")),
        "gpus": gpus,
        "profile": profile.name,
        "issues": issues,
    }
    try:
        __import__("sglang")
        checks["sglang_importable"] = True
    except ImportError:
        issues.append("Python package 'sglang' is not importable; install Nex's sglang fork")
    if args.model and Path(args.model).is_dir():
        try:
            checks["model_config"] = validate_model_config(args.model)
        except ValueError as exc:
            issues.append(str(exc))
    print(json.dumps(checks, indent=2))
    return 2 if issues else 0


def command_serve(args: argparse.Namespace) -> int:
    profile = _profile(args.profile)
    command = build_launch_command(
        profile,
        model=args.model,
        host=args.host,
        port=args.port,
        node_rank=args.node_rank,
        dist_init_addr=args.dist_init_addr,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        enable_nextn=args.nextn,
        loader_threads=args.loader_threads,
        extra_args=args.sglang_arg,
    )
    print(shell_join(command), file=sys.stderr)
    if args.dry_run:
        return 0
    os.execv(command[0], command)
    return 127


def command_bench(args: argparse.Namespace) -> int:
    report = benchmark_endpoint(
        args.endpoint,
        model=args.model,
        prompt=args.prompt,
        requests=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        warmup=args.warmup,
        api_key=args.api_key,
    )
    report["target_tps"] = args.target_tps
    report["passed"] = report["aggregate_tps"] >= args.target_tps
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 3


def _add_launch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=sorted(PROFILES), default="fast-h200x8-fp8")
    parser.add_argument("--model", help="local model directory or Hugging Face model ID")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--dist-init-addr")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--loader-threads", type=int, default=8)
    parser.add_argument("--target-tps", type=float)
    parser.add_argument("--nextn", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sglang-arg", action="append", default=[],
                        help="append one raw SGLang argument; repeat for argument and value")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print a validated hardware and launch plan")
    _add_launch_args(plan)
    plan.set_defaults(func=command_plan)

    doctor = sub.add_parser("doctor", help="validate hardware, runtime, and local model config")
    doctor.add_argument("--profile", choices=sorted(PROFILES), default="fast-h200x8-fp8")
    doctor.add_argument("--model")
    doctor.set_defaults(func=command_doctor)

    serve = sub.add_parser("serve", help="launch Nex-N2-Pro through the optimized SGLang backend")
    _add_launch_args(serve)
    serve.add_argument("--dry-run", action="store_true")
    serve.set_defaults(func=command_serve)

    bench = sub.add_parser("bench", help="measure and enforce an OpenAI endpoint throughput floor")
    bench.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    bench.add_argument("--model", default=MODEL_BF16)
    bench.add_argument("--prompt", default="Explain why deterministic benchmarking matters in two paragraphs.")
    bench.add_argument("--requests", type=int, default=3)
    bench.add_argument("--concurrency", type=int, default=1)
    bench.add_argument("--max-tokens", type=int, default=256)
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--timeout", type=float, default=900)
    bench.add_argument("--target-tps", type=float, default=20.0)
    bench.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    bench.set_defaults(func=command_bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
