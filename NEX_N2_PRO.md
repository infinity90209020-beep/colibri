# Nex-N2-Pro backend for Colibri

This branch adds a validated execution backend for `nex-agi/Nex-N2-Pro` and the official `nex-agi/Nex-N2-Pro-fp8` checkpoint.

It does **not** pass Qwen3.5 weights into Colibri's GLM-specific C kernel. Nex-N2-Pro uses a different hybrid architecture: 60 text layers, 45 linear-attention layers, 15 full-attention layers, 512 experts with 10 selected per token, and a native MTP layer. Correct support therefore uses Nex's customized SGLang runtime while Colibri owns validation, planning, launch policy, diagnostics, and performance acceptance.

## What is included

- Exact local `config.json` architecture validation.
- Reproducible BF16 16×H100 and FP8 8×H100/H200 launch profiles.
- Hybrid-attention `extra_buffer` scheduling.
- Qwen3 reasoning and tool-call parsers.
- Optional native NEXTN/MTP speculative decoding.
- Multithreaded checkpoint loading.
- Hardware diagnostics and explicit rejection reports.
- OpenAI-compatible benchmark harness with a strict 20 token/s throughput gate.
- A calibrated performance estimator derived from measured Colibri hardware data.
- No unsupported claim that 20 token/s is possible on arbitrary hardware.

## Install the runtime

Use the Nex-maintained SGLang fork:

```bash
git clone https://github.com/nex-agi/sglang.git
cd sglang
git checkout v0.5.12
python -m pip install -e python
```

## Plan before launching

```bash
python nex/coli_nex.py plan --profile fast-h200x8-fp8
python nex/coli_nex.py doctor --profile fast-h200x8-fp8 --model /models/Nex-N2-Pro-fp8
python nex/performance_estimator.py
```

The fast profile uses the official FP8 checkpoint because the BF16 checkpoint is about 794 GB. The exact BF16 profile follows Nex's documented 16×H100, two-node deployment.

## Launch FP8 on one 8-GPU HGX-class node

```bash
python nex/coli_nex.py serve \
  --profile fast-h200x8-fp8 \
  --model /models/Nex-N2-Pro-fp8
```

## Launch exact BF16 across two nodes

Run on both machines, changing `--node-rank`:

```bash
python nex/coli_nex.py serve \
  --profile reference-h100x16-bf16 \
  --model /models/Nex-N2-Pro \
  --node-rank 0 \
  --dist-init-addr 10.0.0.10:20000
```

## Enforce the 20 token/s target

```bash
python nex/coli_nex.py bench \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --requests 5 \
  --concurrency 1 \
  --max-tokens 256 \
  --target-tps 20
```

The command exits non-zero when measured **single-request decode throughput** is below the target. With `--concurrency 1`, aggregate throughput and single-request throughput are the same. The benchmark gate is authoritative; an estimate never overrides a measured result.

## Calibrated model geometry

The estimator derives the routed-expert size directly from Nex-N2-Pro's published dimensions:

```text
one routed expert
  = 3 × hidden_size × moe_intermediate_size
  = 3 × 4096 × 1024
  = 12,582,912 parameters

Colibri-style int4 expert
  = packed weights + fp32 row scales
  = 6,316,032 bytes
  = 6.316 MB

routed experts per token
  = 60 layers × top-10
  = 600 expert calls

fully cold routed-expert traffic
  = 600 × 6.316 MB
  = 3.790 GB/token
```

The complete routed-expert store is approximately **194.0 GB** at Colibri-style int4. The text path activates approximately **16.33B weights/token**, corresponding to about **32.7 GFLOP/token** when a multiply-add is counted as two operations.

This creates two different scaling regimes relative to GLM-5.2:

- **Disk-bound:** GLM's measured ~11.4 GB/token divided by Nex's 3.790 GB/token gives about **3.01×** potential scaling.
- **Resident routed-expert compute:** both models make 600 routed-expert calls/token, but Nex experts are about one third the size, again giving about **3.01×** for that component.
- **Dense, attention, output-head, and runtime work:** this does not shrink by the same factor. The estimator uses a more cautious **1.8×** component scale.

A single parameter-count ratio is therefore rejected as an estimation method.

## Measured Colibri anchors

The estimator preserves three main-repository measurements and scales each timing component separately:

| Colibri measurement | Measured GLM rate | How it constrains Nex |
|---|---:|---|
| 12 cores, 25 GB RAM, ~1 GB/s effective random NVMe | 0.05–0.10 tok/s cold | Strong disk-bound anchor; predicts roughly 0.14–0.30 tok/s for Nex. |
| Ryzen AI Max+ 395, 128 GB LPDDR5x, fast Gen4 NVMe, quality-preserving `DRAFT=0` | 1.01–1.10 tok/s sustained | Balanced disk/compute anchor; predicts roughly 2.8–3.8 tok/s because Nex's expert store is nearly half the size. |
| 6× RTX 5090, 251 GiB RAM, 24 physical cores, every expert resident | 6.28–6.84 tok/s | Zero-disk compute anchor; component scaling predicts roughly 14.5–16.0 tok/s before Nex-specific kernel improvements. |

The six-5090 experiment measured the GLM full-resident time split as 56% expert compute, 25% attention, 19% other. The Nex estimate scales those components independently instead of multiplying 6.84 tok/s by an arbitrary model-size ratio.

## Hardware-level estimates

These are **single-request, quality-preserving decode estimates** for a native Colibri-style Nex int4 backend. They are ranges because routing concentration, effective random-read latency, memory bandwidth, thermal throttling, and kernel implementation materially change the result.

| Hardware level | Expected Nex decode | Confidence |
|---|---:|---|
| RTX 2060 6 GB, 16 GB RAM, ~1 GB/s effective random NVMe | **0.14–0.30 tok/s** | medium-high |
| 32 GB RAM, 12–16 cores, 3–5 GB/s measured random reads | **0.8–1.8 tok/s** | medium |
| 64 GB RAM, 16–24 cores, 8–12 GB/s effective PCIe5/RAID reads | **2.2–4.8 tok/s** | medium-low |
| 128 GB fast unified memory, 16 strong cores, CPU-only | **2.8–3.8 tok/s** | medium-high |
| 128 GB RAM plus one RTX 5090 32 GB | **4.5–8.0 tok/s** | medium-low |
| 256 GB RAM, 24–32 AVX-512/VNNI cores, all experts resident | **4.5–8.5 tok/s** | medium |
| 192–256 GB RAM plus two RTX 5090s, all experts resident | **8–13 tok/s** | medium-low |
| 128–192 GB RAM plus four RTX 5090s, all experts resident | **11–17 tok/s** | medium-low |
| Six RTX 5090s plus enough RAM for zero disk misses | **14.5–16.0 tok/s initial native baseline** | medium-high |
| Six to eight high-end GPUs with grouped Tensor Core MoE kernels | **18–26 tok/s optimization target** | low until measured |

The **20 tok/s baseline is credible only in the final full-resident, optimized GPU tier**. On the measured six-5090 GLM host, Nex's component-scaled baseline is about 14.5–16.0 tok/s. Reaching 20 requires approximately a **25–38% kernel/runtime improvement**, or additional accelerator capacity. That is demanding but materially different from claiming it is impossible.

## How to estimate a specific machine

Measure storage using Colibri's random-read benchmark rather than the SSD's advertised sequential speed. Then calculate the theoretical disk ceiling:

```text
disk ceiling tok/s = effective random GB/s / [3.790 × (1 - expert hit rate)]
```

The estimator exposes the same calculation:

```bash
python nex/performance_estimator.py --disk-gbps 5.0 --hit-rate 0.75
```

The result is only a disk ceiling. Actual throughput must also fit under CPU/GPU compute, RAM bandwidth, attention, output-head, scheduling, and synchronization ceilings.

## Estimation rules

1. Always state whether a number is measured, anchor-scaled, or projected.
2. Always state prompt length, generated-token count, quantization, and whether the rate is decode-only or end-to-end.
3. Use `DRAFT=0` as the streaming baseline until batched routed-expert kernels make speculative verification cheaper than separate expert work.
4. Do not use lossy router top-p/top-k results as the quality-preserving baseline.
5. Use effective O_DIRECT/random-read measurements, not vendor sequential bandwidth.
6. Report cache hit rate and whether any expert bytes reached disk.
7. A real benchmark result replaces the estimate immediately.

## Hardware reality for the current laptop

The RTX 2060 6 GB and 16 GB system RAM can plausibly execute a native Nex int4 streaming backend, but it would be a minimum-tier demonstration at roughly **0.14–0.30 tok/s**, not a 20 tok/s system. This is now represented as a quantified estimate rather than a categorical claim that the model cannot load.
