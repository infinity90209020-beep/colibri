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

The command exits non-zero when measured aggregate decode throughput is below the target. This turns “20 tok/s” into a reproducible acceptance test rather than an unsupported promise.

## Hardware reality

The user's RTX 2060 6 GB and 16 GB system RAM cannot hold this 397B-A17B model or approach the target. The backend reports that mismatch instead of silently offloading hundreds of gigabytes to disk and presenting an unusable configuration as success.
