#!/usr/bin/env python3
"""Calibrated Nex-N2-Pro throughput estimates derived from Colibri measurements.

The purpose of this module is to prevent parameter-count-only estimates. It keeps
three separate quantities:

1. cold routed-expert bytes/token (disk-bound scaling),
2. routed-expert compute/token (resident expert scaling), and
3. dense/attention/other work (which does not scale exactly like routed experts).

All quoted ranges are single-request decode rates. They are engineering estimates,
not benchmark results for Nex-N2-Pro, until a real run passes the benchmark gate.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

# Nex-N2-Pro text architecture from the published config and Qwen3.5 MoE source.
HIDDEN_SIZE = 4096
MOE_INTERMEDIATE_SIZE = 1024
NUM_LAYERS = 60
NUM_EXPERTS = 512
TOP_K = 10

# One expert has gate, up, and down projections. Colibri-style int4 stores packed
# nibbles plus one fp32 scale per output row (2*I rows for gate/up + H rows down).
EXPERT_PARAMS = 3 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE
EXPERT_PACKED_BYTES = EXPERT_PARAMS // 2
EXPERT_SCALE_BYTES = (2 * MOE_INTERMEDIATE_SIZE + HIDDEN_SIZE) * 4
EXPERT_BYTES_INT4 = EXPERT_PACKED_BYTES + EXPERT_SCALE_BYTES
EXPERT_CALLS_PER_TOKEN = NUM_LAYERS * TOP_K
COLD_EXPERT_GB_PER_TOKEN = EXPERT_BYTES_INT4 * EXPERT_CALLS_PER_TOKEN / 1e9
EXPERT_STORE_GB = EXPERT_BYTES_INT4 * NUM_EXPERTS * NUM_LAYERS / 1e9

# Exact text-path estimate from published dimensions. Embedding lookup is excluded
# from active matmul weights; the full lm_head is included.
NEX_ACTIVE_WEIGHTS_B = 16.331668736
NEX_ACTIVE_GFLOP_PER_TOKEN = 2 * NEX_ACTIVE_WEIGHTS_B

# Colibri's published GLM measurements/description.
GLM_COLD_EXPERT_GB_PER_TOKEN = 11.4
GLM_ACTIVE_WEIGHTS_B = 40.0

DISK_SCALE = GLM_COLD_EXPERT_GB_PER_TOKEN / COLD_EXPERT_GB_PER_TOKEN
ROUTED_COMPUTE_SCALE = DISK_SCALE  # both models perform 600 routed expert calls/token
DENSE_COMPUTE_SCALE = 1.8  # lower-confidence: hybrid attention and lm-head do not scale identically


@dataclass(frozen=True)
class Anchor:
    name: str
    hardware: str
    glm_low_tps: float
    glm_high_tps: float
    disk_fraction: float
    routed_compute_fraction: float
    dense_other_fraction: float
    cache_advantage_low: float = 1.0
    cache_advantage_high: float = 1.0
    confidence: str = "medium"
    basis: str = ""


@dataclass(frozen=True)
class Estimate:
    name: str
    hardware: str
    low_tps: float
    high_tps: float
    confidence: str
    basis: str


def scale_anchor(anchor: Anchor) -> Estimate:
    fractions = anchor.disk_fraction + anchor.routed_compute_fraction + anchor.dense_other_fraction
    if abs(fractions - 1.0) > 1e-6:
        raise ValueError(f"anchor fractions must sum to 1.0, got {fractions}")
    time_ratio = (
        anchor.disk_fraction / DISK_SCALE
        + anchor.routed_compute_fraction / ROUTED_COMPUTE_SCALE
        + anchor.dense_other_fraction / DENSE_COMPUTE_SCALE
    )
    low = anchor.glm_low_tps / time_ratio * anchor.cache_advantage_low
    high = anchor.glm_high_tps / time_ratio * anchor.cache_advantage_high
    return Estimate(anchor.name, anchor.hardware, low, high, anchor.confidence, anchor.basis)


def disk_ceiling_tps(effective_random_gbps: float, expert_hit_rate: float) -> float:
    """Theoretical Nex int4 disk ceiling before compute/attention overhead."""
    if effective_random_gbps <= 0:
        raise ValueError("effective_random_gbps must be positive")
    if not 0 <= expert_hit_rate <= 1:
        raise ValueError("expert_hit_rate must be between 0 and 1")
    miss_gb = COLD_EXPERT_GB_PER_TOKEN * (1.0 - expert_hit_rate)
    return float("inf") if miss_gb == 0 else effective_random_gbps / miss_gb


ANCHORS = (
    Anchor(
        name="minimum streaming box",
        hardware="12 physical cores, 25 GB RAM, ~1 GB/s effective random NVMe",
        glm_low_tps=0.05,
        glm_high_tps=0.10,
        disk_fraction=0.80,
        routed_compute_fraction=0.10,
        dense_other_fraction=0.10,
        confidence="high",
        basis="Colibri README proven cold baseline; Nex cold expert traffic is 3.01x smaller.",
    ),
    Anchor(
        name="128 GB Strix Halo CPU-only",
        hardware="Ryzen AI Max+ 395, 128 GB LPDDR5x, fast Gen4 NVMe",
        glm_low_tps=1.01,
        glm_high_tps=1.10,
        disk_fraction=0.55,
        routed_compute_fraction=0.25,
        dense_other_fraction=0.20,
        cache_advantage_low=1.05,
        cache_advantage_high=1.25,
        confidence="medium-high",
        basis=(
            "Issue #124 quality-preserving DRAFT=0 sustained result. Nex's ~194 GB expert store "
            "roughly doubles byte coverage for the same RAM, so a bounded cache advantage is included."
        ),
    ),
    Anchor(
        name="six RTX 5090 full-resident",
        hardware="6x RTX 5090 32 GB, 251 GiB RAM, 24 physical Xeon cores",
        glm_low_tps=6.28,
        glm_high_tps=6.84,
        disk_fraction=0.0,
        routed_compute_fraction=0.56,
        dense_other_fraction=0.44,
        confidence="high for scaling, medium for Nex kernels",
        basis=(
            "Measured Colibri full-resident run: 56% expert compute, 25% attention, 19% other, "
            "zero disk wait. This is the strongest baseline for a native Nex backend."
        ),
    ),
)

# Projected hardware tiers. These ranges are constrained by the anchor scaling and
# the 3.790 GB/token cold-byte equation. They intentionally do not use vendor
# sequential SSD bandwidth; users should substitute Colibri iobench O_DIRECT data.
PROJECTED_TIERS = (
    Estimate(
        "RTX 2060 laptop / minimum",
        "RTX 2060 6 GB, 16 GB RAM, ~1 GB/s effective random NVMe",
        0.14, 0.30, "medium-high",
        "Directly bracketed by the proven 25 GB GLM cold baseline and cold-byte scaling.",
    ),
    Estimate(
        "mainstream desktop",
        "32 GB RAM, 12-16 cores, 3-5 GB/s iobench random reads",
        0.8, 1.8, "medium",
        "Small cache; predominantly disk/latency bound despite faster storage.",
    ),
    Estimate(
        "fast-storage workstation",
        "64 GB RAM, 16-24 cores, 8-12 GB/s PCIe5/RAID effective random reads",
        2.2, 4.8, "medium-low",
        "About one-quarter of the int4 expert store can be resident; disk and CPU compute are balanced.",
    ),
    Estimate(
        "large unified-memory workstation",
        "128 GB fast RAM, 16 high-bandwidth cores, fast Gen4 NVMe, CPU-only",
        2.8, 3.8, "medium-high",
        "Scaled from the measured Strix Halo DRAFT=0 sustained run, with larger Nex cache coverage.",
    ),
    Estimate(
        "128 GB plus one 5090",
        "128 GB RAM, RTX 5090 32 GB, 16-24 cores, fast NVMe",
        4.5, 8.0, "medium-low",
        "Near-resident hot working set; exact result depends strongly on routing profile and CPU bandwidth.",
    ),
    Estimate(
        "full-resident CPU workstation",
        "256 GB RAM, 24-32 AVX-512/VNNI cores, no decode-time disk misses",
        4.5, 8.5, "medium",
        "Expert store fits in RAM; limited by int4 matmul bandwidth and hybrid-attention kernels.",
    ),
    Estimate(
        "full-resident two-GPU workstation",
        "192-256 GB RAM plus 2x RTX 5090, all experts in RAM/VRAM",
        8.0, 13.0, "medium-low",
        "Interpolated between CPU residency and the measured six-5090 full-resident anchor.",
    ),
    Estimate(
        "full-resident four-GPU workstation",
        "128-192 GB RAM plus 4x RTX 5090, all experts in RAM/VRAM",
        11.0, 17.0, "medium-low",
        "Most expert bytes execute on GPU; communication and dense placement decide the upper end.",
    ),
    Estimate(
        "six-5090 native baseline",
        "6x RTX 5090 32 GB plus >=96 GB RAM, zero disk misses",
        14.5, 16.0, "medium-high",
        "Component-scaled directly from Colibri's measured 6.28-6.84 tok/s GLM full-resident run.",
    ),
    Estimate(
        "six-to-eight GPU optimized target",
        "6-8x RTX 5090/H100/H200, full residency, grouped Tensor Core MoE kernels",
        18.0, 26.0, "low until measured",
        "Requires 13-37% beyond the six-5090 component-scaled baseline; 20 tok/s is credible but gated.",
    ),
)


def report() -> dict:
    return {
        "model_geometry": {
            "expert_params": EXPERT_PARAMS,
            "expert_bytes_int4": EXPERT_BYTES_INT4,
            "expert_calls_per_token": EXPERT_CALLS_PER_TOKEN,
            "cold_expert_gb_per_token": COLD_EXPERT_GB_PER_TOKEN,
            "expert_store_gb_int4": EXPERT_STORE_GB,
            "active_weights_billion": NEX_ACTIVE_WEIGHTS_B,
            "active_gflop_per_token": NEX_ACTIVE_GFLOP_PER_TOKEN,
        },
        "scaling": {
            "glm_cold_expert_gb_per_token": GLM_COLD_EXPERT_GB_PER_TOKEN,
            "disk_and_routed_expert_scale": DISK_SCALE,
            "dense_other_scale": DENSE_COMPUTE_SCALE,
        },
        "calibrated_anchors": [asdict(scale_anchor(anchor)) for anchor in ANCHORS],
        "projected_tiers": [asdict(tier) for tier in PROJECTED_TIERS],
        "rules": [
            "All rates are single-request decode, not aggregate throughput.",
            "Use effective Colibri iobench O_DIRECT/random throughput, not SSD sequential marketing speed.",
            "DRAFT=0 is the quality-preserving streaming baseline until batched routed-expert kernels amortize MTP.",
            "A real benchmark result overrides every estimate.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-gbps", type=float)
    parser.add_argument("--hit-rate", type=float)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    output = report()
    if args.disk_gbps is not None or args.hit_rate is not None:
        if args.disk_gbps is None or args.hit_rate is None:
            parser.error("--disk-gbps and --hit-rate must be supplied together")
        output["custom_disk_ceiling_tps"] = disk_ceiling_tps(args.disk_gbps, args.hit_rate)
    print(json.dumps(output, indent=None if args.compact else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
