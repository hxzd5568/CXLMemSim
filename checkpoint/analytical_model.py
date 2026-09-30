#!/usr/bin/env python3
"""
Analytical model for the CXLMemSim checkpoint/restore engine (target2 dual-line).

Mirrors the closed-form bounds in include/checkpoint_engine.h
(ckpt::computeBounds / planOffload) and the stage/persist/restore timing model in
src/checkpoint_engine.cpp, so it can rapidly scan topology x channel counts
before running the (slower, but ground-truth) C++ engine.

target2 dual-line model (E0):
    Line 1 (GPU->DRAM) = Gen5 x16 uplink (consumes the switch uplink)
    Line 2 (GPU->CXL)  = Gen5 x16 P2P (expander direct-attach, no uplink, pure TLP)

    B_line1 = min(pcie_eff(line_dram) * write_ratio, DRAM_write)
    B_line2 = min(pcie_eff(line_cxl)  * write_ratio, CXL_write)   [CXL = raw DDR, no flit]
    B_stage = min(num_gpus * pcie_eff(egress), B_line1 + B_line2)   [additive]

Offload plan (section 7): water-filling split at task submission,
    alpha = B_line1 / (B_line1 + B_line2)  (clamped by capacity/affinity).

Usage:
    python3 checkpoint/analytical_model.py [--total-mib 512] [--channels 1,2,4,8,16]
        [--line-dram 64] [--line-cxl 64] [--num-gpus 1] [--gpu-egress 32]
        [--write-ratio 1.0] [--dram-lanes 8] [--dram-write 17.6] [--cxl-lanes 8]
        [--cxl-write 50] [--mps 256] [--tlp-overhead 20]
"""

import argparse

ENCODING_EFFICIENCY = 128.0 / 130.0  # 128b/130b line coding
RESIDUAL_EFFICIENCY = 0.95           # DLLP/ACK/flow-control/DMA-engine residual


class Pool:
    def __init__(self, num_lanes, read_gbps, write_gbps, read_lat_ns, write_lat_ns):
        self.num_lanes = num_lanes
        self.read_gbps = read_gbps      # per-lane
        self.write_gbps = write_gbps    # per-lane
        self.read_lat_ns = read_lat_ns
        self.write_lat_ns = write_lat_ns

    def agg_read(self):
        return self.read_gbps * self.num_lanes

    def agg_write(self):
        return self.write_gbps * self.num_lanes


class Config:
    """Defaults mirror ckpt_test::defaultConfig in tests/checkpoint_test_harness.h."""

    def __init__(self, channels=2, *, line_dram_raw=64.0, line_cxl_raw=64.0,
                 num_gpus=1, gpu_egress=32.0, write_read_ratio=1.0,
                 mps=256, tlp_overhead=20,
                 dram_lanes=8, dram_read=17.6, dram_write=17.6,
                 cxl_lanes=8, cxl_read=50.0, cxl_write=50.0,
                 storage_gbps=2.0):
        self.chunk_size = 4096
        self.channels = channels
        # target2 dual-line: two independent PCIe lines replace the single uplink.
        self.line_dram_raw = line_dram_raw     # Gen5 x16 uplink (consumes uplink)
        self.line_cxl_raw = line_cxl_raw       # Gen5 x16 P2P (expander direct-attach, no uplink)
        self.num_gpus = num_gpus
        self.gpu_egress = gpu_egress           # per-GPU Gen4 x16 raw egress
        self.write_read_ratio = write_read_ratio
        # TLP-structure overhead model (derives the usable payload fraction).
        self.pcie_encoding_efficiency = ENCODING_EFFICIENCY  # 128b/130b (Gen3+)
        self.pcie_max_payload_bytes = mps
        self.pcie_tlp_overhead_bytes = tlp_overhead
        self.pcie_residual_efficiency = RESIDUAL_EFFICIENCY
        # CXL.mem flit framing: deprecated (target2 DMA is pure PCIe TLP, no flit).
        self.cxl_flit_data_bytes = 64
        self.cxl_flit_total_bytes = 64  # 64/64 => no flit overhead
        # Calibration points: per-lane RAW link/channel rates.
        self.dram = Pool(dram_lanes, dram_read, dram_write, 100.0, 100.0)
        self.cxl = Pool(cxl_lanes, cxl_read, cxl_write, 400.0, 400.0)
        self.channel_read_gbps = storage_gbps
        self.channel_write_gbps = storage_gbps
        self.channel_latency_ns = 10000.0

    def pcie_effective(self, raw_gbps):
        tlp_ratio = self.pcie_max_payload_bytes / (self.pcie_max_payload_bytes + self.pcie_tlp_overhead_bytes)
        return raw_gbps * self.pcie_encoding_efficiency * tlp_ratio * self.pcie_residual_efficiency

    def pcie_write(self, raw_gbps):
        return self.pcie_effective(raw_gbps) * self.write_read_ratio

    def cxl_overhead(self):
        # target2: CXL backend is raw DDR bandwidth (pure TLP P2P, no flit).
        return 1.0

    def cxl_agg_eff_read(self):
        return self.cxl.agg_read() * self.cxl_overhead()

    def cxl_agg_eff_write(self):
        return self.cxl.agg_write() * self.cxl_overhead()


def line_bounds(cfg, dram_enabled=True, cxl_enabled=True):
    """Return (B_line1, B_line2, egress) for the dual-line staging model."""
    b1 = min(cfg.pcie_write(cfg.line_dram_raw), cfg.dram.agg_write()) if dram_enabled else 0.0
    b2 = min(cfg.pcie_write(cfg.line_cxl_raw), cfg.cxl_agg_eff_write()) if cxl_enabled else 0.0
    egress = cfg.num_gpus * cfg.pcie_effective(cfg.gpu_egress)
    return b1, b2, egress


def bounds(cfg, dram_enabled=True, cxl_enabled=True):
    cxl_write = cfg.cxl_agg_eff_write() if cxl_enabled else 0.0
    cxl_read = cfg.cxl_agg_eff_read() if cxl_enabled else 0.0
    dram_write = cfg.dram.agg_write() if dram_enabled else 0.0
    dram_read = cfg.dram.agg_read() if dram_enabled else 0.0

    b1, b2, egress = line_bounds(cfg, dram_enabled, cxl_enabled)
    staging = min(egress, b1 + b2)
    channel_write = cfg.channels * cfg.channel_write_gbps
    channel_read = cfg.channels * cfg.channel_read_gbps

    bstage = staging
    bsave = min(staging, dram_read + cxl_read, channel_write)
    brestore = min(channel_read, dram_write + cxl_write, egress)
    return bstage, bsave, brestore


def plan_split(cfg, total_bytes, dram_usable=float("inf"), cxl_usable=float("inf"),
               dram_enabled=True, cxl_enabled=True):
    """Water-filling offload plan (section 7). Returns (dram_bytes, cxl_bytes, alpha).
    Continuous approximation of the C++ planOffload (which quantizes to split units)."""
    b1, b2, _ = line_bounds(cfg, dram_enabled, cxl_enabled)
    total_r = b1 + b2
    if total_r <= 0:
        return (0.0, total_bytes, 0.0)
    x = total_bytes * b1 / total_r
    x = min(max(x, 0.0), float(total_bytes))
    x = min(x, float(dram_usable))
    x = max(x, float(total_bytes - cxl_usable))
    dram = x
    cxl = total_bytes - dram
    return (dram, cxl, dram / total_bytes if total_bytes else 0.0)


def stage_time(cfg, total_bytes, dram_enabled=True, cxl_enabled=True):
    """GPU-visible staging time (phase 1) under the dual-line model."""
    b1, b2, egress = line_bounds(cfg, dram_enabled, cxl_enabled)
    total_r = b1 + b2
    if total_r == 0:
        return float("inf")
    dram_bytes = total_bytes * b1 / total_r
    cxl_bytes = total_bytes * b2 / total_r

    egress_ns = total_bytes / egress if egress > 0 else 0.0
    line1_ns = dram_bytes / cfg.pcie_write(cfg.line_dram_raw) if dram_enabled and dram_bytes > 0 else 0.0
    line2_ns = cxl_bytes / cfg.pcie_write(cfg.line_cxl_raw) if cxl_enabled and cxl_bytes > 0 else 0.0
    dram_ns = dram_bytes / cfg.dram.agg_write() if dram_enabled else 0.0
    cxl_ns = cxl_bytes / cfg.cxl_agg_eff_write() if cxl_enabled else 0.0
    lat = 0.0
    if dram_enabled:
        lat = max(lat, cfg.dram.write_lat_ns)
    if cxl_enabled:
        lat = max(lat, cfg.cxl.write_lat_ns)
    return max(egress_ns, line1_ns, line2_ns, dram_ns, cxl_ns) * 1e-9 + lat / 1e9


def persist_time(cfg, total_bytes, dram_enabled=True, cxl_enabled=True):
    _, bsave, _ = bounds(cfg, dram_enabled, cxl_enabled)
    return total_bytes / bsave * 1e-9 + cfg.channel_latency_ns / 1e9


def restore_time(cfg, total_bytes, dram_enabled=True, cxl_enabled=True):
    _, _, brestore = bounds(cfg, dram_enabled, cxl_enabled)
    return total_bytes / brestore * 1e-9 + cfg.channel_latency_ns / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-kib", type=int, default=4)
    p.add_argument("--total-mib", type=int, default=512)
    p.add_argument("--channels", default="1,2,4,8,16")
    p.add_argument("--line-dram", type=float, default=64.0)
    p.add_argument("--line-cxl", type=float, default=64.0)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--gpu-egress", type=float, default=32.0)
    p.add_argument("--write-ratio", type=float, default=1.0)
    p.add_argument("--dram-lanes", type=int, default=8)
    p.add_argument("--dram-read", type=float, default=17.6)
    p.add_argument("--dram-write", type=float, default=17.6)
    p.add_argument("--cxl-lanes", type=int, default=8)
    p.add_argument("--cxl-read", type=float, default=50.0)
    p.add_argument("--cxl-write", type=float, default=50.0)
    p.add_argument("--mps", type=int, default=256)
    p.add_argument("--tlp-overhead", type=int, default=20)
    p.add_argument("--storage-gbps", type=float, default=2.0)
    args = p.parse_args()

    total_bytes = args.total_mib * 1024 * 1024
    channel_list = [int(x) for x in args.channels.split(",")]

    def make_cfg(c):
        return Config(c, line_dram_raw=args.line_dram, line_cxl_raw=args.line_cxl,
                      num_gpus=args.num_gpus, gpu_egress=args.gpu_egress,
                      write_read_ratio=args.write_ratio, mps=args.mps, tlp_overhead=args.tlp_overhead,
                      dram_lanes=args.dram_lanes, dram_read=args.dram_read, dram_write=args.dram_write,
                      cxl_lanes=args.cxl_lanes, cxl_read=args.cxl_read, cxl_write=args.cxl_write,
                      storage_gbps=args.storage_gbps)

    base = make_cfg(2)
    b1, b2, egress = line_bounds(base)
    dram, cxl, alpha = plan_split(base, total_bytes)

    print("CXLMemSim checkpoint analytical model (target2 dual-line)")
    print(f"  workload: {args.total_mib} MiB, chunk {args.chunk_kib} KiB")
    print(f"  Line1 (DRAM): {args.line_dram} GB/s raw -> {base.pcie_write(args.line_dram):.2f} GB/s eff (uplink)")
    print(f"  Line2 (CXL) : {args.line_cxl} GB/s raw -> {base.pcie_write(args.line_cxl):.2f} GB/s eff (P2P)")
    print(f"  egress: {args.num_gpus} GPU x {base.pcie_effective(args.gpu_egress):.2f} GB/s = {egress:.2f} GB/s")
    print(f"  DRAM {args.dram_lanes} lanes x {args.dram_write}/{args.dram_read} GB/s (w/r) = "
          f"{base.dram.agg_write():.0f}/{base.dram.agg_read():.0f} GB/s")
    print(f"  CXL  {args.cxl_lanes} DDR x {args.cxl_write}/{args.cxl_read} GB/s (w/r) = "
          f"{base.cxl_agg_eff_write():.0f}/{base.cxl_agg_eff_read():.0f} GB/s (raw DDR, no flit)")
    print(f"  offload plan: alpha={alpha:.3f} (dram {dram/1024/1024:.0f} MiB / cxl {cxl/1024/1024:.0f} MiB)\n")

    print("GPU copy bandwidth (staging):")
    for topo, dram_en, cxl_en in [("only DRAM", True, False), ("only CXL", False, True), ("both", True, True)]:
        bstage, _, _ = bounds(make_cfg(2), dram_en, cxl_en)
        print(f"  {topo:<9}: {bstage:6.2f} GB/s")
    print()

    print(f"{'topology':<10}" + "".join(f"{c:>7d}ch" for c in channel_list))
    for topo, dram_en, cxl_en in [("dram", True, False), ("cxl", False, True), ("both", True, True)]:
        row = f"{topo:<10}"
        for c in channel_list:
            _, bsave, _ = bounds(make_cfg(c), dram_en, cxl_en)
            row += f"{bsave:>8.2f}"
        print(row)

    print("\n--- predicted phase times (2 channels, dual-lane) ---")
    cfg = make_cfg(2)
    st = stage_time(cfg, total_bytes, True, True)
    pt = persist_time(cfg, total_bytes, True, True)
    rt = restore_time(cfg, total_bytes, True, True)
    print(f"  staging (GPU stall) : {st*1e3:8.2f} ms")
    print(f"  persist (background): {pt*1e3:8.2f} ms")
    print(f"  restore (cold)      : {rt*1e3:8.2f} ms")
    _, bsave, _ = bounds(cfg, True, True)
    print(f"  B_save upper bound  : {bsave:8.2f} GB/s")


if __name__ == "__main__":
    main()
