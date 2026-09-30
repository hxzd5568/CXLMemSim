/*
 * Lane-based GPU copy bandwidth under the target2 DUAL-LINE model
 * (CXL expander direct-attach, no CXL switch).
 *
 * The single shared PCIe uplink is replaced by two independent Gen5 x16 lines (E0):
 *   Line 1 (GPU->DRAM)       traverses the switch uplink (Gen5 x16, consumes_uplink).
 *   Line 2 (GPU->CXL expander) switch-local P2P (Gen5 x16, no uplink, device DMA,
 *                              pure PCIe TLP -- NO CXL.mem 68B flit).
 * Staging bandwidth is additive and capped by the GPUs' aggregate egress:
 *   B_stage = min(num_gpus * egress, B_line1 + B_line2)
 *
 * Verifies (acceptance A1/A3): with wide backends a single GPU is egress-limited;
 * with a narrow DRAM backend the CXL line saturates the GPU egress; with 4 GPUs
 * the CXL line offloads the DRAM uplink and reaches the full egress (2.0x).
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <cstdio>

using namespace ckpt;

namespace {

constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr uint64_t kGiB = 1024ull * kMiB;

double measure(const CheckpointConfig &cfg, uint64_t chunks) {
    CheckpointEngine e(cfg);
    e.stage(0, chunks);
    uint64_t bytes = chunks * cfg.chunk_size;
    return ckpt_test::bandwidthGbps(bytes, e.stats().gpu_stage_time_ns);
}

} // namespace

int main() {
    // --- (a) wide backends, single GPU: egress-limited in every case ---
    {
        CheckpointConfig base = ckpt_test::defaultConfig(2);
        base.dram.capacity_bytes = 1 * kGiB;
        base.cxl.capacity_bytes = 1 * kGiB;
        ckpt_test::disablePoolPressure(base);

        CheckpointConfig dram_only = base;
        dram_only.cxl_enabled = false;
        CheckpointConfig cxl_only = base;
        cxl_only.dram_enabled = false;

        constexpr uint64_t chunks = (512ull * kMiB) / 4096;
        double bw_d = measure(dram_only, chunks);
        double bw_c = measure(cxl_only, chunks);
        double bw_b = measure(base, chunks);

        double egress = ckpt::pcieEffectiveGbps(base.gpu_egress_gbps, base);
        std::printf("single-GPU wide backends: only-DRAM %.2f, only-CXL %.2f, both %.2f (egress %.2f)\n", bw_d, bw_c,
                    bw_b, egress);
        REQUIRE_MSG(bw_d > 0.98 * egress, "only-DRAM %.2f should ~egress %.2f", bw_d, egress);
        REQUIRE_MSG(bw_c > 0.98 * egress, "only-CXL %.2f should ~egress %.2f", bw_c, egress);
        REQUIRE_MSG(bw_b > 0.98 * egress, "both %.2f should ~egress %.2f", bw_b, egress);
    }

    // --- (b) narrow DRAM backend (1 sub-channel) + CXL DDR: dual saturates egress ---
    {
        CheckpointConfig cfg = ckpt_test::defaultConfig(2);
        cfg.dram.num_lanes = 1;
        cfg.dram.write_gbps = 17.6;
        cfg.dram.read_gbps = 17.6;
        cfg.cxl.num_lanes = 1;
        cfg.cxl.write_gbps = 50.0; // 1 DDR on the expander
        cfg.cxl.read_gbps = 50.0;
        cfg.dram.capacity_bytes = 1 * kGiB;
        cfg.cxl.capacity_bytes = 1 * kGiB;
        ckpt_test::disablePoolPressure(cfg);

        CheckpointConfig dram_only = cfg;
        dram_only.cxl_enabled = false;

        constexpr uint64_t chunks = (512ull * kMiB) / 4096;
        double bw_d = measure(dram_only, chunks);
        double bw_b = measure(cfg, chunks);
        double egress = ckpt::pcieEffectiveGbps(cfg.gpu_egress_gbps, cfg);

        std::printf("single-GPU narrow DRAM (1 sub-ch) + CXL DDR: only-DRAM %.2f, both %.2f (egress %.2f)\n", bw_d, bw_b,
                    egress);
        REQUIRE_MSG(bw_d > 0.98 * 17.6 && bw_d < 1.02 * 17.6, "only-DRAM %.2f should ~17.6 (1 sub-channel)", bw_d);
        REQUIRE_MSG(bw_b > 0.98 * egress, "both %.2f should ~egress %.2f", bw_b, egress);
        REQUIRE_MSG(bw_b > bw_d * 1.05, "both %.2f should exceed only-DRAM %.2f (additivity)", bw_b, bw_d);
    }

    // --- (c) 4 GPUs: CXL line offloads the DRAM uplink (A1/A2) ---
    {
        CheckpointConfig cfg = ckpt_test::defaultConfig(2);
        cfg.num_gpus = 4;
        cfg.split_unit_bytes = 64 * kMiB; // fine split => small quantization error
        cfg.dram.capacity_bytes = 2 * kGiB;
        cfg.cxl.capacity_bytes = 2 * kGiB;
        ckpt_test::disablePoolPressure(cfg);

        CheckpointConfig dram_only = cfg;
        dram_only.cxl_enabled = false;

        constexpr uint64_t chunks = (2ull * kGiB) / 4096;
        double bw_d = measure(dram_only, chunks);
        double bw_b = measure(cfg, chunks);

        std::printf("4-GPU: only-DRAM %.2f (uplink-limited), both %.2f (DRAM uplink + CXL P2P)\n", bw_d, bw_b);
        REQUIRE_MSG(bw_b > bw_d * 1.05, "4-GPU both %.2f should exceed only-DRAM %.2f (CXL offloads uplink)", bw_b, bw_d);
    }

    std::printf("PASS\n");
    return 0;
}
