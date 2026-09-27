/*
 * Balancer + pressure control.
 *
 * Honest staging-bandwidth accounting: the GPU has a single PCIe uplink shared
 * by both lanes, so aggregate staging is min(PCIe, DRAM_write + CXL_write).
 * With realistic DRAM (faster than PCIe), a single DRAM lane already saturates
 * the uplink, so acceptance #2 ("dual path > fastest single path") is NOT met
 * on a single-node topology -- dual-lane only helps against a lane that is
 * itself slower than PCIe (e.g. CXL-only). This matches SimCXL's honest P10
 * conclusion. Also covers #6 (pinned pool reuses but never grows under pressure).
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <cstdio>

using namespace ckpt;

int main() {
    constexpr uint64_t chunks = 1024;
    const uint64_t bytes = chunks * 4096;

    // --- honest #2: dual-lane gives a modest (not 2x) staging gain ---
    // Single-DRAM (50 GB/s effective) cannot saturate the ~54 GB/s effective
    // PCIe uplink, so adding the CXL lane (28 GB/s) uses the remaining PCIe
    // headroom -- but the total is still capped by the uplink.
    {
        CheckpointConfig single_cfg = ckpt_test::defaultConfig(2);
        single_cfg.cxl_enabled = false;
        CheckpointEngine single(single_cfg);
        single.stage(0, chunks);
        double bw_single = ckpt_test::bandwidthGbps(bytes, single.stats().gpu_stage_time_ns);

        CheckpointEngine dual(ckpt_test::defaultConfig(2));
        dual.stage(0, chunks);
        double bw_dual = ckpt_test::bandwidthGbps(bytes, dual.stats().gpu_stage_time_ns);

        // Dual must exceed single, but only modestly (bounded by the shared
        // uplink, not 2x as target.md assumed).
        REQUIRE_MSG(bw_dual > bw_single, "dual %.2f should exceed single DRAM %.2f", bw_dual, bw_single);
        REQUIRE_MSG(bw_dual < bw_single * 1.25, "dual %.2f should be a modest gain over single %.2f", bw_dual, bw_single);
        std::printf("[staging] single DRAM %.2f -> dual %.2f GB/s (%.2fx; both below raw 64 GB/s PCIe)\n", bw_single,
                    bw_dual, bw_dual / bw_single);
    }

    // --- dual-lane gives a larger gain vs a slow lane (CXL-only) ---
    {
        // Single CXL lane staging rate = min(effective PCIe 54.4, CXL.write 28) = 28.
        double pcie_eff = ckpt_test::defaultConfig(2).gpu_pcie_write_gbps * ckpt_test::defaultConfig(2).pcie_efficiency;
        double bw_single_cxl = std::min(pcie_eff, ckpt_test::defaultConfig(2).cxl.write_gbps);

        CheckpointEngine dual(ckpt_test::defaultConfig(2));
        dual.stage(0, chunks);
        double bw_dual = ckpt_test::bandwidthGbps(bytes, dual.stats().gpu_stage_time_ns);

        REQUIRE_MSG(bw_dual > bw_single_cxl, "dual %.2f should exceed single CXL %.2f", bw_dual, bw_single_cxl);
        std::printf("[staging] dual %.2f GB/s > single CXL %.2f GB/s (dual-lane helps vs a slow lane)\n", bw_dual,
                    bw_single_cxl);
    }

    // --- PCIe ceiling: a slow uplink caps dual-lane to the uplink itself ---
    {
        CheckpointConfig limited = ckpt_test::defaultConfig(2);
        limited.gpu_pcie_write_gbps = 20.0; // PCIe uplink slower than both lanes
        limited.pcie_efficiency = 1.0;

        CheckpointConfig single_cfg = limited;
        single_cfg.cxl_enabled = false;
        CheckpointEngine ls(single_cfg);
        ls.stage(0, chunks);
        double bw_single = ckpt_test::bandwidthGbps(bytes, ls.stats().gpu_stage_time_ns);

        CheckpointEngine ld(limited);
        ld.stage(0, chunks);
        double bw_dual = ckpt_test::bandwidthGbps(bytes, ld.stats().gpu_stage_time_ns);

        REQUIRE_MSG(bw_dual <= bw_single * 1.05 + 0.5,
                    "PCIe-limited dual %.2f should not exceed single %.2f by >5%%", bw_dual, bw_single);
        std::printf("[staging] PCIe-limited (20 GB/s uplink): dual %.2f ~= single %.2f GB/s (ceiling honored)\n", bw_dual,
                    bw_single);
    }

    // --- #6: pressure state machine stops pool growth ---
    {
        CheckpointConfig cfg = ckpt_test::defaultConfig(2);
        cfg.cxl_enabled = false;
        cfg.dram.capacity_bytes = 65536; // 16 chunks of 4096 B
        CheckpointEngine e(cfg);

        e.stage(0, chunks); // far more than the pool can hold
        const PinnedPool &dram = e.dramPool();
        REQUIRE(dram.pinnedBytes() <= cfg.dram.capacity_bytes);
        REQUIRE(dram.state() != PoolState::NORMAL);

        uint64_t pinned_after = dram.pinnedBytes();
        REQUIRE(pinned_after > 0);

        // Further staging must not grow the pool (reuse only, no expansion).
        e.stage(1, 256);
        REQUIRE(e.dramPool().pinnedBytes() == pinned_after);

        std::printf("[pressure] pool state=%s pinned=%llu (cap=%llu) does not grow (OK)\n", poolStateName(dram.state()),
                    (unsigned long long)pinned_after, (unsigned long long)cfg.dram.capacity_bytes);
    }

    std::printf("PASS\n");
    return 0;
}
