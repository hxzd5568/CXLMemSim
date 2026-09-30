/*
 * Offload plan + pressure control under the target2 model (expander direct-attach).
 *
 * The balancer is no longer a per-chunk greedy chooser; placement is a static
 * offload plan (section 7) computed once at stage() time, with the memcpy
 * address (split unit) fixing DRAM vs CXL. This test covers:
 *   - the plan's split ratio (alpha) matches the water-filling closed form
 *     (both lines are Gen5 x16, so alpha = 0.5 for wide backends);
 *   - #6: pinned pool reuses but never grows under pressure.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <cstdio>

using namespace ckpt;

int main() {
    constexpr uint64_t kMiB = 1024ull * 1024ull;
    constexpr uint64_t chunks = (512ull * kMiB) / 4096; // 512 MiB
    const uint64_t bytes = chunks * 4096;

    // --- offload plan split ratio (water-fill) ---
    {
        CheckpointConfig cfg = ckpt_test::defaultConfig(2);
        cfg.dram.capacity_bytes = 1 * kMiB * 1024;
        cfg.cxl.capacity_bytes = 1 * kMiB * 1024;
        ckpt_test::disablePoolPressure(cfg);

        double b1 = std::min(ckpt::pcieWriteGbps(cfg.line_dram, cfg), cfg.dram.aggregateWriteGbps());
        double b2 = std::min(ckpt::pcieWriteGbps(cfg.line_cxl, cfg), cfg.cxl.aggregateWriteGbps());

        CheckpointEngine e(cfg);
        e.stage(0, chunks);
        const PlacementPlan &p = e.plan();

        // Both lines are Gen5 x16 (~55.5 GB/s) and both backends exceed that, so
        // the water-fill splits 50/50 (alpha = B1/(B1+B2) = 0.5).
        double alpha_expect = b1 / (b1 + b2);
        std::printf("[offload] plan alpha %.3f (expect ~%.3f), dram %llu MiB / cxl %llu MiB\n", p.alpha, alpha_expect,
                    (unsigned long long)(p.dram_bytes / kMiB), (unsigned long long)(p.cxl_bytes / kMiB));
        REQUIRE(!p.unit_lane.empty());
        REQUIRE_MSG(p.alpha > 0.0 && p.alpha < 1.0, "alpha %.3f should split both lanes", p.alpha);
        REQUIRE(p.dram_bytes > 0 && p.cxl_bytes > 0);
        REQUIRE(p.dram_bytes + p.cxl_bytes == bytes);
        REQUIRE_MSG(p.alpha > 0.45 && p.alpha < 0.55, "alpha %.3f should be ~0.5 (both lines Gen5 x16)", p.alpha);
    }

    // --- #6: pressure state machine stops pool growth ---
    {
        CheckpointConfig cfg = ckpt_test::defaultConfig(2);
        cfg.cxl_enabled = false;
        cfg.dram.capacity_bytes = 65536; // 16 chunks of 4096 B
        CheckpointEngine e(cfg);

        e.stage(0, 1024); // far more than the pool can hold
        const PinnedPool &dram = e.dramPool();
        REQUIRE(dram.pinnedBytes() <= cfg.dram.capacity_bytes);
        REQUIRE(dram.state() != PoolState::NORMAL);

        uint64_t pinned_after = dram.pinnedBytes();
        REQUIRE(pinned_after > 0);

        e.stage(1, 256);
        REQUIRE(e.dramPool().pinnedBytes() == pinned_after);

        std::printf("[pressure] pool state=%s pinned=%llu (cap=%llu) does not grow (OK)\n", poolStateName(dram.state()),
                    (unsigned long long)pinned_after, (unsigned long long)cfg.dram.capacity_bytes);
    }

    std::printf("PASS\n");
    return 0;
}
