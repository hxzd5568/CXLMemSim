/*
 * Offload plan (section 7) unit tests: water-filling split, capacity and
 * placement-affinity constraints, partial split units, infeasibility, and the
 * drift detector that triggers a re-plan.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <cmath>
#include <cstdio>

using namespace ckpt;

namespace {
constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr uint64_t kUnit = 256ull * kMiB;
} // namespace

int main() {
    // --- water-fill: all-EITHER data splits proportional to B_line1/(B1+B2) ---
    {
        OffloadSpec spec;
        spec.total_bytes = 20 * kUnit; // 5 GiB, 20 split units
        spec.split_unit_bytes = kUnit;
        spec.affinity.assign(20, Placement::EITHER);

        PlacementPlan p = planOffload(spec, 63.0, 35.0, 8 * 1024 * kMiB, 8 * 1024 * kMiB);
        REQUIRE(!p.unit_lane.empty());
        REQUIRE(p.unit_lane.size() == 20);
        REQUIRE(p.dram_bytes + p.cxl_bytes == spec.total_bytes);

        double alpha_expect = 63.0 / (63.0 + 35.0); // ~0.6429
        // Quantization to whole 256 MiB units => within 1/20 of the closed form.
        REQUIRE_MSG(std::fabs(p.alpha - alpha_expect) < (1.0 / 20.0) + 1e-9, "alpha %.4f ~= %.4f", p.alpha, alpha_expect);
        std::printf("[offload] water-fill alpha %.4f (expect ~%.4f), dram %llu MiB / cxl %llu MiB\n", p.alpha, alpha_expect,
                    (unsigned long long)(p.dram_bytes / kMiB), (unsigned long long)(p.cxl_bytes / kMiB));
    }

    // --- placement affinity: pinned units stay on their lane ---
    {
        OffloadSpec spec;
        spec.total_bytes = 10 * kUnit;
        spec.split_unit_bytes = kUnit;
        spec.affinity.assign(10, Placement::EITHER);
        spec.affinity[0] = Placement::DRAM;
        spec.affinity[1] = Placement::DRAM;
        spec.affinity[2] = Placement::CXL;

        PlacementPlan p = planOffload(spec, 63.0, 35.0, 8 * 1024 * kMiB, 8 * 1024 * kMiB);
        REQUIRE(p.unit_lane[0] == LaneId::DRAM);
        REQUIRE(p.unit_lane[1] == LaneId::DRAM);
        REQUIRE(p.unit_lane[2] == LaneId::CXL);
        std::printf("[offload] affinity: 2 DRAM + 1 CXL pinned units honored (OK)\n");
    }

    // --- capacity constraint: a small DRAM pool pushes the split toward CXL ---
    {
        OffloadSpec spec;
        spec.total_bytes = 20 * kUnit;
        spec.split_unit_bytes = kUnit;
        spec.affinity.assign(20, Placement::EITHER);

        uint64_t dram_usable = 3 * kUnit; // only 3 GiB of DRAM
        PlacementPlan p = planOffload(spec, 63.0, 35.0, dram_usable, 8 * 1024 * kMiB);
        REQUIRE(p.dram_bytes <= dram_usable);
        std::printf("[offload] capacity clamp: dram %llu MiB <= usable %llu MiB (OK)\n",
                    (unsigned long long)(p.dram_bytes / kMiB), (unsigned long long)(dram_usable / kMiB));
    }

    // --- partial final split unit ---
    {
        OffloadSpec spec;
        spec.total_bytes = 2 * kUnit + 128 * kMiB; // 640 MiB => 3 units (last partial)
        spec.split_unit_bytes = kUnit;
        spec.affinity.assign(3, Placement::EITHER);

        PlacementPlan p = planOffload(spec, 63.0, 35.0, 8 * 1024 * kMiB, 8 * 1024 * kMiB);
        REQUIRE(!p.unit_lane.empty());
        REQUIRE(p.unit_lane.size() == 3);
        REQUIRE(p.dram_bytes + p.cxl_bytes == spec.total_bytes);
        std::printf("[offload] partial unit: 640 MiB = dram %llu + cxl %llu (OK)\n",
                    (unsigned long long)(p.dram_bytes / kMiB), (unsigned long long)(p.cxl_bytes / kMiB));
    }

    // --- infeasibility: affinity exceeds capacity => empty plan ---
    {
        OffloadSpec spec;
        spec.total_bytes = 4 * kUnit;
        spec.split_unit_bytes = kUnit;
        spec.affinity.assign(4, Placement::DRAM);

        PlacementPlan p = planOffload(spec, 63.0, 35.0, 2 * kUnit, 8 * 1024 * kMiB); // DRAM too small
        REQUIRE(p.unit_lane.empty());
        std::printf("[offload] infeasible affinity -> empty plan (OK)\n");
    }

    // --- drift detector: a full pool triggers re-plan ---
    {
        CheckpointConfig cfg = ckpt_test::defaultConfig(2);
        cfg.cxl_enabled = false;
        cfg.dram.capacity_bytes = 64 * kMiB; // 16384 chunks
        CheckpointEngine e(cfg);
        e.stage(0, 16384); // fill DRAM past FROZEN (~80%)
        REQUIRE(e.dramPool().state() != PoolState::NORMAL);
        REQUIRE(e.shouldReplan()); // free < drift_cap_threshold (0.20)
        std::printf("[offload] drift: DRAM pool past FROZEN triggers re-plan (free %.0f%%)\n",
                    100.0 * static_cast<double>(e.dramPool().freeBytes()) / static_cast<double>(e.dramPool().capacityBytes()));
    }

    std::printf("PASS\n");
    return 0;
}
