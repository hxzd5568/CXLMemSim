/*
 * vhot dual-path pipelined restore (20% CXL hot standby + dual-NVMe cold load).
 *
 * The system is multi-backend: the cold tail of a checkpoint is restored through
 * TWO independent NVMe devices that each feed their own landing pool --
 *     nvme0 -> host DRAM -> GPU
 *     nvme1 -> CXL DRAM  -> GPU
 * -- while the first 20% of chunks are served from the CXL hot standby (no NVMe).
 * All three streams overlap (Little's law), so restore time is bounded by the
 * slowest stream, not the sum.
 *
 * Compares, for 512 MiB:
 *   single-path restore 0% hot   (one storage backend -> DRAM -> GPU)
 *   single-path restore 20% hot
 *   dual-path   restore 0% hot   (nvme0->DRAM + nvme1->CXL, in parallel)
 *   dual-path   restore 20% hot
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <cmath>
#include <cstdio>
#include <memory>

using namespace ckpt;

namespace {
constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr uint64_t kGiB = 1024ull * kMiB;

CheckpointConfig vhotConfig() {
    CheckpointConfig cfg = ckpt_test::defaultConfig(2);
    cfg.dram.capacity_bytes = 1 * kGiB;
    cfg.cxl.capacity_bytes = 1 * kGiB;
    cfg.storage_capacity_per_channel_bytes = 1 * kGiB;
    ckpt_test::disablePoolPressure(cfg);
    return cfg;
}
} // namespace

int main() {
    constexpr uint64_t chunks = (512ull * kMiB) / 4096; // 131072 chunks = 512 MiB
    constexpr uint64_t total_bytes = chunks * 4096;
    constexpr uint64_t hot20 = chunks / 5; // 20%

    CheckpointConfig cfg = vhotConfig();
    cfg.hot_capacity_chunks = chunks;

    // Build a persisted checkpoint and promote `hot_count` chunks into the CXL
    // hot standby (LRU cache of the disk copy).
    auto build = [&](uint64_t hot_count) {
        auto e = std::make_unique<CheckpointEngine>(cfg);
        e->stage(1, chunks);
        e->persist(1);
        e->commitGeneration(1);
        for (uint64_t i = 0; i < hot_count; ++i) {
            e->promoteChunk(1, static_cast<uint32_t>(i));
        }
        return e;
    };

    double t_single_0, t_single_20, t_dual_0, t_dual_20;
    DualRestoreStats d0, d20;

    {
        auto e = build(0);
        t_single_0 = e->restore(1, chunks);
        std::printf("[single  0%% hot] restore %.3f ms (hit %.2f)\n", t_single_0 / 1e6, e->stats().hot_hit_rate);
    }
    {
        auto e = build(hot20);
        t_single_20 = e->restore(1, chunks);
        std::printf("[single 20%% hot] restore %.3f ms (hit %.2f)\n", t_single_20 / 1e6, e->stats().hot_hit_rate);
    }
    {
        auto e = build(0);
        t_dual_0 = e->restoreDualPath(1, chunks, 0.0);
        d0 = e->stats().dual_restore;
        std::printf("[dual    0%% hot] restore %.3f ms (nvme0 %.1f / nvme1 %.1f / gpu %.1f ms)\n",
                    t_dual_0 / 1e6, d0.nvme0_fill_ns / 1e6, d0.nvme1_fill_ns / 1e6, d0.gpu_drain_ns / 1e6);
    }
    {
        auto e = build(hot20);
        t_dual_20 = e->restoreDualPath(1, chunks, 0.20);
        d20 = e->stats().dual_restore;
        std::printf("[dual   20%% hot] restore %.3f ms (hot %.1f MiB, nvme0 %.1f / nvme1 %.1f MiB, gpu %.1f ms)\n",
                    t_dual_20 / 1e6, d20.hot_bytes / double(kMiB), d20.nvme0_bytes / double(kMiB),
                    d20.nvme1_bytes / double(kMiB), d20.gpu_drain_ns / 1e6);
    }

    double speedup_dual_vs_single = t_single_0 / t_dual_0;
    double speedup_vhot = t_single_0 / t_dual_20;
    std::printf("[vhot] dual-path 0%% hot is %.2fx faster than single-path; +20%% hot => %.2fx total\n",
                speedup_dual_vs_single, speedup_vhot);

    // Dual path (two NVMe backends in parallel) must beat single path.
    REQUIRE_MSG(t_dual_0 < t_single_0 * 0.6, "dual-path should be ~2x single, got %.3f vs %.3f ms", t_dual_0 / 1e6,
                t_single_0 / 1e6);

    // 20% hot further reduces restore time (cold tail shrinks to 80%).
    REQUIRE_MSG(t_dual_20 < t_dual_0, "20%% hot should speed up restore, got %.3f vs %.3f ms", t_dual_20 / 1e6,
                t_dual_0 / 1e6);

    // Data integrity on the dual path.
    {
        auto e = build(hot20);
        e->restoreDualPath(1, chunks, 0.20);
        REQUIRE_MSG(e->stats().mismatches == 0 && e->stats().errors == 0, "dual-path restore must be byte-correct");
        REQUIRE_MSG(e->stats().hot_hit_rate > 0.19 && e->stats().hot_hit_rate < 0.21,
                    "20%% hot hit rate expected, got %.2f", e->stats().hot_hit_rate);
    }

    // Placement accounting: 20% hot, 80% cold split ~50/50 across the two NVMe.
    {
        const auto &d = d20;
        uint64_t expect_hot = hot20 * 4096;
        uint64_t expect_cold = total_bytes - expect_hot;
        REQUIRE_MSG(d.hot_bytes == expect_hot, "hot bytes %llu != %llu", (unsigned long long)d.hot_bytes,
                    (unsigned long long)expect_hot);
        REQUIRE_MSG(d.nvme0_bytes + d.nvme1_bytes == expect_cold, "cold bytes %llu + %llu != %llu",
                    (unsigned long long)d.nvme0_bytes, (unsigned long long)d.nvme1_bytes,
                    (unsigned long long)expect_cold);
        // The two cold halves are balanced (off by at most one chunk).
        int64_t diff = (int64_t)d.nvme0_bytes - (int64_t)d.nvme1_bytes;
        REQUIRE_MSG(std::llabs(diff) <= 4096, "nvme0/nvme1 imbalance %lld bytes", (long long)diff);
    }

    // Pipelining: the two NVMe streams and the GPU sink overlap. Restore time is
    // bounded by the slowest stream (max), NOT the sum of the three.
    {
        const auto &d = d0;
        double sum_ms = (d.nvme0_fill_ns + d.nvme1_fill_ns + d.gpu_drain_ns) / 1e6;
        std::printf("[vhot] pipelining: restore %.3f ms << sum(nvme0+nvme1+gpu) %.3f ms\n", t_dual_0 / 1e6, sum_ms);
        REQUIRE_MSG(t_dual_0 / 1e6 < sum_ms, "restore should overlap the three streams (%.3f < %.3f ms)", t_dual_0 / 1e6,
                    sum_ms);
    }

    std::printf("PASS\n");
    return 0;
}
