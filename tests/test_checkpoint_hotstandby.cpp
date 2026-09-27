/*
 * CXL hot standby. Covers acceptance #4: restore time falls monotonically as the
 * hot-standby hit rate rises (0/25/50/100%).
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <cstdio>
#include <utility>

using namespace ckpt;

int main() {
    constexpr uint64_t chunks = 512;
    CheckpointConfig cfg = ckpt_test::defaultConfig(2);
    cfg.hot_capacity_chunks = chunks;

    // Run an independent save/restore for each hot fraction (fresh hot cache).
    auto run = [&](uint64_t hot_count) {
        CheckpointEngine e(cfg);
        e.stage(1, chunks);
        e.persist(1);
        e.commitGeneration(1);
        for (uint64_t i = 0; i < hot_count; ++i) {
            e.promoteChunk(1, static_cast<uint32_t>(i));
        }
        double t = e.restore(1, chunks);
        return std::make_pair(t, e.stats().hot_hit_rate);
    };

    auto [t0, hr0] = run(0);
    auto [t25, hr25] = run(chunks / 4);
    auto [t50, hr50] = run(chunks / 2);
    auto [t100, hr100] = run(chunks);

    std::printf("[hot   0%%] restore %.3f ms (hit %.2f)\n", t0 / 1e6, hr0);
    std::printf("[hot  25%%] restore %.3f ms (hit %.2f)\n", t25 / 1e6, hr25);
    std::printf("[hot  50%%] restore %.3f ms (hit %.2f)\n", t50 / 1e6, hr50);
    std::printf("[hot 100%%] restore %.3f ms (hit %.2f)\n", t100 / 1e6, hr100);

    // Restore time must fall monotonically with hit rate (acceptance #4).
    REQUIRE(t0 > t25);
    REQUIRE(t25 > t50);
    REQUIRE(t50 > t100);

    // Hit rate must track the promoted fraction.
    REQUIRE_MSG(hr0 == 0.0, "0%% should have zero hits, got %.2f", hr0);
    REQUIRE_MSG(hr100 > 0.99, "100%% should be all hits, got %.2f", hr100);

    std::printf("[hot] 100%% hit is %.1fx faster than 0%% hit (OK)\n", t0 / t100);

    std::printf("PASS\n");
    return 0;
}
