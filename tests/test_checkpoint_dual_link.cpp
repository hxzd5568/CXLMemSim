/*
 * target2 S1/S2 acceptance scenarios (acceptance A1).
 *
 * S1 (single GPU, 1 DRAM channel + 1 CXL lane): the two independent lines let a
 * single GPU saturate its Gen4 x16 egress (~27.75 GB/s effective) instead of
 * being stuck at the ~17.6 GB/s single-channel backend limit.
 *
 * S2 (4 GPUs, DRAM 4 DIMM + 2 CXL lanes): the CXL line is switch-local P2P and
 * does not consume the Gen5 uplink, so dual-line staging exceeds the DRAM-only
 * uplink bound (~55.5 GB/s).
 *
 * Both assert speedup > 1.3 (section 8 criterion).
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
    return ckpt_test::bandwidthGbps(chunks * cfg.chunk_size, e.stats().gpu_stage_time_ns);
}
} // namespace

int main() {
    // ---------------- S1: single GPU dual-path (1 DRAM ch + 1 CXL DDR) ---------
    {
        CheckpointConfig cfg = ckpt_test::defaultConfig(2);
        cfg.num_gpus = 1;
        cfg.dram.num_lanes = 1;
        cfg.dram.write_gbps = 17.6;
        cfg.dram.read_gbps = 17.6;
        cfg.cxl.num_lanes = 1;
        cfg.cxl.write_gbps = 50.0; // 1 DDR on the expander
        cfg.cxl.read_gbps = 50.0;
        cfg.dram.capacity_bytes = 1 * kGiB;
        cfg.cxl.capacity_bytes = 1 * kGiB;
        ckpt_test::disablePoolPressure(cfg);

        CheckpointConfig single = cfg;
        single.cxl_enabled = false;

        constexpr uint64_t chunks = (512ull * kMiB) / 4096;
        double bw_single = measure(single, chunks);
        double bw_dual = measure(cfg, chunks);
        double speedup = bw_dual / bw_single;

        std::printf("[S1] single GPU: B_single %.2f GB/s, B_dual %.2f GB/s, speedup %.2fx (bound dual %.2f)\n", bw_single,
                    bw_dual, speedup, computeBounds(cfg).bstage_gbps);
        REQUIRE_MSG(speedup > 1.3, "S1 speedup %.2f should exceed 1.3", speedup);
    }

    // -------- S2: 4 GPUs dual-path (DRAM 4 DIMM + CXL 8 DDR) ------------------
    {
        CheckpointConfig cfg = ckpt_test::defaultConfig(2);
        cfg.num_gpus = 4;
        cfg.split_unit_bytes = 64 * kMiB; // finer split => smaller quantization error
        cfg.dram.capacity_bytes = 2 * kGiB;
        cfg.cxl.capacity_bytes = 2 * kGiB;
        ckpt_test::disablePoolPressure(cfg);

        CheckpointConfig dram_only = cfg;
        dram_only.cxl_enabled = false;

        constexpr uint64_t chunks = (2ull * kGiB) / 4096;
        double bw_dram = measure(dram_only, chunks);
        double bw_dual = measure(cfg, chunks);
        double speedup = bw_dual / bw_dram;

        std::printf("[S2] 4 GPUs: B_dram_only %.2f GB/s, B_dual %.2f GB/s, speedup %.2fx (bound dual %.2f)\n", bw_dram,
                    bw_dual, speedup, computeBounds(cfg).bstage_gbps);
        REQUIRE_MSG(speedup > 1.3, "S2 speedup %.2f should exceed 1.3", speedup);
    }

    std::printf("PASS\n");
    return 0;
}
