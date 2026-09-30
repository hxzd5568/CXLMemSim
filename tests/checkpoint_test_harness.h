/*
 * Shared harness for CXLMemSim checkpoint-engine tests.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#ifndef CXLMEMSIM_TESTS_CHECKPOINT_TEST_HARNESS_H
#define CXLMEMSIM_TESTS_CHECKPOINT_TEST_HARNESS_H

#include "checkpoint_engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#define REQUIRE(condition)                                                                                             \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::cerr << "Requirement failed: " #condition << "\n";                                                    \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

#define REQUIRE_MSG(condition, fmt, ...)                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::fprintf(stderr, "Requirement failed: %s (" fmt ")\n", #condition, ##__VA_ARGS__);                     \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace ckpt_test {

constexpr uint64_t kMiB = 1024 * 1024;

// Example target2 (dual-line) configuration for the acceptance tests. Bandwidths
// are CALIBRATION POINTS (PER-LANE raw link/channel rates; the CXL backend is raw
// DDR bandwidth -- NO flit, see cxlOverheadFactor -> 1.0).
//  - Line 1 (DRAM): Gen5 x16 uplink raw 64 GB/s -> ~55.5 GB/s effective (128b/130b,
//    MPS 256 B, header+LCRC 20 B, ~5% residual).
//  - Line 2 (CXL): Gen5 x16 P2P raw 64 GB/s -> ~55.5 GB/s effective, does NOT
//    consume the uplink, device-initiated (expander-side DMA).
//  - GPU egress: per-GPU Gen4 x16 raw 32 GB/s -> ~27.75 GB/s effective.
//  - DRAM: 8 sub-channels (4 DIMM x 2) x 17.6 GB/s => ~140.8 GB/s.
//  - CXL:  8 DDR x 50 GB/s = 400 GB/s (raw DDR, no flit).
//  - storage 2 GB/s/channel with 10 us flash latency (persist bottleneck).
inline ckpt::CheckpointConfig defaultConfig(uint32_t num_channels = 2) {
    ckpt::CheckpointConfig cfg;
    cfg.chunk_size = 4096;
    cfg.num_storage_channels = num_channels;

    // target2 dual-line links (E0/E3): BOTH Gen5 x16.
    cfg.line_dram.raw_gbps = 64.0;       // Gen5 x16 uplink
    cfg.line_dram.consumes_uplink = true;
    cfg.line_dram.device_initiated = false;
    cfg.line_dram.write_read_ratio = 1.0;
    cfg.line_cxl.raw_gbps = 64.0;        // Gen5 x16 P2P (expander direct-attach)
    cfg.line_cxl.consumes_uplink = false;
    cfg.line_cxl.device_initiated = true;
    cfg.line_cxl.write_read_ratio = 1.0;
    cfg.num_gpus = 1;
    cfg.gpu_egress_gbps = 32.0;          // per-GPU Gen4 x16 raw egress

    cfg.split_unit_bytes = 256u * 1024u * 1024u; // 256 MiB memcpy split unit

    cfg.dram.num_lanes = 8;   // 4 DIMM x 2 DDR5 sub-channels
    cfg.dram.read_gbps = 17.6; // per sub-channel (DDR5-4400 32-bit)
    cfg.dram.write_gbps = 17.6;
    cfg.dram.read_latency_ns = 100.0;
    cfg.dram.write_latency_ns = 100.0;
    cfg.dram.capacity_bytes = 64 * kMiB;

    cfg.cxl.num_lanes = 8;     // 8 DDR on the CXL expander
    cfg.cxl.read_gbps = 50.0;  // per-DDR bandwidth (raw, no flit)
    cfg.cxl.write_gbps = 50.0;
    cfg.cxl.read_latency_ns = 400.0;
    cfg.cxl.write_latency_ns = 400.0;
    cfg.cxl.capacity_bytes = 64 * kMiB;

    cfg.storage_channel.read_gbps = 2.0;
    cfg.storage_channel.write_gbps = 2.0;
    cfg.storage_channel.base_latency_ns = 10000.0;
    cfg.storage_channel.max_outstanding = 32;
    cfg.storage_capacity_per_channel_bytes = 16 * kMiB;

    cfg.hot_capacity_chunks = 0;
    cfg.hot_read_latency_ns = 400.0;
    return cfg;
}

// Bandwidth of N chunks (bytes) completed in `time_ns`.
inline double bandwidthGbps(uint64_t bytes, double time_ns) {
    if (time_ns <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bytes) / time_ns; // bytes/ns == GB/s
}

// Disable the pinned-pool pressure state machine so a bandwidth test can fill the
// pool to capacity without the FROZEN/SHRINK/STOPPED thresholds stopping new pins.
inline void disablePoolPressure(ckpt::CheckpointConfig &cfg) {
    cfg.dram.frozen_threshold = 1.5;
    cfg.dram.shrink_threshold = 1.5;
    cfg.dram.stop_threshold = 1.5;
    cfg.cxl.frozen_threshold = 1.5;
    cfg.cxl.shrink_threshold = 1.5;
    cfg.cxl.stop_threshold = 1.5;
}

inline void requireClose(const char *label, double got, double want, double tol) {
    if (std::fabs(got - want) > tol) {
        std::cerr << label << ": got " << got << " want " << want << " (tol " << tol << ")\n";
        std::exit(1);
    }
}

} // namespace ckpt_test

#endif // CXLMEMSIM_TESTS_CHECKPOINT_TEST_HARNESS_H
