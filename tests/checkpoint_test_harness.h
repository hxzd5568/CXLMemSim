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

// A realistic single-node configuration for the acceptance tests.
// Bandwidths are EFFECTIVE (post-overhead) values:
//  - PCIe Gen5 x16 raw 64 GB/s, but TLP headers + 128b/130b encoding + DLLP/
//    flow-control leave ~85% usable => effective uplink ~54 GB/s.
//  - DRAM 50 GB/s: GPU DMA writes traverse root complex + coherency + memory
//    controller, so the effective write bandwidth is well below raw DDR5
//    (~78% of raw PCIe; it cannot saturate the link by itself).
//  - CXL.mem 28 GB/s: Gen4 x16 (~32 raw) with CXL link overhead.
//  - storage 2 GB/s/channel with 10 us flash latency (persist bottleneck).
inline ckpt::CheckpointConfig defaultConfig(uint32_t num_channels = 2) {
    ckpt::CheckpointConfig cfg;
    cfg.chunk_size = 4096;
    cfg.num_storage_channels = num_channels;

    cfg.gpu_pcie_write_gbps = 64.0; // PCIe Gen5 x16 raw
    cfg.gpu_pcie_read_gbps = 64.0;
    cfg.pcie_efficiency = 0.85;     // ~54 GB/s usable payload

    cfg.dram.read_gbps = 50.0;
    cfg.dram.write_gbps = 50.0;
    cfg.dram.read_latency_ns = 100.0;
    cfg.dram.write_latency_ns = 100.0;
    cfg.dram.capacity_bytes = 16 * kMiB; // 4096 chunks

    cfg.cxl.read_gbps = 28.0;
    cfg.cxl.write_gbps = 28.0;
    cfg.cxl.read_latency_ns = 400.0;
    cfg.cxl.write_latency_ns = 400.0;
    cfg.cxl.capacity_bytes = 64 * kMiB; // 16384 chunks

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

inline void requireClose(const char *label, double got, double want, double tol) {
    if (std::fabs(got - want) > tol) {
        std::cerr << label << ": got " << got << " want " << want << " (tol " << tol << ")\n";
        std::exit(1);
    }
}

} // namespace ckpt_test

#endif // CXLMEMSIM_TESTS_CHECKPOINT_TEST_HARNESS_H
