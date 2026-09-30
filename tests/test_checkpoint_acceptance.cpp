/*
 * Full acceptance matrix for the target2 dual-line checkpoint engine (A1-A9).
 * Runs all nine criteria end-to-end and prints a PASS/FAIL matrix.
 *
 * A1  dual-line staging bandwidth additive (S1 single-GPU speedup)
 * A2  two lines do not share the switch uplink (per-line counters)
 * A3  multi-channel backend scales (2 storage channels > 1)
 * A4  PCIe physical layer (write ~= read * write_read_ratio)
 * A5  on-device DMA / P2P semantics (CXL line device-initiated)
 * A6  data consistency (CRC/byte)
 * A7  per-line stats
 * A8  CXL hot standby speeds restore
 * A9  pressure reuse / no unpin in flight
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace ckpt;

namespace {

constexpr uint64_t kMiB = 1024ull * 1024ull;
constexpr uint64_t kGiB = 1024ull * kMiB;

struct Result {
    std::string name;
    bool pass;
    std::string detail;
};

// A1: dual-line staging exceeds the fastest single path (S1 single GPU).
bool a1(double *dual, double *single) {
    constexpr uint64_t bytes = 512ull * kMiB;
    CheckpointConfig cfg = ckpt_test::defaultConfig(2);
    cfg.dram.num_lanes = 1;
    cfg.dram.write_gbps = 17.6;
    cfg.dram.read_gbps = 17.6;
    cfg.cxl.num_lanes = 1;
    cfg.cxl.write_gbps = 50.0; // 1 DDR on the expander
    cfg.cxl.read_gbps = 50.0;
    cfg.dram.capacity_bytes = 1 * kGiB;
    cfg.cxl.capacity_bytes = 1 * kGiB;

    CheckpointConfig single_cfg = cfg;
    single_cfg.cxl_enabled = false;

    constexpr uint64_t chunks = bytes / 4096;
    CheckpointEngine es(single_cfg);
    es.stage(0, chunks);
    *single = ckpt_test::bandwidthGbps(bytes, es.stats().gpu_stage_time_ns);

    CheckpointEngine ed(cfg);
    ed.stage(0, chunks);
    *dual = ckpt_test::bandwidthGbps(bytes, ed.stats().gpu_stage_time_ns);
    return *dual > *single * 1.05;
}

// A2: the CXL line (P2P) does not count toward the switch uplink.
bool a2(std::string *d) {
    CheckpointConfig cfg = ckpt_test::defaultConfig(2);
    cfg.dram.capacity_bytes = 1 * kGiB;
    cfg.cxl.capacity_bytes = 1 * kGiB;
    ckpt_test::disablePoolPressure(cfg);

    CheckpointEngine e(cfg);
    e.stage(0, (512ull * kMiB) / 4096);
    const auto &l = e.stats().links;
    bool dram_uplink = l[0].counts_toward_uplink && l[0].write_bytes > 0;
    bool cxl_no_uplink = !l[1].counts_toward_uplink && l[1].write_bytes > 0;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "line1 uplink=%d bytes=%llu, line2 uplink=%d bytes=%llu", l[0].counts_toward_uplink,
                  (unsigned long long)l[0].write_bytes, l[1].counts_toward_uplink, (unsigned long long)l[1].write_bytes);
    *d = buf;
    return dram_uplink && cxl_no_uplink;
}

// A3: 2 storage channels > 1 channel.
bool a3(double *bw2, double *bw1) {
    uint64_t bytes = 1024ull * 4096;
    CheckpointEngine e1(ckpt_test::defaultConfig(1));
    e1.stage(0, 1024);
    e1.persist(0);
    *bw1 = ckpt_test::bandwidthGbps(bytes, e1.stats().durable_completion_time_ns);

    CheckpointEngine e2(ckpt_test::defaultConfig(2));
    e2.stage(0, 1024);
    e2.persist(0);
    *bw2 = ckpt_test::bandwidthGbps(bytes, e2.stats().durable_completion_time_ns);
    return *bw2 > *bw1;
}

// A4: PCIe write path is ~write_read_ratio x the read path (posted vs non-posted).
bool a4(std::string *d) {
    CheckpointConfig cfg = ckpt_test::defaultConfig(2);
    cfg.line_dram.write_read_ratio = 0.65;
    double w = pcieWriteGbps(cfg.line_dram, cfg);
    double r = pcieReadGbps(cfg.line_dram, cfg);
    char buf[96];
    std::snprintf(buf, sizeof(buf), "read %.2f GB/s, write %.2f GB/s (ratio %.2f)", r, w, w / r);
    *d = buf;
    return w < r && std::fabs(w - r * 0.65) < 1e-9;
}

// A5: the CXL line is device-initiated (expander-side P2P DMA).
bool a5(std::string *d) {
    CheckpointConfig cfg = ckpt_test::defaultConfig(2);
    cfg.dram.capacity_bytes = 1 * kGiB;
    cfg.cxl.capacity_bytes = 1 * kGiB;
    ckpt_test::disablePoolPressure(cfg);

    CheckpointEngine e(cfg);
    e.stage(0, (512ull * kMiB) / 4096);
    const auto &l = e.stats().links;
    char buf[128];
    std::snprintf(buf, sizeof(buf), "cxl device=%llu host=%llu; dram host=%llu device=%llu",
                  (unsigned long long)l[1].device_initiated_bytes, (unsigned long long)l[1].host_initiated_bytes,
                  (unsigned long long)l[0].host_initiated_bytes, (unsigned long long)l[0].device_initiated_bytes);
    *d = buf;
    return l[1].device_initiated_bytes > 0 && l[1].host_initiated_bytes == 0;
}

// A6: data consistency (round trip CRC + byte).
bool a6() {
    CheckpointEngine e(ckpt_test::defaultConfig(2));
    constexpr uint64_t chunks = 256;
    e.stage(1, chunks);
    e.persist(1);
    e.commitGeneration(1);
    e.restore(1, chunks);
    return e.stats().mismatches == 0 && e.stats().errors == 0;
}

// A7: per-line + per-lane statistics populated.
bool a7(std::string *d) {
    CheckpointEngine e(ckpt_test::defaultConfig(2));
    e.stage(1, 256);
    e.persist(1);
    const auto &ls = e.stats().lanes;
    const auto &lk = e.stats().links;
    bool lanes = ls[0].completed_bytes > 0 && ls[1].completed_bytes > 0;
    bool links = lk[0].write_bytes > 0 && lk[1].write_bytes > 0;
    auto ss = e.storage().stats();
    bool per_channel = ss.size() == 2 && ss[0].bytes_written > 0 && ss[1].bytes_written > 0;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "lanes dram=%lluB cxl=%lluB; links line1=%lluB line2=%lluB; chan=%llu/%lluB",
                  (unsigned long long)ls[0].completed_bytes, (unsigned long long)ls[1].completed_bytes,
                  (unsigned long long)lk[0].write_bytes, (unsigned long long)lk[1].write_bytes,
                  (unsigned long long)ss[0].bytes_written, (unsigned long long)ss[1].bytes_written);
    *d = buf;
    return lanes && links && per_channel;
}

// A8: CXL hot standby speeds restore monotonically with hit rate.
bool a8(double *speedup) {
    constexpr uint64_t chunks = 512;
    CheckpointConfig cfg = ckpt_test::defaultConfig(2);
    cfg.hot_capacity_chunks = chunks;
    auto run = [&](uint64_t hot_count) {
        CheckpointEngine e(cfg);
        e.stage(1, chunks);
        e.persist(1);
        e.commitGeneration(1);
        for (uint64_t i = 0; i < hot_count; ++i) {
            e.promoteChunk(1, static_cast<uint32_t>(i));
        }
        return e.restore(1, chunks);
    };
    double t0 = run(0);
    double t50 = run(chunks / 2);
    double t100 = run(chunks);
    *speedup = t0 / t100;
    return t0 > t50 && t50 > t100;
}

// A9: pressure reuse (no growth) + no unpin while in flight.
bool a9() {
    CheckpointConfig cfg = ckpt_test::defaultConfig(2);
    cfg.cxl_enabled = false;
    cfg.dram.capacity_bytes = 65536;
    CheckpointEngine e(cfg);
    e.stage(0, 512);
    if (e.dramPool().state() == PoolState::NORMAL) {
        return false;
    }
    uint64_t pinned = e.dramPool().pinnedBytes();
    e.stage(1, 256);
    bool no_growth = e.dramPool().pinnedBytes() == pinned && pinned <= cfg.dram.capacity_bytes;

    CheckpointEngine e2(ckpt_test::defaultConfig(2));
    e2.stage(7, 16);
    if (e2.unpinChunk(7, 0)) {
        return false;
    }
    e2.persistBegin(7);
    if (e2.unpinChunk(7, 0)) {
        return false;
    }
    e2.persistFinish(7);
    bool unpin_ok = e2.unpinChunk(7, 0);
    return no_growth && unpin_ok;
}

} // namespace

int main() {
    std::vector<Result> results;

    double dual, single;
    bool r1 = a1(&dual, &single);
    char d1[128];
    std::snprintf(d1, sizeof(d1), "dual %.2f vs single %.2f GB/s", dual, single);
    results.push_back({"A1 dual-line staging additive", r1, d1});

    std::string d2;
    results.push_back({"A2 CXL line does not use uplink", a2(&d2), d2});

    double bw2, bw1;
    bool r3 = a3(&bw2, &bw1);
    char d3[128];
    std::snprintf(d3, sizeof(d3), "2ch %.2f vs 1ch %.2f GB/s", bw2, bw1);
    results.push_back({"A3 multi-channel backend scales", r3, d3});

    std::string d4;
    results.push_back({"A4 PCIe write/read asymmetry", a4(&d4), d4});

    std::string d5;
    results.push_back({"A5 on-device DMA (P2P)", a5(&d5), d5});

    results.push_back({"A6 data consistency", a6(), ""});

    std::string d7;
    results.push_back({"A7 per-line/per-lane stats", a7(&d7), d7});

    double speedup;
    bool r8 = a8(&speedup);
    char d8[128];
    std::snprintf(d8, sizeof(d8), "hot/cold restore speedup %.1fx", speedup);
    results.push_back({"A8 CXL hot standby speeds restore", r8, d8});

    results.push_back({"A9 pressure reuse / no unpin", a9(), ""});

    std::printf("[P acceptance matrix]\n");
    int passed = 0;
    for (const auto &r : results) {
        std::printf("  %-40s %s", r.name.c_str(), r.pass ? "PASS" : "FAIL");
        if (!r.detail.empty()) {
            std::printf("  (%s)", r.detail.c_str());
        }
        std::printf("\n");
        if (r.pass) {
            passed++;
        }
    }

    std::printf("[P acceptance matrix] %d/9 criteria PASS\n", passed);
    if (passed < 9) {
        std::fprintf(stderr, "Acceptance matrix FAILED (%d/9 passed)\n", passed);
        return 1;
    }
    return 0;
}
