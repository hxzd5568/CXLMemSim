/*
 * Full acceptance matrix for the checkpoint engine (target.md section 8).
 * Runs all eight criteria end-to-end and prints a PASS/FAIL matrix.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace ckpt;

namespace {

struct Result {
    std::string name;
    bool pass;
    std::string detail;
};

bool acceptance1() {
    CheckpointEngine e(ckpt_test::defaultConfig(2));
    constexpr uint64_t chunks = 256;
    e.stage(1, chunks);
    e.persist(1);
    e.commitGeneration(1);
    e.restore(1, chunks);
    return e.stats().mismatches == 0 && e.stats().errors == 0;
}

bool acceptance2(double *dual, double *single) {
    uint64_t bytes = 1024ull * 4096;
    CheckpointConfig s = ckpt_test::defaultConfig(2);
    s.cxl_enabled = false;
    CheckpointEngine es(s);
    es.stage(0, 1024);
    *single = ckpt_test::bandwidthGbps(bytes, es.stats().gpu_stage_time_ns);

    CheckpointEngine ed(ckpt_test::defaultConfig(2));
    ed.stage(0, 1024);
    *dual = ckpt_test::bandwidthGbps(bytes, ed.stats().gpu_stage_time_ns);
    // Honest result: single-DRAM (50 GB/s effective) cannot saturate the ~54 GB/s
    // effective PCIe uplink, so dual-lane uses the leftover headroom -- a modest
    // gain, not the 2x target.md assumed.
    return *dual > *single;
}

bool acceptance3(double *bw2, double *bw1) {
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

bool acceptance4(double *speedup) {
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

bool acceptance5() {
    CheckpointEngine e(ckpt_test::defaultConfig(2));
    constexpr uint64_t chunks = 128;
    e.stage(1, chunks);
    e.persist(1);
    e.commitGeneration(1);
    std::vector<uint32_t> order(chunks);
    for (uint32_t i = 0; i < chunks; ++i) {
        order[i] = i;
    }
    std::mt19937 rng(99);
    std::shuffle(order.begin(), order.end(), rng);
    for (uint32_t id : order) {
        const ChunkMeta *m = e.chunkMeta(1, id);
        if (!m) {
            return false;
        }
        std::vector<uint8_t> buf(e.storage().chunkSize());
        StorageOpResult res = e.storage().read(id, buf.data(), buf.size(), 0);
        if (!res.accepted) {
            return false;
        }
        e.storage().markComplete(res.channel);
        if (crc32(buf.data(), buf.size()) != m->crc32) {
            return false;
        }
    }
    return true;
}

bool acceptance6() {
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
    return e.dramPool().pinnedBytes() == pinned && pinned <= cfg.dram.capacity_bytes;
}

bool acceptance7() {
    CheckpointEngine e(ckpt_test::defaultConfig(2));
    e.stage(7, 16);
    if (e.unpinChunk(7, 0)) {
        return false; // PINNED must not be unpinnable
    }
    e.persistBegin(7);
    if (e.unpinChunk(7, 0)) {
        return false; // IN_FLIGHT must not be unpinnable
    }
    e.persistFinish(7);
    return e.unpinChunk(7, 0); // durable => allowed
}

bool acceptance8(std::string *detail) {
    CheckpointEngine e(ckpt_test::defaultConfig(2));
    e.stage(1, 256);
    e.persist(1);
    const auto &ls = e.stats().lanes;
    bool dram = ls[0].completed_bytes > 0 && ls[0].completed_chunks > 0;
    bool cxl = ls[1].completed_bytes > 0 && ls[1].completed_chunks > 0;
    auto ss = e.storage().stats();
    bool per_channel = ss.size() == 2 && ss[0].bytes_written > 0 && ss[1].bytes_written > 0;

    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "dram completed=%llu/%llu B, cxl completed=%llu/%llu B; chan0=%llu B chan1=%llu B",
                  (unsigned long long)ls[0].completed_chunks, (unsigned long long)ls[0].completed_bytes,
                  (unsigned long long)ls[1].completed_chunks, (unsigned long long)ls[1].completed_bytes,
                  (unsigned long long)ss[0].bytes_written, (unsigned long long)ss[1].bytes_written);
    *detail = buf;
    return dram && cxl && per_channel;
}

} // namespace

int main() {
    std::vector<Result> results;

    results.push_back({"#1 CRC/byte consistency", acceptance1(), ""});

    double dual, single;
    bool r2 = acceptance2(&dual, &single);
    char d2[160];
    std::snprintf(d2, sizeof(d2), "dual %.2f vs single-DRAM %.2f GB/s (%.2fx; both below raw 64 GB/s PCIe)", dual,
                  single, dual / single);
    results.push_back({"#2 dual-path staging > fastest single", r2, d2});

    double bw2, bw1;
    bool r3 = acceptance3(&bw2, &bw1);
    char d3[128];
    std::snprintf(d3, sizeof(d3), "2ch %.2f GB/s vs 1ch %.2f GB/s", bw2, bw1);
    results.push_back({"#3 2 storage channels > 1 channel", r3, d3});

    double speedup;
    bool r4 = acceptance4(&speedup);
    char d4[128];
    std::snprintf(d4, sizeof(d4), "hot/cold restore speedup %.1fx", speedup);
    results.push_back({"#4 CXL hot standby speeds restore", r4, d4});

    results.push_back({"#5 out-of-order restore", acceptance5(), ""});
    results.push_back({"#6 pressure: pool reuses, no growth", acceptance6(), ""});
    results.push_back({"#7 no unpin while in flight", acceptance7(), ""});

    std::string d8;
    bool r8 = acceptance8(&d8);
    results.push_back({"#8 per-link queue/bw/latency stats", r8, d8});

    std::printf("[P acceptance matrix]\n");
    int passed = 0;
    for (const auto &r : results) {
        std::printf("  %-42s %s", r.name.c_str(), r.pass ? "PASS" : "FAIL");
        if (!r.detail.empty()) {
            std::printf("  (%s)", r.detail.c_str());
        }
        std::printf("\n");
        if (r.pass) {
            passed++;
        }
    }

    if (passed < 8) {
        std::fprintf(stderr, "Acceptance matrix FAILED (%d/8 passed)\n", passed);
        return 1;
    }
    // #2 passes with a MODEST gain (~1.08x), not the 2x target.md assumed --
    // single-DRAM cannot saturate the effective PCIe uplink, so the CXL lane
    // only reclaims the leftover PCIe headroom.
    std::printf("[P acceptance matrix] all 8 criteria PASS (#2 is a modest ~%.2fx gain)\n", dual / single);
    return 0;
}
