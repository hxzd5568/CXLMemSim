/*
 * ParallelStorage: multi-channel striping + bandwidth scaling + backpressure.
 * Covers acceptance #3 (two channels > one channel) and the storage half of
 * acceptance #8 (per-channel bytes/ops/reject stats).
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include "parallel_storage.h"

#include <cstdio>
#include <vector>

using namespace ckpt;

int main() {
    // --- #3: persist throughput with 1 vs 2 storage channels ---
    {
        constexpr uint64_t chunks = 1024;
        uint64_t bytes = chunks * 4096;

        CheckpointEngine e1(ckpt_test::defaultConfig(1));
        e1.stage(0, chunks);
        e1.persist(0);
        double bw1 = ckpt_test::bandwidthGbps(bytes, e1.stats().durable_completion_time_ns);

        CheckpointEngine e2(ckpt_test::defaultConfig(2));
        e2.stage(0, chunks);
        e2.persist(0);
        double bw2 = ckpt_test::bandwidthGbps(bytes, e2.stats().durable_completion_time_ns);

        REQUIRE_MSG(bw2 > bw1, "2ch persist %.2f GB/s should exceed 1ch %.2f GB/s", bw2, bw1);
        std::printf("[storage] 1ch persist %.2f GB/s < 2ch persist %.2f GB/s (OK)\n", bw1, bw2);
    }

    // --- striping: channel = chunk_id % N, slot = chunk_id / N ---
    {
        ParallelStorageConfig sc;
        sc.num_channels = 4;
        sc.chunk_size = 4096;
        sc.channel.read_gbps = 2.0;
        sc.channel.write_gbps = 2.0;
        sc.channel.base_latency_ns = 10000.0;
        sc.channel.max_outstanding = 32;
        sc.capacity_per_channel_bytes = 1024 * 1024;

        ParallelStorage ps(sc);
        for (uint32_t c = 0; c < 8; ++c) {
            std::vector<uint8_t> buf(sc.chunk_size);
            for (size_t i = 0; i < sc.chunk_size; ++i) {
                buf[i] = static_cast<uint8_t>(c * 7 + i);
            }
            ps.write(c, buf.data(), sc.chunk_size, 0);
            REQUIRE(ps.stripeChannel(c) == c % 4);
            REQUIRE(ps.channelSlot(c) == c / 4);
        }
        // Verify bytes landed on the correct channel/slot.
        for (uint32_t c : {0u, 4u}) {
            const uint8_t *p = ps.peek(c, 0);
            REQUIRE(p != nullptr);
            bool ok = true;
            for (size_t i = 0; i < sc.chunk_size; ++i) {
                if (p[i] != static_cast<uint8_t>(c * 7 + i)) {
                    ok = false;
                    break;
                }
            }
            REQUIRE(ok);
        }
        // Each of 4 channels got exactly 2 chunks (balanced striping).
        auto st = ps.stats();
        REQUIRE(st.size() == 4);
        for (const auto &s : st) {
            REQUIRE(s.bytes_written == 2 * sc.chunk_size);
            REQUIRE(s.num_writes == 2);
        }
        std::printf("[storage] 4-channel striping balance + byte placement OK\n");
    }

    // --- backpressure: queue depth exceeded -> reject (acceptance #8) ---
    {
        ParallelStorageConfig sc;
        sc.num_channels = 1;
        sc.chunk_size = 4096;
        sc.channel.max_outstanding = 2;
        sc.channel.read_gbps = 2.0;
        sc.channel.write_gbps = 2.0;
        sc.channel.base_latency_ns = 10000.0;
        sc.capacity_per_channel_bytes = 1024 * 1024;

        ParallelStorage ps(sc);
        std::vector<uint8_t> buf(sc.chunk_size, 0xAA);
        int accepted = 0, rejected = 0;
        for (uint32_t c = 0; c < 5; ++c) {
            auto r = ps.write(c, buf.data(), sc.chunk_size, 0);
            if (r.accepted) {
                accepted++;
            } else {
                rejected++;
            }
        }
        REQUIRE(accepted == 2); // max_outstanding == 2
        REQUIRE(rejected == 3);
        REQUIRE(ps.stats()[0].rejected == 3);
        std::printf("[storage] queue-depth backpressure OK (accepted=%d rejected=%d)\n", accepted, rejected);
    }

    std::printf("PASS\n");
    return 0;
}
