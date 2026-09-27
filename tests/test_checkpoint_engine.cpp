/*
 * Checkpoint engine: full round-trip consistency, out-of-order restore, and
 * the in-flight / unpin safety rule. Covers acceptance #1 (CRC/byte
 * consistency), #5 (out-of-order completion recovery), and #7 (no unpin while
 * DMA is in flight), plus generation rollback on fault.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using namespace ckpt;

int main() {
    // --- #1/#5: round trip + out-of-order restore correctness ---
    {
        CheckpointEngine e(ckpt_test::defaultConfig(2));
        constexpr uint64_t chunks = 256;

        e.stage(1, chunks);
        e.persist(1);
        e.commitGeneration(1);
        e.restore(1, chunks);

        REQUIRE(e.stats().mismatches == 0);
        REQUIRE(e.stats().completed_chunks == chunks);
        REQUIRE(e.stats().errors == 0);

        // Out-of-order: read every chunk in a shuffled order and verify each
        // per-chunk CRC against the manifest (completion matched by chunk_id).
        std::vector<uint32_t> order(chunks);
        for (uint32_t i = 0; i < chunks; ++i) {
            order[i] = i;
        }
        std::mt19937 rng(12345);
        std::shuffle(order.begin(), order.end(), rng);

        for (uint32_t id : order) {
            const ChunkMeta *m = e.chunkMeta(1, id);
            REQUIRE(m != nullptr);
            REQUIRE(m->state == ChunkState::DISK_COMMITTED);
            std::vector<uint8_t> buf(e.storage().chunkSize());
            StorageOpResult r = e.storage().read(id, buf.data(), buf.size(), 0);
            REQUIRE(r.accepted);
            e.storage().markComplete(r.channel);
            REQUIRE(crc32(buf.data(), buf.size()) == m->crc32);
        }
        std::printf("[roundtrip] save/restore CRC + byte consistency OK (%llu chunks, out-of-order OK)\n",
                    (unsigned long long)chunks);
    }

    // --- #7: in-flight DMA forbids unpin ---
    {
        CheckpointEngine e(ckpt_test::defaultConfig(2));
        constexpr uint64_t chunks = 64;

        e.stage(7, chunks);
        // Staged but not durable => unpin forbidden.
        REQUIRE(e.chunkMeta(7, 0)->state == ChunkState::PINNED);
        REQUIRE(!e.unpinChunk(7, 0));

        e.persistBegin(7);
        REQUIRE(e.chunkMeta(7, 0)->state == ChunkState::IN_FLIGHT);
        REQUIRE(!e.unpinChunk(7, 0)); // still forbidden while in flight

        e.persistFinish(7);
        REQUIRE(e.chunkMeta(7, 0)->state == ChunkState::DISK_COMMITTED);
        REQUIRE(e.unpinChunk(7, 0)); // durable => allowed
        REQUIRE(e.chunkMeta(7, 0)->state == ChunkState::FREE);
        std::printf("[lifecycle] unpin forbidden until durable (OK)\n");
    }

    // --- fault injection + generation rollback ---
    {
        CheckpointEngine e(ckpt_test::defaultConfig(2));
        constexpr uint64_t chunks = 128;

        e.stage(1, chunks);
        e.persist(1);
        e.commitGeneration(1);
        REQUIRE(e.committedGeneration() == 1);

        // gen 2 crashes after staging only half its chunks.
        e.stage(2, chunks / 2);
        e.rollback(2); // discard incomplete generation
        REQUIRE(e.committedGeneration() == 1);
        REQUIRE(e.chunkMeta(2, 0)->state == ChunkState::FREE);

        // gen 1 still restores byte-correctly.
        e.restore(1, chunks);
        REQUIRE(e.stats().mismatches == 0);
        std::printf("[fault] incomplete generation discarded, rollback to gen 1 OK\n");
    }

    std::printf("PASS\n");
    return 0;
}
