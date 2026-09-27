/*
 * Checkpoint payload generator and data-integrity primitives for CXLMemSim.
 *
 * Models the "GPU checkpoint source" described in target.md: instead of
 * counting bytes, the engine produces a deterministic, reproducible payload
 *
 *     payload[i] = PRNG(checkpoint_id, chunk_id, i)
 *
 * so that save/restore can be verified byte-for-byte (or via per-chunk CRC32
 * plus a whole-checkpoint SHA-256) without keeping a reference copy in memory.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#ifndef CXLMEMSIM_CHECKPOINT_PAYLOAD_H
#define CXLMEMSIM_CHECKPOINT_PAYLOAD_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ckpt {

/*
 * Deterministic splitmix64-style hash. Gives a stable 32-bit element value for
 * a (checkpoint_id, chunk_id, byte_index) triple.
 */
uint32_t payloadElement(uint64_t checkpoint_id, uint32_t chunk_id, uint64_t index);

// Fill dst[0..size) with payloadElement(checkpoint_id, chunk_id, offset + i).
void payloadFill(uint64_t checkpoint_id, uint32_t chunk_id, uint8_t *dst, size_t size, uint64_t offset = 0);

// Verify a buffer against the deterministic payload. Returns false on the first
// mismatching byte (also reports the byte offset through `mismatch_offset`).
bool payloadVerify(uint64_t checkpoint_id, uint32_t chunk_id, const uint8_t *src, size_t size, uint64_t offset = 0,
                   uint64_t *mismatch_offset = nullptr);

// IEEE 802.3 CRC-32 (same polynomial/init as zlib's crc32).
uint32_t crc32(const uint8_t *data, size_t size);
uint32_t crc32Update(uint32_t crc, const uint8_t *data, size_t size);

// FIPS 180-4 SHA-256 digest.
struct Sha256Digest {
    uint8_t bytes[32];

    bool operator==(const Sha256Digest &other) const;
    bool operator!=(const Sha256Digest &other) const { return !(*this == other); }
    std::string hex() const;
};

Sha256Digest sha256(const uint8_t *data, size_t size);
// Convenience overload for byte vectors.
Sha256Digest sha256(const std::vector<uint8_t> &data);

} // namespace ckpt

#endif // CXLMEMSIM_CHECKPOINT_PAYLOAD_H
