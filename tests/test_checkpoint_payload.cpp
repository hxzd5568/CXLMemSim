/*
 * Checkpoint payload primitives: CRC32 + SHA-256 test vectors + determinism.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_test_harness.h"

#include "checkpoint_payload.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace ckpt;

int main() {
    // CRC-32 known-answer (matches zlib crc32 of "123456789").
    const char *s = "123456789";
    REQUIRE(crc32(reinterpret_cast<const uint8_t *>(s), 9) == 0xCBF43926u);

    // SHA-256 known-answer tests (FIPS 180-4).
    Sha256Digest empty = sha256(nullptr, 0);
    REQUIRE(empty.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    Sha256Digest abc = sha256(reinterpret_cast<const uint8_t *>("abc"), 3);
    REQUIRE(abc.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // Determinism: same (checkpoint_id, chunk_id) regenerates identical bytes.
    std::vector<uint8_t> a(1024), b(1024);
    payloadFill(42, 7, a.data(), a.size());
    payloadFill(42, 7, b.data(), b.size());
    REQUIRE(std::memcmp(a.data(), b.data(), a.size()) == 0);

    // Different chunk ids decorrelate.
    std::vector<uint8_t> c(1024);
    payloadFill(42, 8, c.data(), c.size());
    REQUIRE(std::memcmp(a.data(), c.data(), a.size()) != 0);

    // Verify round-trip.
    uint64_t mm = 0;
    REQUIRE(payloadVerify(42, 7, a.data(), a.size(), 0, &mm));
    a[123] ^= 0xff;
    REQUIRE(!payloadVerify(42, 7, a.data(), a.size(), 0, &mm));
    REQUIRE(mm == 123);

    std::printf("PASS\n");
    return 0;
}
