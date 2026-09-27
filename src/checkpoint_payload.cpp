/*
 * Checkpoint payload generator and data-integrity primitives for CXLMemSim.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_payload.h"

#include <cstdio>

namespace ckpt {

namespace {

// MurmurHash3-style finalizer, used to mix the (checkpoint_id, chunk_id, index)
// tuple into a single 64-bit seed.
uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

constexpr uint32_t kCrc32Polynomial = 0xEDB88320u;

uint32_t g_crc_table[256];
bool g_crc_table_ready = false;

void ensureCrcTable() {
    if (g_crc_table_ready) {
        return;
    }
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int k = 0; k < 8; ++k) {
            crc = (crc >> 1) ^ (kCrc32Polynomial & (0u - (crc & 1u)));
        }
        g_crc_table[i] = crc;
    }
    g_crc_table_ready = true;
}

// SHA-256 round constants and helper functions.
constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

} // namespace

uint32_t payloadElement(uint64_t checkpoint_id, uint32_t chunk_id, uint64_t index) {
    // Derive a per-element 64-bit state so adjacent elements decorrelate.
    uint64_t state = mix64(checkpoint_id) ^ mix64(static_cast<uint64_t>(chunk_id) + 0x9e3779b97f4a7c15ULL) ^
                     mix64(index);
    state ^= 0xa5a5a5a5a5a5a5a5ULL;
    uint64_t mixed = splitmix64(state);
    return static_cast<uint32_t>(mixed ^ (mixed >> 32));
}

void payloadFill(uint64_t checkpoint_id, uint32_t chunk_id, uint8_t *dst, size_t size, uint64_t offset) {
    size_t i = 0;
    // Fill 4-byte words where possible.
    while (i + 4 <= size) {
        uint32_t e = payloadElement(checkpoint_id, chunk_id, offset + i);
        dst[i + 0] = static_cast<uint8_t>(e & 0xff);
        dst[i + 1] = static_cast<uint8_t>((e >> 8) & 0xff);
        dst[i + 2] = static_cast<uint8_t>((e >> 16) & 0xff);
        dst[i + 3] = static_cast<uint8_t>((e >> 24) & 0xff);
        i += 4;
    }
    while (i < size) {
        uint32_t e = payloadElement(checkpoint_id, chunk_id, offset + i);
        dst[i] = static_cast<uint8_t>(e & 0xff);
        ++i;
    }
}

bool payloadVerify(uint64_t checkpoint_id, uint32_t chunk_id, const uint8_t *src, size_t size, uint64_t offset,
                   uint64_t *mismatch_offset) {
    // Matches payloadFill: each 4-byte little-endian word is generated from one
    // element payloadElement(checkpoint_id, chunk_id, word_index).
    for (size_t i = 0; i < size; ++i) {
        uint64_t pos = offset + i;
        uint32_t e = payloadElement(checkpoint_id, chunk_id, pos - (pos % 4));
        uint8_t expected = static_cast<uint8_t>((e >> (8 * (pos % 4))) & 0xff);
        if (src[i] != expected) {
            if (mismatch_offset) {
                *mismatch_offset = pos;
            }
            return false;
        }
    }
    return true;
}

uint32_t crc32Update(uint32_t crc, const uint8_t *data, size_t size) {
    ensureCrcTable();
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) {
        crc = g_crc_table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    }
    return ~crc;
}

uint32_t crc32(const uint8_t *data, size_t size) { return crc32Update(0u, data, size); }

bool Sha256Digest::operator==(const Sha256Digest &other) const {
    for (int i = 0; i < 32; ++i) {
        if (bytes[i] != other.bytes[i]) {
            return false;
        }
    }
    return true;
}

std::string Sha256Digest::hex() const {
    static const char *digits = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (int i = 0; i < 32; ++i) {
        out.push_back(digits[(bytes[i] >> 4) & 0xf]);
        out.push_back(digits[bytes[i] & 0xf]);
    }
    return out;
}

Sha256Digest sha256(const uint8_t *data, size_t size) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    uint64_t bit_len = static_cast<uint64_t>(size) * 8;
    // Padding: 0x80, zeros, then 64-bit big-endian length. Total multiple of 64.
    uint64_t padded = size + 1;
    while (padded % 64 != 56) {
        padded++;
    }
    padded += 8;

    std::vector<uint8_t> msg(padded, 0);
    for (size_t i = 0; i < size; ++i) {
        msg[i] = data[i];
    }
    msg[size] = 0x80;
    for (int i = 0; i < 8; ++i) {
        msg[padded - 1 - i] = static_cast<uint8_t>((bit_len >> (8 * i)) & 0xff);
    }

    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<uint32_t>(msg[off + i * 4]) << 24) | (static_cast<uint32_t>(msg[off + i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(msg[off + i * 4 + 2]) << 8) | static_cast<uint32_t>(msg[off + i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = hh + S1 + ch + kSha256K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    Sha256Digest out;
    for (int i = 0; i < 8; ++i) {
        out.bytes[i * 4 + 0] = static_cast<uint8_t>((h[i] >> 24) & 0xff);
        out.bytes[i * 4 + 1] = static_cast<uint8_t>((h[i] >> 16) & 0xff);
        out.bytes[i * 4 + 2] = static_cast<uint8_t>((h[i] >> 8) & 0xff);
        out.bytes[i * 4 + 3] = static_cast<uint8_t>(h[i] & 0xff);
    }
    return out;
}

Sha256Digest sha256(const std::vector<uint8_t> &data) { return sha256(data.data(), data.size()); }

} // namespace ckpt
