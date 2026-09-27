/*
 * Multi-channel ParallelStorage backend for CXLMemSim.
 *
 * Models a logical checkpoint namespace striped across N independent flash
 * channels. Each channel has its own read/write bandwidth, base flash latency,
 * queue depth (max outstanding), and a byte-addressable backing store, so the
 * simulator can reproduce the "two channels > one channel" bandwidth scaling
 * and per-channel queue/backpressure behavior described in target.md.
 *
 * Striping (matches SimCXL ParallelStorage::mapAddr and target.md):
 *     channel        = chunk_id % num_channels
 *     channel_offset = (chunk_id / num_channels) * chunk_size + offset%chunk_size
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#ifndef CXLMEMSIM_PARALLEL_STORAGE_H
#define CXLMEMSIM_PARALLEL_STORAGE_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace ckpt {

struct StorageChannelConfig {
    double read_gbps = 1.0;  // GB/s (1 GB/s == 1 byte/ns)
    double write_gbps = 1.0; // GB/s
    double base_latency_ns = 10000.0; // whole-page flash latency (e.g. 10us)
    uint32_t max_outstanding = 32;     // per-channel queue depth
};

struct ParallelStorageConfig {
    uint32_t num_channels = 2;
    uint32_t chunk_size = 4096;        // bytes
    StorageChannelConfig channel{};    // shared per-channel parameters
    uint64_t capacity_per_channel_bytes = 0; // backing capacity per channel
};

struct StorageOpResult {
    bool accepted = false;     // false => channel queue full (backpressure)
    uint64_t complete_time_ns = 0;
    uint32_t channel = 0;
    uint64_t channel_offset = 0;
};

struct StorageChannelStats {
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    uint64_t num_reads = 0;
    uint64_t num_writes = 0;
    uint64_t rejected = 0;   // ops dropped due to queue-full
    uint64_t retries = 0;    // caller-side retries after rejection
    double latency_sum_ns = 0.0;
    std::vector<uint64_t> latencies; // completed op latencies for P95
};

class ParallelStorage {
public:
    explicit ParallelStorage(ParallelStorageConfig config);

    uint32_t numChannels() const { return config_.num_channels; }
    uint32_t chunkSize() const { return config_.chunk_size; }

    // Static stripe mapping, independent of instance timing.
    uint32_t stripeChannel(uint64_t chunk_id) const { return static_cast<uint32_t>(chunk_id % config_.num_channels); }
    uint64_t channelSlot(uint64_t chunk_id) const { return chunk_id / config_.num_channels; }
    uint64_t channelOffset(uint64_t chunk_id, uint64_t offset_within_chunk) const {
        return channelSlot(chunk_id) * config_.chunk_size + (offset_within_chunk % config_.chunk_size);
    }

    // Timing-modelled data movement. `now_ns` is the issue time.
    StorageOpResult write(uint64_t chunk_id, const uint8_t *data, size_t size, uint64_t now_ns);
    StorageOpResult read(uint64_t chunk_id, uint8_t *dst, size_t size, uint64_t now_ns);

    // Release one outstanding slot on a channel (completion accounting).
    void markComplete(uint32_t channel);

    // Direct backing-store access (no timing) for correctness checks.
    const uint8_t *peek(uint64_t chunk_id, uint64_t offset_within_chunk) const;
    uint8_t *mutablePeek(uint64_t chunk_id, uint64_t offset_within_chunk);

    bool validChunk(uint64_t chunk_id) const;

    std::vector<StorageChannelStats> stats() const;
    void resetStats();

private:
    ParallelStorageConfig config_;
    std::vector<std::vector<uint8_t>> backing_; // per-channel byte store

    // Per-channel timing state.
    struct ChannelTiming {
        uint64_t busy_read_until_ns = 0;
        uint64_t busy_write_until_ns = 0;
        uint32_t outstanding = 0;
    };
    mutable std::mutex mutex_;
    std::vector<ChannelTiming> timing_;
    std::vector<StorageChannelStats> stats_;

    StorageOpResult op(uint64_t chunk_id, uint8_t *buffer, size_t size, uint64_t now_ns, bool is_read);
};

} // namespace ckpt

#endif // CXLMEMSIM_PARALLEL_STORAGE_H
