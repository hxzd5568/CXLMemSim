/*
 * Multi-channel ParallelStorage backend for CXLMemSim.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "parallel_storage.h"

#include <algorithm>
#include <cstring>

namespace ckpt {

ParallelStorage::ParallelStorage(ParallelStorageConfig config) : config_(std::move(config)) {
    if (config_.num_channels == 0) {
        config_.num_channels = 1;
    }
    if (config_.chunk_size == 0) {
        config_.chunk_size = 4096;
    }
    uint64_t cap = config_.capacity_per_channel_bytes;
    if (cap == 0) {
        // Default: room for a reasonable number of slots.
        cap = static_cast<uint64_t>(config_.chunk_size) * 4096;
    }
    backing_.resize(config_.num_channels, std::vector<uint8_t>(cap, 0));
    timing_.resize(config_.num_channels);
    stats_.resize(config_.num_channels);
}

bool ParallelStorage::validChunk(uint64_t chunk_id) const {
    uint32_t ch = stripeChannel(chunk_id);
    uint64_t off = channelOffset(chunk_id, 0);
    return ch < backing_.size() && off + config_.chunk_size <= backing_[ch].size();
}

StorageOpResult ParallelStorage::op(uint64_t chunk_id, uint8_t *buffer, size_t size, uint64_t now_ns, bool is_read) {
    StorageOpResult result;
    uint32_t ch = stripeChannel(chunk_id);
    uint64_t off = channelOffset(chunk_id, 0);

    if (size > config_.chunk_size) {
        size = config_.chunk_size;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    result.channel = ch;
    result.channel_offset = off;

    if (ch >= backing_.size() || off + size > backing_[ch].size()) {
        result.accepted = false;
        return result;
    }

    ChannelTiming &t = timing_[ch];
    StorageChannelStats &s = stats_[ch];

    // Backpressure: reject when the channel queue is full.
    if (t.outstanding >= config_.channel.max_outstanding) {
        result.accepted = false;
        s.rejected++;
        return result;
    }

    double gbps = is_read ? config_.channel.read_gbps : config_.channel.write_gbps;
    if (gbps <= 0.0) {
        gbps = 1.0;
    }
    double service_ns = static_cast<double>(size) / gbps;

    uint64_t &busy = is_read ? t.busy_read_until_ns : t.busy_write_until_ns;
    uint64_t start = std::max(now_ns, busy);
    // Bandwidth serialization: the channel is occupied for `service_ns` only;
    // the flash latency overlaps with the next transfer (Little's law).
    busy = start + static_cast<uint64_t>(service_ns);
    uint64_t finish = start + static_cast<uint64_t>(service_ns) + static_cast<uint64_t>(config_.channel.base_latency_ns);

    t.outstanding++;
    result.accepted = true;
    result.complete_time_ns = finish;

    // Byte movement (visible immediately for correctness; timing modelled above).
    if (is_read) {
        std::memcpy(buffer, backing_[ch].data() + off, size);
        s.bytes_read += size;
        s.num_reads++;
    } else {
        std::memcpy(backing_[ch].data() + off, buffer, size);
        s.bytes_written += size;
        s.num_writes++;
    }

    // Record latency; drain one outstanding slot at completion time (the caller
    // decrements outstanding on completion via markComplete()).
    uint64_t latency = finish - now_ns;
    s.latency_sum_ns += static_cast<double>(latency);
    s.latencies.push_back(latency);

    return result;
}

StorageOpResult ParallelStorage::write(uint64_t chunk_id, const uint8_t *data, size_t size, uint64_t now_ns) {
    // Copy into a temporary buffer so the caller may reuse `data`.
    std::vector<uint8_t> tmp(size);
    std::memcpy(tmp.data(), data, size);
    StorageOpResult r = op(chunk_id, tmp.data(), size, now_ns, false);
    // Immediate copy into backing already happened; tmp can be freed.
    return r;
}

StorageOpResult ParallelStorage::read(uint64_t chunk_id, uint8_t *dst, size_t size, uint64_t now_ns) {
    return op(chunk_id, dst, size, now_ns, true);
}

void ParallelStorage::markComplete(uint32_t channel) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (channel < timing_.size() && timing_[channel].outstanding > 0) {
        timing_[channel].outstanding--;
    }
}

const uint8_t *ParallelStorage::peek(uint64_t chunk_id, uint64_t offset_within_chunk) const {
    uint32_t ch = stripeChannel(chunk_id);
    uint64_t off = channelOffset(chunk_id, offset_within_chunk);
    if (ch >= backing_.size() || off >= backing_[ch].size()) {
        return nullptr;
    }
    return backing_[ch].data() + off;
}

uint8_t *ParallelStorage::mutablePeek(uint64_t chunk_id, uint64_t offset_within_chunk) {
    uint32_t ch = stripeChannel(chunk_id);
    uint64_t off = channelOffset(chunk_id, offset_within_chunk);
    if (ch >= backing_.size() || off >= backing_[ch].size()) {
        return nullptr;
    }
    return backing_[ch].data() + off;
}

std::vector<StorageChannelStats> ParallelStorage::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void ParallelStorage::resetStats() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &s : stats_) {
        s = StorageChannelStats{};
    }
}

} // namespace ckpt
