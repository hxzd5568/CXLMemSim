/*
 * GPU training checkpoint/restore data-movement engine for CXLMemSim.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#include "checkpoint_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace ckpt {

namespace {

double gbpsToBytesPerNs(double gbps) { return gbps; } // 1 GB/s == 1 byte/ns

} // namespace

const char *laneName(LaneId lane) { return lane == LaneId::DRAM ? "DRAM" : "CXL"; }

const char *chunkStateName(ChunkState state) {
    switch (state) {
    case ChunkState::FREE:
        return "FREE";
    case ChunkState::PINNED:
        return "PINNED";
    case ChunkState::IN_FLIGHT:
        return "IN_FLIGHT";
    case ChunkState::DISK_COMMITTED:
        return "DISK_COMMITTED";
    case ChunkState::HOT:
        return "HOT";
    case ChunkState::EVICTABLE:
        return "EVICTABLE";
    }
    return "?";
}

const char *poolStateName(PoolState state) {
    switch (state) {
    case PoolState::NORMAL:
        return "NORMAL";
    case PoolState::FROZEN:
        return "FROZEN";
    case PoolState::SHRINK:
        return "SHRINK";
    case PoolState::STOPPED:
        return "STOPPED";
    }
    return "?";
}

BandwidthBounds computeBounds(const CheckpointConfig &cfg) {
    BandwidthBounds b;
    // target2 dual-line model: two independent PCIe data lines replace the
    // single shared uplink. Line 1 (DRAM) traverses the switch uplink; Line 2
    // (CXL) is switch-local P2P and does NOT consume it. Staging bandwidth is
    // therefore additive: B_stage = min(num_gpus * egress, B_line1 + B_line2).
    double cxl_eff = cxlOverheadFactor(cfg);
    double dram_write = cfg.dram_enabled ? cfg.dram.aggregateWriteGbps() : 0.0;
    double dram_read = cfg.dram_enabled ? cfg.dram.aggregateReadGbps() : 0.0;
    double cxl_write = cfg.cxl_enabled ? cfg.cxl.aggregateWriteGbps() * cxl_eff : 0.0;
    double cxl_read = cfg.cxl_enabled ? cfg.cxl.aggregateReadGbps() * cxl_eff : 0.0;

    double line1 = cfg.dram_enabled ? std::min(pcieWriteGbps(cfg.line_dram, cfg), dram_write) : 0.0;
    double line2 = cfg.cxl_enabled ? std::min(pcieWriteGbps(cfg.line_cxl, cfg), cxl_write) : 0.0;
    double egress = cfg.num_gpus * pcieEffectiveGbps(cfg.gpu_egress_gbps, cfg);

    double channel_write = cfg.num_storage_channels * cfg.storage_channel.write_gbps;
    double channel_read = cfg.num_storage_channels * cfg.storage_channel.read_gbps;

    b.bstage_gbps = std::min(egress, line1 + line2);
    b.bsave_gbps = std::min({b.bstage_gbps, dram_read + cxl_read, channel_write});
    b.brestore_gbps = std::min({channel_read, dram_write + cxl_write, egress});
    return b;
}

PlacementPlan planOffload(const OffloadSpec &spec, double B_line1, double B_line2,
                          uint64_t dram_usable_bytes, uint64_t cxl_usable_bytes) {
    PlacementPlan plan;
    if (spec.total_bytes == 0 || spec.split_unit_bytes == 0 || spec.affinity.empty()) {
        return plan; // empty unit_lane => undefined/infeasible
    }
    const uint64_t unit = spec.split_unit_bytes;
    const size_t S = spec.affinity.size();

    // Per-unit actual sizes; the last unit may be partial.
    std::vector<uint64_t> unit_bytes(S, 0);
    {
        uint64_t remaining = spec.total_bytes;
        for (size_t i = 0; i < S; ++i) {
            unit_bytes[i] = std::min<uint64_t>(unit, remaining);
            remaining -= unit_bytes[i];
        }
    }

    uint64_t dram_min = 0, cxl_min = 0, free_bytes = 0;
    for (size_t i = 0; i < S; ++i) {
        if (spec.affinity[i] == Placement::DRAM) {
            dram_min += unit_bytes[i];
        } else if (spec.affinity[i] == Placement::CXL) {
            cxl_min += unit_bytes[i];
        } else {
            free_bytes += unit_bytes[i];
        }
    }
    if (dram_min > dram_usable_bytes || cxl_min > cxl_usable_bytes) {
        return plan; // affinity alone exceeds capacity => infeasible
    }

    // Water-filling on the free bytes (section 7.3): balance t1 == t2.
    const double b1 = B_line1 > 0.0 ? B_line1 : 1.0;
    const double b2 = B_line2 > 0.0 ? B_line2 : 1.0;
    double x = (b1 * static_cast<double>(cxl_min + free_bytes) - b2 * static_cast<double>(dram_min)) / (b1 + b2);
    x = std::clamp(x, 0.0, static_cast<double>(free_bytes));

    // Capacity clamps on both sides.
    const uint64_t dram_room = dram_usable_bytes - dram_min;
    const uint64_t cxl_room = cxl_usable_bytes - cxl_min;
    if (x > static_cast<double>(dram_room)) {
        x = static_cast<double>(dram_room);
    }
    if ((static_cast<double>(free_bytes) - x) > static_cast<double>(cxl_room)) {
        x = static_cast<double>(free_bytes) - static_cast<double>(cxl_room);
    }
    x = std::clamp(x, 0.0, static_cast<double>(free_bytes));

    // Greedy unit assignment: DRAM-affinity units first, then EITHER units up to
    // the water-fill DRAM budget. Handles partial units naturally.
    plan.unit_lane.assign(S, LaneId::CXL);
    uint64_t dram_free_budget = static_cast<uint64_t>(x);
    for (size_t i = 0; i < S; ++i) {
        if (spec.affinity[i] == Placement::DRAM) {
            plan.unit_lane[i] = LaneId::DRAM;
        } else if (spec.affinity[i] == Placement::CXL) {
            plan.unit_lane[i] = LaneId::CXL;
        } else if (dram_free_budget >= unit_bytes[i]) {
            plan.unit_lane[i] = LaneId::DRAM;
            dram_free_budget -= unit_bytes[i];
        } else {
            plan.unit_lane[i] = LaneId::CXL;
        }
    }

    uint64_t dram_bytes = 0, cxl_bytes = 0;
    for (size_t i = 0; i < S; ++i) {
        if (plan.unit_lane[i] == LaneId::DRAM) {
            dram_bytes += unit_bytes[i];
        } else {
            cxl_bytes += unit_bytes[i];
        }
    }
    plan.dram_bytes = dram_bytes;
    plan.cxl_bytes = cxl_bytes;
    plan.alpha = spec.total_bytes > 0 ? static_cast<double>(dram_bytes) / static_cast<double>(spec.total_bytes) : 0.0;
    return plan;
}

// ---------------------------------------------------------------------------
// PinnedPool
// ---------------------------------------------------------------------------

PinnedPool::PinnedPool(PoolConfig cfg, uint32_t chunk_size, LaneId lane)
    : cfg_(cfg), chunk_size_(chunk_size), lane_(lane) {
    uint64_t capacity = cfg.capacity_bytes;
    if (capacity == 0) {
        capacity = static_cast<uint64_t>(chunk_size) * 4096; // default slots
    }
    size_t slots = static_cast<size_t>(capacity / chunk_size);
    if (slots == 0) {
        slots = 1;
    }
    data_.resize(slots * chunk_size, 0);
    slot_allocated_.assign(slots, false);
    slot_in_flight_.assign(slots, false);
    for (size_t i = 0; i < slots; ++i) {
        free_slots_.push_back(static_cast<int64_t>(i));
    }
}

int64_t PinnedPool::allocate() {
    if (state_ != PoolState::NORMAL) {
        return -1; // FROZEN/SHRINK/STOPPED deny new pins
    }
    if (free_slots_.empty()) {
        return -1;
    }
    int64_t slot = free_slots_.front();
    free_slots_.pop_front();
    slot_allocated_[slot] = true;
    pinned_bytes_ += chunk_size_;
    updatePressure();
    return slot;
}

bool PinnedPool::release(int64_t slot) {
    if (slot < 0 || static_cast<size_t>(slot) >= slot_allocated_.size()) {
        return false;
    }
    if (slot_in_flight_[slot]) {
        return false; // in-flight DMA: unpin forbidden
    }
    if (!slot_allocated_[slot]) {
        return false;
    }
    slot_allocated_[slot] = false;
    pinned_bytes_ -= chunk_size_;
    free_slots_.push_back(slot);
    updatePressure();
    return true;
}

bool PinnedPool::releaseIfFree(int64_t slot) {
    if (slot < 0 || static_cast<size_t>(slot) >= slot_allocated_.size() || !slot_allocated_[slot]) {
        return false;
    }
    return release(slot);
}

uint8_t *PinnedPool::slotData(int64_t slot) {
    if (slot < 0 || static_cast<size_t>(slot) >= slot_allocated_.size()) {
        return nullptr;
    }
    return data_.data() + static_cast<size_t>(slot) * chunk_size_;
}

const uint8_t *PinnedPool::slotData(int64_t slot) const {
    if (slot < 0 || static_cast<size_t>(slot) >= slot_allocated_.size()) {
        return nullptr;
    }
    return data_.data() + static_cast<size_t>(slot) * chunk_size_;
}

void PinnedPool::setInFlight(int64_t slot, bool in_flight) {
    if (slot >= 0 && static_cast<size_t>(slot) < slot_in_flight_.size()) {
        slot_in_flight_[slot] = in_flight;
    }
}

bool PinnedPool::isInFlight(int64_t slot) const {
    if (slot < 0 || static_cast<size_t>(slot) >= slot_in_flight_.size()) {
        return false;
    }
    return slot_in_flight_[slot];
}

void PinnedPool::updatePressure() {
    double util = capacityBytes() > 0 ? static_cast<double>(pinned_bytes_) / capacityBytes() : 0.0;
    if (util >= cfg_.stop_threshold) {
        state_ = PoolState::STOPPED;
    } else if (util >= cfg_.shrink_threshold) {
        state_ = PoolState::SHRINK;
    } else if (util >= cfg_.frozen_threshold) {
        state_ = PoolState::FROZEN;
    } else if (util <= cfg_.recover_threshold) {
        state_ = PoolState::NORMAL; // hysteresis: recover only below threshold
    }
}

// ---------------------------------------------------------------------------
// Balancer
// ---------------------------------------------------------------------------

Balancer::Balancer(const CheckpointConfig &cfg) : cfg_(cfg) { tokens_ = cfg_.token_burst_chunks; }

double Balancer::laneRate(LaneId lane) const {
    const PoolConfig &pool = (lane == LaneId::DRAM) ? cfg_.dram : cfg_.cxl;
    double read = pool.aggregateReadGbps();
    double write = pool.aggregateWriteGbps();
    if (lane == LaneId::CXL) {
        // target2: CXL backend is raw DDR bandwidth (no flit); factor is 1.0.
        read *= cxlOverheadFactor(cfg_);
        write *= cxlOverheadFactor(cfg_);
    }
    return std::min(read, write);
}

LaneId Balancer::chooseLane(const PinnedPool &dram, const PinnedPool &cxl) {
    uint32_t chunk = cfg_.chunk_size;
    bool dram_headroom = cfg_.dram_enabled && dram.state() == PoolState::NORMAL && dram.freeBytes() >= chunk;
    bool cxl_headroom = cfg_.cxl_enabled && cxl.state() == PoolState::NORMAL && cxl.freeBytes() >= chunk;

    if (!dram_headroom && !cxl_headroom) {
        return LaneId::DRAM; // both full: caller will deny
    }

    auto predicted = [&](LaneId lane) {
        double r = laneRate(lane);
        if (r <= 0.0) {
            r = 1.0;
        }
        const LaneStats &s = lanes_[static_cast<size_t>(lane)];
        uint64_t backlog = s.queued_bytes + s.outstanding * cfg_.chunk_size;
        return static_cast<double>(backlog + cfg_.chunk_size) / r;
    };

    if (!dram_headroom) {
        return LaneId::CXL;
    }
    if (!cxl_headroom) {
        return LaneId::DRAM;
    }
    return predicted(LaneId::DRAM) <= predicted(LaneId::CXL) ? LaneId::DRAM : LaneId::CXL;
}

void Balancer::recordCompletion(LaneId lane, uint64_t bytes, double latency_ns) {
    LaneStats &s = lanes_[static_cast<size_t>(lane)];
    s.completed_chunks++;
    s.completed_bytes += bytes;
    double bw = latency_ns > 0 ? gbpsToBytesPerNs(bytes) / (latency_ns) : 0.0; // bytes/ns == GB/s
    // EWMA bandwidth
    constexpr double alpha = 0.2;
    s.ewma_bandwidth_gbps = (s.ewma_bandwidth_gbps == 0.0) ? bw : alpha * bw + (1.0 - alpha) * s.ewma_bandwidth_gbps;
}

void Balancer::recordRetry(LaneId lane) { lanes_[static_cast<size_t>(lane)].retries++; }

void Balancer::recordQueueFull(LaneId lane) { lanes_[static_cast<size_t>(lane)].queue_full++; }

void Balancer::addQueued(LaneId lane, uint64_t bytes) { lanes_[static_cast<size_t>(lane)].queued_bytes += bytes; }

void Balancer::removeQueued(LaneId lane, uint64_t bytes) {
    LaneStats &s = lanes_[static_cast<size_t>(lane)];
    s.queued_bytes = (s.queued_bytes >= bytes) ? (s.queued_bytes - bytes) : 0;
}

bool Balancer::allowNextChunk(uint64_t now_ns) {
    if (cfg_.token_rate_chunks_per_sec <= 0.0) {
        return true;
    }
    if (last_token_ns_ == 0) {
        last_token_ns_ = now_ns;
    }
    double rate_per_ns = cfg_.token_rate_chunks_per_sec / 1e9;
    double elapsed = static_cast<double>(now_ns - last_token_ns_);
    tokens_ = std::min(cfg_.token_burst_chunks, tokens_ + rate_per_ns * elapsed);
    last_token_ns_ = now_ns;
    if (tokens_ >= 1.0) {
        tokens_ -= 1.0;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// HotStandby
// ---------------------------------------------------------------------------

HotStandby::HotStandby(uint64_t capacity_chunks) : capacity_(capacity_chunks) {}

bool HotStandby::contains(uint64_t checkpoint_id, uint32_t chunk_id) const {
    return entries_.count({checkpoint_id, chunk_id}) > 0;
}

void HotStandby::insert(uint64_t checkpoint_id, uint32_t chunk_id, const std::vector<uint8_t> &data) {
    Key k{checkpoint_id, chunk_id};
    if (entries_.count(k)) {
        entries_[k].data = data;
        touch(checkpoint_id, chunk_id);
        return;
    }
    while (entries_.size() >= capacity_ && !lru_.empty()) {
        Key victim = lru_.front();
        lru_.pop_front();
        entries_.erase(victim);
    }
    entries_[k].data = data;
    lru_.push_back(k);
}

const std::vector<uint8_t> *HotStandby::lookup(uint64_t checkpoint_id, uint32_t chunk_id) {
    Key k{checkpoint_id, chunk_id};
    auto it = entries_.find(k);
    if (it == entries_.end()) {
        misses_++;
        return nullptr;
    }
    hits_++;
    touch(checkpoint_id, chunk_id);
    return &it->second.data;
}

void HotStandby::touch(uint64_t checkpoint_id, uint32_t chunk_id) {
    Key k{checkpoint_id, chunk_id};
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        if (*it == k) {
            lru_.erase(it);
            lru_.push_back(k);
            return;
        }
    }
    // Not present in LRU (possible on insert of a new key): append.
    lru_.push_back(k);
}

void HotStandby::evict(uint64_t checkpoint_id, uint32_t chunk_id) { erase({checkpoint_id, chunk_id}); }

void HotStandby::erase(const Key &k) {
    entries_.erase(k);
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        if (*it == k) {
            lru_.erase(it);
            break;
        }
    }
}

double HotStandby::hitRate() const {
    uint64_t total = hits_ + misses_;
    return total > 0 ? static_cast<double>(hits_) / total : 0.0;
}

// ---------------------------------------------------------------------------
// CheckpointEngine
// ---------------------------------------------------------------------------

CheckpointEngine::CheckpointEngine(CheckpointConfig cfg)
    : cfg_(std::move(cfg)), dram_(cfg_.dram, cfg_.chunk_size, LaneId::DRAM),
      cxl_(cfg_.cxl, cfg_.chunk_size, LaneId::CXL), storage_(ParallelStorageConfig{
                                                           cfg_.num_storage_channels, cfg_.chunk_size,
                                                           cfg_.storage_channel, cfg_.storage_capacity_per_channel_bytes}),
      balancer_(cfg_), hot_(cfg_.hot_capacity_chunks) {}

PinnedPool &CheckpointEngine::poolOf(LaneId lane) { return lane == LaneId::DRAM ? dram_ : cxl_; }

const PinnedPool &CheckpointEngine::poolOf(LaneId lane) const { return lane == LaneId::DRAM ? dram_ : cxl_; }

const ChunkMeta *CheckpointEngine::chunkMeta(uint64_t checkpoint_id, uint32_t chunk_id) const {
    auto it = manifest_.find(keyOf(checkpoint_id, chunk_id));
    return it == manifest_.end() ? nullptr : &it->second;
}

void CheckpointEngine::stageOne(uint64_t checkpoint_id, uint32_t chunk_id, LaneId lane) {
    PinnedPool &pool = poolOf(lane);

    int64_t slot = pool.allocate();
    if (slot < 0) {
        // Try the other lane before giving up.
        LaneId other = (lane == LaneId::DRAM) ? LaneId::CXL : LaneId::DRAM;
        bool other_enabled = (other == LaneId::DRAM) ? cfg_.dram_enabled : cfg_.cxl_enabled;
        if (!other_enabled) {
            stats_.errors++;
            balancer_.recordQueueFull(lane);
            return;
        }
        PinnedPool &opool = poolOf(other);
        int64_t oslot = opool.allocate();
        if (oslot < 0) {
            stats_.errors++;
            balancer_.recordQueueFull(lane);
            return;
        }
        lane = other;
        pool = opool;
        slot = oslot;
    }

    uint8_t *buf = pool.slotData(slot);
    payloadFill(checkpoint_id, chunk_id, buf, cfg_.chunk_size);

    ChunkMeta meta;
    meta.checkpoint_id = checkpoint_id;
    meta.chunk_id = chunk_id;
    meta.logical_offset = static_cast<uint64_t>(chunk_id) * cfg_.chunk_size;
    meta.length = cfg_.chunk_size;
    meta.crc32 = crc32(buf, cfg_.chunk_size);
    meta.source = 0;
    meta.state = ChunkState::PINNED;
    meta.lane = lane;
    meta.pool_slot = static_cast<uint32_t>(slot);

    balancer_.addQueued(lane, cfg_.chunk_size);
    manifest_[keyOf(checkpoint_id, chunk_id)] = meta;
}

double CheckpointEngine::stage(uint64_t checkpoint_id, uint64_t num_chunks) {
    uint64_t dram_chunks = 0;
    uint64_t cxl_chunks = 0;

    // ---- Offload plan (section 7): decided ONCE, at task submission ----
    // The checkpoint size is fixed, so the DRAM/CXL split is a static water-fill
    // computed from the two lines' bandwidths and pool capacities. The memcpy
    // address (split unit) then fixes each chunk's lane; the hot path is an O(1)
    // plan lookup instead of a per-chunk greedy chooseLane().
    uint64_t total_bytes = num_chunks * cfg_.chunk_size;
    OffloadSpec spec;
    spec.total_bytes = total_bytes;
    // Adaptive split unit: keep at least a few units even for small checkpoints so
    // the split remains meaningful (the default 256 MiB unit is for large models).
    constexpr uint64_t kMinSplitUnits = 4;
    uint64_t eff_split = cfg_.split_unit_bytes;
    if (eff_split > 0 && total_bytes / eff_split < kMinSplitUnits) {
        eff_split = std::max<uint64_t>(cfg_.chunk_size, total_bytes / kMinSplitUnits);
        if (eff_split == 0) {
            eff_split = cfg_.chunk_size;
        }
    }
    spec.split_unit_bytes = eff_split;
    uint64_t S = (total_bytes + eff_split - 1) / eff_split;
    if (S == 0) {
        S = 1;
    }
    spec.affinity.assign(static_cast<size_t>(S), Placement::EITHER);

    double b1 = cfg_.dram_enabled
                    ? std::min(pcieWriteGbps(cfg_.line_dram, cfg_), cfg_.dram.aggregateWriteGbps())
                    : 0.0;
    double b2 = cfg_.cxl_enabled
                    ? std::min(pcieWriteGbps(cfg_.line_cxl, cfg_),
                               cfg_.cxl.aggregateWriteGbps() * cxlOverheadFactor(cfg_))
                    : 0.0;
    plan_ = planOffload(spec, b1, b2, cfg_.dram_enabled ? dram_.capacityBytes() : 0,
                        cfg_.cxl_enabled ? cxl_.capacityBytes() : 0);
    next_unit_ = 0;

    for (uint64_t i = 0; i < num_chunks; ++i) {
        if (!balancer_.allowNextChunk(now_)) {
            stats_.errors++; // token-bucket throttle counts as a denied issue
        }
        uint32_t chunk_id = static_cast<uint32_t>(i);
        uint64_t unit = (i * cfg_.chunk_size) / eff_split;
        LaneId lane = LaneId::DRAM;
        if (!plan_.unit_lane.empty() && unit < plan_.unit_lane.size()) {
            lane = plan_.unit_lane[static_cast<size_t>(unit)];
        }
        stageOne(checkpoint_id, chunk_id, lane);
        const ChunkMeta *meta = chunkMeta(checkpoint_id, chunk_id);
        if (meta && meta->state == ChunkState::PINNED) {
            if (meta->lane == LaneId::DRAM) {
                dram_chunks++;
            } else {
                cxl_chunks++;
            }
        }
    }
    next_unit_ = S;

    // ---- GPU-visible staging time (dual-line + egress) ----
    // The stall is the max over: (a) all bytes leaving the GPUs through their
    // aggregate egress, (b) each line serializing its bytes on its own link,
    // (c) each backend absorbing its share -- plus one write latency. With two
    // independent lines, staging is additive (bounded by egress), unlike the old
    // single-shared-uplink model.
    double dram_bytes = static_cast<double>(dram_chunks) * cfg_.chunk_size;
    double cxl_bytes = static_cast<double>(cxl_chunks) * cfg_.chunk_size;
    double staged_bytes = dram_bytes + cxl_bytes;
    double egress_eff = pcieEffectiveGbps(cfg_.gpu_egress_gbps, cfg_);
    double egress_ns = (egress_eff > 0.0 && staged_bytes > 0.0) ? staged_bytes / (cfg_.num_gpus * egress_eff) : 0.0;

    double line1_link = cfg_.dram_enabled ? pcieWriteGbps(cfg_.line_dram, cfg_) : 0.0;
    double line2_link = cfg_.cxl_enabled ? pcieWriteGbps(cfg_.line_cxl, cfg_) : 0.0;
    double line1_emit = (line1_link > 0.0 && dram_chunks > 0) ? dram_bytes / line1_link : 0.0;
    double line2_emit = (line2_link > 0.0 && cxl_chunks > 0) ? cxl_bytes / line2_link : 0.0;

    double dram_absorb = cfg_.dram_enabled ? dram_bytes / cfg_.dram.aggregateWriteGbps() : 0.0;
    double cxl_absorb = cfg_.cxl_enabled
                            ? cxl_bytes / (cfg_.cxl.aggregateWriteGbps() * cxlOverheadFactor(cfg_))
                            : 0.0;

    double write_latency = 0.0;
    if (cfg_.dram_enabled) {
        write_latency = std::max(write_latency, cfg_.dram.write_latency_ns);
    }
    if (cfg_.cxl_enabled) {
        write_latency = std::max(write_latency, cfg_.cxl.write_latency_ns);
    }
    double stall = std::max({egress_ns, line1_emit, line2_emit, dram_absorb, cxl_absorb}) + write_latency;

    stats_.gpu_stage_time_ns = stall;
    stats_.gpu_stall_time_ns = stall;
    stats_.lanes = balancer_.laneStats();
    stats_.lanes[static_cast<size_t>(LaneId::DRAM)].pool_free_bytes = dram_.freeBytes();
    stats_.lanes[static_cast<size_t>(LaneId::CXL)].pool_free_bytes = cxl_.freeBytes();

    // Per-line statistics (A2/A7): CXL line is P2P and does not count toward the
    // switch uplink; CXL bytes are device-initiated when line_cxl.device_initiated.
    stats_.links[0].write_bytes += static_cast<uint64_t>(dram_bytes);
    stats_.links[0].host_initiated_bytes += static_cast<uint64_t>(dram_bytes);
    stats_.links[0].counts_toward_uplink = cfg_.line_dram.consumes_uplink;
    stats_.links[1].write_bytes += static_cast<uint64_t>(cxl_bytes);
    stats_.links[1].counts_toward_uplink = cfg_.line_cxl.consumes_uplink;
    if (cfg_.line_cxl.device_initiated) {
        stats_.links[1].device_initiated_bytes += static_cast<uint64_t>(cxl_bytes);
    } else {
        stats_.links[1].host_initiated_bytes += static_cast<uint64_t>(cxl_bytes);
    }
    return stall;
}

bool CheckpointEngine::shouldReplan() const {
    const LaneStats &d = balancer_.laneStats(LaneId::DRAM);
    const LaneStats &c = balancer_.laneStats(LaneId::CXL);
    double b1 = cfg_.dram_enabled
                    ? std::min(pcieWriteGbps(cfg_.line_dram, cfg_), cfg_.dram.aggregateWriteGbps())
                    : 0.0;
    double b2 = cfg_.cxl_enabled
                    ? std::min(pcieWriteGbps(cfg_.line_cxl, cfg_),
                               cfg_.cxl.aggregateWriteGbps() * cxlOverheadFactor(cfg_))
                    : 0.0;
    auto drift = [](double measured, double assumed) {
        return (assumed > 0.0 && measured > 0.0) ? std::fabs(measured - assumed) / assumed : 0.0;
    };
    if (cfg_.dram_enabled && drift(d.ewma_bandwidth_gbps, b1) > cfg_.drift_bw_threshold) {
        return true;
    }
    if (cfg_.cxl_enabled && drift(c.ewma_bandwidth_gbps, b2) > cfg_.drift_bw_threshold) {
        return true;
    }
    if (d.retries + d.queue_full > cfg_.drift_retry_threshold || c.retries + c.queue_full > cfg_.drift_retry_threshold) {
        return true;
    }
    if (cfg_.dram_enabled && dram_.capacityBytes() > 0 &&
        static_cast<double>(dram_.freeBytes()) / static_cast<double>(dram_.capacityBytes()) < cfg_.drift_cap_threshold) {
        return true;
    }
    if (cfg_.cxl_enabled && cxl_.capacityBytes() > 0 &&
        static_cast<double>(cxl_.freeBytes()) / static_cast<double>(cxl_.capacityBytes()) < cfg_.drift_cap_threshold) {
        return true;
    }
    return false;
}

void CheckpointEngine::persistBegin(uint64_t checkpoint_id) {
    for (auto &kv : manifest_) {
        ChunkMeta &meta = kv.second;
        if (meta.checkpoint_id == checkpoint_id && meta.state == ChunkState::PINNED) {
            meta.state = ChunkState::IN_FLIGHT;
            poolOf(meta.lane).setInFlight(meta.pool_slot, true);
        }
    }
}

double CheckpointEngine::persistFinish(uint64_t checkpoint_id) {
    uint64_t start = now_;
    uint64_t max_finish = now_;

    for (auto &kv : manifest_) {
        ChunkMeta &meta = kv.second;
        if (meta.checkpoint_id != checkpoint_id || meta.state != ChunkState::IN_FLIGHT) {
            continue;
        }
        PinnedPool &pool = poolOf(meta.lane);
        const uint8_t *payload = pool.slotData(meta.pool_slot);

        // Pool read.
        double read_bw = (meta.lane == LaneId::DRAM) ? cfg_.dram.aggregateReadGbps()
                                                     : cfg_.cxl.aggregateReadGbps() * cxlOverheadFactor(cfg_);
        double read_service = static_cast<double>(cfg_.chunk_size) / read_bw;
        double read_latency = (meta.lane == LaneId::DRAM) ? cfg_.dram.read_latency_ns : cfg_.cxl.read_latency_ns;
        uint64_t read_start = std::max(now_, pool.busy_read_until_ns);
        pool.busy_read_until_ns = read_start + static_cast<uint64_t>(read_service);
        uint64_t read_finish = read_start + static_cast<uint64_t>(read_service) + static_cast<uint64_t>(read_latency);

        // Storage write (the durable copy). Retry on queue-full.
        StorageOpResult res = storage_.write(meta.chunk_id, payload, cfg_.chunk_size, read_finish);
        if (!res.accepted) {
            balancer_.recordQueueFull(meta.lane);
            balancer_.recordRetry(meta.lane);
            res = storage_.write(meta.chunk_id, payload, cfg_.chunk_size,
                                 read_finish + static_cast<uint64_t>(cfg_.storage_channel.base_latency_ns));
            if (!res.accepted) {
                stats_.errors++;
            }
        }
        uint64_t complete = res.accepted ? res.complete_time_ns : read_finish;
        storage_.markComplete(res.channel);

        meta.state = ChunkState::DISK_COMMITTED;
        meta.source = 0;
        pool.setInFlight(meta.pool_slot, false);

        max_finish = std::max(max_finish, complete);
        stats_.completed_chunks++;
        LaneStats &ls = balancer_.laneStats(meta.lane);
        ls.completed_chunks++;
        ls.completed_bytes += cfg_.chunk_size;
        ls.queued_bytes = (ls.queued_bytes >= cfg_.chunk_size) ? (ls.queued_bytes - cfg_.chunk_size) : 0;
        balancer_.recordCompletion(meta.lane, cfg_.chunk_size, static_cast<double>(complete - read_start));
    }

    double durable = static_cast<double>(max_finish - start);
    stats_.durable_completion_time_ns = durable;
    stats_.storage_backlog_bytes = balancer_.laneStats(LaneId::DRAM).queued_bytes +
                                   balancer_.laneStats(LaneId::CXL).queued_bytes;
    stats_.lanes = balancer_.laneStats();
    stats_.lanes[static_cast<size_t>(LaneId::DRAM)].pool_free_bytes = dram_.freeBytes();
    stats_.lanes[static_cast<size_t>(LaneId::CXL)].pool_free_bytes = cxl_.freeBytes();
    return durable;
}

double CheckpointEngine::persist(uint64_t checkpoint_id) {
    persistBegin(checkpoint_id);
    return persistFinish(checkpoint_id);
}

void CheckpointEngine::commitGeneration(uint64_t checkpoint_id) {
    committed_gen_ = checkpoint_id;
    version_++;
}

void CheckpointEngine::rollback(uint64_t checkpoint_id) {
    // Discard any chunk of this generation that is not yet durable.
    for (auto &kv : manifest_) {
        ChunkMeta &meta = kv.second;
        if (meta.checkpoint_id != checkpoint_id) {
            continue;
        }
        if (meta.state == ChunkState::PINNED || meta.state == ChunkState::IN_FLIGHT) {
            poolOf(meta.lane).setInFlight(meta.pool_slot, false);
            poolOf(meta.lane).releaseIfFree(meta.pool_slot);
            meta.state = ChunkState::FREE;
        }
    }
}

void CheckpointEngine::promoteChunk(uint64_t checkpoint_id, uint32_t chunk_id) {
    auto it = manifest_.find(keyOf(checkpoint_id, chunk_id));
    if (it == manifest_.end() || it->second.state != ChunkState::DISK_COMMITTED) {
        return;
    }
    const uint8_t *src = storage_.peek(chunk_id, 0);
    if (!src) {
        return;
    }
    std::vector<uint8_t> data(src, src + cfg_.chunk_size);
    hot_.insert(checkpoint_id, chunk_id, data);
    it->second.state = ChunkState::HOT;
}

void CheckpointEngine::promoteHot(uint64_t checkpoint_id, uint64_t num_chunks) {
    uint64_t limit = std::min<uint64_t>(num_chunks, hot_.capacity());
    for (uint64_t i = 0; i < limit; ++i) {
        promoteChunk(checkpoint_id, static_cast<uint32_t>(i));
    }
}

bool CheckpointEngine::unpinChunk(uint64_t checkpoint_id, uint32_t chunk_id) {
    auto it = manifest_.find(keyOf(checkpoint_id, chunk_id));
    if (it == manifest_.end()) {
        return false;
    }
    ChunkMeta &meta = it->second;
    if (meta.state == ChunkState::PINNED || meta.state == ChunkState::IN_FLIGHT) {
        return false; // in-flight DMA: unpin forbidden
    }
    poolOf(meta.lane).release(meta.pool_slot);
    meta.state = ChunkState::FREE;
    return true;
}

double CheckpointEngine::restore(uint64_t checkpoint_id, uint64_t num_chunks) {
    uint64_t start = now_;
    uint64_t max_finish = now_;
    uint64_t gpu_busy = now_;
    stats_.mismatches = 0;
    stats_.errors = 0;

    std::vector<uint8_t> scratch(cfg_.chunk_size);

    for (uint64_t i = 0; i < num_chunks; ++i) {
        uint32_t chunk_id = static_cast<uint32_t>(i);
        auto it = manifest_.find(keyOf(checkpoint_id, chunk_id));
        if (it == manifest_.end() || it->second.state == ChunkState::FREE) {
            stats_.errors++;
            continue;
        }

        uint64_t ready = now_;
        std::vector<uint8_t> buf(cfg_.chunk_size);

        const std::vector<uint8_t> *hot_data = hot_.lookup(checkpoint_id, chunk_id);
        if (hot_data) {
            // CXL hot path: no flash access.
            std::memcpy(buf.data(), hot_data->data(), cfg_.chunk_size);
            ready = now_ + static_cast<uint64_t>(cfg_.hot_read_latency_ns);
        } else {
            // Cold path: storage read -> landing pool -> GPU.
            StorageOpResult res = storage_.read(chunk_id, buf.data(), cfg_.chunk_size, now_);
            if (!res.accepted) {
                balancer_.recordQueueFull(it->second.lane);
                balancer_.recordRetry(it->second.lane);
                res = storage_.read(chunk_id, buf.data(), cfg_.chunk_size,
                                    now_ + static_cast<uint64_t>(cfg_.storage_channel.base_latency_ns));
                if (!res.accepted) {
                    stats_.errors++;
                }
            }
            storage_.markComplete(res.channel);
            ready = res.complete_time_ns;

            // Landing-pool write (use DRAM by default on cold restore).
            PinnedPool &land = dram_;
            double w_service = static_cast<double>(cfg_.chunk_size) / cfg_.dram.aggregateWriteGbps();
            uint64_t w_start = std::max(ready, land.busy_write_until_ns);
            land.busy_write_until_ns = w_start + static_cast<uint64_t>(w_service);
            uint64_t w_finish = w_start + static_cast<uint64_t>(w_service) + static_cast<uint64_t>(cfg_.dram.write_latency_ns);
            ready = w_finish;
        }

        // GPU read of the restored chunk.
        double g_service = static_cast<double>(cfg_.chunk_size) / pcieEffectiveGbps(cfg_.gpu_pcie_read_gbps, cfg_);
        uint64_t g_start = std::max(ready, gpu_busy);
        uint64_t g_finish = g_start + static_cast<uint64_t>(g_service);
        gpu_busy = g_finish;
        max_finish = std::max(max_finish, g_finish);

        // Verify per-chunk CRC + regenerated payload (acceptance #1/#5).
        uint32_t crc = crc32(buf.data(), cfg_.chunk_size);
        if (crc != it->second.crc32) {
            stats_.mismatches++;
        }
        payloadFill(checkpoint_id, chunk_id, scratch.data(), cfg_.chunk_size);
        uint64_t mm = 0;
        if (!payloadVerify(checkpoint_id, chunk_id, buf.data(), cfg_.chunk_size, 0, &mm)) {
            stats_.mismatches++;
        }
    }

    double resume = static_cast<double>(max_finish - start);
    stats_.restore_time_ns = resume;
    stats_.hot_hit_rate = hot_.hitRate();
    stats_.lanes = balancer_.laneStats();
    return resume;
}

void CheckpointEngine::reset() {
    now_ = 0;
    version_ = 0;
    committed_gen_ = 0;
    manifest_.clear();
    stats_ = EngineStats{};
    plan_ = PlacementPlan{};
    next_unit_ = 0;
    dram_.busy_read_until_ns = dram_.busy_write_until_ns = 0;
    cxl_.busy_read_until_ns = cxl_.busy_write_until_ns = 0;
    storage_.resetStats();
}

} // namespace ckpt
