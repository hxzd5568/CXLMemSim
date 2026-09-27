/*
 * GPU training checkpoint/restore data-movement engine for CXLMemSim.
 *
 * Re-implements the target.md architecture (originally prototyped in the
 * SimCXL gem5 full-system simulator) as a self-contained, time-modelled C++20
 * simulation library:
 *
 *     GPU payload --stage--> DRAM / CXL pinned pools
 *                 --persist--> ParallelStorage (multi-channel flash)
 *                 --restore--> DRAM / CXL --> GPU
 *
 *   Phase 1 (synchronous, GPU-visible): GPU writes state to DRAM/CXL staging.
 *   Phase 2 (asynchronous, background): DRAM/CXL -> storage is persisted
 *     without blocking the GPU.
 *
 * The engine models bandwidth/latency contention, per-channel queue depth and
 * backpressure, a lane balancer (分流调度), a pinned-pool pressure state machine
 * (NORMAL/FROZEN/SHRINK/STOPPED), a versioned manifest with generation
 * rollback, CXL hot standby (a cache of disk chunks), and fault injection --
 * covering all eight acceptance criteria from target.md.
 *
 * SPDX-License-Identifier: (LGPL-2.1 OR BSD-2-Clause)
 * Copyright 2025 Regents of the University of California
 * UC Santa Cruz Sluglab.
 */

#ifndef CXLMEMSIM_CHECKPOINT_ENGINE_H
#define CXLMEMSIM_CHECKPOINT_ENGINE_H

#include "checkpoint_payload.h"
#include "parallel_storage.h"

#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ckpt {

// Which memory pool a chunk is staged into.
enum class LaneId : uint8_t { DRAM = 0, CXL = 1 };

// Manifest chunk lifecycle (target.md section 4.3 / P9).
enum class ChunkState : uint8_t {
    FREE = 0,
    PINNED = 1,        // staged into a pinned pool, not yet persisted
    IN_FLIGHT = 2,     // persistence DMA in flight (cannot be unpinned)
    DISK_COMMITTED = 3, // durable in storage
    HOT = 4,           // resident in CXL hot standby (cache of disk)
    EVICTABLE = 5,     // hot copy evictable, disk copy still authoritative
};

// Pinned-pool pressure state machine (target.md section 6 / P4).
enum class PoolState : uint8_t { NORMAL = 0, FROZEN = 1, SHRINK = 2, STOPPED = 3 };

const char *laneName(LaneId lane);
const char *chunkStateName(ChunkState state);
const char *poolStateName(PoolState state);

struct PoolConfig {
    // Calibration points (like CXLMemSim's --mlc-bandwidth): these are EFFECTIVE
    // measured bandwidths, not raw peak. Fill them from a benchmark (MLC or a
    // GPU->host DMA microbenchmark) on the target machine. Defaults are example
    // values only.
    double read_gbps = 30.0;  // measured read bandwidth (GB/s)
    double write_gbps = 30.0; // measured write bandwidth (GB/s)
    double read_latency_ns = 100.0;
    double write_latency_ns = 100.0;
    uint64_t capacity_bytes = 0;  // 0 => derived from chunk slots

    // Pressure thresholds as fraction of pool capacity (target.md section 6).
    double frozen_threshold = 0.80;
    double shrink_threshold = 0.90;
    double stop_threshold = 0.95;
    double recover_threshold = 0.75;
};

struct ChunkMeta {
    uint64_t checkpoint_id = 0;
    uint32_t chunk_id = 0;
    uint64_t logical_offset = 0; // chunk_id * chunk_size
    uint32_t length = 0;
    uint32_t crc32 = 0;
    uint8_t source = 0;  // 0 = DISK, 1 = CXL_HOT
    ChunkState state = ChunkState::FREE;
    LaneId lane = LaneId::DRAM;
    uint32_t pool_slot = 0; // pinned-pool slot within `lane`
};

struct CheckpointConfig {
    uint32_t chunk_size = 4096;
    uint32_t num_storage_channels = 2;

    // GPU <-> host PCIe uplink bandwidth (GB/s). This is a SINGLE shared
    // budget: on a single node the GPU has one PCIe link, so staging to
    // DRAM and CXL (and reading back) both traverse it. The aggregate staging
    // rate is therefore min(pcie_effective, dram.write + cxl.write), NOT
    // dram.write + cxl.write by themselves.
    double gpu_pcie_write_gbps = 64.0; // PCIe Gen5 x16 raw line rate (32 GT/s x16)
    double gpu_pcie_read_gbps = 64.0;

    // PCIe link overhead model: derives the achievable payload fraction from
    // the TLP structure instead of a magic efficiency constant.
    //   effective = raw * (128/130) * MPS/(MPS+overhead) * residual
    double pcie_encoding_efficiency = 128.0 / 130.0; // 128b/130b line coding
    uint32_t pcie_max_payload_bytes = 256;           // TLP MaxPayloadSize
    uint32_t pcie_tlp_overhead_bytes = 20;           // TLP header (16 B) + LCRC (4 B)
    double pcie_residual_efficiency = 0.95; // DLLP/ACK/flow-control/DMA-engine residual

    PoolConfig dram;
    PoolConfig cxl;
    StorageChannelConfig storage_channel{};
    uint64_t storage_capacity_per_channel_bytes = 0;

    // CXL hot standby: max chunks resident (cache of disk, not authoritative).
    uint64_t hot_capacity_chunks = 0;
    // Hot standby served with this CXL-side latency (fast path).
    double hot_read_latency_ns = 400.0;

    // Optional global token bucket (chunks/sec). 0 disables rate limiting.
    double token_rate_chunks_per_sec = 0.0;
    double token_burst_chunks = 0.0;

    bool cxl_enabled = true;
};

// Derives the achievable PCIe payload bandwidth from the TLP-structure
// parameters: 128b/130b encoding, TLP header/LCRC overhead, and a small
// residual (DLLP/ACK/flow-control/DMA engine). This replaces a flat magic
// efficiency constant with a formula whose inputs are physical link settings.
inline double pcieEffectiveGbps(double raw_gbps, const CheckpointConfig &cfg) {
    double tlp_ratio = static_cast<double>(cfg.pcie_max_payload_bytes) /
                       static_cast<double>(cfg.pcie_max_payload_bytes + cfg.pcie_tlp_overhead_bytes);
    return raw_gbps * cfg.pcie_encoding_efficiency * tlp_ratio * cfg.pcie_residual_efficiency;
}

struct LaneStats {
    uint64_t queued_bytes = 0;
    uint64_t outstanding = 0;
    uint64_t pool_free_bytes = 0;
    double ewma_bandwidth_gbps = 0.0;
    uint64_t p95_latency_ns = 0;
    uint64_t retries = 0;
    uint64_t queue_full = 0;
    uint64_t completed_chunks = 0;
    uint64_t completed_bytes = 0;
};

struct EngineStats {
    double gpu_stage_time_ns = 0.0;          // phase-1 GPU-visible staging time
    double gpu_stall_time_ns = 0.0;          // alias for stage time (phase 1)
    double durable_completion_time_ns = 0.0; // phase-2 background persist time
    double restore_time_ns = 0.0;            // phase-3 time-to-resume
    uint64_t storage_backlog_bytes = 0;
    uint64_t completed_chunks = 0;
    uint64_t errors = 0;
    uint64_t mismatches = 0;
    double hot_hit_rate = 0.0;
    std::array<LaneStats, 2> lanes{};
};

// Bandwidth-model bound helpers (target.md section 5, P8 analysis model).
struct BandwidthBounds {
    double bsave_gbps = 0.0;   // persist upper bound
    double bstage_gbps = 0.0;  // staging upper bound
    double brestore_gbps = 0.0;
};

BandwidthBounds computeBounds(const CheckpointConfig &cfg);

/*
 * A pinned memory pool backed by real bytes (so the round trip can be verified
 * byte-for-byte). Slots are fixed-size (chunk_size). Tracks pinned bytes and
 * drives the NORMAL/FROZEN/SHRINK/STOPPED pressure state machine.
 */
class PinnedPool {
public:
    PinnedPool(PoolConfig cfg, uint32_t chunk_size, LaneId lane);

    uint32_t chunkSize() const { return chunk_size_; }
    uint64_t capacityBytes() const { return data_.size(); }
    uint64_t pinnedBytes() const { return pinned_bytes_; }
    uint64_t freeBytes() const { return capacityBytes() - pinned_bytes_; }
    PoolState state() const { return state_; }

    // Allocate one slot. Returns slot index or -1 if out of room / frozen.
    int64_t allocate();
    // Release a slot. Returns false if slot is currently in-flight (protected).
    bool release(int64_t slot);
    bool releaseIfFree(int64_t slot);

    uint8_t *slotData(int64_t slot);
    const uint8_t *slotData(int64_t slot) const;

    void setInFlight(int64_t slot, bool in_flight);
    bool isInFlight(int64_t slot) const;

    // Advance the pressure state machine based on current utilization.
    void updatePressure();

    uint64_t slotBytes() const { return chunk_size_; }
    size_t numSlots() const { return slot_in_flight_.size(); }

    // Bandwidth-serialization bookkeeping (ns) for the time model.
    uint64_t busy_write_until_ns = 0;
    uint64_t busy_read_until_ns = 0;

private:
    PoolConfig cfg_;
    uint32_t chunk_size_;
    LaneId lane_;
    std::vector<uint8_t> data_;
    std::vector<bool> slot_allocated_;
    std::vector<bool> slot_in_flight_;
    std::deque<int64_t> free_slots_;
    uint64_t pinned_bytes_ = 0;
    PoolState state_ = PoolState::NORMAL;
};

/*
 * Lane balancer (分流调度器): chooses DRAM or CXL per chunk using a
 * predicted-finish heuristic plus headroom, exactly as described in target.md.
 */
class Balancer {
public:
    Balancer(const CheckpointConfig &cfg);

    // Choose a lane for the next chunk given current queued bytes.
    LaneId chooseLane(const PinnedPool &dram, const PinnedPool &cxl);

    void recordCompletion(LaneId lane, uint64_t bytes, double latency_ns);
    void recordRetry(LaneId lane);
    void recordQueueFull(LaneId lane);
    void addQueued(LaneId lane, uint64_t bytes);
    void removeQueued(LaneId lane, uint64_t bytes);

    // Token bucket: returns false if the rate limit denies the next chunk.
    bool allowNextChunk(uint64_t now_ns);

    const std::array<LaneStats, 2> &laneStats() const { return lanes_; }
    LaneStats &laneStats(LaneId lane) { return lanes_[static_cast<size_t>(lane)]; }

private:
    CheckpointConfig cfg_;
    std::array<LaneStats, 2> lanes_;
    double tokens_ = 0.0;
    uint64_t last_token_ns_ = 0;
    double laneRate(LaneId lane) const;
};

/*
 * CXL hot standby: an LRU cache of disk chunks held in CXL memory. It is a
 * cache of the disk copy, never the authoritative copy.
 */
class HotStandby {
public:
    explicit HotStandby(uint64_t capacity_chunks);

    bool contains(uint64_t checkpoint_id, uint32_t chunk_id) const;
    void insert(uint64_t checkpoint_id, uint32_t chunk_id, const std::vector<uint8_t> &data);
    const std::vector<uint8_t> *lookup(uint64_t checkpoint_id, uint32_t chunk_id);
    void touch(uint64_t checkpoint_id, uint32_t chunk_id);
    void evict(uint64_t checkpoint_id, uint32_t chunk_id);
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    double hitRate() const;
    size_t size() const { return entries_.size(); }
    size_t capacity() const { return capacity_; }

private:
    struct Entry {
        std::vector<uint8_t> data;
    };
    struct Key {
        uint64_t checkpoint_id;
        uint32_t chunk_id;
        bool operator==(const Key &o) const { return checkpoint_id == o.checkpoint_id && chunk_id == o.chunk_id; }
    };
    struct KeyHash {
        size_t operator()(const Key &k) const {
            return std::hash<uint64_t>{}(k.checkpoint_id) ^ (std::hash<uint32_t>{}(k.chunk_id) << 1);
        }
    };

    uint64_t capacity_;
    std::unordered_map<Key, Entry, KeyHash> entries_;
    std::deque<Key> lru_;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;

    void erase(const Key &k);
};

/*
 * The checkpoint engine. Orchestrates stage / persist / restore, maintains the
 * manifest, and exposes per-lane statistics and the acceptance-matrix helpers.
 */
class CheckpointEngine {
public:
    explicit CheckpointEngine(CheckpointConfig cfg);

    // ---- Manifest / lifecycle ----
    uint64_t manifestVersion() const { return version_; }
    uint64_t committedGeneration() const { return committed_gen_; }
    const ChunkMeta *chunkMeta(uint64_t checkpoint_id, uint32_t chunk_id) const;

    // ---- Phase 1: GPU-visible staging ----
    // Stages `num_chunks` chunks of `checkpoint_id` into DRAM/CXL via the
    // balancer. Returns total GPU stall time (ns).
    double stage(uint64_t checkpoint_id, uint64_t num_chunks);

    // ---- Phase 2: background persist ----
    // Marks PINNED chunks IN_FLIGHT (DMA window; unpin forbidden). No timing.
    void persistBegin(uint64_t checkpoint_id);
    // Performs pool-read -> storage-write for all IN_FLIGHT chunks, then marks
    // them DISK_COMMITTED. Returns durable completion time (ns).
    double persistFinish(uint64_t checkpoint_id);
    // Convenience: persistBegin() + persistFinish().
    double persist(uint64_t checkpoint_id);

    // ---- Generation commit / rollback ----
    void commitGeneration(uint64_t checkpoint_id);
    // Discard any generation newer than the last committed one (fault recovery).
    void rollback(uint64_t checkpoint_id);

    // ---- Phase 3: restore ----
    // Restores `num_chunks` chunks of a committed checkpoint. Hot chunks are
    // served from CXL hot standby. Verifies per-chunk CRC + regenerated payload.
    // Returns time-to-resume (ns).
    double restore(uint64_t checkpoint_id, uint64_t num_chunks);

    // ---- In-flight / unpin safety (acceptance #7) ----
    bool unpinChunk(uint64_t checkpoint_id, uint32_t chunk_id);

    // ---- Hot standby promotion ----
    void promoteChunk(uint64_t checkpoint_id, uint32_t chunk_id);
    void promoteHot(uint64_t checkpoint_id, uint64_t num_chunks);

    // ---- Stats ----
    const EngineStats &stats() const { return stats_; }
    const PinnedPool &dramPool() const { return dram_; }
    const PinnedPool &cxlPool() const { return cxl_; }
    const ParallelStorage &storage() const { return storage_; }
    ParallelStorage &storage() { return storage_; }
    const HotStandby &hotStandby() const { return hot_; }

    uint64_t nowNs() const { return now_; }
    void reset();

private:
    using ChunkKey = uint64_t; // (checkpoint_id << 32) | chunk_id
    static ChunkKey keyOf(uint64_t checkpoint_id, uint32_t chunk_id) {
        return (checkpoint_id << 32) | chunk_id;
    }

    CheckpointConfig cfg_;
    PinnedPool dram_;
    PinnedPool cxl_;
    ParallelStorage storage_;
    Balancer balancer_;
    HotStandby hot_;

    std::unordered_map<ChunkKey, ChunkMeta> manifest_;
    uint64_t version_ = 0;
    uint64_t committed_gen_ = 0;
    uint64_t now_ = 0;

    EngineStats stats_;

    PinnedPool &poolOf(LaneId lane);
    const PinnedPool &poolOf(LaneId lane) const;

    // Generate payload + CRC for a chunk and write it into the chosen pool.
    void stageOne(uint64_t checkpoint_id, uint32_t chunk_id);
    // Persist a single chunk (pool read -> storage write). Returns completion ns.
    uint64_t persistOne(const ChunkMeta &meta, const uint8_t *payload);
    // Restore a single chunk (storage/hot read -> pool -> GPU). Returns resume ns.
    uint64_t restoreOne(const ChunkMeta &meta, uint8_t *dst);
};

} // namespace ckpt

#endif // CXLMEMSIM_CHECKPOINT_ENGINE_H
