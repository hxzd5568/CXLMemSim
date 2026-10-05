# exec_summary_vhot.md —— CXLMemSim checkpoint 引擎 vhot 改造（双 NVMe 双路恢复 + 20% CXL 热备）执行总结

> 续篇：`checkpoint/exec_summary_v2.md`（target2 双线 staging，E0–E5，9/9）。
> 本文记录：在 target2 基础上**新增 restore 阶段的双路（dual-backend）流水线模型**与
> **20% CXL 热备场景**的开发、参数设定、成果（实测数据）、困难与决策。
> 代码改动仅落在 checkpoint 引擎（`include/checkpoint_engine.h`、`src/checkpoint_engine.cpp`、
> `tests/test_checkpoint_vhot.cpp`、`CMakeLists.txt`），**不影响主仓库 `cxlmemsim` 库**
> （checkpoint 引擎自包含，仅依赖标准库；`cxlmemsim` 库仍因缺 `bpf/bpf.h` 预置失败，与本改动无关）。

---

## 1. 核心结论（一句话）

旧 `restore()` 是**单路冷恢复**（一个 `ParallelStorage` 后端 → 单一 DRAM landing pool → GPU），
且冷路径没有 CXL landing。vhot 改造把它升级为**三路重叠的流水线恢复**：

```
20% 热 chunk  ── CXL hot standby ──────────────┐
                                               ├──> GPU（共享汇点，按 chunk 序 drain）
冷 chunk（偶）── nvme0 ──> host DRAM ──────────┤
冷 chunk（奇）── nvme1 ──> CXL DRAM  ──────────┘
```

三条流按 Little's law 重叠（每设备读带宽串行化 + 每 pool landing 串行化 + 共享 GPU sink），
恢复时间 = **max（三条流）而非三条流之和**。

实测（512 MiB）：

| 场景 | 恢复时间 | 有效恢复带宽 | 加速比 |
|---|---|---|---|
| single 0% hot（单 NVMe → DRAM） | 134.2 ms | 4.0 GB/s | 1.00× |
| single 20% hot | 107.4 ms | 5.0 GB/s | 1.25× |
| **dual 0% hot（nvme0→DRAM + nvme1→CXL）** | **67.1 ms** | **8.0 GB/s** | **2.00×** |
| **dual 20% hot** | **53.7 ms** | **10.0 GB/s** | **2.50×** |

**结论**：双路恢复（2 个独立 NVMe 后端各 4 GB/s）把冷恢复带宽从 4 → 8 GB/s（**恰好 2×**）；
再叠加 20% 热备（冷数据降到 80%，热数据走 CXL 快路径）总加速 **2.50×**。恢复时间是
**流水线后的延迟**：`restore 67.1 ms ≪ nvme0 + nvme1 + gpu 之和 201.4 ms`，证明三条流重叠、
由最慢流（NVMe）封顶，而非串行相加。

---

## 2. 回答：旧 `test_checkpoint_hotstandby` 的延迟是不是"流水线之后的"？

- **部分流水线，但只有单路冷路径**。旧 `restore()` 的 `busy_until` 时间模型已经让
  「SSD→DRAM」与「DRAM/CXL→GPU」跨 chunk 重叠（Little's law：带宽按 service 串行化、flash
  延迟做常数偏移），所以 50% hot 的 0.272 ms ≈ 仅冷 chunk 的 NVMe 时间（256 chunk / 4 GB/s），
  **热 chunk 的 cxl→gpu 确实与冷 chunk 的 ssd→dram 重叠**（GPU 是共享汇点，由 `gpu_busy` 串行化）。
- **但没有"双路"**：旧模型冷数据全部走一个 `ParallelStorage`（2 通道 = 4 GB/s）落到**唯一**
  landing pool（DRAM），没有「nvme1 → CXL DRAM」这条第二路；`land = dram_` 是写死的。
- 因此旧模型的"流水线"是**单后端内的流水线**，不是**多后端（nvme0/nvme1 × DRAM/CXL）的
  双路流水线**。vhot 补上的正是后者。

---

## 3. 新的目标

1. 恢复阶段显式建模**双后端**：两条互不争抢的 NVMe 读路径
   `nvme0 → host DRAM → GPU` 与 `nvme1 → CXL DRAM → GPU`，各自独立 landing pool。
2. 加入 **20% CXL 热备**场景：20% chunk 从 CXL hot standby 直接服务（无 NVMe），80% 冷数据
   双路加载。
3. 让三条流（热 / nvme0 / nvme1）**流水线重叠**，恢复时间由 `max` 而非 `sum` 决定，并用
   数字证明（`restore ≪ nvme0+nvme1+gpu`）。
4. 保持数据正确性（CRC + 逐字节 payload 校验）与既有验收（A1–A9）不回归。

---

## 4. 参数设定（标定点）

沿用 `exec_summary_v2.md` 口径（`tests/checkpoint_test_harness.h::defaultConfig`），新增字段：

| 参数 | 值 | 含义 |
|---|---|---|
| `num_storage_channels` | 2（**每设备** 2 通道） | 每台 NVMe 的并行通道数 |
| `storage_channel.read_gbps` | 2.0 | 每通道读带宽 → **每设备 4 GB/s** |
| `storage_channel.base_latency_ns` | 10000（10 µs） | NAND 整页读固定延迟（Little's law 每设备只算一次） |
| `num_storage_devices`（语义） | 2（nvme0 / nvme1） | 双后端；冷 chunk 按 `(chunk_id - hot)%2` 轮转分到两设备 |
| nvme0 landing | `dram.aggregateWriteGbps()` = 140.8 GB/s | host DRAM 8 子通道 |
| nvme1 landing | `cxl.aggregateWriteGbps()` = 400 GB/s | CXL expander 8 DDR |
| 热备 `hot_fraction` | 0.20 | 前 20% chunk 提升进 CXL hot standby |
| `hot_read_latency_ns` | 400 | CXL 热路径延迟（无 NVMe） |
| GPU sink | `pcieEffectiveGbps(64)` ≈ 55.53 GB/s | 共享 GPU 读汇点 |
| 数据规模 | 512 MiB = 131072 chunk × 4 KiB | 稳态带宽量级（摆脱固定开销） |
| 冷分片策略 | **轮转（round-robin）** `(i-hot)%2` | 对齐 `ParallelStorage` RAID-0 条带，避免"前半全 nvme0 / 后半全 nvme1"导致的 GPU 有序 drain 尾巴 |

恢复时间模型（每条流 `busy_until` 串行化 + 延迟常数偏移）：

```
T_restore = max( nvme0_fill, nvme1_fill, gpu_drain )
nvme0_fill = nvme0_bytes / (num_channels × read_gbps) + base_latency
nvme1_fill = nvme1_bytes / (num_channels × read_gbps) + base_latency
gpu_drain  = total_bytes / pcieEffectiveGbps(gpu_pcie_read_gbps)
```

---

## 5. 关键实现改动

- **新增 `DualRestoreStats`**（`include/checkpoint_engine.h`）：记录 `hot/nvme0/nvme1` 字节、
  chunk 数、`nvme0_fill_ns/nvme1_fill_ns/gpu_drain_ns`；挂进 `EngineStats::dual_restore`。
- **新增 `CheckpointEngine::restoreDualPath(checkpoint_id, num_chunks, hot_fraction)`**
  （`src/checkpoint_engine.cpp`）：三路重叠的时间模型 + 逐 chunk CRC/payload 校验。
  - 热 chunk（`i < hot_count`）走 `hot_.lookup()`（内存速 + 400 ns 延迟），无 NVMe。
  - 冷 chunk 按 `(i - hot_count) % 2` 轮转到 nvme0（→DRAM）或 nvme1（→CXL），每设备用
    `dev_read_bw = num_storage_channels × read_gbps` 串行化读服务；flash 延迟做常数偏移
    （Little's law，不 per-chunk）。
  - landing 池各用独立 `busy_until`（`dram_busy` / `cxl_busy`），GPU 用共享 `gpu_busy` 按
    chunk 序 drain；`max_finish = max(gpu_finish)`。
  - 数据读取复用 `storage_.peek()`（无计时）取持久化字节，校验 CRC + 重放 payload。
  - `stats_.hot_hit_rate` 报告**热服务字节占比**（20%），而非 cache 查中率（冷 chunk 绕过
    cache，查中率会恒为 1，语义不符）。
- **轮转分片而非连续分片（关键决策，见 §7）**：连续分片（前半 nvme0 / 后半 nvme1）会让
  GPU 有序 drain 先追 nvme0 到 67 ms、再以 GPU 全速 drain 已就绪的 nvme1 后半 → 多出
  ~4.8 ms 尾巴（71.9 ms）；轮转让两设备交错喂给有序 drain，消除尾巴（67.1 ms = 理论 2×）。
- **`tests/test_checkpoint_vhot.cpp`**：4 个场景对比 + 数据正确性 + 放置记账 + 流水线断言。
- **`CMakeLists.txt`**：注册 `test_checkpoint_vhot`。

---

## 6. 实测成果（`./test_checkpoint_vhot`）

```
[single  0% hot] restore 134.228 ms (hit 0.00)
[single 20% hot] restore 107.385 ms (hit 0.20)
[dual    0% hot] restore  67.119 ms (nvme0 67.1 / nvme1 67.1 / gpu 67.1 ms)
[dual   20% hot] restore  53.698 ms (hot 102.4 MiB, nvme0 204.8 / nvme1 204.8 MiB, gpu 53.7 ms)
[vhot] dual-path 0% hot is 2.00x faster than single-path; +20% hot => 2.50x total
[vhot] pipelining: restore 67.119 ms << sum(nvme0+nvme1+gpu) 201.358 ms
PASS
```

- 双路 0% hot：67.1 ms = 512 MiB / 8 GB/s，**恰好 2.00×**（两设备各 4 GB/s 并行）。
- 双路 20% hot：53.7 ms；冷 409.6 MiB @ 8 GB/s = 51.2 ms + 热尾巴，有效恢复带宽 ~10 GB/s。
- 流水线证据：`restore 67.1 ≪ nvme0 67.1 + nvme1 67.1 + gpu 67.1 = 201.4`，三条流重叠、
  由 `max` 封顶。
- 数据正确性：CRC + 逐字节 payload 全对（`mismatches==0 && errors==0`），热占比 0.20，
  nvme0/nvme1 各 204.8 MiB（平衡，差 ≤1 chunk）。

`test_checkpoint_acceptance` 仍 **9/9 PASS**；`test_checkpoint_hotstandby` / `storage` /
`engine` 等既有测试全绿（不回归）。

---

## 7. 困难与决策

1. **`CheckpointEngine` 不可移动/复制**（`ParallelStorage` 含 `std::mutex`），测试里
   `return CheckpointEngine` 编译失败。**决策**：改用 `std::make_unique<CheckpointEngine>`。

2. **连续分片产生有序 drain 尾巴**（核心决策）。初版把冷数据连续切成两半（nvme0 前半、
   nvme1 后半）：nvme1 早早填满后半却要等 GPU 先把 nvme0 的前半按序 drain 完，多出
   ~4.8 ms（71.9 ms vs 理论 67.1 ms）。**决策**：改成轮转 `(i-hot)%2`，对齐
   `ParallelStorage` 的 RAID-0 条带语义，两设备交错喂给有序 GPU drain，恢复恰好 2×。

3. **`hot_hit_rate` 语义冲突**。旧 `restore()` 对每个 chunk 都 `lookup()`，故
   `hits/(hits+misses)` 能反映命中率；`restoreDualPath` 里冷 chunk 直接走 NVMe、不查 cache，
   若沿用 `hot_.hitRate()` 会恒为 1.0。**决策**：`restoreDualPath` 里 `hot_hit_rate` 改为
   「热服务字节 / 总字节」（20% 语义），并注释说明。

4. **`round()` 与整型截断不一致**。`hot_count = round(0.20 × N)` 与测试侧 `chunks/5` 在
   512 MiB 下都等于 26214，但 `total_bytes/5`（整型除法）≠ `(chunks/5)×4096`（差 1638 B），
   导致记账断言误报。**决策**：断言统一用 `hot20 × 4096` 口径。

5. **环境缺 `bpf/bpf.h`**（预置问题）：`cmake --build build` 全量会因 `cxlmemsim` 库失败，
   但 checkpoint 引擎及其测试自包含、单独 build 全绿（README 已记录该限制，与本改动无关）。

---

## 8. 改动文件清单

```
include/checkpoint_engine.h       DualRestoreStats + EngineStats::dual_restore + restoreDualPath 声明
src/checkpoint_engine.cpp         restoreDualPath（三路重叠时间模型 + CRC/payload 校验）
tests/test_checkpoint_vhot.cpp    4 场景对比 + 正确性/记账/流水线断言
CMakeLists.txt                    注册 test_checkpoint_vhot
checkpoint/exec_summary_vhot.md   本文
```
