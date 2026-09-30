# cxlmem_target2.md —— CXLMemSim checkpoint 引擎 GPU 双线 staging 加速（CXL 同域拓扑，expander 直连）改造计划

> 本文是 `checkpoint/README.md`（P0–P10 的 CXLMemSim 重实现）的续篇，同时是
> `../SimCXL/configs/example/gem5_library/target2.md` 的 **CXLMemSim 侧对应计划**。
> 阅读前提：先读 `checkpoint/README.md` 的架构与「诚实结论」（验收 #2 单机单上行不成立），
> 再读 SimCXL `target2.md`（拓扑变更 + E0–E5）。
>
> 一句话目标：把 checkpoint 引擎的 staging 带宽模型从「单共享 PCIe 上行封顶」改成
> 「两条独立 Gen5 x16 数据线」，让 GPU 在**不争抢同一条上行**的前提下并行向 Host DRAM
> （4 DIMM / 8 子通道）与 **CXL expander（8 DDR，直连 PCIe switch，无 CXL switch）**
> staging。CXL 线的 DMA 由 expander 侧 on-device DMA 主导、**纯 PCIe TLP 协议（无 68B
> flit）**，CPU 只发 CXL.io 控制面。从而把 `B_stage` 从 `min(上行, …)` 升级为
> `min(GPU egress, B_line1 + B_line2)` 的可加性，多卡时上行瓶颈被 CXL 线完全分担。

---

## 一、背景：拓扑从「单共享上行 + CXL flit」变成「CXL 同域双 Gen5 线（expander 直连）」

### 1.1 改造前（当前 CXLMemSim 的建模现状）

CXLMemSim checkpoint 引擎用**一个共享的 PCIe 上行预算**串行化 staging，且 CXL 线按
CXL.mem 68B flit 折算（见 `include/checkpoint_engine.h` 的 `CheckpointConfig::gpu_pcie_write_gbps`
与 `cxl_flit_data_bytes/cxl_flit_total_bytes`，以及 `src/checkpoint_engine.cpp` 的
`computeBounds`）：

```
GPU payload ──(1 条共享 PCIe uplink)──┬─ Host DRAM（4 lanes，128 GB/s）
                                       └─ CXL Type-3（2 links × 68B-flit 折算，59.3 GB/s）
```

`computeBounds()`（`src/checkpoint_engine.cpp`）写死这一假设：

```
B_stage = min( pcie_eff, DRAM_write + CXL_write × 64/68 )   # 单上行共享预算 + CXL flit
```

这正是 `checkpoint/README.md` 记录的「诚实结论」根因：DRAM 与 CXL 聚合带宽都超过单上行
有效值（~55.5 GB/s），所以 only-DRAM / only-CXL / both 全被 PCIe 封顶在 ~55.5 GB/s，
验收 #2「双路 staging > 最快单路径」在单机单上行上**不成立**（与 SimCXL P10 一致）。

### 1.2 改造后（新拓扑：CXL expander 直连，无 CXL switch）

```
                        CPU ── Host DRAM Controller
                                 ├─ DDR0
                                 ├─ DDR1
                                 ├─ ...
                                 └─ DDR7        （4 DIMM / 8 条 DDR 子通道）
                         │
                     Root Complex   (PCIe 5.0 上行)
                         │
                     PCIe Switch    (PCIe 5.0)
                   ├── GPU 0 (PCIe 4.0)
                   ├── GPU 1 (PCIe 4.0)
                   ├── GPU 2 (PCIe 4.0)
                   ├── GPU 3 (PCIe 4.0)
                   └── Gen5 x16 CXL Expander  (8 DDR，共 400 GB/s)
                         ├─ DDR 0
                         ├─ DDR 1
                         ├─ DDR 2
                         ├─ DDR 3
                         ├─ DDR 4
                         ├─ DDR 5
                         ├─ DDR 6
                         └─ DDR 7
```

与旧 target2 拓扑的**关键差异**：

1. **没有 CXL switch**。CXL expander 作为端点**直接挂在 PCIe switch 之下**（Gen5 x16），
   不再经过一个独立的 CXL switch 层级。
2. **数据线是纯 PCIe TLP**。GPU → expander 的 staging 由 expander 的 **on-device DMA**
   主导：CPU 只发 **CXL.io** 给 expander（控制面：下发 DMA 描述符、门铃、完成），真正的
   数据搬运是 expander 发起的 **PCIe P2P TLP**（读 GPU HBM → 写自己的 DDR）。因此这条线
   **没有 CXL.mem 的 68B flit 开销**，只有 PCIe TLP 的线编码 + header/LCRC 开销。
3. **CXL 后端是 8 DDR = 400 GB/s**（50 GB/s/DDR），比 host DRAM（4 DIMM / 8 子通道 ≈
   140.8 GB/s）更宽。

于是产生两条**互不争抢上行**的 staging 数据线，且**都是 Gen5 x16**：

- **Line 1（GPU→DRAM）**：`GPU → PCIe Switch → Root Complex（上行）→ Host DRAM Controller
  → 4 DIMM`。**占用 switch 上行**。
- **Line 2（GPU→CXL）**：`GPU → PCIe Switch → CXL Expander（直连，P2P）→ 8 DDR`。这条线是
  **switch 本地 P2P**，**不经过 switch 上行、不经过 Root Complex/CPU**，且由 expander 的
  DMA 主导、**纯 TLP 无 flit**。

> CXLMemSim 的 checkpoint 引擎是**时间模型**（`busy_until` 带宽串行化 + 每操作延迟按
> Little's Law 重叠），没有 gem5 的 PCI 设备 / DMA 端口 / Ruby 目录。因此「双独立 Gen5 线」在
> CXLMemSim 里不是改 `PcieLink`/`DmaPort`，而是**把配置里的单条共享上行拆成两条独立链路
> 描述符**，并**移除 CXL flit 折算**，让 `computeBounds`/`stage` 用「每 line 独立 min、两 line
> 相加」取代「单上行 min」。这是对 `checkpoint/README.md` 第 1 节「与 SimCXL 的映射」里
> `PcieLink` 那一行的细化。

---

## 二、理论依据（论文 + 规范）

论文：**"Replacing NVMe Staging in LLM Inference with a High-Bandwidth CXL
Memory Expander with an On-Device DMA Controller"**（Burru, Nalla, Prasad,
SIGCOMM 2026，DOI 10.1145/3789240.3822568）。新拓扑比旧 target2 **更直接地**命中论文的
机制：

1. **CXL 内存扩展器带 on-device DMA controller**：用它替代 NVMe staging。LLM inference 的
   KV cache / 中间张量溢出时 stage 到 CXL expander（本拓扑的 8 DDR）。
2. **当代 GPU 原生不参与 CXL 协议**：论文的做法是**设备端 DMA 控制器直接驱动 PCIe P2P
   传输、直击 GPU HBM**；CXL 只在 host 侧用于容量扩展与设备管理。这正是本拓扑
   「CPU 发 CXL.io 给 expander、expander 主导 GPU→CXL DDR 的 DMA、数据走纯 PCIe TLP P2P」
   的技术含义——**数据路径没有 CXL.mem 的 68B flit**。
3. 实测带宽：expander 达 **~51 GB/s 读（≈Gen5 x16 峰值 80%）**、**~33 GB/s 写**；写比读慢
   ~1.5×，归因于 PCIe 写完成语义（posted vs non-posted）。
4. **多线程描述符提交**把达到峰值带宽的 block size 从 8 MB 降到 256 KB，支撑 KV cache 更新
   这类细粒度、高频 staging 负载。

配套数值（本计划合理性标定）：

| 项 | 数值 |
|---|---|
| PCIe Gen5 x16 峰值（128b/130b） | 32 GT/s × 16 × 128/130 ≈ **63.0 GB/s** |
| PCIe Gen4 x16 峰值 | 16 GT/s × 16 × 128/130 ≈ **31.5 GB/s** |
| CXL expander 实测读 / 写（论文） | ~51 / ~33 GB/s（写≈读×0.65，写完成开销） |
| DDR5-4400 单 32-bit 子通道 | 4400 MT/s × 4B ≈ **17.6 GB/s** |
| CXL expander 后端（本拓扑） | 8 DDR × 50 ≈ **400 GB/s** |
| Host DRAM 后端（本拓扑） | 4 DIMM / 8 子通道 × 17.6 ≈ **140.8 GB/s** |

**关键结论（本拓扑的核心修正）**：数据线（GPU→CXL DDR）是 **PCIe P2P TLP**，因此 CXL 线的
有效带宽**不再乘 CXL.mem 的 68B flit 折算（64/68）**，而只按 PCIe TLP 的线编码 + header/LCRC
折算——**与 DRAM 线完全一致**。CXL 的 68B flit 只出现在 CXL.mem 事务里；本拓扑的 DMA 走
PCIe P2P，没有这条开销。这就是旧 target2 把 CXL 线算低（~29.7/27.75 GB/s）的根因，也是本次
改造要修正的地方。

---

## 三、目标澄清：优化的是 staging，不是 storage 落盘

沿用 `checkpoint/README.md` 第 2 节的分层结论，把优先级钉死在 **staging** 上：

- **阶段 1（同步，GPU 可见）**：GPU 把状态写入 DRAM / CXL（staging），写完立刻回到训练/
  推理。**首要优化对象**，指标是 **staging 带宽** 与 **GPU stall time**。
- **阶段 2（异步，后台）**：DRAM / CXL → storage 慢慢落盘，不阻塞 GPU，次要目标。

**本拓扑（双独立 Gen5 上行 + 多通道后端）的全部价值都落在阶段 1**：两条独立 Gen5 x16 线让
staging 带宽可加性成立（`B_line1 + B_line2`），多通道后端（4 DIMM + 8 DDR）让每条线都能吃到
多通道并发。storage 落盘（`ParallelStorage`）保持不变、不进关键路径——这与现有
`checkpoint/README.md` 的结论方向一致，只是把它从「单上行封顶 → 收益为零」修正为
「双线可加 → 收益为正（多卡时上行瓶颈被 CXL 线完全分担）」。

因此本计划的验收矩阵**以 staging 带宽 / GPU stall 为主指标**，storage 吞吐、CXL 热备等原
矩阵项降为从属指标（见第五节）。

---

## 四、与 checkpoint/README.md 的关系（继承 P0–P10，新增 E 阶段）

`checkpoint/README.md` 已交付（全部复用）：`ckpt::payloadFill`/`crc32`/`sha256`、
`ParallelStorage`（N 通道条带 + 反压）、`CheckpointEngine`（stage/persist/restore）、
`Balancer`（predicted-finish 分流）、`PinnedPool` 压力状态机、manifest 版本/代际回退、
`HotStandby`（CXL 热备）、`LaneStats` 分链路统计、`analytical_model.py`。这些**全部复用**。

本计划要推翻/修正的是两个建模假设：

1. **P10/README 的「单共享上行」**：`CheckpointConfig` 用单个 `gpu_pcie_write_gbps` 表达
   「GPU 只有一条 PCIe 上行、DRAM 与 CXL 都走它」。本计划改成**每条线一个独立链路描述符
   （`line_dram` / `line_cxl`）**，DRAM lane 与 CXL lane 分别接不同链路，其中 CXL line 不占
   上行（P2P）。
2. **README 的「CXL 68B flit 折算」**：`cxlOverheadFactor`（`128/130 × 64/68`）把 CXL 线
   按 CXL.mem flit 折低。本计划**移除 flit 折算**——CXL 线的 DMA 是纯 PCIe TLP，只按 TLP
   线编码/header/LCRC 折算（与 DRAM 线同式）。同时 CXL 后端由「2 links × 32 raw」改为
   「**8 DDR = 400 GB/s**」（expander 直连、无 CXL switch）。

新增阶段编号沿用 **E0–E5（epoch 2）**，但实现载体是 CXLMemSim 的时间模型 + 单元测试 +
分析模型（无 gem5 全系统）。

---

## 五、微调后的验收矩阵

保留原有「正确性」与「分链路统计」的骨架，把「双路径带宽」升级为"两条独立 Gen5 x16 线可加"，
并新增 3 条拓扑/物理层现实性指标。**粗体为主（staging）指标，普通字体为保留的从属指标。**

| # | 验收项 | 指标 / 判据 | 对应阶段 | 与旧矩阵的关系 |
|---|---|---|---|---|
| **A1** | **双独立 Gen5 x16 线 staging 带宽可加** | 覆盖两个场景：**S1 单 GPU 双路径（DRAM 窄后端）加速比**；**S2 4 GPU 双路径（DRAM 4 DIMM + CXL 8 DDR）加速比**。判据 `B_stage ≈ min(GPU_egress, B_line1 + B_line2)`，且 `B_stage > max(B_line1, B_line2)`，同时 `B_stage > 旧共享上行拓扑的 B_stage` | E0/E4 | 升级旧 #2 |
| **A2** | **两条线不共享 switch 上行** | CXL line 的流量不经过 Root Complex 上行（per-line 计数器证明 CXL lane 的字节不计入 uplink 方向） | E0/E2 | 新增 |
| **A3** | **多通道后端线性扩展** | host DRAM 4 DIMM（8 子通道）与 CXL expander 8 DDR 各自的多通道并发使单线带宽随通道数上升；S2 里 8 个 CXL DDR 同时被使用 | E1/E4 | 拆自旧 #2/#3 |
| **A4** | **PCIe 物理层现实性（纯 TLP，无 flit）** | 两条线都按 Gen5 x16 + 128b/130b + TLP header/LCRC 建模（**无 68B flit**）；GPU 上行 Gen4 x16；单 Gen5 x16 写 ≈ 33 GB/s、读 ≈ 51 GB/s 量级（对齐论文） | E3 | 新增（并修正旧 flit） |
| **A5** | **On-device DMA / P2P 语义** | CXL 线由 expander 侧 DMA 发起、直击 GPU HBM（device-initiated）；CPU 只发 CXL.io 控制面，不 host-initiate 数据搬运 | E2 | 新增 |
| A6 | 数据一致（GPU→DRAM/CXL→…→GPU CRC/SHA 一致） | 每 chunk CRC32 + 全 checkpoint SHA-256 + GPU 端逐字节重放比对 | E4 | 保留旧 #1/#5 |
| A7 | 分链路统计 | 每条线独立统计排队/带宽/延迟/重试/完成字节（`LaneStats` + 新增 `LinkStats`） | E0 | 保留旧 #8 |
| A8 | CXL 热备命中恢复加速 | 命中率 0/25/50/100% 下恢复时间单调下降 | E4 | 保留旧 #4 |
| A9 | 后台落盘与压力/生命周期 | 2 存储通道 > 单通道；压力下 pinned pool 只复用不扩大；在飞 DMA 完成前禁 unpin | E4 | 保留旧 #3/#6/#7 |

> 主指标 A1–A5 全部围绕 staging；A6–A9 保证不回归 `checkpoint/README.md` 已交付的正确性
> 与从属功能。A1 是核心，A2/A3/A5 是"为何能加速"的机理，A4 是"算得对不对"（本版把 flit
> 折算从 A4 里拿掉，改为纯 TLP）。

---

## 六、开发计划（E0–E5）

原则：**最小改动复用 P0–P10 成果，只在带宽上界与数据通路配置上动手**；每个阶段都给出
关键接口、文件位置与验证方法，且都用宿主机（无 gem5）单测闭环。

### E0：双独立 Gen5 x16 链路骨架（推翻"单共享上行"）

目标：把 `CheckpointConfig` 从「单共享上行」改回「每 line 一条 Gen5 x16 链路」，DRAM line
与 CXL line 分别建模，`computeBounds`/`stage` 用可加性取代单上行 min。

改动（CXLMemSim 侧，全部集中在时间模型与配置，无设备/DMA 端口）：

1. `include/checkpoint_engine.h`：新增 `PcieLinkConfig`：
   ```cpp
   struct PcieLinkConfig {
       double raw_gbps = 64.0;              // raw 链路速率（Gen5 x16 = 64 / Gen4 x16 = 32）
       bool   consumes_uplink = true;       // Line1(DRAM) 占上行；Line2(CXL P2P) 不占
       bool   device_initiated = false;     // Line2: expander 侧 DMA 发起（P2P）
       double write_read_ratio = 1.0;       // 写/读不对称（E2/E3 落点，先 1.0 对称）
   };
   ```
   `CheckpointConfig` 增加 `PcieLinkConfig line_dram, line_cxl;`、`uint32_t num_gpus = 1;`、
   `double gpu_egress_gbps = 32.0;`（每 GPU PCIe Gen4 x16 egress，源端上界）。
   **两条线都默认 Gen5 x16（raw 64）**：`line_dram` 走上行，`line_cxl` 直连 expander（P2P）。
2. `src/checkpoint_engine.cpp`：
   - `computeBounds()` 改为：
     ```cpp
     double line1 = dram_enabled ? std::min(pcieWriteGbps(line_dram), dram_write) : 0.0;
     double line2 = cxl_enabled  ? std::min(pcieWriteGbps(line_cxl),  cxl_write)  : 0.0; // cxl_write 无 flit
     b.bstage_gbps = std::min(num_gpus * gpu_egress_eff, line1 + line2);
     ```
   - `stage()` 把单条 `gpu_emission_ns = total_bytes / pcie_eff` 拆成
     `egress_ns = staged_bytes / (num_gpus * egress)`、`line1_ns = dram_bytes / pcieWrite(line_dram)`、
     `line2_ns = cxl_bytes / pcieWrite(line_cxl)`，
     `stall = max(egress_ns, line1_ns, line2_ns, dram_absorb, cxl_absorb) + write_latency`。
3. `include/checkpoint_engine.h` 新增 `LinkStats`（每 line 独立：`write/read/completed_bytes`、
   `device_initiated_bytes`/`host_initiated_bytes`、`counts_toward_uplink`），挂进 `EngineStats`。
4. `checkpoint/analytical_model.py`：`Config` 增加 `line_dram/line_cxl/num_gpus/gpu_egress`，
   `bounds()` 同步改为可加性；`stage_time()` 改用 per-line 发射。

验证（宿主机，无 gem5）：

- 更新 `tests/test_checkpoint_lanes.cpp`：only-DRAM / only-CXL / both 三条路径；在
  `num_gpus=4`、`line_dram=Gen5 x16`、`line_cxl=Gen5 x16(不占上行)` 下，断言
  `both > max(only-DRAM, only-CXL)`，且 `both ≈ min(4×egress, line1+line2)`。
- 新增「旧共享上行」对照组：临时令 `line_cxl.consumes_uplink=true` 且
  `line_cxl.raw_gbps = line_dram.raw_gbps`（两 line 复用一条 link），直接量化 A1 的「新 > 旧」。
- `./test_checkpoint_lanes` 与 `ctest -R test_checkpoint --output-on-failure` 全绿。

### E1：多通道后端（4 DIMM host DRAM + 8 DDR CXL expander）

目标：每条线吃到多通道并发，让单线带宽不再被单通道上限卡死。CXLMemSim 的
`PoolConfig::num_lanes` 本就是标定输入，这一步主要是**改标定值 + 验证取 min 逻辑**。

改动：

1. `tests/checkpoint_test_harness.h` 的 `defaultConfig()`（及后续 S1/S2 配置）：
   - host DRAM：`dram.num_lanes = 8`、`dram.write_gbps = 17.6`——4 DIMM、每 DIMM 2 条
     32-bit 子通道 = 8 子通道（对齐拓扑图 DDR0..DDR7，Σ ≈ 140.8 GB/s）。
   - CXL expander：`cxl.num_lanes = 8`、`cxl.write_gbps = 50.0`（每 DDR 50 GB/s）——
     **8 DDR = 400 GB/s，且不再乘 flit 折算**（expander 直连、纯 TLP）。
2. `analytical_model.py`：确认 `B_line ≤ min(link, Σ channel_bw)` 的 min 逻辑；打印每条
   line 的 `min(link, aggregate)` 分解。

验证：

- `test_checkpoint_lanes` 的 single-line 模式：line_dram（8 子通道）与 line_cxl（8 DDR）
  各自单线带宽随通道数上升（对照 E1 之前单通道基线）。
- `python3 checkpoint/analytical_model.py --dram-lanes 8 --dram-write 17.6 --cxl-lanes 4 --cxl-write 50`
  输出 `B_line1`/`B_line2` 分解。

### E2：On-device DMA / P2P 语义 + 移除 CXL flit 折算

目标：让 CXL 线在语义上"由 expander 的 on-device DMA 发起、直击 GPU HBM"，且**数据线是纯
PCIe TLP（无 CXL.mem flit）**——CPU 只发 CXL.io 控制面。CXLMemSim 时间模型没有 host CPU /
Ruby 目录，这一步用**语义标记 + 统计区分 + 移除 flit 系数**表达。

改动：

1. **移除 flit 折算**：删除 `CheckpointConfig::cxl_flit_data_bytes/cxl_flit_total_bytes` 与
   `cxlOverheadFactor` 对数据线的折低。CXL 线带宽 = `cxl.aggregateWriteGbps()`（不再乘
   64/68）；链路侧只按 `pcieEffectiveGbps`（128b/130b + TLP header/LCRC）折算，与 DRAM 线同式。
   （如需保留 CXL.io 控制面开销，仅作为一次常数延迟，不进带宽公式。）
2. `PcieLinkConfig` 增加 `bool device_initiated = false;`（`line_cxl` 置 `true`）；`LinkStats`
   增加 `device_initiated_bytes`/`host_initiated_bytes` 两套计数器（A5/A7 依据）。
3. `line_cxl.consumes_uplink = false`：`computeBounds`/`stage` 对 `line_cxl` 不扣减上行预算；
   `line_dram` 保持 `consumes_uplink = true`。
4. **写完成开销建模**：`pcieWriteGbps` 写方向乘 `write_read_ratio`（posted vs non-posted 写
   完成语义），初始 `0.65`（对齐论文 ~1.5× 读/写差）。staging 用 `pcieWriteGbps`，restore 用
   `pcieReadGbps`。

验证：

- `test_checkpoint_dual_link`：仅 line_cxl staging 时，`line_dram` 的 `counts_toward_uplink`/
  `completed_bytes` 为 0，只有 `line_cxl` 计数器增长（A2/A5 判据）；且 `line_cxl` 的
  `device_initiated_bytes > 0`、`host_initiated_bytes == 0`（A5）。
- 宿主机：`pcieWriteGbps(line_dram) ≈ pcieReadGbps(line_dram) × 0.65`；`cxl` 线带宽不乘
  64/68（= 纯 TLP 值）。

### E3：PCIe 物理层精细化 + 对标论文

目标：把链路建模精度提到论文可对标水平，且两条线对称（都 Gen5 x16、都纯 TLP）。

改动：

1. `PcieLinkConfig` 参数化 GPU link 为 Gen4（`gpu_egress_gbps = 32`，Gen4 x16）、switch 上行
   与 expander 直连都为 Gen5（`line_dram.raw_gbps = line_cxl.raw_gbps = 64`）。**两条数据线
   速率一致**，差异只在 `consumes_uplink`（DRAM 占上行 / CXL 不占）。
2. **MPS / 多描述符提交建模**（对应论文"多线程描述符提交把峰值 block size 从 8 MB 降到
   256 KB"）：把 `CheckpointConfig` 的 staging 批量提交（每 lane 的 in-flight chunk 数）与
   `pcie_max_payload_bytes`（MPS）联动，而非固定一个 magic 常数。
3. 读/写不对称参数化（E2-4 落点）：`write_read_ratio` 可 CLI/配置标定。

验证（对标论文，宿主机单 lane Gen5 x16 测量）：

| 项 | 期望 |
|---|---|
| 单 Gen5 x16 读 | ~51 GB/s（≈80% 峰值） |
| 单 Gen5 x16 写 | ~33 GB/s（写完成开销） |
| 峰值 block size（多描述符） | 从 ~8 MB 降到 ~256 KB |

### E4：双线 staging 验收 + 分流策略重标定

目标：跑通第五节 A1–A9，落实第八节 S1/S2 两个核心加速比场景。

改动：

1. 新测试 `tests/test_checkpoint_dual_link.cpp`：
   - **S1**：`num_gpus=1`；DRAM 窄后端（1 子通道 17.6）vs dual（+ 1 CXL DDR）；输出
     `B_dual / B_single`。
   - **S2**：`num_gpus=4`；DRAM 4 DIMM（8 子通道）基线 vs dual（DRAM 8 子通道 + CXL 8 DDR）；
     输出 `B_dual / B_dram_only`。
   - 二者都带"旧共享上行"对照组（临时 `line_cxl.consumes_uplink=true` + 复用 line_dram
     link），直接量化 A1 的「新 > 旧」。
2. `Balancer` 改为**任务级 offload plan**（见第七节）：删除热路径里的 per-chunk
   `chooseLane` 贪心，换成 `PlacementPlan` 表查询。`laneRate()` 的 link 上限并入
   `planOffload()`：
   ```cpp
   R_dram = min(pcieWrite(line_dram), dram_read, dram_write)   // 仅 dram_enabled 时
   R_cxl  = min(pcieWrite(line_cxl),  cxl_read,  cxl_write)    // 仅 cxl_enabled 时（无 flit）
   ```
   `split_unit_bytes`（默认 256 MiB）在 memcpy 选地址时定死 chunk→lane；`stage()` 只做
   O(1) 表查，不再每 chunk 重算 predicted-finish；重计划由漂移触发（见第七节 7.5）。
3. `analytical_model.py`：新增 `bounds_dual_link()`（`B_stage = min(num_gpus * egress,
   line1 + line2)`）、`stage_time()` per-line 发射；给 `topology` 概念加"每 line 独立
   link"的候选路径（DRAM→host 上行 / CXL→switch 本地 P2P）。

验证：

- 宿主机 `test_checkpoint_dual_link`：S1（512 MiB 单卡）与 S2（4×512 MiB）全量毫秒级验证
  均衡 split 与可加性上界（时间模型单测很快，无需周期仿真）。
- `test_checkpoint_acceptance` 更新：A1–A9 全跑，输出分链路统计与 S1/S2 的 staging 带宽与
  加速比，期望 `B_stage ≈ B_line1 + B_line2`（受 GPU egress 封顶）且显著高于"共享上行"
  对照组。旧 #2 的诚实 FAIL 翻转为 A1 PASS。

### E5：分析模型协同 + 扩展（多 CXL expander、多 GPU）

目标：把时间模型抽样的结论回填分析模型，扩展到拓扑图里的完整形态。

改动：

1. `analytical_model.py` 支持 N 个 CXL expander（`cxl.num_lanes` 泛化为 N 条独立 DDR）、
   M 个 GPU 的扫描（每 GPU 独立 Gen4 x16 egress，switch 上行 Gen5 x16 为多 GPU 去 DRAM 的
   共享瓶颈；CXL expander 直连、P2P 不占上行）。
2. C++ 引擎抽样验证分析模型在关键拓扑点（1/4 GPU、1/2/4 CXL DDR）的预测（沿用 README
   第 3 节"分析模型 vs 引擎实测并排"的输出格式）。
3. 输出论文级分链路报告（staging 带宽、GPU stall time、每 line P95、上行占用率）。

验证：

- 分析模型与 C++ 引擎在 1 GPU/4 GPU、1/2/4 CXL DDR 等关键点的误差 <5%。

---

## 七、任务级分流（offload plan）算法

> 本节取代 SimCXL `target.md` 第五节 / 本计划 E4 里的「在线 predicted-finish 分流」。
> 核心观点：LLM checkpoint 数据量固定、DRAM/CXL 的容量与放置也固定，因此分流不是每个
> TLP / 每个 4 KiB chunk 的在线决策，而是在**任务提交时一次性解出静态放置计划**，中间只
> **周期性监控 + 漂移触发重计划**。分流粒度是 `split_unit_bytes`（默认 256 MiB）——`memcpy`
> 时选中的目标地址就已经决定这块数据去 DRAM 还是 CXL。

### 7.1 为什么在线 per-chunk 分流是错的

现有 `Balancer::chooseLane`（`src/checkpoint_engine.cpp`）每来一个 chunk 就算一次
`predicted_finish_i = (queued_bytes + chunk_size) / R_i` 并选更短者。问题有三：

1. **决策次数荒谬**：5 GiB / 4 KiB ≈ 130 万次决策，而真正要决定的只是 ~20 个 256 MiB
   split unit 的去向（`20 × 256 MiB` 这个量级）。
2. **问题本身是静态凸优化**：`C` 固定、`B_line1`/`B_line2` 固定、容量固定，最优 split 有
   闭式解（水填），不需要贪心在线近似；在线贪心还会引入 per-chunk 抖动与决策开销。
3. **分流本质是放置不是路由**：memcpy 时选中的目标地址就已经决定 DRAM 还是 CXL；per-TLP
   路由是过度设计。

### 7.2 输入（关键变量）

| 变量 | 含义 | 类型/例子 |
|---|---|---|
| **内存数据集** `MemoryLayout` | DRAM/CXL 两池容量、每 lane 写/读带宽、延迟、lane 数，及扣除压力预留后的 `usable_capacity` | `dram{cap, 8 子通道×17.6, write_lat}`、`cxl{cap, 8 DDR×50, 无 flit}` |
| **checkpoint 总数据大小** `C` | 固定值 + `split_unit_bytes`（256 MiB）+ `chunk_size`（4 KiB） | `C = 5 GiB`，`S = 20` 个 split unit |
| **offload 卸载** `OffloadSpec` | 卸载量 + 放置亲和：`offload_bytes`、`affinity[unit]∈{EITHER,DRAM,CXL}`、可选 `hot_set`（→CXL 热备） | optimizer/KV → CXL；大冷权重 → DRAM |
| 链路/出口速率 | `B_line1`（GPU→DRAM，Gen5 上行）、`B_line2`（GPU→CXL，Gen5 P2P 纯 TLP）、`gpu_egress`、`num_gpus` | 来自 E0 配置 |

> **offload 卸载**是关键自由度：它既给出要 stage 的数据量（卸载多少），又给出**哪些 tensor
> 优先去 CXL**（需频繁重读、走热备）还是**去 DRAM**（大批量冷数据）。因此 split 不是纯带宽
> 均分，而是「容量 + 带宽平衡 + 放置亲和」三者约束下的静态规划。新拓扑里 `B_line1 ≈ B_line2`
> （都 Gen5 x16），水填退化为近似 50:50，容量/亲和成为主要约束。

### 7.3 一次性规划（任务提交时）

**Step 1 容量可行性**：
```
C_dram_min = Σ {亲和=DRAM 的 unit 字节}
C_cxl_min  = Σ {亲和=CXL 的 unit 字节}
C_free     = C - C_dram_min - C_cxl_min      // 亲和=EITHER 的字节
要求 C_dram_min ≤ dram.usable 且 C_cxl_min ≤ cxl.usable（否则拒绝任务 / 回退落盘）
```

**Step 2 带宽水填（解 C_free 的分配）**：设 `x` 为「free 字节分给 DRAM」的量，staging 时间
（GPU 可见）是三条串行化源的最大值：
```
t1(x)  = (C_dram_min + x) / B_line1
t2(x)  = (C_cxl_min + (C_free - x)) / B_line2
t_egr  = C / (num_gpus * gpu_egress)
stall(x) = max(t1, t2, t_egr)
```
无约束最优在 `t1 = t2`（水填）：
```
x* = (B_line1 * (C_cxl_min + C_free) - B_line2 * C_dram_min) / (B_line1 + B_line2)
```
再夹到容量上界：
```
0 ≤ x* ≤ C_free
x* ≤ dram.usable - C_dram_min
C_free - x* ≤ cxl.usable - C_cxl_min
```
这是**闭式解、一次算完**，无在线搜索。最终 `alpha = (C_dram_min + x*) / C`。新拓扑里
`B_line1 = B_line2 = 55.53`，无容量/亲和约束时 `x* = C_free/2`（50:50）。

**Step 3 生成放置计划**：`unit_lane[s] ∈ {DRAM,CXL}`（s=0..S-1）+ `dram_bytes`/`cxl_bytes`
/`alpha`。memcpy 按 `unit_lane[s]` 选目标地址——**选地址即分流**；之后逐 4 KiB chunk 只做
O(1) 表查 `unit_lane[chunk_id * chunk_size / split_unit_bytes]`。

### 7.4 稳态（热路径无 per-chunk 决策）

`CheckpointEngine::stage` 逐 chunk 写 payload + CRC，但 lane 由计划表查得，不再调用
`chooseLane`、不算 predicted-finish。每 lane 照常累计 `LaneStats`（EWMA 带宽、retry、
queue_full、pool free），供监控用。

### 7.5 周期性监控 + 漂移触发重计划

每 `monitor_period_units` 个 split unit（或每 `monitor_period_ns`），对每 line 计算漂移：
```
drift(line) = |B_measured(line) - B_assumed(line)| / B_assumed(line)
trigger = drift > δ_bw
       OR retries + queue_full > δ_q        // 重发激增
       OR pool_free < δ_cap                  // 容量逼近
       OR 链路故障/降速                      // 线路数据变化过大
```
- **触发后只重计划未发出的尾部 unit**：用实测 `B_line1`/`B_line2` 与剩余容量重跑 7.3 的
  Step 2，更新 tail 的 `unit_lane`。**已 pin 的 unit 不搬**（避免双重写 / 在飞 unpin）。
- 重计划成本 O(S)，摊到 S 个 unit 上可忽略；监控是周期性的，不碰热路径。

### 7.6 接口与验证

- C++（`include/checkpoint_engine.h` / `src/checkpoint_engine.cpp`）：新增 `OffloadPlanner`
  /`PlacementPlan`（`planOffload`/`replan`）、`MemoryLayout`、`OffloadSpec`；`Balancer` 保留
  `LaneStats` 统计与 token bucket，`chooseLane` 改为计划表查询 + `shouldReplan()` 漂移判定；
  `CheckpointConfig` 增 `split_unit_bytes`、`monitor_period_units`、`δ_bw/δ_q/δ_cap`。
- `checkpoint/analytical_model.py`：新增 `plan_split(...)`（水填闭式）与 `replan(...)`，和
  C++ 同式。
- 验证 `tests/test_checkpoint_offload.cpp`：① 计划满足容量/亲和约束；② `alpha` 等于水填闭式
  解；③ 无漂移时 plan 稳定、决策次数 == S（而非 N_chunk）；④ 注入链路降速后 `replan` 把 tail
  移向健康 line；⑤ S1/S2 的 `B_stage` 与在线贪心一致或更优（至少不差）。

---

## 八、两个核心验收测试场景（S1 / S2）

A1 的两个加速比场景。两者都以"写 staging"为方向（GPU→内存），用 steady-state 带宽度量；
**单卡数据量 ≥256 MiB，默认/推荐 512 MiB**（`checkpoint/README.md` 与 `readme_cxl.md`
结论：512 MiB 才能摆脱 ~1 ms 固定开销、到达稳态）。CXLMemSim 时间模型无冷启动/尾部排出
的固定开销问题（`busy_until` 串行化 + 延迟做常数偏移），但为了与 SimCXL 口径一致、且让
`chunk_size` 条带在 8 子通道/8 DDR 上铺满，仍沿用 512 MiB 量级。

两个场景的分流比例 `alpha` 均由第七节的 offload plan 一次性算出：新拓扑里
`B_line1 = B_line2 = 55.53`（都 Gen5 x16 纯 TLP），无容量/亲和约束时 `alpha = 0.5`。

### S1：单 GPU 双路径（DRAM 窄后端）加速比

问题：单卡只有一个 Gen4 x16 egress（≈27.76 GB/s 有效），而单条 DDR5 子通道只有 ~17.6 GB/s。
若 GPU 只写单条 DRAM 子通道，链路远未吃满；把数据分流到两条独立后端（1 条 DRAM 子通道 +
1 条 CXL DDR），让两个后端控制器并行吸收，从而吃到 GPU egress 的全部带宽。

| 项 | 值 |
|---|---|
| 配置 | `num_gpus=1`，`gpu_egress_gbps=32`（Gen4 x16 → 27.76 有效）；`line_dram`=Gen5 x16 上行 55.53；`line_cxl`=Gen5 x16 P2P 55.53（纯 TLP） |
| 后端 | Line1 = 1 DRAM 子通道（17.6）；Line2 = 1 CXL DDR（50，无 flit） |
| 基线 | 单路径：仅 DRAM 1 子通道，`B_single` |
| 实验 | 双路径：1 DRAM 子通道 + 1 CXL DDR 并发 `B_dual` |
| 数据量 | **≥256 MiB，推荐 512 MiB**（每 GPU） |
| 指标 | `speedup = B_dual / B_single` |
| 期望 | `B_single ≈ 17.6`；`B_dual ≈ min(27.76, 17.6+50)=27.76`；**speedup ≈ 1.5–1.6×**（GPU egress 限） |

判据：`speedup > 1.3` 且 `B_dual` 逼近单卡 Gen4 x16 egress 上限（证明两路不因共享上行互相
踩踏，单卡就能靠双后端并行受益）。

### S2：4 GPU 双路径（DRAM 4 DIMM + CXL 8 DDR）加速比

问题：4 卡合计 egress ≈ 4×27.76 = 111 GB/s，而 switch 上行只有 Gen5 x16 ≈ 55.53 GB/s。旧
拓扑"只写 DRAM"时，4 卡全挤同一条上行，上行就是硬瓶颈。新拓扑把一半 staging 迁到 switch
本地的 CXL expander（8 DDR，不占上行），给 DRAM 线腾出上行，总 staging 带宽越过上行上限、
直达 4 卡 egress 上限。

| 项 | 值 |
|---|---|
| 配置 | `num_gpus=4`，每卡 `gpu_egress_gbps=32`（合计 111）；`line_dram`=Gen5 x16 上行 55.53；`line_cxl`=Gen5 x16 P2P 55.53（纯 TLP） |
| 后端 | Line1 = 4 DIMM（8 子通道 ≈140.8）；Line2 = 8 CXL DDR（8×50 = 400，无 flit） |
| 基线 | 传统只写 DRAM：4 GPU 全走上行 → 4 DIMM，`B_dram_only`（上行限 ≈55.53） |
| 实验 | 双路径：分流到 DRAM 4 DIMM + CXL 8 DDR 并发 `B_dual` |
| 数据量 | **每 GPU ≥256 MiB（合计 ≥1 GiB），推荐每 GPU 512 MiB（合计 2 GiB）** |
| 指标 | `speedup = B_dual / B_dram_only` |
| 期望 | `B_dram_only ≈ 55.53`（上行限）；`B_dual ≈ min(111, 55.53+55.53)=111`；**speedup ≈ 2.0×**（上行瓶颈被 CXL 线完全分担，直达 egress 上限） |

判据：`speedup > 1.3` 且 CXL line 的 per-line 统计显示其流量不计入上行方向（联动 A2）；8 个
CXL DDR 确实被同时使用（联动 A3）。

### 数据量与仿真规模（可行性与一致性）

- **稳态阈值**：512 MiB 是摆脱固定开销、到达稳态带宽的最低量级，S1 必须满足；S2 按每 GPU
  256 MiB 起步。
- **时间模型成本**：CXLMemSim 引擎是闭式 + `busy_until` 串行化，S1/S2 全量（4×512 MiB）在
  宿主机毫秒级即可跑完，**无需像 gem5 那样抽样**。这是 CXLMemSim 相对 SimCXL 的显著优势：
  没有周期仿真成本，S2 全量直接跑。
- **计时口径**：沿用 `EngineStats::gpu_stage_time_ns`（`stage()` 返回的 GPU stall 时间），
  度量"最后一字节写进 pinned pool"的端到端 staging 时间，而非注入吞吐；与 `readme_cxl.md`
  的 `bytesWritten == copy_bytes` 修正口径等价。

---

## 九、合理性论证（为什么这个模拟内容既现实、又能加速 staging）

1. **两线可加性是物理真实，不是假设**。论文的核心机制就是"expander 直连 switch 本地用 P2P
   直击 GPU HBM、host 侧 CXL 只做容量与管理"，因此 CXL 线天然不占上行。我们的 E0/E2 把这个
   机理显式建模成"两条独立 `PcieLinkConfig` + CXL 线 `consumes_uplink=false` + on-device
   DMA"，是论文拓扑的直接仿真，而非拍脑袋。
2. **纯 TLP、无 flit，比旧 target2 更准确**。旧 target2 把 CXL 线按 CXL.mem 68B flit 折低，
   但本拓扑的 DMA 走 PCIe P2P TLP，根本没有 flit。E2 移除 flit 折算后，CXL 线与 DRAM 线同式
   （都 Gen5 x16、都按 128b/130b + TLP header/LCRC），对标论文（读 51/写 33 GB/s）是**可复现
   的收敛目标**。
3. **多通道后端（4 DIMM + 8 DDR）匹配现实服务器**。拓扑图 DDR0..DDR7 = 4 个 DDR5 DIMM（每
   DIMM 2 子通道），CXL expander 8 DDR = 400 GB/s；CXLMemSim 的 `PoolConfig::num_lanes` 已是
   标定输入，直接填 8 子通道 / 8 DDR 即可，无需新代码（只动 `defaultConfig`）。
4. **加速来源清晰、可量化**。旧拓扑 `B_stage ≤ 上行（Gen5 x16 ≈ 55.53）`；新拓扑
   `B_stage ≈ min(GPU_egress, B_line1 + B_line2)`，且 Line 2 不占上行、与 Line 1 同速。多卡时
   `B_line1 = B_line2 = 55.53`，`B_stage` 直达 `num_gpus × egress`（4 卡 = 111），**speedup
   ≈ 2.0×**。A1 的"新 > 旧"对照组直接度量这个加速。
5. **复用已交付资产，不重复造轮子**。payload/CRC/SHA、`ParallelStorage`、`Balancer`、
   `PinnedPool` 压力机、manifest 代际回退、`HotStandby`、`LaneStats`、`analytical_model.py`
   全部沿用；本计划只动"配置的链路描述 + 移除 flit + 带宽上界 + staging 时间模型"这一层，
   风险可控。
6. **每个阶段可独立验证**。E0–E3 各有宿主机单测闭环，E4 一次性跑验收矩阵，E5 才做全量
   扩展——符合 README 一贯的"宿主机毫秒级单测"方法（本计划比 SimCXL 更进一步：连 S2 全量
   都无需抽样，因为时间模型无周期仿真成本）。

---

## 十、参考资料

- 论文（本计划理论依据）：*Replacing NVMe Staging in LLM Inference with a High-Bandwidth
  CXL Memory Expander with an On-Device DMA Controller*, Burru, Nalla, Prasad, SIGCOMM 2026,
  DOI 10.1145/3789240.3822568。
- 同组配套论文：*CXL Memory for LLM Inference Staging via an On-Device DMA Controller*,
  DOI 10.1145/3837053.3837356。
- PCIe 基础规范（128b/130b、Gen4=16 GT/s、Gen5=32 GT/s、MaxPayloadSize、posted/non-posted
  写完成语义、P2P TLP）；CXL 3.x 规范（CXL.io 控制面、CXL.mem 68B flit、Type-3 设备、HDM
  解码——**本拓扑数据线走 PCIe P2P TLP，不涉及 CXL.mem flit**）。
- SimCXL 前作：`../SimCXL/configs/example/gem5_library/target.md`（P0–P10 与验收矩阵）、
  `../SimCXL/configs/example/gem5_library/target2.md`（本计划的 SimCXL 对应版）、
  `../SimCXL/readme_cxl.md`（GPU→CXL 带宽方法与 512 MiB 稳态结论）、`../SimCXL/exec_summary.md`。
- 仓库内（CXLMemSim）：`checkpoint/README.md`（P0–P10 重实现 + 诚实结论）、
  `include/checkpoint_engine.h`（`CheckpointConfig`/`computeBounds`/`pcieEffectiveGbps`/
  `LaneStats`）、`src/checkpoint_engine.cpp`（`computeBounds`/`stage`/`Balancer`）、
  `include/parallel_storage.h` / `src/parallel_storage.cpp`（N 通道条带 + 反压）、
  `checkpoint/analytical_model.py`（闭式上界 + 扫描）、`tests/checkpoint_test_harness.h`
  （`defaultConfig`）、`tests/test_checkpoint_{lanes,balancer,acceptance}.cpp`。
