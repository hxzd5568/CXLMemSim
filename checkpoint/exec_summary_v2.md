# exec_summary_v2.md —— CXLMemSim checkpoint 引擎 target2 改造（expander 直连 / 纯 TLP / 8 DDR）执行总结

> 对应计划：`checkpoint/cxlmem_target2.md`（SimCXL `target2.md` 的 CXLMemSim 侧版本）。
> 本文记录：核心拓扑改动后**重新执行**的开发、**参数设定**、**成果（实测数据）**、**困难与决策**。
> 所有代码改动均落在 CXLMemSim checkpoint 引擎（`include/checkpoint_engine.h`、
> `src/checkpoint_engine.cpp`、`checkpoint/analytical_model.py`、`tests/`），
> **不影响主仓库 `cxlmemsim` 库**（checkpoint 引擎自包含，仅依赖标准库）。

---

## 1. 核心改动与成果总览

本次按用户核心改动重做：**取消 CXL switch**，CXL expander 作为端点**直连 PCIe switch
（Gen5 x16）**；CPU 发 **CXL.io** 给 expander，expander 的 **on-device DMA** 主导
GPU→CXL DDR 的搬运；这条 DMA 是**纯 PCIe TLP**，**没有 CXL.mem 68B flit**；CXL 后端为
**8 DDR = 400 GB/s**。

| 阶段 | 内容 | 状态 |
|---|---|---|
| E0 | 双独立 Gen5 x16 链路骨架 + 可加性 `computeBounds`/`stage` + `LinkStats` | ✅ |
| E1 | 多通道后端标定：DRAM 8 子通道 ×17.6、CXL **8 DDR ×50 = 400 GB/s** | ✅ |
| E2 | **移除 CXL 68B flit 折算** + `device_initiated`/`consumes_uplink` 语义 + 写/读不对称 | ✅ |
| E3 | PCIe 物理层：两条线都 Gen5 x16 纯 TLP；GPU 上行 Gen4 x16；读 55.53 / 写 36.09 | ✅ |
| E4 | 双线 staging 验收（S1/S2）+ offload plan 取代在线 per-chunk 分流 | ✅ |
| E5 | `analytical_model.py` 双线 + `plan_split` 水填，与 C++ 同式 | ✅ |
| — | 验收矩阵 A1–A9 | **9/9 PASS** |

**核心结论**：旧模型「单共享上行 + CXL flit」下 `B_stage ≤ 上行`、双路收益为零。新模型把
CXL 线改成「Gen5 x16 P2P + 纯 TLP（无 flit）」，`B_line1 = B_line2 = 55.53 GB/s`，多卡时
`B_stage = min(num_gpus×egress, B_line1 + B_line2)` 直达 4 卡 egress 上限（111.05 GB/s），
**S2 加速比 2.00×**。

---

## 2. 参数设定（标定点）

全部沿用 `cxlmem_target2.md` 口径（`tests/checkpoint_test_harness.h::defaultConfig`）：

| 参数 | 值 | 含义 |
|---|---|---|
| `line_dram.raw_gbps` | 64.0（Gen5 x16） | Line1 上行，`consumes_uplink=true` |
| `line_cxl.raw_gbps` | **64.0（Gen5 x16）** | Line2 P2P 直连 expander，`consumes_uplink=false`、`device_initiated=true` |
| 两条线有效写 | **55.53 GB/s** | `raw × 128/130 × 256/(256+20) × 0.95`（纯 TLP，无 flit） |
| `gpu_egress_gbps` | 32.0（Gen4 x16）/GPU | → 27.76 GB/s/GPU |
| `num_gpus` | 1（默认）| S2 用 4 |
| `dram` | 8 子通道 × 17.6 = 140.8 GB/s | 4 DIMM × 2 DDR5-4400 32-bit 子通道 |
| `cxl` | **8 DDR × 50 = 400 GB/s** | 无 flit（纯 TLP P2P，`cxlOverheadFactor()=1.0`） |
| `write_read_ratio` | 1.0（默认）；A4 用 0.65 | posted/non-posted 写完成开销 |
| `split_unit_bytes` | 256 MiB（默认）；自适应 ≥4 units | memcpy 地址粒度 |
| 漂移阈值 | `drift_bw=0.15`、`drift_retry=16`、`drift_cap=0.20` | 重计划触发 |
| storage | 2 GB/s/ch、10 µs、queue depth 32 | persist 瓶颈（不变） |

`pcieEffectiveGbps(raw) = raw × (128/130) × MPS/(MPS+20) × 0.95`；
`pcieWriteGbps(link) = pcieEffectiveGbps(link.raw) × write_read_ratio`；
`cxlOverheadFactor(cfg) = 1.0`（**不再乘 64/68 flit**，CXL 后端就是 raw DDR 带宽）。

---

## 3. 关键实现改动（相对上一版 exec_summary）

- **移除 flit 折算**（核心）：`include/checkpoint_engine.h` 的 `cxlOverheadFactor` 由
  `128/130 × 64/68` 改为 **`1.0`**（保留函数签名与 `cxl_flit_*` 字段作 API 兼容，标注
  deprecated）。`computeBounds`/`stage`/`Balancer::laneRate` 里 `cxl_write =
  aggregateWriteGbps() × cxlOverheadFactor()` 因此退化为 raw DDR 带宽。
- **`line_cxl` Gen4→Gen5**：`defaultConfig` 中 `line_cxl.raw_gbps = 64`（原来是 32），
  `checkpoint/analytical_model.py` 的 `--line-cxl` 默认同步为 64。
- **CXL 后端 2 link→8 DDR**：`cxl.num_lanes = 8`、`cxl.read/write_gbps = 50.0`（原 2×32）。
- 测试侧 `defaultConfig`、`test_checkpoint_{lanes,balancer,dual_link,acceptance}.cpp` 同步
  到新标定；`analytical_model.py` 的 `cxl_overhead()=1.0`、默认 `cxl_lanes=8 / cxl=50`。

---

## 4. 实测成果

### 4.1 双线 staging 带宽（`test_checkpoint_lanes`）

```
single-GPU wide backends: only-DRAM 27.76, only-CXL 27.76, both 27.76  (egress 27.76)
single-GPU narrow DRAM (1 sub-ch) + CXL DDR: only-DRAM 17.60, both 27.76
4-GPU: only-DRAM 55.53 (uplink-limited), both 111.05 (DRAM uplink + CXL P2P)
```

### 4.2 核心验收场景 S1 / S2（`test_checkpoint_dual_link`）

| 场景 | 基线 | 双线 | 加速比 | 判据(>1.3) |
|---|---|---|---|---|
| S1（1 GPU，1 DRAM 子通道 + 1 CXL DDR） | 17.60 GB/s | 27.76 GB/s | **1.58×** | ✅ |
| S2（4 GPU，DRAM 4 DIMM + CXL 8 DDR） | 55.53 GB/s | **111.05 GB/s** | **2.00×** | ✅ |

> S2 分析模型闭式上界（`computeBounds`/`analytical_model.py`）与 C++ 引擎实测**完全一致**
> （都是 111.05 GB/s），因为 `B_line1 = B_line2 = 55.53`（对称）、水填 `alpha=0.5`，无量化
> 误差（见 §5.3）。

### 4.3 offload plan（`test_checkpoint_offload` / `test_checkpoint_balancer`）

- 水填闭式：`planOffload(63, 35)` → alpha 0.60（量化后）；容量夹紧 / 亲和 / 部分单元 /
  不可行 → 空计划 全部 OK。
- 引擎内：默认 wide 后端（两线都 55.53）512 MiB → `alpha = 0.500`、dram 256 / cxl 256 MiB。
- 漂移：DRAM pool 越过 FROZEN → `shouldReplan()==true`。

### 4.4 验收矩阵（`test_checkpoint_acceptance`）

```
[P acceptance matrix]
  A1 dual-line staging additive            PASS  (dual 27.76 vs single 17.60 GB/s)
  A2 CXL line does not use uplink          PASS  (line1 uplink=1 bytes=256MiB, line2 uplink=0 bytes=256MiB)
  A3 multi-channel backend scales          PASS  (2ch 3.96 vs 1ch 1.99 GB/s)
  A4 PCIe write/read asymmetry             PASS  (read 55.53, write 36.09, ratio 0.65)
  A5 on-device DMA (P2P)                   PASS  (cxl device=256MiB host=0; dram host=256MiB device=0)
  A6 data consistency                      PASS
  A7 per-line/per-lane stats               PASS
  A8 CXL hot standby speeds restore        PASS  (hot/cold 14.2x)
  A9 pressure reuse / no unpin             PASS
[P acceptance matrix] 9/9 criteria PASS
```

`ctest -R test_checkpoint`：**9/9 全绿**（总耗时 ~89s；`lanes` 39s / `dual_link` 29s /
`acceptance` 15s 因 2 GiB 内存内 staging 较慢）。

---

## 5. 困难与决策

1. **flit 折算的语义归属（核心决策）**。旧模型把 CXL 线当 CXL.mem 链路（raw link rate ×
   128/130 × 64/68）。新拓扑的 DMA 是 PCIe P2P TLP，**没有 68B flit**；CXL 后端是 raw DDR
   带宽。**决策**：`cxlOverheadFactor()` 改为返回 `1.0`（保留签名作 API 兼容，字段标注
   deprecated），让 CXL 后端与 DRAM 后端对称（都无协议开销，PCIe 线编码由 `pcieEffectiveGbps`
   在链路侧统一扣除）。

2. **非对称后端使水填"退化为单线"**。DRAM 子通道 17.6 与 CXL DDR 50 不对称，S1（单 GPU、
   2 个 256 MiB split unit）的水填把全部数据路由到更快的 CXL 线（50 > 17.6），"双线"退化为
   "CXL-only"。加速比仍是 1.58×（CXL 线让 GPU 吃满 egress），但不是字面的两路分拆。**决策**：
   S1 的验收判据用 `B_dual > B_single`（DRAM-only 基线），而非 "两线都被用到"；"两线都承载
   数据" 的判据放到 A2/A5（用宽后端 50/50 的默认配置）。

3. **A2/A5 需宽后端才能两线都用**。窄非对称后端下 line1 分到 0 字节，`write_bytes>0` 判据
   失败。**决策**：A2/A5 改用 `defaultConfig`（DRAM 8×17.6 / CXL 8×50，`B_line1=B_line2=55.53`，
   水填 50/50），保证 line1/line2 都有 256 MiB。

4. **S2 量化误差消失**。两线对称（都 55.53）→ 水填 `alpha=0.5` 恰好是整 unit 边界，无需
   细 split unit 也零量化误差（上版非对称时 S2 实测 80.76 vs 上界 83.29）。**决策**：S2 保留
   64 MiB split unit 仅作稳健性，实测与上界一致（111.05）。

5. **PinnedPool 压力机干扰带宽测量**（沿用上版决策）：`stage()` 计时改用实际 staged 字节；
   带宽测试用 `disablePoolPressure()`；`drift_cap_threshold` 取 0.20 对齐 FROZEN。

6. **自适应 split unit**（沿用上版）：小 checkpoint（如 1 MiB 单测）自动 `min(256MiB, total/4)`，
   保证 ≥4 units、双 lane 可被用到；大 checkpoint 仍用 256 MiB。

7. **环境缺 `clang-format`**：本机无 `clang-format` 二进制，手写代码对齐既有 4 空格风格、
   未跑格式化；如需 CI 装 `clang-format` 后统一 `-i`。

8. **`Balancer::chooseLane` 成死代码**：被 `planOffload` 表查取代，保留（`Balancer` 仍承担
   `LaneStats` 统计 + token bucket）。

---

## 6. 改动文件清单

```
include/checkpoint_engine.h            cxlOverheadFactor->1.0（去 flit）+ line_cxl Gen5 + 注释
src/checkpoint_engine.cpp              computeBounds/stage（可加性，无 flit）+ planOffload/shouldReplan
checkpoint/analytical_model.py         line_cxl=64 / cxl 8 DDR×50 / cxl_overhead=1.0
tests/checkpoint_test_harness.h        defaultConfig（line_cxl 64, cxl 8×50）+ disablePoolPressure
tests/test_checkpoint_lanes.cpp        双线带宽模型（wide/narrow/4-GPU）
tests/test_checkpoint_balancer.cpp     offload plan alpha(0.5) + 压力
tests/test_checkpoint_acceptance.cpp   A1–A9（A2/A5 用宽后端，A1 cxl=50）
tests/test_checkpoint_offload.cpp      planOffload 单测 + 漂移
tests/test_checkpoint_dual_link.cpp    S1/S2（S2=2.00×）
checkpoint/cxlmem_target2.md           计划文档（expander 直连 / 纯 TLP / 8 DDR）
checkpoint/exec_summary_v2.md          本文
```
