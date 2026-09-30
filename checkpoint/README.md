# Checkpoint / Restore Data-Movement Engine (CXLMemSim)

> 摘要：本目录把 SimCXL `configs/example/gem5_library/target.md` 里的 **GPU 训练
> checkpoint / restore 数据搬运引擎**（P0–P10）在 CXLMemSim 中重新实现。CXLMemSim
> 没有 gem5 的 PCI 设备 + 全系统 VM，取而代之的是自包含的 C++20 仿真库 + 单元测试 +
> 分析模型：用确定性的 payload 生成、DRAM/CXL 双路 staging、多通道闪存后端、分流调度、
> pinned pool 压力状态机、manifest 版本/代际回退、CXL 热备与故障注入。八条验收中 7 条
> 通过；**#2（双路 staging > 最快单路径）在单机单 PCIe 上行上不成立**——内存侧按 lane
> 建模（DRAM 4 通道、CXL 2 链路），其聚合带宽都超过 PCIe 上行，因此 GPU 拷贝带宽被
> PCIe 封顶（~55.5 GB/s），与内存 lane 拓扑无关（诚实结论，与 SimCXL P10 一致）。

## 1. 与 SimCXL 的映射

SimCXL 用「gem5 硬件 + VM 软件」实现；CXLMemSim 是「C++20 服务器侧仿真库」。两者概念
一一对应，但载体不同：

| target.md 模块 | SimCXL（gem5） | CXLMemSim（本实现） |
| --- | --- | --- |
| GPU checkpoint source | `GpuDmaEngine`（C++ 设备模型） | `ckpt::payloadFill`（确定性 PRNG，`include/checkpoint_payload.h`） |
| 数据校验 | 设备 CRC32 + guest SHA-256 | `ckpt::crc32` / `ckpt::sha256` |
| DRAM/CXL pinned pool | gem5 DRAM + `CXLMemCtrl` + guest `mbind` | `ckpt::PinnedPool`（带宽/延迟 + 字节级 backing） |
| 多通道存储 | `ParallelStorage`（gem5） | `ckpt::ParallelStorage`（`include/parallel_storage.h`） |
| DMA 引擎 / MemCopy | `SimCkptDevice` | `ckpt::CheckpointEngine::stage/persist/restore` |
| 分流调度器 | guest `balancer.{h,c}` | `ckpt::Balancer` |
| pinned pool 压力状态机 | guest `ckptd` | `ckpt::PinnedPool::updatePressure` |
| Manifest / 代际回退 | guest `ckptd` | `ckpt::CheckpointEngine` manifest + `commitGeneration`/`rollback` |
| CXL 热备 | guest `ckptd` LRU | `ckpt::HotStandby`（CXL 缓存磁盘副本） |
| 分析模型 | `analytical_model.py` | `checkpoint/analytical_model.py` |
| 周期仿真 | gem5 Timing CPU | C++ 时间模型（ns 级带宽/延迟/队列） |

关键差异：CXLMemSim 没有 VM 与 PCI 驱动，所以「内核模块 + ioctl + 中断」这一层被省略；
取而代之的是一个**纳秒粒度的时间模型**（每资源 `busy_until` 带宽串行化 + 每次操作延迟
按 Little's Law 重叠），以及直接可复用的 `SharedMemoryManager`/SSD 流式后端（本实现为
自包含，不依赖 spdlog/perf/BPF，故在缺少 bpf 头文件的环境也能编译）。

## 2. 架构与数据通路

```
GPU payload ──stage(同步, GPU 可见)──> DRAM / CXL pinned pools
             │                          │ persist(异步, 后台)
             │                          └─> ParallelStorage (N 通道闪存)
             │                                 │
             └────restore<─────────────────────┘
                  (热 chunk 走 CXL 热备, 冷 chunk 走 storage)
```

- **阶段 1（staging，GPU 可见）**：GPU 把状态并行写入 DRAM/CXL 两路 pinned pool，写完
  立即回到训练。首要目标是**最小化 GPU stall**（= staging 时间）。
- **阶段 2（persist，后台）**：DRAM/CXL → storage 异步落盘，不阻塞 GPU。
- **阶段 3（restore）**：storage/CXL 热备 → landing pool → GPU。

条带化与 SimCXL `ParallelStorage::mapAddr` 一致：

```
channel        = chunk_id % num_channels
channel_offset = (chunk_id / num_channels) * chunk_size + offset % chunk_size
```

### 时间模型与理论基础

每个资源维护 `busy_until`（带宽串行化，只按 service 时间推进）+ 每操作延迟（与下一次
传输重叠，符合 Little's Law）。这样：

- 单通道持久带宽 ≈ `chunk_size / (chunk_size / write_gbps)` = `write_gbps`；
- N 通道 ≈ `N × write_gbps`（存储是瓶颈时），从而「2 通道 > 1 通道」（验收 #3）。

模型的三块理论依据（详见本节末尾）：

1. **瓶颈/最小上界**：稳态吞吐 ≤ 最慢串行资源，`B = min(各段带宽)`（排队论稳定条件）。
2. **Little's Law**：带宽 B × 延迟 L = 所需在途字节；只要 outstanding 足够，延迟不
   串行化吞吐（`busy_until` 只按 service 推进、延迟做常数偏移）。
3. **确定性单服务器队列 D/D/1 + max-plus 闭包**：staging 时间 `max(GPU 发射, DRAM 吸收,
   CXL 吸收) + 延迟` 是确定型延迟上界的标准写法。

带宽上界（`ckpt::computeBounds`，对应 target.md 五节）：

```
PCIe_eff = PCIe_raw × 线编码 × MPS/(MPS+TLP头+LCRC) × residual        # 由 TLP 结构推导
CXL_eff  = CXL_raw  × 线编码 × 64/68                                  # 68B flit 只带 64B 数据
B_stage   = min( PCIe_eff, DRAM_write + CXL_eff )      # GPU 单条 PCIe 上行是共享预算
B_save    = min( B_stage, DRAM_read + CXL_eff_read, N × channel_write )
B_restore = min( N × channel_read, DRAM_write + CXL_eff_write, PCIe_eff )
```

> **实际在线字节 > 逻辑字节（重要）**：拷贝 4 MiB 逻辑数据，线上会走更多字节——PCIe
> 线编码（128b/130b，Gen3+；8b/10b=0.8 仅 Gen1/2）+ TLP 头/LCRC + CXL 的 68B flit
> 只携带 64B 数据（2 CRC + 2 头）。因此有效带宽必须按这些开销折算，不能拿 raw 速率当
> 逻辑带宽：
>
> - PCIe 线编码：Gen3+ = 128b/130b（128/130≈0.985），Gen1/2 = 8b/10b（8/10=0.8）；按
>   `lspci LnkSta` 的链路速度选。
> - TLP 头+LCRC：MPS/(MPS+overhead)，MPS=256B、overhead=20B → 0.928。
> - CXL flit：64/68 ≈ 0.941（在 PCIe 线编码之上）。
> - 合成：PCIe_eff = 64 × 0.985 × 0.928 × 0.95 ≈ 55.5 GB/s；CXL 每链路 = 32 × 0.985 ×
>   0.941 ≈ 29.7 GB/s。

> **单机节点 PCIe 上界（重要，建模必须完整）**：GPU 只有一条 PCIe 上行，DRAM 与 CXL
> 两条 staging 路径都走它，且这条上行到不了理论峰值（上述开销）。内存侧按 lane 显式
> 建模：DRAM = 4 条 DDR5 通道、CXL = 2 条链路（每条 raw 32 GB/s，经 flit/编码折成
> 29.7 GB/s）。用标定参数（PCIe 64→55.5、DRAM 4×32=128、CXL 2×29.7=59.3 GB/s）时，
> **三条路径的 GPU 拷贝带宽都是 ~55.5 GB/s（PCIe 封顶）**——DRAM 4 通道（128）与 CXL
> 2 链路（59.3）的聚合带宽都超过 PCIe 有效值：
>
> | 场景 | 聚合内存带宽 | GPU 拷贝带宽 | 瓶颈 |
> | --- | --- | --- | --- |
> | only DRAM | 4×32 = 128 GB/s | ~55.5 GB/s | PCIe 上行 |
> | only CXL | 2×29.7 = 59.3 GB/s | ~55.5 GB/s | PCIe 上行（59.3 > 55.5） |
> | both | 128+59.3 = 187 GB/s | ~55.5 GB/s | PCIe 上行 |
>
> 只有当内存 lane 本身比 PCIe 窄时才自受限：例如 CXL 改成 2×16 raw（=29.7 有效）时，
> only-CXL 掉到 ~29.7 GB/s（`test_checkpoint_lanes` 验证）。因此 target.md 声称的
> 「双路 staging 带宽翻倍」在单机单上行上**不成立**（与 SimCXL P10 诚实结论一致）——
> GPU 拷贝带宽由 PCIe 上行决定，与内存 lane 拓扑无关（只要聚合内存带宽 > 上行）。此外
> SimCXL 实测还发现 DDR5 写缓冲（`write_buffer_size=64`）会在双路并发写时被击穿，进一步
> 压低并发带宽——本实现是带宽/延迟模型，不建模写缓冲这一级，属后续限制项。

### 带宽标定（calibration）

DRAM/CXL 的读/写带宽与 lane 数是**标定输入**（`PoolConfig::num_lanes/read_gbps/write_gbps`
为每-lane 值），不是推导值，默认 DRAM 4 通道、CXL 2 链路只作示例。参照 CXLMemSim 的
`--mlc-bandwidth`，应在目标机器上实测后填入：

```bash
# 宿主机/guest 上测每 lane 的 GPU->DRAM / GPU->CXL 实际带宽（MLC 或 DMA 微基准），
# 然后把读数写进 CheckpointConfig 或 analytical_model.py 的 --dram-*/--cxl-*。
# CXL 的 --cxl-read/--cxl-write 填 raw 链路速率，flit(64/68) 与线编码由模型折算。
python3 checkpoint/analytical_model.py \
    --dram-lanes 4 --dram-read 38 --dram-write 32 \
    --cxl-lanes 2 --cxl-read 32 --cxl-write 32 \
    --pcie-raw 64 --mps 256 --tlp-overhead 20
```

标定的关键：**带宽用实测、延迟用实测、PCIe 有效值用 TLP 结构推导**——这样模型结构靠
排队论，参数靠测量/推导，没有拍脑袋的常数。

## 3. 分析手册（analytical model）

`checkpoint/analytical_model.py` 用与 C++ 相同的闭式上界快速扫描「拓扑 × 通道数」：

```bash
python3 checkpoint/analytical_model.py --total-mib 512 --channels 1,2,4,8,16
```

默认参数（`tests/checkpoint_test_harness.h::defaultConfig`，均可经 CLI 标定）：
PCIe 64 GB/s 原始 → 55.5 GB/s 有效（128b/130b + MPS 256 + TLP 头 20B + residual）、
DRAM 4 通道 × 32 GB/s 写 / 38 读、CXL 2 链路 × 32 GB/s raw → 29.7 有效（flit 64/68）、
闪存 2 GB/s/通道、4 KiB chunk、10 µs flash 延迟。GPU 拷贝带宽（staging，512 MiB 实测）
与 B_save 上界：

```
GPU copy bandwidth (staging):
  only DRAM:  55.53 GB/s   (PCIe-limited)
  only CXL :  55.53 GB/s   (PCIe-limited)
  both     :  55.53 GB/s   (PCIe-limited)

topology        1ch      2ch      4ch      8ch     16ch
dram          2.00    4.00    8.00   16.00   32.00
cxl           2.00    4.00    8.00   16.00   32.00
both          2.00    4.00    8.00   16.00   32.00
```

结论：**storage（闪存）在低通道数下是持久化瓶颈**（B_save = N×2 GB/s），直到通道数足够
大时才回到内存路径（~32 GB/s）。GPU 拷贝带宽（staging）由 PCIe 上行封顶（~55.5 GB/s），
与内存 lane 拓扑无关——这与 target.md 的论断一致：双内存路径的价值不在落盘吞吐，staging
也受 PCIe 上行封顶，因此单机上双路 staging 的真实收益**为零**（dual == 单路 DRAM）。

分析模型预测的 phase 时间与 C++ 引擎实测一致：例如 512 MiB、2 通道双路，staging
≈ 9.67 ms（= 总字节 / 55.5 GB/s 有效 PCIe）、persist ≈ 134 ms（≈ B_save 4 GB/s）；
C++ 引擎 `test_checkpoint_storage` 实测 2ch persist 3.96 GB/s、1ch 1.99 GB/s，与分析
模型 4 / 2 GB/s 吻合（差一个 flash 页延迟的常数项）。

## 4. 执行手册（execution manual）

### 4.1 构建

checkpoint 引擎自包含，仅依赖标准库，可独立构建（无需 spdlog/perf/BPF）：

```bash
cd CXLMemSim
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target cxlmemsim_ckpt \
    test_checkpoint_payload \
    test_checkpoint_storage \
    test_checkpoint_engine \
    test_checkpoint_balancer \
    test_checkpoint_hotstandby \
    test_checkpoint_acceptance -j
```

> 说明：主仓库 `cxlmemsim` 库依赖 `bpf/bpf.h`（`libbpf-dev`），本环境缺该头文件时该库
> 编译失败——这是**预先存在**的依赖问题，与 checkpoint 引擎无关；`cxlmemsim_ckpt` 及其
> 测试不依赖它。

### 4.2 运行测试

```bash
cd build && ctest -R test_checkpoint --output-on-failure
# 或逐个运行
./test_checkpoint_payload     # CRC32/SHA-256 已知答案 + payload 确定性
./test_checkpoint_storage     # 多通道带宽 + 条带 + 队列反压
./test_checkpoint_engine      # 往返一致性 + 乱序恢复 + 在飞禁止 unpin + 代际回退
./test_checkpoint_balancer    # 双路 staging 带宽 + 压力状态机
./test_checkpoint_hotstandby  # 热备命中率 -> 恢复时间
./test_checkpoint_lanes       # only-DRAM / only-CXL / both 的 GPU 拷贝带宽
./test_checkpoint_acceptance  # 完整 8 条验收矩阵
```

`test_checkpoint_lanes` 输出（GPU 拷贝带宽，512 MiB，DRAM 4 通道 / CXL 2 链路）：

```
GPU copy bandwidth (PCIe 64 raw -> 55.53 effective)
  DRAM: 4 lanes x 32 GB/s = 128 GB/s aggregate
  CXL : 2 lanes x 32 GB/s raw -> 29.7 effective (flit 64/68 x 128b/130b) = 59.3 GB/s aggregate
  only DRAM : 55.53 GB/s (bound 55.53)
  only CXL  : 55.52 GB/s (bound 55.53)
  both      : 55.52 GB/s (bound 55.53)
  only CXL (2x16 raw = 29.7 effective): 29.65 GB/s (CXL-limited)
```

`test_checkpoint_acceptance` 输出示例（诚实 7/8，#2 见上）：

```
[P acceptance matrix]
  #1 CRC/byte consistency                    PASS
  #2 dual-path staging > fastest single      FAIL  (dual 55.23 vs single-DRAM 55.45 GB/s (both PCIe-limited))
  #3 2 storage channels > 1 channel          PASS  (2ch 3.96 GB/s vs 1ch 1.99 GB/s)
  #4 CXL hot standby speeds restore          PASS  (hot/cold restore speedup 14.2x)
  #5 out-of-order restore                    PASS
  #6 pressure: pool reuses, no growth        PASS
  #7 no unpin while in flight                PASS
  #8 per-link queue/bw/latency stats         PASS
[P acceptance matrix] 7/8 criteria PASS (#2 not achievable on single-node)
```

## 5. 验收标准映射（target.md 八条）

| # | 验收标准 | 实现位置 | 测试 |
| --- | --- | --- | --- |
| 1 | GPU→DRAM/CXL→storage→…→GPU 数据 CRC 一致 | `stage` 生成 payload+CRC，`restore` 校验 | `test_checkpoint_engine` |
| 2 | 双内存路径 staging 带宽 > 最快单路径 | `stage` 单 PCIe 上界 + lane 聚合带宽 | `test_checkpoint_balancer`（诚实结论：单机单上行不成立，dual == 单路 DRAM） |
| 3 | 两个存储通道带宽 > 单通道 | `ParallelStorage` 每通道独立带宽 | `test_checkpoint_storage` |
| 4 | CXL 热备命中时恢复时间明显下降 | `HotStandby` LRU 快路径 | `test_checkpoint_hotstandby` |
| 5 | chunk 乱序完成仍能正确恢复 | manifest 按 `chunk_id` 匹配 + 逐 chunk CRC | `test_checkpoint_engine` |
| 6 | 压力触发后 pinned pool 只复用不扩大 | `PinnedPool::updatePressure` | `test_checkpoint_balancer` |
| 7 | 在飞 DMA 完成前禁止 unpin | `unpinChunk` 拒绝 PINNED/IN_FLIGHT | `test_checkpoint_engine` |
| 8 | 每条链路排队/带宽/延迟/重试分别统计 | `LaneStats` + `StorageChannelStats` | `test_checkpoint_acceptance` |

## 6. 代码结构

```
include/checkpoint_payload.h     PRNG payload、CRC32、SHA-256
include/parallel_storage.h       N 通道闪存后端 + 条带 + 队列/反压
include/checkpoint_engine.h      PinnedPool/Balancer/HotStandby/CheckpointEngine + lane 聚合
src/checkpoint_payload.cpp
src/parallel_storage.cpp
src/checkpoint_engine.cpp
tests/checkpoint_test_harness.h  共享 REQUIRE/配置
tests/test_checkpoint_*.cpp       7 个测试（见上）
checkpoint/analytical_model.py    分析模型（含 lane + GPU 拷贝带宽）
```

## 7. 参考

- SimCXL 目标与实现：`../SimCXL/configs/example/gem5_library/target.md`、
  `../SimCXL/exec_summary.md`（P0–P10）。
- CXLMemSim：`README.md`、`detail.md`（Type-3 服务器 / SharedMemoryManager / SSD 后端）。
- 论文：Cohet (HPCA 2026)、CXL-DMSim (TCAD 2025)；CXLMemSim (YArch'23)。
