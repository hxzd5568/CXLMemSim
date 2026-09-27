# hetGPU / Type-2 / Server-SSD 架构问答

本文基于仓库 `README.md`、CXLMemSim 服务端源码，以及已拉取的 `lib/qemu` 子模块源码整理。

> 更新：`lib/qemu` 子模块现已拉取（commit `d986b2f591c00cbf4e676863ba8199523ceda3f0`，分支 `remotes/origin/codex/pgas-fast-poll`）。Q1/Q3 已按 `lib/qemu/hw/cxl/cxl_hetgpu.c`、`cxl_type2.c`、`cxl_type2_coherency.c` 的真实源码重新分析。

- 原始提问中的第 2 问粘贴内容显示为 `[Pasted ~3 lines]`，未成功传入，已在 Q2 处留占位符。

---

## Q1. 代码里的 hetGPU 是什么意思？什么功能？有普通 GPU 能用它干什么？普通操作（如普通 cudaMemcpy）还能不能模拟？好处与理论收益是什么？

### 1.1 hetGPU 是什么（按真实源码）

名称上有 **两个** 东西被叫 hetGPU，别混淆：

1. **QEMU 树内的桥接层**：`lib/qemu/hw/cxl/cxl_hetgpu.c` + `lib/qemu/include/hw/cxl/cxl_hetgpu.h`。它实现了一整套 `hetgpu_*` C API（init / malloc / free / memcpy_htod|dtoh|dtod / memset / load_ptx / get_function / launch_kernel / create_context / create_coherent_region / flush&invalidate_cache …），是 CXL Type 2 设备与“GPU 后端”之间的接口层（`cxl_hetgpu.h:175-465`）。
2. **外部 CUDA-ABI 库（`libnvcuda.so`）**：由 `hetgpu-lib` 参数传入，QEMU 桥接层用 `dlopen()` 加载它并 `dlsym()` 出 `cuInit/cuDeviceGet/cuCtxCreate/cuMemAlloc/cuMemcpyHtoD/...`（`cxl_hetgpu.c:184-206`）。这个外部库才是把 CUDA 调用“翻译到 Intel/AMD/Tenstorrent”的 PTX→原生后端。

后端枚举是权威定义（`cxl_hetgpu.h:24-31`，名字见 `cxl_hetgpu.c:1222-1240`）：

```
AUTO=0, INTEL=1 (Level Zero), AMD=2 (HIP/ROCm), NVIDIA=3, TENSTORRENT=4, SIMULATION=5
```

桥接层的关键机制（`cxl_hetgpu.c`）：

- **真实 GPU 路径**：`hetgpu_init` 按顺序尝试 `dlopen()`：给定 `hetgpu-lib` → `/usr/lib/x86_64-linux-gnu/libcuda.so` → `/usr/lib64/libcuda.so` → `libcuda.so.1`（`cxl_hetgpu.c:150,165-177`）。所以 `hetgpu-lib` 既可以是真实 NVIDIA 驱动，也可以是 hetGPU 的 `libnvcuda.so`——桥接层对二者一视同仁。真实路径下 `malloc/memcpy/launch_kernel` 直接调用 `cu*` 真机执行（`cxl_hetgpu.c:641-662,753-768,1133-1160`）。
- **hetGPU managed 模式**：若外部库的 `cuCtxCreate` 返回 NULL context（`libnvcuda.so` 的典型行为），桥接层不报错，转为由该库内部管理上下文，继续调用 `cuMemAlloc/cuLaunchKernel`（`cxl_hetgpu.c:268-297`）。
- **模拟路径（backend=5）**：完全不 `dlopen`，直接在进程内模拟：`hetgpu_malloc` 用 `g_malloc0` 分配真实 host 缓冲区并发放伪造 device ptr（从 `0x100000` 起，`cxl_hetgpu.c:664-687`）；`memcpy_htod/dtoh/dtod` 就是对这块缓冲区 `memcpy`（`:770-886`）；`memset` 同理（`:897-908`）；`load_ptx` 返回假模块句柄 `0x12345678`；`launch_kernel` 只记日志、不执行（`:1162-1167`）。设备属性是内置的 `default_props`（名字 `"Virtual GPU (TMatmul)"`，4GB，`cxl_hetgpu.c:22-39`）。
- **真实 GPU 失败不会静默退回模拟**：`simulation_fallback` 标签明确返回 `HETGPU_ERROR_NO_DEVICE`，只提示检查 nvidia-smi / 权限 / libcuda（`cxl_hetgpu.c:366-379`）。想用模拟必须显式 `hetgpu-backend=5`。
- 该文件里的 coherency 相关接口是**桩**：`hetgpu_flush_cache`/`hetgpu_invalidate_cache` 只加计数、直接返回成功（`cxl_hetgpu.c:972-998`）；`hetgpu_set_coherency_callback` 只保存回调指针给 `cxl_type2.c` 使用。真正的一致性在 `cxl_type2_coherency.c`（见 Q3）。

一句话：**hetGPU = 让 CXL Type 2 设备能挂载“任意 GPU 后端（含纯软件模拟）”的适配层**，由 QEMU 内桥接层 + 可选外部 CUDA-ABI 库组成。

### 1.2 它在整条数据通路中的位置

```
guest CUDA workload
  -> qemu_integration/guest_libcuda/libcuda.so.1   (guest 侧 CUDA Driver API shim)
  -> QEMU cxl-type2 PCI device  +  BAR2 MMIO 命令寄存器
  -> cxl_hetgpu backend        (host 侧 hetGPU 库)
  -> host NVIDIA CUDA driver / 其它厂商后端 / 模拟后端
  -> 真实 GPU 或 模拟
```

（`README.md:206-216` 的 Type 2 Data Path，以及 `README.md:320-330` 的 Runtime Stack。）

### 1.3 有普通 GPU 能拿它干什么

取决于你选的 backend：

| 你的硬件 | 建议设置 | 效果 |
| --- | --- | --- |
| NVIDIA 显卡 | `hetgpu-backend=3`（或 `0` auto），`hetgpu-lib` 指向真实 `libcuda.so` | 真实 CUDA 在你的 GPU 上执行；设备内存/一致性路径仍由 CXL Type 2 模拟 |
| Intel / AMD / Tenstorrent | `hetgpu-backend=1/2/4`，`hetgpu-lib` 指向 hetGPU 的 `libnvcuda.so` | hetGPU 做 PTX → 原生指令翻译，让非 NVIDIA 卡跑同一份 CUDA 工作负载 |
| 没有任何 GPU | `hetgpu-backend=5`（simulation，多 GPU sweep 脚本的默认值） | 全功能模拟，无需真实 GPU |

要真实 GPU 时，QEMU 侧典型参数见 `README.md:743-753`：

```
-device cxl-type2,id=cxl-gpu0,cache-size=128M,mem-size=4G,\
    hetgpu-lib=/usr/lib/x86_64-linux-gnu/libcuda.so,hetgpu-device=0
```

注意一个限制：真实 GPU 初始化失败时，当前实现**报错而不是静默退回模拟**（`README.md:286`）。想用模拟必须显式选 backend=5（或 hetGPU 自身的 sim 后端）。

### 1.4 普通操作能不能模拟（例如普通 cudaMemcpy）

能。guest shim `libcuda.c` 实现的就是一整套 CUDA Driver API 子集，并把每个调用翻译成 BAR2 MMIO 命令：

- `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2`：按 `CXL_GPU_DATA_SIZE`（1MB）分块通过数据缓冲区搬运（`qemu_integration/guest_libcuda/libcuda.c:517-579`）。
- `cuMemcpyDtoD_v2`（`:779`）、`cuMemsetD8/D32_v2`（`:818,849`）、`cuMemAlloc/cuMemFree`（`:487,505`）、`cuModuleLoadData`/`cuModuleGetFunction`/`cuLaunchKernel`（`:581,611,634`）。
- 命令枚举本身位于 `cxl_gpu_cmd.h:88-167`（`MEM_COPY_HTOD/DTOH/DTOD`、`MEM_SET`、`LAUNCH_KERNEL`、`BULK_*`、`CACHE_*`、`P2P_*`、`COHERENT_*`、`DCD/GFAM/MHSLD` 等）。

这些命令最终都落到 QEMU 内的 hetGPU 桥接层（`cxl_type2_hetgpu_*` → `hetgpu_*`）。所以：

- **backend=5（模拟）**：`cuMemAlloc` → `g_malloc0` 一块真实 host 缓冲区并返回伪造 device ptr；`cuMemcpyHtoD/DtoH/DtoD` → 对这块缓冲区 `memcpy`；`cuMemset` → `memset`；`cuModuleLoadData` → 返回假句柄；`cuLaunchKernel` → 只打日志、不执行（`lib/qemu/hw/cxl/cxl_hetgpu.c:664-687,770-886,897-908,1030,1162-1167`）。
- **backend=3/0 + 真实库**：`cuMemAlloc/cuMemcpy*/cuLaunchKernel` 真正调用 host CUDA（或 hetGPU 的 `libnvcuda.so` 翻译后端）（`cxl_hetgpu.c:641-662,753-768,1133-1160`）。

即“普通操作”既能**真跑**，也能**纯模拟**，取决于 backend；模拟模式下算子的空间（malloc/copy/set）真实存在，但**计算（kernel launch）不执行**。

### 1.5 用它（而不是裸跑 GPU）的好处

1. **无需 CXL 硬件即可研究 CXL Type 2 语义**：设备内存孔径走模拟 CXL 内存，一致性走模拟目录，能观察/调参延迟、路由、目录压力、bias、P2P 等（`README.md:3-8`）。
2. **GPU 厂商无关**：同一份 CUDA 工作负载可跑在 NVIDIA / Intel / AMD / Tenstorrent / 模拟后端上，便于横向对比或没有对应硬件时做实验。
3. **可复现、可插桩的功能模拟**（backend=5）：没有真实 GPU 也能回归测试、做 coherency litmus、放大规模扫描（见 `qemu_integration/launch_qemu_type2_multigpu.sh` 的 `NUM_TYPE2` 1/2/4/6/8 扩展）。
4. **内存容量扩展 + 分层研究**：设备内存放到模拟 CXL 上，可研究超出 HBM 容量的工作集、KV cache offload（`README.md:771-789` 的 llama.cpp-cxl `--cxl-kv`）、内存池化、与 Type 3 的 P2P DMA、coherent shared pool 等。

### 1.6 理论收益

- **容量收益**：用 CXL 内存承接 GPU 工作集/KV cache，解决 HBM 容量瓶颈（能放下更大模型/更长上下文），代价是把一部分访问从 HBM 搬到延迟/带宽更差的 CXL 内存。
- **收益是负载相关的**：收益 = “因容量变大而少掉的换出/重算/通信” 减去 “CXL 访问相对 HBM 的额外延迟/带宽惩罚”。对容量受限型负载（LLM KV cache、图、向量检索）收益显著；对带宽/延迟敏感型负载可能为负。
- 该路径是**实验性质**的：`README.md:913-917` 明确说 Type 2 GPU 路径依赖修改版 QEMU、shim 只实现 Driver API 子集、不应视为周期精确硬件。因此这里的“收益”是**建模/研究数值**，不是可直接照搬到生产的加速比。

---

## Q2. [原始粘贴内容缺失]

此问的 3 行粘贴文本在传输中丢失（显示为 `[Pasted ~3 lines]`），无法据此作答。请将该段内容补发，我会追加到本文件。

---

## Q3. 为什么会区分 “Type 2 GPU” 和 “Type 2 coherency model”？它们在定义上不是有重叠吗？

结论：**两者是同一设备上的两个不同角色/协议层，不是重复定义；重叠发生在“同一个 Type 2 设备 + 同一片地址孔径”上，但建模的是不同协议。**

拿到真实源码后可以更清楚地看到：QEMU 的 Type 2 设备里其实有**三条相对独立、又在 BAR 窗口处交汇**的路径。

### 3.1 “Type 2 GPU 栈”是加速器/命令路径

它描述的是“**GPU 操作怎么发下去、计算在哪跑**”：

```
guest CUDA shim -> BAR2 MMIO 命令寄存器 -> QEMU cxl-type2 -> cxl_hetgpu 桥接层 -> 外部 GPU 库 / 模拟
```

- 设备状态里对应 `gpu_cmd` 结构（status/cmd_result/params/results/data/modules/functions）与 `gpu_cmd_region`、`bulk_transfer_region`（`lib/qemu/include/hw/cxl/cxl_type2.h:257-278`）。
- 命令协议：alloc / memcpy / memset / module load / kernel launch / bulk transfer / P2P 等（`qemu_integration/guest_libcuda/cxl_gpu_cmd.h:88-167`）。
- 落到 `cxl_type2_hetgpu_malloc/memcpy_htod/dtoh/launch_kernel`（声明见 `cxl_type2.h:389-402`，实现见 `lib/qemu/hw/cxl/cxl_type2.c:781-847`），由它调用 Q1 讲的 hetGPU 桥接层。

### 3.2 “Type 2 coherency model”是缓存一致/目录路径——而且 QEMU 里有两级

它描述的是“**谁拥有哪条 cache line、CPU 与设备之间怎么保持一致**”。真实源码里这不是一个东西，而是两层：

1. **设备侧 cache-line 模型 `CXLType2CoherencyState`**（`cxl_type2.h:88-97`）：`GHashTable *cache_lines` + `cache_hits/cache_misses/coherency_ops/snoops`。这是每个 cache line（`CXLCacheLine`：tag/state/dirty/data[64]/timestamp，`cxl_type2.h:44-58`）的 MESI 状态模型，对应 `cxl_type2_coherency_init/lookup/insert/invalidate/writeback/snoop_request`（`cxl_type2.h:373-381`）。
2. **BAR 区域 CPU↔GPU 一致协议 `CXLBARCoherencyState`**（`lib/qemu/hw/cxl/cxl_type2_coherency.c` + `cxl_type2_coherency.h`）：这是一个更完整的 CXL.cache 模型，包含
   - 一致性域 `CPU / GPU / CXL_HOST / CXL_DEV`（`cxl_type2_coherency.h:23-28`）；
   - CXL.cache 请求类型 `RD_SHARED / RD_OWN / RD_ANY / WR_INV / WR_CUR / CLEAN_EVICT / DIRTY_EVICT`（`:31-40`）；
   - snoop filter（`GHashTable *snoop_filter` → `CXLSnoopEntry`）、BAR 区域表、pending 事务、**back-invalidation 队列**、bias（host/device）、原子操作、以及 `stats`（snoop_hits/snoop_misses/back_invalidations/writebacks/bias_flips…）（`:57-159,179-255`）。
   - 由 `cxl_bar_coherent_read/write`、`cxl_bar_snoop_*`、`cxl_bar_back_invalidate`、`cxl_bar_set_bias` 等驱动（`cxl_type2_coherency.h:187-255`）。

3. 另外还有 **CFMWS / HDM 路径**：host 通过 `cxl_type2_cfmws_read/write` 访问设备内存（`cxl_type2.h:404-409`），以及到 `cxlmemsim_server` 的 Type 3 一致性协议（`CXLType2MemSimConn` + `CXL_T2_MSG_*`，`cxl_type2.h:99-113,348-371`；协议 v2 见 `cxl_memsim_v2.c`）。

### 3.3 为什么会有“重叠”的错觉

CXL 规范里 Type 2 设备本身就是 **Type 1（cache/加速器）+ Type 3（内存）** 的合体，源码注释直接这么写（`cxl_type2.h:32`）。所以同一个设备**同时**承担：

- **加速器 + 命令面**（`gpu_cmd_region`）→ 对应 “Type 2 GPU”；
- **CXL.cache 一致性参与者**（BAR snoop filter / back-invalidation / bias）→ 对应 “Type 2 coherency model”；
- **CXL.mem 主机**（CFMWS / device_mem）→ 设备内存孔径；
- （可选）**到 Type 3 服务器的一致性内存访问**（memsim / memsim_v2）。

因此“Type 2 GPU”和“Type 2 coherency model”共享同一设备、同一片 BAR/HDM 地址范围，自然重叠；但它们**建模对象不同**：

| 维度 | Type 2 GPU 栈 | Type 2 coherency model |
| --- | --- | --- |
| 关心 | 命令与计算、数据搬运 | 一致性状态、所有权、snoop/invalidate |
| 接口 | BAR2 MMIO 命令寄存器 + `gpu_cmd` | BAR snoop filter + CXL.cache 消息 |
| 实现 | `cxl_hetgpu.*` + `cxl_type2_hetgpu_*` | `cxl_type2_coherency.*`（+ `cxl_type2.c` 里的 per-line 模型） |
| 状态 | context/module/function/stream | 域、MESI、sharer、dirty、bias、back-invalidation 队列 |
| 触发者 | guest CUDA 调用 | host/device 的读、写、原子 |

两者的交汇点被显式建模：GPU 访问时会通知一致性层（`cxl_bar_notify_gpu_access`，调用点见 `cxl_type2.c:638`），配合 `hetgpu_set_coherency_callback` 让 GPU 写入触发失效、GPU 读取前回写 CPU 侧 dirty line（`README.md:303`）。也就是说：**GPU 栈是“动作”，coherency model 是“这些动作必须遵守的秩序”**，二者分层且互相调用，而非重复定义。

---

## Q4. Runtime Stack 中 `cxlmemsim_server -> SharedMemoryManager` 的边界在哪？是 SSD 磁盘的 DMA 吗？支持多通道吗？

### 4.1 边界在哪

**逻辑边界是 `SharedMemoryManager` 的公共 API**，调用点在服务器请求处理函数里：

- `ThreadPerConnectionServer::handle_request()`（`src/main_server.cc:1548`）先做地址合法性校验（`is_valid_address`，`main_server.cc:1629-1630`），然后调用：
  - 读：`shm_manager->read_cacheline(...)`（`main_server.cc:1681`）
  - 写：`shm_manager->write_cacheline(...)`（`main_server.cc:1738`、`1898`）
  - 原子：`atomic_fetch_add_uint64` / `atomic_compare_exchange_uint64`（`main_server.cc:1989,2017`）
  - flush：`flush_cacheline` / `flush`（`main_server.cc:1745,2050`）
- `SharedMemoryManager` 是对外统一接口，内部按 `BackingMode` 决定落到哪种存储：`SharedMemory` / `FileMmap` / `SsdStream`（`include/shared_memory_manager.h:56-60`，`src/shared_memory_manager.cc:79-101`）。

所以：**server 只认 `SharedMemoryManager` 的 read/write/flush API；至于这些字节是来自 POSIX shm、普通文件 mmap，还是 SSD 后端，server 不关心。** 这就是边界。

### 4.2 “SSD 磁盘的 DMA”是边界吗？

**不是同一个边界，而是边界之下的一层。** 要分两种模式：

- **默认 `SharedMemory` / `FileMmap`**：`read_cacheline`/`write_cacheline` 就是 `memcpy` 到 mmap 区域，写后再 `msync`（`src/shared_memory_manager.cc:452-607`）。这里**没有磁盘 DMA**，只是内存拷贝 + msync。
- **`SsdStream`**：`SharedMemoryManager` 把请求转给 `SsdStreamingBackend`（`src/shared_memory_manager.cc:416-449`、`506-542`、`621-651`）。真正做块 I/O 的是 `SsdStreamingBackend::read_page` / `write_page`（`src/ssd_streaming_backend.cpp:425,450`）：
  - 优先用 **io_uring**：`IORING_OP_READ_FIXED` / `IORING_OP_WRITE_FIXED`（`ssd_streaming_backend.cpp:439,455`）；
  - 缓冲对齐（`O_DIRECT` + `posix_memalign` 到 4KB，`ssd_streaming_backend.cpp:41,115-125,166-171`）；
  - 失败回退 `pread`/`pwrite`（`ssd_streaming_backend.cpp:447,462`）。

因此准确说法是：

- **逻辑边界** = `SharedMemoryManager` API（server 调用它）；
- **“磁盘 DMA”边界** = `SsdStreamingBackend::read_page/write_page` 及其 io_uring/O_DIRECT 提交（比 `SharedMemoryManager` 低一层，且仅在 `--ssd-backing-file` 选 SSD streaming 时才存在）。

（`README.md:107` 描述的 `--ssd-backing-file`、`--ssd-cache-mb`、64KB read-ahead、io_uring、O_DIRECT 等就是这层。）

### 4.3 支持多通道吗？

**当前不支持真正的多通道/多队列。** 依据 `src/ssd_streaming_backend.cpp`：

- 只打开**一个** backing fd（`fd_`，`ssd_streaming_backend.cpp:155-190`）。
- io_uring 只注册**一个固定文件**：`IORING_REGISTER_FILES` 传 `&fixed_fd, 1U`（`ssd_streaming_backend.cpp:294-305`）。
- 只注册**一个固定缓冲**：`IORING_REGISTER_BUFFERS` 传 `&iov, 1U`（`ssd_streaming_backend.cpp:307-321`）。
- 单个 ring，深度固定 `io_uring_setup(64U, ...)`（`ssd_streaming_backend.cpp:227`）。
- 所有 I/O 经过同一个 `io_mutex_` 串行化（声明 `ssd_streaming_backend.cpp:137`，加锁 `:438`、`:453`）。
- 提交后立即等待完成（submit 1 个再 `io_uring_enter` 等 CQE，`ssd_streaming_backend.cpp:394-421`），**并没有异步流水线**——虽然是 io_uring，但用法是同步阻塞的“单通道”语义。

已有的并行/优化手段只有：

- **read-ahead**：默认 16 页 = 64KB（`ssd_streaming_backend.cpp:700-728`，`--ssd-read-ahead-pages`）。
- **页缓存**：CLOCK-like 淘汰（`ssd_streaming_backend.cpp:635-662`，`--ssd-cache-mb`）。
- **`backing_latency_ns`**：可注入确定性 backing 读延迟用于建模（`include/ssd_streaming_backend.h:39-45`，`ssd_streaming_backend.cpp:426-435`）。

若要“多通道”，目前只能靠**多个 server 实例 / 多个 backing 文件**在更高层做并行，例如分布式模式里的 `--ssd-backing-file=...{node}...` 每节点一个（`README.md:589-603`），而不是单个 `SsdStreamingBackend` 内的多队列条带化。

另外区分一下“通道”的两种可能含义：

- **NVMe/SSD 多队列（多通道）**：不支持，见上。
- **CXL 地址交织（interleave）通道**：`HDM decoder` 支持 range/interleaved/hybrid 解码（`README.md:165-169`），但这是地址解码层，与 `SsdStreamingBackend` 的单 fd 块 I/O 是两回事，并不会把 SSD I/O 分片到多个设备。

### 4.4 小结

- `cxlmemsim_server -> SharedMemoryManager` 的边界 = `read_cacheline/write_cacheline/read_range/write_range/atomics/flush` 这组 API（`main_server.cc` 调用，`shared_memory_manager.cc` 实现）。
- “SSD 磁盘 DMA”不是这个边界本身，而是 `SsdStream` 模式下 `SharedMemoryManager` **内部**转发的 `SsdStreamingBackend` 块 I/O（io_uring/O_DIRECT）。
- 该 SSD 后端目前是**单 fd、单固定文件、单固定缓冲、单 ring、单 io_mutex** 的同步模型，**不支持多通道/多队列**；并行只能靠多实例/多 backing 文件。

---

## Q4.5（补充）SharedMemoryManager 与相邻两层的边界到底在哪？为什么要把 `SharedMemoryManager` 独立出来？

README 的 Type 3 Runtime Stack 写成：

```
cxlmemsim_server
  -> SharedMemoryManager
  -> CXLController, HDM decoder, coherency engine, topology model
```

`SharedMemoryManager` 的**两个邻居**分别是：

- **上层邻居**：`cxlmemsim_server`（含 TCP / SHM / PGAS-SHM / distributed 传输与请求协议）
- **下层邻居**：真正的字节落点（POSIX shm / 文件 mmap / SSD streaming 后端）

（README 里并列写的 `CXLController / HDM decoder / coherency engine / topology model` 其实是**与存储并列的建模组件**，不是“存储的下一层”；见下方修正。）

### 4.5.1 上层边界：`cxlmemsim_server` ↔ `SharedMemoryManager`

**边界就是 `SharedMemoryManager` 的一组公共方法。** 调用点在 `handle_request()`（`src/main_server.cc:1548`）：

- 读：`shm_manager->read_cacheline()`（`main_server.cc:1681`）
- 写：`shm_manager->write_cacheline()` / `flush_cacheline()`（`main_server.cc:1738,1745`）
- 原子：`atomic_fetch_add_uint64()` / `atomic_compare_exchange_uint64()`（`main_server.cc:1989,2017`）
- 地址校验：`shm_manager->is_valid_address()`（`main_server.cc:1629-1630`）
- 每行元数据：`shm_manager->get_cacheline_metadata()`（`main_server.cc:1649`）

分工：

| 属于 cxlmemsim_server（上层） | 属于 SharedMemoryManager（下层） |
| --- | --- |
| 传输/协议帧：TCP、SHM ring、PGAS-SHM、distributed | 字节存储：`read/write_cacheline`、`read/write_range` |
| `op_type` 分派、LSA/DCD/GFAM/原子分派 | 地址→偏移解码：`is_valid_address`、`address_to_backend_offset`、`cacheline_to_index` |
| 一致性状态机推进：`handle_read/write_coherency`（`main_server.cc:1678,1735`） | 持久化：`flush()`（`msync` / 后端 flush） |
| 延迟核算：`controller->dramlatency + fabric_latency_ns`、`record_cxl_access`（`main_server.cc:1642,1706`） | 每行元数据容器：`metadata_cache` + `get_cacheline_metadata`（`shared_memory_manager.h:112,166`） |
| fabric/GFAM 访问检查（`main_server.cc:1607`） | 内存区域/allocation 记账（`allocate_region`、`regions`） |

一句话：**server 关心“这是什么请求、该走什么一致性/时序语义、回什么延迟”；manager 关心“字节存在哪、地址怎么翻译、怎么刷盘”。**

> 注意一个边界上的细节：**一致性元数据本身（state/sharers/owner/lock）由 manager 持有**（`shared_memory_manager.h:35-52,112`），但**状态迁移逻辑在 server**。也就是说 manager 是元数据的“容器”，server 是元数据的“解释器/状态机”。

### 4.5.2 下层边界：`SharedMemoryManager` ↔ 落盘/落内存后端

manager 内部按 `BackingMode` 三选一（`shared_memory_manager.h:56-60`，`shared_memory_manager.cc:79-101`）：

- `SharedMemory`：`shm_open` + `mmap`，读写即 `memcpy`，写后 `msync`（`shared_memory_manager.cc:452-607`）。
- `FileMmap`：普通文件 `mmap`，同上。
- `SsdStream`：转发给 `SsdStreamingBackend`（`shared_memory_manager.cc:416-449,506-542`），由它做真正的块 I/O（io_uring / `pread`/`pwrite`，见 Q4.2）。

所以**下层边界 = `SsdStreamingBackend` 的 `read/write/flush/prefetch/pin/evict_after` 接口**，只有 `SsdStream` 模式才会触达这一层。上层（server、coherence-v2、PGAS）只通过 manager 的同一组 API，感知不到下面到底是 RAM、shm、普通文件还是 SSD——唯一一处显式分支是 PGAS 初始化时用 `is_ssd_streaming()` 决定“数据放 manager 还是放 shm”（`main_server.cc:2441,2478`）。

### 4.5.3 为什么要把它独立出来

1. **一个 API 覆盖多种介质，解耦“策略”与“存储”**：同一个 `read_cacheline/write_cacheline` 既能落在内存(shm/file)，也能落在持久大容量(SSD)。加一种 backing 不需要动 server/一致性逻辑（BackingMode 分派集中在 manager）。
2. **单一地址解码权威，被多条前端复用**：
   - legacy TCP 路径：`main_server.cc:1681,1738`
   - PGAS-SHM 路径：`main_server.cc:2636,2677`
   - coherence-v2：`SharedMemoryCoherenceBackend` 只依赖 `SharedMemoryManager&`（`src/coherence_server_runtime_v2.cpp:15-28`，`include/coherence_server_runtime_v2.h:14-16`）
   
   地址翻译只写一份，避免各前端各自实现导致不一致。
3. **前端可切换、数据不动**：TCP / SHM / PGAS-SHM / distributed 都指向同一份存储（README:574,587 说明 PGAS/分布式在 SSD 模式下只把控制面放 shm，数据由 manager 提供）。
4. **支持持久化与大容量**：SSD streaming 让逻辑 CXL 容量不受 RAM 限制，且持久（`README:559-603`）；shm 模式还会复用已有对象并靠 header 的 magic/version 校验（`shared_memory_manager.cc:134-177,256-262`）。
5. **可替换、可测试**：一致性引擎面向抽象 `CoherenceMemoryBackend`（`include/coherence_memory_backend.h:10-18`），`SharedMemoryManager` 只是其一个实现；测试可直接构造 manager 做单测（见 `tests/test_coherence_server_runtime_v2.cpp:46`）。
6. **职责单一**：把“字节在哪、怎么持久化、地址怎么翻译、每行元数据怎么放”收拢到一个类，server 才只需专注协议、时序与一致性语义。这也正是 README 把它单列一层、与 CXLController/HDM/coherency engine 并列的原因。

### 4.5.4 为什么 stack 一定要区分这两层（本质）

本质上是**控制面/建模（server）** 与 **数据面/存储（manager）** 的分离：

- `cxlmemsim_server` ≈ **CXL 控制器/端点的语义**：收发协议帧、解释 `op_type`、跑一致性状态机、算延迟/拓扑/fabric 策略、处理 LSA/DCD/GFAM。
- `SharedMemoryManager` ≈ **被模拟的那片 CXL DRAM 的 backing store**：字节到底放在哪、地址怎么翻译、什么时候刷盘、每行元数据放哪。

必须分开的理由：

1. **同一份存储要被多个前端/一致性实现复用**：TCP、SHM、PGAS-SHM、distributed，以及 coherence-v2（`SharedMemoryCoherenceBackend` 直接持有 `SharedMemoryManager&`）。若不抽出来，每个前端都要各写一份 mmap/地址解码/持久化。
2. **存储介质可替换而不动协议**：RAM(shm/file) ↔ 持久大容量(SSD) 切换只影响 manager，server 与一致性逻辑零改动。
3. **地址空间映射与持久化是“存储属性”，不是“协议属性”**：`base_addr`、cacheline 编号、`magic/version`、`msync`/`fsync` 都属于存储；把它们塞进 server 会让协议代码被存储细节污染。
4. **可独立测试/替换**：一致性引擎面向 `CoherenceMemoryBackend` 抽象，manager 只是其一实现，便于单测。

类比：server 是“内存控制器 + 协议引擎”，manager 是“插上去的那条内存条（以及它的落盘方式）”。所以 README 把它们画成相邻两层。

### 4.5.5 关于 README 那行“下层邻居”的澄清

README 把 `CXLController, HDM decoder, coherency engine, topology model` 画在 `SharedMemoryManager` 下面，容易被读成“manager 的下一层”。实际代码里：

- `HDM decoder` / `coherency engine` / `CXLController` 是**地址解码、一致性、延迟/拓扑的建模组件**，与 server 的请求处理并行协作；
- 真正位于 `SharedMemoryManager` **之下**的，是 backing store（shm / file / `SsdStreamingBackend`）。

所以更准确的两条边界是：**上 = server 的请求处理 API；下 = backing store（SSD backend）接口**，而 CXLController 等是与存储解耦的旁路建模层。

---

## Q5. 为什么 GPU 命令用 BAR2 寄存器？BAR2 的作用是什么？为什么不用 BAR0？

### 5.1 三个 BAR 的实际布局（`lib/qemu/hw/cxl/cxl_type2.c:4293-4354`）

| BAR | 大小 | 内容 | 注册代码 |
| --- | --- | --- | --- |
| **BAR0** | `2*0x10000` = 128KB | offset 0：CXL component registers；offset `0x10000`：CXL device registers（CCI/mailbox/LSA 等） | `cxl_type2.c:4294-4298` |
| **BAR2** | `cache-size`（默认 128MB，prefetchable、64-bit） | **Type 1 缓存窗口** `cache_mem`（RAM）+ I/O overlay `cache_io`；**GPU 命令协议就实现在这里** | `cxl_type2.c:4300-4314` |
| **BAR4** | `device_mem_size`（默认 4GB，prefetchable、64-bit） | **Type 3 设备内存** `device_mem`；offset 0 起 64MB 是 bulk staging，BAR4 顶部是 coherent pool | `cxl_type2.c:4316-4354` |

关键点：GPU 命令寄存器并不是单独的一个 PCI BAR，而是**搭在 BAR2 的缓存窗口上**。QEMU 在 `cache_mem` 之上叠了一个优先级更高的 I/O region `cache_io`（`memory_region_add_subregion_overlap`，`:4310`），并用 `cxl_type2_cache_ops`（`cxl_type2.c:2014-2016`）拦截 guest 的每次访问：`cxl_type2_cache_read/write`（`:1845,1896`）中：

- `CXL_GPU_REG_MAGIC` 返回 `CXL_GPU_MAGIC`（`"CXL2"`＝`0x43584C32`，`:3849-3850`）；
- `CXL_GPU_DATA_OFFSET`（0x1000）起 1MB 是数据缓冲区（`:3979-3980,4031-4032`）。

所以“**BAR2 MMIO 命令寄存器**”准确说是：设备把 Type 1 缓存窗口（BAR2）复用成了加速器命令接口，guest 的 CUDA shim 映射 `resource2`、校验 magic 后就用它收发命令（`qemu_integration/guest_libcuda/libcuda.c:220-275`）。设备结构里虽然有个 `gpu_cmd_region` 字段，但代码注释明确说命令接口直接在 cache_read/write 里处理（`cxl_type2.c:4362`）。

### 5.2 为什么用 BAR2，而不是 BAR0

1. **BAR0 是 CXL 规范定义的寄存器块，不能挪用**。BAR0 里是 component registers（CXL.cache/HDM/RAS 能力）和 device registers（CCI mailbox、LSA 等），由 Linux CXL 驱动在枚举/发 mailbox 命令时按规范读取（`cxl_type2.c:4094` 的 `RBI_CXL_DEVICE_REG | CXL_COMPONENT_REG_BAR_IDX`）。把厂商私有的 GPU 命令协议塞进 BAR0，既不符合 CXL 规范，也会和 CXL 驱动冲突。
2. **容量不够**。BAR0 只有 128KB，而命令块（`CXL_GPU_CMD_REG_SIZE ≈ 1MB+4KB`）+ 1MB 数据缓冲区根本放不下；BAR2 默认 128MB，绰绰有余。
3. **BAR2 本来就是加速器的合规接口**。Type 1 函数暴露的“缓存/一致性窗口”就是一个大可预取 MMIO memory BAR，最适合做加速器寄存器+数据窗口；QEMU 正是靠 overlay 在这里拦截并实现命令语义。
4. **BAR4 已另有用途**。BAR4 是 Type 3 设备内存（数据面：设备内存、64MB bulk staging、coherent pool），不适合再放控制寄存器。
5. **guest ABI 已按 BAR2 固定**。CUDA shim 扫描 `vendor=0x8086/device=0x0d92`、打开 `resource2`、校验 `CXL2` magic（`libcuda.c:220-275`），协议偏移（`cxl_gpu_cmd.h` 全部相对 BAR2 base）也以此为准。

一句话：**BAR0 = 规范的 CXL 寄存器/邮箱；BAR2 = 加速器缓存窗口兼命令接口；BAR4 = 设备内存数据面。** 命令接口放大容量、可预取的 BAR2 上，既合规又放得下。

---

## Q6. “地址孔径（address aperture）”应该怎么理解？它出现是为了说明什么？

### 6.1 一句话定义

**孔径 = 一个地址空间里的一段“窗口”，穿过它就能看到并访问另一个地址空间。** 它不是一个存储实体，而是**映射/边界**：窗口这一侧是主机（或设备）的地址，窗口另一侧是设备（或另一片内存）的真实存储。

最直观的类比：`mmap()` 返回的那段虚拟地址就是“文件孔径”——你看到的是一段连续地址，背后其实是文件里的偏移；窗口大小和文件真实大小可以不一样。CXL/PCIe 里的 aperture 就是这个概念的硬件版本。

### 6.2 本仓库里出现的三类孔径

| 孔径 | 窗口在哪 | 背后是什么 | 代码 |
| --- | --- | --- | --- |
| **PCI BAR 孔径** | 主机 MMIO/I/O 地址空间 | 设备内部寄存器或内存 | BAR0=寄存器窗口(128KB)、BAR2=缓存窗口(128MB)、BAR4=设备内存窗口(4GB)（`cxl_type2.c:4293-4354`） |
| **CXL.mem 窗口（CFMWS/HDM）** | 主机物理地址空间（HPA） | 设备物理地址（DPA）上的存储 | `cxl_type2_cfmws_read/write`、`HDMDecoder`（`lib/qemu/include/hw/cxl/cxl_type2.h:404-434`、`include/hdm_decoder.h`） |
| **CXL.cache 一致窗口** | 设备的可缓存域 | 主机内存 | BAR2 缓存窗口 + snoop filter（`cxl_type2_coherency.*`） |

（服务端还有它的镜像：`SharedMemoryManager::address_to_backend_offset()` 把 CXL 地址翻成 backing 偏移，`cacheline_to_index()` 做 `(addr-base)/64`。那是软件版的孔径解码。）

### 6.3 从代码看“解码”就是孔径的本质

孔径的关键动作是 **decode：窗口内地址 → (目标设备, 目标内偏移)**。

- `HDMDecoder::decode(addr)` 返回 `DecodeResult{ target_id, local_offset, is_remote, hop_count }`（`include/hdm_decoder.h:36-48`）：一个主机地址先被解成“发给哪个 endpoint、在它内部的偏移是多少、是否远端、跨几跳”。
- `HDMRange{base_addr, size, target_id, is_remote}` 和 `HDMInterleaveConfig{base_addr, total_size, granularity, target_ids}`（`hdm_decoder.h:26-45`）：这就是窗口的定义——基址、大小、背后目标、交织粒度。
- `cxl_type2_cfmws_read/write(pdev, hwaddr dpa, ...)`（`cxl_type2.h:404-409`）：host 侧访问带的地址是 **dpa（device physical address）**，`cxl_type2_cfmws_shape_valid` 要求 `window_size==256MB && dpa_base==0 && device_size==256MB`（`cxl_type2.h:416-434`）——即“256MB 窗口 → 256MB 设备区域”。窗口地址和 DPA 之间就是一次固定映射。
- 所以代码里到处能看到 **HPA/窗口地址** 与 **DPA/local_offset** 成对出现、并用一次 `decode` 连接。

### 6.4 它出现是为了说明什么

1. **主机地址 ≠ 设备地址，必须显式映射。** 设备有自己的容量和编址（DPA）；主机只通过一个窗口看到它。孔径把“我看到哪片”与“数据实际在哪”分开。这就是为什么 `decode` 要输出 `local_offset` 和 `target_id`。
2. **窗口大小 ≠ 容量。** 一个孔径可以只覆盖设备容量的一部分；同一个窗口背后的东西还能换（DCD 动态加/减 extent，`CXLType2DCDState`，`cxl_type2.h:177-196`）。换句话说孔径支持**超配/动态容量**——这也是“窗口”和“内存条”的根本区别。
3. **让聚合与池化成为可能。** 多个 endpoint 可被同一片（或交织的）窗口覆盖；`HDMDecoder` 的 interleave 模式把一段窗口地址按粒度轮转分发到多个 target。这样主机软件看到的是一段连续地址，背后却是多设备/远端的池化内存。
4. **把“策略/属性”挂到路径上而不是字节上。** 交错的粒度、`hop_count`/`is_remote`、GFAM 权限、fabric 延迟，都是**窗口/路径**的属性，不是存储介质的属性。这也解释了为什么 README 把 `HDM decoder / topology model` 画在存储旁边：它们是孔径与路径的模型。
5. **解耦上层与底层。** 只要窗口语义不变，背后可以是 HBM、CXL DRAM、远端节点，甚至 shm/文件/SSD（服务端 manager 的 backing 切换）。上层代码（guest 程序、server 请求处理）不需要知道孔径背后是什么。

### 6.5 与“孔径”容易混的两个词

- **DPA (Device Physical Address)**：设备内部的物理地址，是孔径的“另一侧”。`cxl_type2_cfmws_*` 的参数就写成 `dpa`。
- **BAR**：PCIe 里的孔径实现方式——BAR 就是“向 CPU I/O 空间申请的一段窗口”。所以 BAR0/2/4 本质是三个不同用途的孔径。

一句话总结：**孔径回答的是“主机/设备怎样隔着一段有限窗口去访问对方的内存”，它出现是为了把‘地址视图’和‘真实存储’解耦——由此才有映射解码、交织聚合、动态容量、路径属性与后端可替换这些能力。**

---

## Q7. 这个软件可以模拟多通道（multi-channel）SSD DMA 吗？

**结论：不能。** 当前的 SSD streaming 后端是**单通道、单队列、同步**模型；仓库里也没有 NVMe 多队列 / 多设备条带 / 并发 DMA 的实现。但可以用“多进程/多实例”在系统层面近似出多条通道。

### 7.1 为什么说不能（`src/ssd_streaming_backend.cpp`）

- **只有一个后端 fd**：`open_backing()` 打开单个文件或块设备（`ssd_streaming_backend.cpp:155-190`）。
- **io_uring 只注册 1 个文件**：`IORING_REGISTER_FILES` 传 `&fixed_fd, 1U`（`:294-305`）。
- **只注册 1 个固定缓冲**：`IORING_REGISTER_BUFFERS` 传 `&iov, 1U`（`:307-321`）。
- **只有 1 个 ring，深度固定 64**：`io_uring_setup(64U, ...)`（`:227`）。
- **所有 I/O 被单锁串行化**：`io_mutex_`（声明 `:137`，加锁 `:438`、`:453`）——同一时刻只有一次真正的块 I/O。
- **提交后立即等待完成**：`submit_io_uring_rw` 提交 1 个 SQE 后马上 `io_uring_enter` 等 CQE（`:360-423`）。虽然用了 io_uring，但用法是同步阻塞的，**在途 DMA 数 = 1**，没有异步流水线/队列深度。
- **没有多队列/多设备条带**：全仓没有 NVMe 多队列、libaio、SPDK、跨盘 striping 的实现（`coherence_shm_transport_v2.cpp` 里的 “channel” 是一致性 SHM 的主机通道，`bw.cpp` 的 `MAX_CH` 是 CPU 内存通道，都与 SSD 无关）。
- 设计文档也把“优化 SSD backend”列为 Non-Goals（`docs/superpowers/specs/2026-06-18-ssd-streaming-two-qemu-benchmark-design.md:73-76`）。

### 7.2 它现在能做什么（近似手段）

- **read-ahead + 页缓存**：默认 16 页 = 64KB 顺序预读（`:700-728`），CLOCK-like 淘汰（`:635-662`）。这是“隐藏延迟”，不是“多通道带宽”。
- **固定 backing 延迟注入** `backing_latency_ns`：在 `read_page` 里 `sleep_for` 一个确定值来建模 backing 读延迟（`include/ssd_streaming_backend.h:39-45`，`src/ssd_streaming_backend.cpp:426-435`）。它是**每次读的常数延迟**，不是队列深度/带宽/多通道模型。（注意它发生在 `io_mutex_` 之外，所以并发 prefetch 可以重叠这段睡眠，但真正的磁盘 I/O 仍被 `io_mutex_` 串行化。）
- **多实例 = 多通道（进程级）**：起 N 个 `cxlmemsim_server`，各自 `--ssd-backing-file` 指向不同盘/文件，再用 HDM 交织或分布式模式把地址分派到不同实例。这能拿到 N 条独立通道，但粒度是“进程/设备”，不是“单设备多队列”。
  - 分布式模式支持 `--ssd-backing-file=/nvme/....swp`（README:589-603），每个节点一个后端。

### 7.3 若要真正模拟“多通道 SSD DMA”，需要改什么

大致需要给 `SsdStreamingBackend` 增加：多 fd/多设备（每个通道一个）、多注册缓冲与多 SQE 的**异步在途** I/O（去掉单 `io_mutex_` 的串行、维护每通道 in-flight/队列深度）、按通道/交织粒度把地址分片，以及一个带宽/队列延迟模型（而不是单一 `backing_latency_ns` 常数）。当前 API（`read/write/flush/prefetch/pin`）里没有 channel 维度，所以这是需要新增的功能，而非现有开关。

---

## Q8. 为什么先经过 `libcuda.so`（这不已经是原语层了吗）？hetGPU 不该先做类型路由吗？另外 `cuMemcpy` 是 Runtime API 才对吧？

### 8.1 `cu*` 是 **Driver API**，`cuda*` 才是 Runtime API

CUDA 有**两套** API，靠前缀区分，别混淆：

| | Runtime API | Driver API |
| --- | --- | --- |
| 前缀 | `cuda`（`cudaMemcpy`/`cudaMalloc`/`<<<>>>`） | `cu`（`cuMemcpyHtoD`/`cuMemAlloc`/`cuLaunchKernel`） |
| 库 | `libcudart.so` | **`libcuda.so`** |
| 层次 | 高层，隐式 context/device | 低层/原语，显式 context |

所以要纠正一下：`cuMemcpy*`（带 `u`）**正是 Driver API**；`cudaMemcpy`（不带 `u`）才是 Runtime API。本仓库 shim 实现的全部是 `cu*`（`libcuda.c:302-960`），README 也写明是 “CUDA Driver API shim”（`README.md:259`）。库名 `libcuda.so` 本身就是 Driver API 库。

**关键点：Runtime API 是在 Driver API 之上实现的**——`libcudart.so` 内部会调用 `libcuda.so`。因此：

- 只要在 `libcuda.so`（Driver ABI）这一层做拦截，**直接调用 `cu*` 的程序**和**走 Runtime API 的程序**都能覆盖；
- 反过来只在 `cudart` 层拦截，会漏掉直接调 `cu*` 的程序。

这就是为什么 shim 必须落在 “原语层”（Driver API）——它是最低公共 ABI，而不是选错了层。

### 8.2 图中其实有**两个**不同的 `libcuda.so`

1. **guest 侧** `libcuda.so.1` = **shim 本身**（由 `qemu_integration/guest_libcuda/Makefile` 从 `libcuda.c` 编出来，`:20-24`）。guest CUDA 程序解析 `cu*` 符号时命中的就是它；它**不碰真实 GPU**，只把命令写进 BAR2 MMIO（`libcuda.c:517-579`）。
2. **host 侧** 真实 `libcuda.so` = NVIDIA 驱动库，由 **QEMU 的 hetGPU 桥接层** `dlopen()`（或 hetGPU 的 `libnvcuda.so`）（`cxl_hetgpu.c:150,165-177`）。

所以“先经过 libcuda.so”指的是 **guest 侧 shim**（ABI 拦截层），不是已经落到 GPU 原语硬件。

### 8.3 正确的顺序：hetGPU 确实在路由，但在 host 侧、在 BAR2 之后

```
guest app
  -> [guest] libcuda.so.1 (shim, Driver ABI)        # 拦截点
  -> BAR2 MMIO 命令
  -> [QEMU] cxl-type2 device
  -> [QEMU] cxl_type2_hetgpu_init -> hetgpu_init(backend, ...)   # <== hetGPU 在这里选后端
       - backend==5: 直接模拟（不 dlopen）
       - 否则: dlopen(hetgpu-lib) -> cu*
  -> host libcuda.so (真 NVIDIA) 或 libnvcuda.so (hetGPU 翻译到 Intel/AMD/...)
  -> GPU
```

- 后端选择由 `hetgpu-backend` 属性驱动，入口在 `cxl_type2_hetgpu_init`：`hetgpu_init(hetgpu, ct2d->gpu_info.hetgpu_backend, ...)`（`lib/qemu/hw/cxl/cxl_type2.c:664-695`），枚举见 `cxl_hetgpu.h:24-31`。
- `HETGPU_BACKEND_SIMULATION(=5)` 时桥接层完全跳过 `dlopen`（`cxl_hetgpu.c:134-142`）；其它值才加载 CUDA-ABI 库。

### 8.4 为什么不让 hetGPU “在最前面”选类型，而要先过 `libcuda.so`

1. **拦截点必须低于 Runtime API**：因为 Runtime 建在 Driver 之上，只有 Hook 到 `libcuda.so`（Driver ABI）才能覆盖全部 CUDA 程序（见 8.1）。
2. **hetGPU 是 host 侧的事，guest 不该知道**：guest 里只有一块 PCI 设备，没有 GPU 驱动；选 Intel/AMD/NVIDIA/模拟需要 host 的驱动与 QEMU 设备状态，所以路由只能发生在 QEMU 侧（BAR2 之后）。
3. **`libnvcuda.so` 本身就是 `libcuda.so` ABI 兼容库**：也就是说“hetGPU 选择厂商”这件事，是**在被 QEMU 加载的那个库里内部完成**的；从 QEMU 视角看，它只是“加载一个 CUDA 库并调 `cu*`”。`hetgpu-backend` 是告诉桥接层用哪种模式的开关；真正把 PTX 翻译到 Intel/AMD 的后端分派发生在 `libnvcuda.so` 内部。
4. 所以顺序不是“先 libcuda、再 hetGPU 路由”，而是：**guest shim 是强制的 Driver-ABI 拦截层（因为必须在最底层公共 ABI 上截），hetGPU（桥接层 + 可选翻译库）位于 BAR2 之后，正是路由发生的地方。**

一句话：**“原语层”恰恰是正确且唯一的通用拦截点；`libcuda.so` 在 guest 侧是 shim，在 host 侧才是真正的驱动库，而 hetGPU 的后端选择就发生在二者之间的 QEMU 里。**


shim 不是缩写/首字母词（不像 API、ABI），它是普通英文单词，本义是“塞在两个部件之间、让它们贴合/填缝的薄垫片”。
在软件里的含义：一个薄薄的适配/拦截层，夹在两个组件中间，不改动双方，把一侧的调用翻译/转发给另一侧。
在这里（qemu_integration/guest_libcuda/libcuda.c 编译成 libcuda.so.1/libcuda.so）：
- 它对外假装自己是真正的 CUDA Driver API 库 libcuda.so，导出 cuInit/cuMemAlloc/cuMemcpyHtoD/cuLaunchKernel 等 cu* 符号（libcuda.c:302-960），让未修改的 guest CUDA 程序能正常链接/加载它。
- 但它不实现 CUDA、也不碰 GPU，只是把每个调用翻译成 BAR2 MMIO 命令发给 QEMU 的 CXL Type 2 设备（:517-579）。
所以叫 “shim” 是因为它是“guest 程序的 CUDA Driver ABI”与“CXL Type 2 命令协议”之间的一层薄转接层：只有转发/翻译，没有真正的实现，也不需要改动 guest 程序（靠 LD_PRELOAD 或替换 libcuda.so）。

---

## 参考

- `README.md`（Type 2 GPU 栈 `184-303`，Runtime Stack `305-341`，hetGPU 构建 `362-366`，SSD streaming `105-107,559-603`）
- `qemu_integration/guest_libcuda/libcuda.c`、`cxl_gpu_cmd.h`
- `qemu_integration/launch_qemu_type2_hetgpu.sh`、`launch_qemu_type2_multigpu.sh`
- `src/main_server.cc`、`src/shared_memory_manager.cc`、`include/shared_memory_manager.h`
- `src/ssd_streaming_backend.cpp`、`include/ssd_streaming_backend.h`
- `src/coherence_server_runtime_v2.cpp`、`include/coherence_memory_backend.h`
- `lib/qemu/hw/cxl/cxl_hetgpu.c`、`lib/qemu/include/hw/cxl/cxl_hetgpu.h`
- `lib/qemu/hw/cxl/cxl_type2.c`、`lib/qemu/include/hw/cxl/cxl_type2.h`
- `lib/qemu/hw/cxl/cxl_type2_coherency.c`、`lib/qemu/include/hw/cxl/cxl_type2_coherency.h`
- `docs/superpowers/specs/2026-08-09-type2-coherent-domain-litmus-design.md`
