---
status: suspended
created: 2026-07-01
updated: 2026-07-01
---

# Section 3 新增关键词中低体量子关键词的命中观察

## 背景

为覆盖 only_torch 关注的 CPU 底层加速，Section 3「ML 库 / CPU 效率」从 4 组扩到 7 组，新增三组（见 `justfile`）：

- 硬件内核侧：`SIMD/AVX-512/vectorized inference`
- 硬件内核侧：`fast matrix multiplication/sparse matrix multiplication/cache-efficient`
- 算法数学侧：`linear attention/low-rank compression/Winograd convolution`

新增前对候选短语做过 arXiv abs 命中体检（全时段累计），并本地 `just default` 实测跑通（exit 0）。

## 现象 / 影响

部分子关键词全时段累计命中偏低，长期看某月可能拉不满 8 篇甚至零命中：

| 子关键词 | arXiv abs 累计命中 |
|---|---|
| Winograd convolution | ≈28 |
| vectorized inference | ≈10 |
| AVX-512 | ≈48 |
| low-rank compression | ≈71 |

程序用 `sortBy=lastUpdatedDate` 取最新、不限月份，且同组按 `/` 拆分为独立请求、结果去重合并，因此单个低体量词零命中不会导致整组失败（`src/arxiv.rs`：部分子关键词失败仅 warn，整组全失败才 bail）。影响面有限，属"观察项"而非阻塞。

## 已尝试

- 初版算法数学侧锚点用 `low-rank approximation`（≈1383），实测顶部被科学计算 DLRA（弹性导波、Lindblad、湍流、动理学方程）噪声占满，偏离"ML 数学层加速"初衷。
- 已换锚为 `linear attention`（≈474，纯 ML）+ `low-rank compression`（≈71，ML 模型压缩），实测标题确认相关性达标（含 "Optimizing Winograd Convolution on ARMv8 processors"、"Deep Learning Convolutions on Energy-constrained CPUs" 等正中 only_torch 的命中）。

## 当前卡点

无技术卡点。低体量子关键词是否会在真实每月运行中稳定产出，需要时间序列观察，当下无法一次性判定。

## 暂缓原因

需要真实 CI 月度运行数据（至少两轮）才能判断某组是否需要替换同义词；现在改属过早优化。

## 下次恢复条件

- 时间点：约 2026-09（本条目创建后约两个月，覆盖 2026-08 / 2026-09 两轮 CI 月更）。
- 触发信号：某个新增 group 连续两月零命中，或整组相关性明显偏离预期。

## 下一步建议

- 回看时先查 CI 生成的 `README.md` 历史（第 11 / 12 / 13 section），确认各组是否有稳定产出与相关性。
- 若 `linear attention/low-rank compression/Winograd convolution` 组因 Winograd 太稀疏而整体偏弱，可评估替换为更高体量同义词：`fast Fourier convolution`（≈29）、`low-rank factorization`（≈316）、`structured matrix`（≈250）等，但需重新体检相关性避免引入噪声。
- 硬件内核侧两组体量充足，预计无需调整。

## 相关文件 / 命令 / 对话

- `justfile`：`keywords` 变量 Section 3 的三组新增关键词
- `AGENTS.md` §4.1（三层 group 清单）、§5 Active Context（最近变更 / 下一步）
- `src/arxiv.rs`：子关键词 `/` 拆分（第 177 行）、AND/OR 由整组空格数决定（`src/main.rs` 第 178 行）、部分失败仅 warn / 全失败 bail 的错误处理
- 复现体检：`curl -sL 'https://export.arxiv.org/api/query?search_query=abs:"<phrase>"&max_results=1'` 读 `opensearch:totalResults`
