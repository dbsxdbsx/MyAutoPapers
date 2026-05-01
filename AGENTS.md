# AGENTS.md - MyAutoPapers

> 项目级 AI agent onboarding 入口。
> 本文件只记录项目内可共享的事实、约定和入口；个人环境中的规则、Skills 与本机路径不在此列。
> 最近更新：2026-05-02。

## 1. Project Identity

- **类型**：CLI 工具（Windows 优先，跨平台 Rust binary）
- **主语言 / 关键框架**：Rust 2021 + tokio + reqwest + feed-rs
- **职责**：每月从 arXiv API 抓取关键词相关论文，生成 `README.md` 与 `.github/ISSUE_TEMPLATE.md`，由 GitHub Actions 自动 commit + 创建 issue
- **阶段**：Stable（CI 已稳定运行多月，本仓库本身代码不再频繁变动；主要演进发生在 keywords 调整）
- **仓库**：本地仓库（`origin` 指向 `github.com/dbsxdbsx/MyAutoPapers`）

## 2. Project Map

```text
MyAutoPapers/
├── src/                       # Rust 源码（4 个文件）
│   ├── main.rs                # CLI 入口、参数解析、README/ISSUE 生成主流程
│   ├── arxiv.rs               # arXiv API 请求、子关键词拆分、过滤、重试
│   ├── types.rs               # Paper、Config 数据结构与 markdown 渲染
│   └── utils.rs               # 时间、备份、表格生成工具函数
├── justfile                   # 关键词配置 + 任务运行入口（关键词唯一来源）
├── .github/
│   ├── workflows/update.yaml  # 每月 1 日 UTC 16:00 自动跑（北京时间次日 00:00）
│   └── ISSUE_TEMPLATE.md      # 程序生成（不要手改）
├── README.md                  # 程序生成（不要手改，每月被覆盖）
├── Cargo.toml / Cargo.lock    # Rust 依赖
└── target/                    # 构建产物（已 .gitignore）
```

关键入口：

- 主程序：[`src/main.rs`](src/main.rs)
- 关键词配置：[`justfile`](justfile)（变量 `keywords` / `exclude` / `per_keyword`）
- CI：[`.github/workflows/update.yaml`](.github/workflows/update.yaml)

语义索引：

- 改**关键词**（拉哪些论文）→ 看 `justfile` 的 `keywords` 变量
- 改**抓取逻辑 / 重试 / 过滤**→ 看 `src/arxiv.rs`
- 改**输出格式**（README/ISSUE 表格）→ 看 `src/types.rs` 的 `to_readme_markdown` / `to_issue_markdown`
- 改**整体流程**（备份 / 写文件 / 错误退出）→ 看 `src/main.rs`
- 改**CI 时机 / 提交策略**→ 看 `.github/workflows/update.yaml`
- **不要**手改 `README.md` 或 `.github/ISSUE_TEMPLATE.md`，它们每次都被程序覆盖

## 3. Working Commands

| 任务 | 命令 | 备注 |
|---|---|---|
| 试跑（dev，本地验证关键词命中） | `just default` | 等价于 `cargo run --` 加 justfile 里的关键词参数；约 5–6 分钟（35 次 arxiv 请求 × 5 秒间隔） |
| 用现成 release 二进制跑 | `just run target/release/my_auto_papers.exe` | CI 中使用；要先 `cargo build --release` |
| Release 构建 | `cargo build --release` | LTO + codegen-units=1，约 1–3 分钟 |
| Lint | `cargo clippy` | 无项目级自定义配置 |
| Format | `cargo fmt` | 无 `rustfmt.toml` |

> 不要在本地手动 commit 由 `just default` 生成的 `README.md` / `.github/ISSUE_TEMPLATE.md`；这两份文件由 CI 工作流统一覆盖，本地试跑后请用 `git checkout -- README.md .github/ISSUE_TEMPLATE.md` 复原。

## 4. Project-specific Norms

### 4.1 关键词管理（核心约束，本项目最常被改的地方）

**关键词唯一来源**：`justfile` 的 `keywords` 变量。改关键词只改这里，**禁止**手改 `README.md` 或 `.github/ISSUE_TEMPLATE.md`。

**层级结构**：
- 用 `,` 分隔不同 **group**（README 里渲染为 `### N. <group>` 一个 section）
- 用 `/` 在同一 group 内分隔 **同义/近义子关键词**

**arXiv API 命中机制**（必须理解，否则容易写出零召回的关键词）：
- 每个子关键词被当成**整段引号短语精确匹配**（`ti:"<sub_keyword>" {AND|OR} abs:"<sub_keyword>"`，见 `src/arxiv.rs:17`）
- AND/OR 的选择：**整 group** 字符串按空格分词，单 word 用 AND（title∧abstract 都要含），多 word 用 OR（title∨abstract 任一含，见 `src/main.rs:178-182`）
- 子关键词之间是**独立请求**，每次请求间隔 5 秒（`ARXIV_REQUEST_INTERVAL_SECS`，见 `src/arxiv.rs:10`），结果按论文 link 去重合并

**关键词撰写规则**：
- 每个子关键词控制在 **2–4 个英文词**，太长几乎零命中（精确短语匹配）
- 单 group 内子关键词数 **≤ 4**，超过会拖慢请求且稀释命中（每多一个就多 5 秒 + 8 篇配额浪费）
- `per_keyword_max_result` 是**每个 group** 的上限（不是每个子关键词），合理范围 5–15
- 加新方向时优先复用现有 5 个 section 的归类

**当前 5 个 section / 16 个 group**（与 `justfile` 同步）：

1. **强化学习效率**（3 组）：efficient RL / model-based / offline
2. **图像处理效率**（3 组）：efficient ViT / efficient classification·detection·segmentation / efficient diffusion
3. **ML 库 / CPU 效率**（4 组）：efficient cpu inference / quantization / pruning·KD / tensor compilation·graph opt·operator fusion
4. **其他前沿**（4 组）：image SR / video SR / quant trading·RL trading / stock prediction·portfolio
5. **神经演化 / NAS**（2 组）：neuroevolution·NEAT / NAS·multi-objective NAS

新增/调整关键词时同步检查：
- 新增 section 是否在上面 5 类的覆盖之内（不在则评估是否要扩张）
- 全局 `exclude_keywords=multi-agent,multiagent` 是子串匹配（见 `src/arxiv.rs:227-249`），会同时影响所有 group；改这个值会影响包括量化交易方向在内的全部论文

### 4.2 错误处理约定

- 单个 group 全部子关键词都失败 → bail（见 `src/arxiv.rs:204`）
- 部分子关键词失败 → 保留成功结果，仅打 warn
- 整轮一篇都没拉到且至少一个 group 异常 → `main.rs` 末尾 bail，避免 CI 提交空 README

### 4.3 时区与时间

所有展示时间使用**亚洲/上海时区**（见 `src/utils.rs` 的 `chrono_tz::Asia::Shanghai`），CI cron 用 UTC（每月 1 日 16:00 UTC = 北京时间 2 日 00:00）。

## 5. Active Context

- **进行中**：关键词体系全面对齐使用者 4 个核心关注点 + 1 个隐性关注（neuroevolution / NAS），从原 13 组旧关键词（含 typo `casual`、与需求脱节的 chinese chess / code llm / speech / theorem proving 等）重构为 5 section / 16 group
- **最近变更**：`6a20239 ✨增强 arXiv API 请求处理，添加错误处理和过滤功能`（错误处理 + 过滤 pipeline）
- **阻塞**：无（无 `.issue/` 目录）
- **下一步**：观察 2026-05 第一轮新关键词的 arxiv 命中情况；若某 group 连续两月零命中，考虑泛化或合并

## 6. Knowledge Index

无 `.doc/` 目录。本项目代码量小（约 600 行 Rust），关键约束都集中在本文件 §4。如未来出现需要长期沉淀的设计/坑点，按 `/maintain-docs` 流程在 `.doc/` 下建专题文件并回链到这里。

## 7. Subprojects & repo boundaries

无 git submodule、无 workspace、无嵌套 `AGENTS.md`。单仓单包结构。
