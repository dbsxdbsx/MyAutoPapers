# Issue Records

`.issue/` 是项目内的**未闭环问题日志**，面向协作者和自动化工具：保留暂时无法收尾、但不能丢失上下文的现场，避免下次继续时重复排查。

它不是任务清单，也不是长期知识库:

- 普通待办放在 `README.md` TODOs、项目管理工具或当前任务计划。
- 已验证、可复用的架构、配置、排障经验沉淀到 `.doc/`。
- 多轮调试后仍未闭环、需要暂缓的问题记录到 `.issue/items/`。

设计参考：[Architecture Decision Records](https://adr.github.io/) 的目录与文件约定、Diátaxis 的 reference 分类。

## Directory Contract

```text
.issue/
├── README.md                                       # 本文件:契约 + 模板（唯一权威源）
├── items/                                          # 当前仍需关注的问题条目，平铺
│   └── 2026-07-01_arxiv_low_volume_keywords.md
├── assets/                                         # 当前问题附件，按条目 stem 分组
└── _archive/                                       # 已闭环 / 已替代的问题历史
    ├── items/
    └── assets/
```

约定:

- `items/` 是**当前仍需关注**的问题条目目录;不要把条目直接放在 `.issue/` 根。
- `assets/<item-file-stem>/` 与条目文件名（去掉 `.md`）一一对应;条目内用相对路径引用，例如 `../assets/2026-07-01_arxiv_low_volume_keywords/screenshot.png`。
- `_archive/items/` 存放已闭环或已被替代的历史条目;归档后对应附件移动到 `_archive/assets/<item-file-stem>/`。
- 归档前必须先把 frontmatter 的 `status` 改为 `resolved` 或 `superseded`，刷新 `updated`，并在正文补齐结论、根因、修复 / 替代方式和验证结果。

## Naming

- 条目文件名:`YYYY-MM-DD_<topic>.md`，topic 用英文 `snake_case`，与 `.doc/` 一致。
  - `YYYY-MM-DD` 为问题首次记录日期;不同日期发生的相似问题分别建条目，不要复用旧文件。
- 附件文件名:英文 `snake_case`，扩展名按实际类型;当前条目附件放在 `assets/<item-file-stem>/` 下，归档条目附件放在 `_archive/assets/<item-file-stem>/` 下。
- 一个条目只记录一个未闭环问题。

## Entry Template

每个条目以 YAML frontmatter 开头，其余为 Markdown 正文。复制以下模板到 `items/<date>_<topic>.md` 即可:

```markdown
---
status: suspended
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# 问题标题

## 背景

## 现象 / 影响

## 已尝试

## 当前卡点

## 暂缓原因

## 下次恢复条件

## 下一步建议

## 相关文件 / 命令 / 对话
```

字段说明:

- `status`:`suspended`（暂缓，待续查）｜`blocked`（被权限/环境/上游/数据/决策阻塞）｜`resolved`（已解决，可归档）｜`superseded`（被新条目、外部 issue 或文档替代，可归档）。
- `created`:首次记录日期（ISO 8601，YYYY-MM-DD）。
- `updated`:最近一次修改日期;状态变化、补充证据、添加附件时都要更新。

> 这是**最小集**字段。如果某个项目需要更细的元数据（severity、tags、related、owner 等），可在该项目的 `.issue/README.md` 中扩展；本 README 只锁定上述三项。

## When to Add

适合新增 `.issue/items/` 记录:

- bug、lint、测试、构建、部署或依赖问题多轮排查后仍未解决。
- 当前缺少权限、环境、数据、上游修复或关键决策，必须暂停。
- 已经尝试过若干路径，下次继续时需要避免重复踩坑。
- 问题对后续开发有影响，但暂时不适合写进 `.doc/`。

不适合新增:

- 立即可执行的小 TODO。
- 已有明确根因和修复方案的问题（继续修，不要先建条目）。
- 纯想法、功能愿望或产品规划。
- 已经沉淀为稳定结论的知识（应放 `.doc/`）。

## When to Archive

适合从 `items/` 移入 `_archive/items/`:

- 问题已经解决，条目已补齐根因、修复方式和验证结果。
- 问题被新条目、外部 issue、PR 或 `.doc/` 文档替代，且条目已用 `superseded` 指向替代来源。
- 条目不再影响当前开发、构建、测试、部署或发布，但仍有历史追溯价值。

不适合归档:

- 仍会影响当前开发、构建、测试、部署或发布的问题。
- 还没有写清楚结论 / 根因 / 验证结果的 `resolved` 条目。
- 含敏感信息且尚未脱敏的条目或附件;这类内容应先处理隐私风险，再决定保留、归档或删除。

归档操作:

1. 更新条目 frontmatter:`status: resolved` 或 `status: superseded`，刷新 `updated`。
2. 在正文补充 `## 结论`、`## 根因`、`## 修复方式 / 替代来源`、`## 验证结果`。
3. 将条目从 `items/` 移到 `_archive/items/`。
4. 若存在附件，将 `assets/<item-file-stem>/` 移到 `_archive/assets/<item-file-stem>/`，并同步修正条目内相对链接。
5. 从当前入口文档、问题索引或外部 tracker 中移除或更新该条目引用;若产生长期知识，再提炼到 `.doc/` 或 `README.md`。
