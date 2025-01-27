# 自动论文推送
本项目自动从 arXiv 获取最新的论文，基于关键词进行筛选。

点击 'Watch' 按钮可以接收自动推送的邮件通知。

<<<<<<< HEAD
## 最后更新：2025-01-27 09:16
=======
## 最后更新：2025-01-27 00:05
>>>>>>> 38f20309dcf68c683ada4c9e68b8843a07fe6f30
**本次更新执行命令**
```
target\debug\my_auto_papers.exe --keywords=
             efficient RL,video super resolution,
             partial observable markov decision process/pomdp,sparse reward reinforcement learning,
             2.5d fighting game/fighting game ai/game ai/fighting game reinforcement learning,
             combinatorial game theory/xiangqi/chinese chess,
             code llm,
             speech recognition,
             zero shot tracking/few shot tracking/pose tracking/pose estimation,
             text to 3d/image to 3d/text to texture,
             casual inference,
             automated theorem proving/interactive theorem proving/formal verification
              --exclude-keywords=multi-agent --per-keyword-max-result=50
```

**参数详解**
- 关键词：`efficient RL`, `video super resolution`, `partial observable markov decision process/pomdp`, `sparse reward reinforcement learning`, `2.5d fighting game/fighting game ai/game ai/fighting game reinforcement learning`, `combinatorial game theory/xiangqi/chinese chess`, `code llm`, `speech recognition`, `zero shot tracking/few shot tracking/pose tracking/pose estimation`, `text to 3d/image to 3d/text to texture`, `casual inference`, `automated theorem proving/interactive theorem proving/formal verification`
- 排除关键词：`multi-agent`
- 每关键词最大结果：`50`
- 目标领域：`cs`, `stat`
- 每关键词重试次数：`3`



