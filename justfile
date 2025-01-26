# 设置使用bash作为shell解释器
set shell := ["C:\\Program Files\\Git\\bin\\bash.exe", "-c"]

# 定义常用任务
default:
    cargo run -- \
        --keywords="{{keywords}}" \
        --exclude-keywords="{{exclude}}" \
        --per-keyword-max-result={{per_keyword}}

# 使用 release 构建运行
run:
    cargo run --release -- \
        --keywords="{{keywords}}" \
        --exclude-keywords="{{exclude}}" \
        --per-keyword-max-result={{per_keyword}}

# 设置默认参数值
per_keyword := "50"
keywords := "
             efficient RL,video super resolution,
             2.5d fighting game/fighting game ai/game ai/fighting game reinforcement learning,
             combinatorial game theory/xiangqi/chinese chess,
             code llm,
             speech recognition,
             zero shot tracking/few shot tracking/pose tracking/pose estimation,
             text to 3d/image to 3d/text to texture,
             casual inference,
             automated theorem proving/interactive theorem proving/formal verification
             "
exclude := "multi-agent"
target_fields := "cs,stat"
