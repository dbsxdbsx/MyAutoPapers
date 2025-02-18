# 设置使用bash作为shell解释器
set shell := ["C:\\Program Files\\Git\\bin\\bash.exe", "-c"]

# 定义常用任务
default:
    cargo run -- \
        --keywords="{{keywords}}" \
        --exclude-keywords="{{exclude}}" \
        --per-keyword-max-result={{per_keyword}}

# 新增直接运行二进制文件的任务
run bin_path:
    "{{bin_path}}" \
        --keywords="{{keywords}}" \
        --exclude-keywords="{{exclude}}" \
        --per-keyword-max-result={{per_keyword}}

# 设置默认参数值
keywords := "
             efficient RL,
             partial observable markov decision process/pomdp,sparse reward reinforcement learning,
             casual RL/counterfactual RL/casual reinforcement learning,
             causal inference/causal discovery/counterfactual reasoning,
             video super resolution,
             knowledge graph/knowledge distillation/knowledge representation/knowledge transfer/knowledge embedding,
             combinatorial game theory/xiangqi/chinese chess,
             code llm,
             speech recognition,
             zero shot tracking/few shot tracking/pose tracking/pose estimation,
             text to 3d/image to 3d/text to texture,
             automated theorem proving/interactive theorem proving/formal verification
             "
exclude := "multi-agent,multiagent"
per_keyword := "8"
target_fields := "cs,stat"
