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

# 关键词分 5 个 section，共 16 组（修改时请同步更新 AGENTS.md 的 Project-specific Norms）
# Section 1: 强化学习效率（RL Efficiency）
# Section 2: 图像处理效率（Image Processing Efficiency）
# Section 3: ML 库 / CPU 效率（ML Library / CPU Efficiency，对应 only_torch 项目）
# Section 4: 其他前沿（超分 + 量化投资）
# Section 5: 神经演化 / NAS（Neuroevolution / NAS）
keywords := "
             efficient reinforcement learning/sample efficient reinforcement learning,
             model-based reinforcement learning/world model,
             offline reinforcement learning,
             efficient vision transformer/mobile vit/lightweight vit,
             efficient image classification/efficient object detection/efficient semantic segmentation,
             efficient diffusion model/one-step diffusion/distillation diffusion,
             efficient cpu inference/on-device inference/edge inference,
             model quantization/low-bit quantization/binary neural network,
             network pruning/sparse neural network/knowledge distillation,
             tensor compilation/computation graph optimization/operator fusion,
             image super resolution/efficient super resolution,
             video super resolution,
             quantitative trading/algorithmic trading/reinforcement learning for trading,
             stock prediction/portfolio optimization/financial time series forecasting,
             neuroevolution/NEAT/evolutionary neural network,
             neural architecture search/multi-objective neural architecture search
             "
exclude := "multi-agent,multiagent"
per_keyword := "8"
target_fields := "cs,stat"
