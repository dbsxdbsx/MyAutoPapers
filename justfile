# 设置使用bash作为shell解释器
set shell := ["C:\\Program Files\\Git\\bin\\bash.exe", "-c"]

# 定义常用任务
default:
    cargo run -- \
        --keywords="{{keywords}}" \
        --exclude-keywords="{{exclude}}" \
        --per-keyword-max-result={{per_keyword}}

# 设置默认参数值
per_keyword := "50"
keywords := "
             efficient RL,video super resolution,
            
             "
exclude := "multi-agent"
target_fields := "cs,stat"
