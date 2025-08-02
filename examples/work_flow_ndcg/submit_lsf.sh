#!/bin/bash
#BSUB -J gru_ndcg_job            # 作业名
#BSUB -q gpu                 # 队列名，可根据系统调整为gpu、short等
#BSUB -n 1                      # 使用 CPU 核心数
#BSUB -gpu "num=1"              # 请求 GPU 数量（如不需要可删除此行）
#BSUB -o grundcg_%J.out    # 标准输出文件（%J为Job ID）
#BSUB -e grundcg_%J.err    # 错误输出文件
#BSUB -m "gpu01|gpu02|gpu03|gpu04|gpu05|gpu06" # 指定 gpu

# 设置默认学习率，如果没传参就用 0.001
LR=${1:-0.001}

# 打印传入参数
echo "Running GRUNDCG training with learning rate: $LR"

# 创建日志目录
mkdir -p logs

# 设置日志文件（带时间戳）
TIME_TAG=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/grundcg_run_lr${LR}_${TIME_TAG}.log"

# 运行 Python 脚本，替换其中的学习率参数
python workflow_ndcg.py --lr "$LR" > "$LOG_FILE" 2>&1
