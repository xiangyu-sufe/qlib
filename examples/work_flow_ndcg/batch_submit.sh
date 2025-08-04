#!/bin/bash

# 检查参数数量
if [ $# -lt 5 ]; then
    echo "用法: $0 <learning_rate> <weight_start> <weight_end> <weight_step> <max_jobs> [wait_minutes]"
    echo "示例: $0 0.01 0 1 0.05 5 3"
    echo "      表示学习率0.01，weight从0到1，步长0.05，最多同时5个任务，每次检查间隔3分钟"
    exit 1
fi

LR=$1
WEIGHT_START=$2
WEIGHT_END=$3
WEIGHT_STEP=$4
MAX_JOBS=$5
WAIT_MINUTES=${6:-3}  # 默认等待3分钟

# 生成所有weight值
weights=()
for w in $(seq $WEIGHT_START $WEIGHT_STEP $WEIGHT_END); do
    # 使用printf控制精度，避免浮点数精度问题
    weight=$(printf "%.2f" $w)
    weights+=("$weight")
done

echo "总共需要运行 ${#weights[@]} 个任务，weight值: ${weights[@]}"

# 提交初始批次的任务
for ((i=0; i<MAX_JOBS && i<${#weights[@]}; i++)); do
    weight=${weights[i]}
    echo "提交任务: 学习率=$LR, weight=$weight"
    ./submit_lsf.sh $LR $weight
    sleep 1  # 避免同时提交太多任务
    unset weights[i]  # 从数组中移除已提交的任务
done

# 监控并补充任务
while [ ${#weights[@]} -gt 0 ]; do
    # 获取当前运行中的作业数
    running_jobs=$(bjobs -w | grep " $USER " | grep -c " RUN\|PEND")
    echo "当前运行中的作业数: $running_jobs"
    
    # 计算可以提交的新任务数
    available_slots=$((MAX_JOBS - running_jobs))
    
    if [ $available_slots -gt 0 ] && [ ${#weights[@]} -gt 0 ]; then
        echo "有 $available_slots 个可用槽位，准备提交新任务..."
        
        # 提交新任务
        for ((i=0; i<available_slots && i<${#weights[@]}; )); do
            if [ -n "${weights[i]}" ]; then
                weight=${weights[i]}
                echo "提交新任务: 学习率=$LR, weight=$weight"
                ./submit_lsf.sh $LR $weight
                unset weights[i]  # 从数组中移除已提交的任务
                sleep 1  # 避免同时提交太多任务
                ((i++))
            else
                ((i++))
            fi
        done
        
        # 重新索引数组
        weights=("${weights[@]}")
    fi
    
    # 如果还有任务未提交，等待一段时间再检查
    if [ ${#weights[@]} -gt 0 ]; then
        echo "等待 $WAIT_MINUTES 分钟后再次检查..."
        sleep $((WAIT_MINUTES * 60))
    fi
done

echo "所有任务已提交完成！"
echo "使用 'bjobs -w' 命令查看作业状态"
