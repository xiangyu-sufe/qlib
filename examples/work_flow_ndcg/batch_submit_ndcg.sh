#!/bin/bash

# 用法: ./batch_submit_ndcg.sh <learning_rate> <weight_start> <weight_end> <weight_step> <max_jobs> [wait_minutes]
# 例: ./batch_submit_ndcg.sh 0.01 0 1 0.05 5 3

if [ $# -lt 5 ]; then
    echo "用法: $0 <learning_rate> <weight_start> <weight_end> <weight_step> <max_jobs> [wait_minutes]"
    exit 1
fi

LR=$1
WEIGHT_START=$2
WEIGHT_END=$3
WEIGHT_STEP=$4
MAX_JOBS=$5
WAIT_MINUTES=${6:-3}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_SAVE_DIR="./experiments/ndcg_exp_${TIMESTAMP}"
mkdir -p "$BASE_SAVE_DIR"

# 生成所有weight值
weights=()
for w in $(seq $WEIGHT_START $WEIGHT_STEP $WEIGHT_END); do
    weight=$(printf "%.2f" $w)
    weights+=("$weight")
done

# 提交初始批次任务
for ((i=0; i<MAX_JOBS && i<${#weights[@]}; i++)); do
    weight=${weights[i]}
    SAVE_PATH="${BASE_SAVE_DIR}/weight_${weight}"
    mkdir -p "$SAVE_PATH"
    echo "提交任务: lr=$LR, weight=$weight, save_path=$SAVE_PATH"
    bsub -J ndcg_w${weight} -q gpu -n 1 -m "gpu01 gpu02 gpu03 gpu04 gpu05 gpu06" -gpu "num=1" -o "${SAVE_PATH}/stdout.log" -e "${SAVE_PATH}/stderr.log" \
        python workflow_ic_ndcg.py --lr "$LR" --weight "$weight" --save_path "$SAVE_PATH"
    unset weights[i]
    sleep 5
done

# 动态补充任务
while [ ${#weights[@]} -gt 0 ]; do
    running_jobs=$(bjobs -w | grep "$USER" | grep -c " RUN\\|PEND")
    available_slots=$((MAX_JOBS - running_jobs))
    if [ $available_slots -gt 0 ]; then
        for ((i=0; i<available_slots && i<${#weights[@]}; )); do
            if [ -n "${weights[i]}" ]; then
                weight=${weights[i]}
                SAVE_PATH="${BASE_SAVE_DIR}/weight_${weight}"
                mkdir -p "$SAVE_PATH"
                echo "提交任务: lr=$LR, weight=$weight, save_path=$SAVE_PATH"
                bsub -J ndcg_w${weight} -q gpu -n 1 -gpu "num=1" -o "${SAVE_PATH}/stdout.log" -e "${SAVE_PATH}/stderr.log" \
                    python workflow_ndcg.py --lr "$LR" --weight "$weight" --save_path "$SAVE_PATH"
                unset weights[i]
                sleep 1
                ((i++))
            else
                ((i++))
            fi
        done
        weights=("${weights[@]}")
    fi
    if [ ${#weights[@]} -gt 0 ]; then
        echo "等待 $WAIT_MINUTES 分钟后再次检查..."
        sleep $((WAIT_MINUTES * 60))
    fi
done

echo "所有任务已提交完成！"
