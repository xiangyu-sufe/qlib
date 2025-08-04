#!/bin/bash

# 监控和自动补充LSF作业的脚本
# 用法: ./monitor_and_resubmit.sh <lr> <weight> <max_jobs> <wait_minutes>

# 学习率和weight参数
LR=${1:-0.001}
WEIGHT=${2:-0.7}
MAX_JOBS=${3:-1}  # 最大并发作业数
WAIT_MINUTES=${4:-3}  # 等待时间（分钟）

# 作业名前缀
JOB_PREFIX="gru_ndcg_job"

# 日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "开始监控LSF作业，学习率: $LR, weight: $WEIGHT, 最大作业数: $MAX_JOBS, 等待时间: $WAIT_MINUTES 分钟"

while true; do
    # 获取当前运行和等待的作业数
    RUNNING_JOBS=$(bjobs -J "$JOB_PREFIX*" | grep -c "RUN")
    PENDING_JOBS=$(bjobs -J "$JOB_PREFIX*" | grep -c "PEND")
    TOTAL_ACTIVE_JOBS=$((RUNNING_JOBS + PENDING_JOBS))
    
    echo "当前活跃作业数: $TOTAL_ACTIVE_JOBS (运行: $RUNNING_JOBS, 等待: $PENDING_JOBS)"
    
    # 如果活跃作业数小于最大作业数，提交新作业
    if [ $TOTAL_ACTIVE_JOBS -lt $MAX_JOBS ]; then
        # 检查是否有已完成的作业（DONE或EXIT状态）
        DONE_JOBS=$(bjobs -a -J "$JOB_PREFIX*" | grep -c "DONE")
        EXITED_JOBS=$(bjobs -a -J "$JOB_PREFIX*" | grep -c "EXIT")
        TOTAL_COMPLETED_JOBS=$((DONE_JOBS + EXITED_JOBS))
        
        if [ $TOTAL_COMPLETED_JOBS -gt 0 ]; then
            echo "检测到已完成作业，准备提交新作业..."
        fi
        
        # 生成唯一作业名
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        JOB_NAME="${JOB_PREFIX}_${TIMESTAMP}"
        
        # 修改bsub命令中的作业名
        sed "s/#BSUB -J gru_ndcg_job/#BSUB -J $JOB_NAME/" submit_lsf.sh > temp_submit.sh
        
        # 提交作业
        echo "提交新作业: $JOB_NAME"
        bsub < temp_submit.sh "$LR" "$WEIGHT"
        
        # 删除临时文件
        rm temp_submit.sh
    else
        echo "已达到最大作业数限制 ($MAX_JOBS)，无需提交新作业"
    fi
    
    # 等待一段时间再检查
    echo "等待 $WAIT_MINUTES 分钟后再次检查..."
    sleep $((WAIT_MINUTES * 60))
done
