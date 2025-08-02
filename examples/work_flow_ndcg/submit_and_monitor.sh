#!/bin/bash

max_running_jobs=5          # 最大同时运行任务数
submit_interval=180          # 每次检查提交间隔（秒）
monitor_interval=30         # 监控状态间隔（秒）
command_file="bsub_template.txt"

if [ "$#" -lt 2 ]; then
    echo "❌ 错误：请至少传入学习率和sigma参数，例如：0.001 3.03 0.0005 2.5"
    echo "用法: $0 <lr1> <sigma1> [<lr2> <sigma2> ...]"
    exit 1
fi

# 检查参数个数是否为偶数
if [ $(( $# % 2 )) -ne 0 ]; then
    echo "❌ 错误：参数个数必须为偶数（学习率和sigma成对出现）"
    echo "用法: $0 <lr1> <sigma1> [<lr2> <sigma2> ...]"
    exit 1
fi

# 将参数分为学习率和sigma数组
lrs=()
sigmas=()
i=1
while [ $i -le $# ]; do
    lrs+=("${!i}")
    ((i++))
    sigmas+=("${!i}")
    ((i++))
done

if [ ! -f "$command_file" ]; then
    echo "❌ 错误：找不到命令模板文件 $command_file"
    exit 1
fi

# 读取 bsub 模板命令（多行支持，空行分隔）
mapfile -t raw_lines < "$command_file"
templates=()
current_block=""
for line in "${raw_lines[@]}"; do
    if [[ -z "$line" ]]; then
        if [[ -n "$current_block" ]]; then
            templates+=("$current_block")
            current_block=""
        fi
    else
        current_block+="$line"$'\n'
    fi
done
if [[ -n "$current_block" ]]; then
    templates+=("$current_block")
fi

declare -A lr_job_ids
declare -A lr_status

i=0
total=${#lrs[@]}

while [ $i -lt $total ]; do
    running=$(bjobs -u "$USER" 2>/dev/null | grep RUN | wc -l)

    if [ "$running" -lt "$max_running_jobs" ]; then
        lr=${lrs[$i]}
        sigma=${sigmas[$i]}
        for template in "${templates[@]}"; do
            # 替换模板中占位符 __LR__ 和 __SIGMA__ 为当前值
            cmd="${template//__LR__/$lr}"
            cmd="${cmd//__SIGMA__/$sigma}"
            echo "🚀 提交任务 lr=$lr, sigma=$sigma（当前运行 $running 个）"
            job_output=$(eval "$cmd" 2>&1)
            echo "$job_output"

            if [[ "$job_output" =~ \<([0-9]+)\> ]]; then
                job_id="${BASH_REMATCH[1]}"
                lr_job_ids["$lr-$sigma"]=$job_id
                lr_status["$lr-$sigma"]="RUNNING"
                echo "✅ 提交成功：lr=$lr, sigma=$sigma, job_id=$job_id"
            else
                echo "❌ 提交失败：lr=$lr, sigma=$sigma"
                lr_status["$lr-$sigma"]="FAILED"
            fi
        done
        ((i++))
    else
        echo "⏸️ 当前运行任务数已达上限（$running），等待空位中..."
    fi

    sleep $submit_interval
done

echo "✅ 所有任务提交完毕，开始监控..."

while true; do
    echo "🔍 [$(date '+%Y-%m-%d %H:%M:%S')] 监控任务状态..."

    for key in "${!lr_job_ids[@]}"; do
        job_id=${lr_job_ids[$key]}
        if [[ "${lr_status[$key]}" != "FINISHED" ]]; then
            bjobs_out=$(bjobs "$job_id" 2>&1)

            if echo "$bjobs_out" | grep -q "not found"; then
                echo "⚠️ 任务 $key (job_id=$job_id) 已结束或异常退出"
                lr_status["$key"]="FINISHED"
            else
                state=$(echo "$bjobs_out" | awk 'NR==2 {print $3}')
                echo "📡 $key 状态=$state"
                lr_status["$key"]="$state"
            fi
        fi
    done

    sleep $monitor_interval
done
