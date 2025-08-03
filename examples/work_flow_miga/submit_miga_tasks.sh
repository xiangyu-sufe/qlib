#!/bin/bash

# 配置参数
max_gpu_jobs=4              # 最大GPU任务数（每个GPU一个任务）
submit_interval=180           # 提交检查间隔（秒）
monitor_interval=60          # 监控状态间隔（秒）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_error() {
    echo -e "${RED}❌ 错误: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ 成功: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  警告: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  信息: $1${NC}"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <参数类型> <参数列表>"
    echo ""
    echo "参数类型:"
    echo "  lr_omega     - 学习率和omega参数的笛卡尔积"
    echo "  custom       - 自定义参数组合"
    echo ""
    echo "示例:"
    echo "  $0 lr_omega \"0.001,0.005,0.01\" \"0.001,0.002,0.003\""
    echo "  $0 custom \"0.001:0.001,0.005:0.002,0.01:0.003\""
    echo ""
    echo "配置:"
    echo "  最大GPU任务数: $max_gpu_jobs"
    echo "  提交检查间隔: ${submit_interval}秒"
    echo "  监控状态间隔: ${monitor_interval}秒"
}

# 检查GPU可用性
check_gpu_availability() {
    local gpu_id="$1"
    
    # 检查GPU是否存在
    if ! nvidia-smi -i "$gpu_id" >/dev/null 2>&1; then
        return 1
    fi
    
    # 检查GPU内存使用情况
    local memory_info=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)
    if [ $? -ne 0 ]; then
        return 1
    fi
    
    local memory_used=$(echo "$memory_info" | cut -d',' -f1)
    local memory_total=$(echo "$memory_info" | cut -d',' -f2)
    
    # 如果内存使用率超过80%，认为不可用
    if [ "$memory_used" -gt $((memory_total * 8 / 10)) ]; then
        return 1
    fi
    
    # 检查GPU利用率
    local gpu_util=$(nvidia-smi -i "$gpu_id" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ $? -eq 0 ] && [ "$gpu_util" -gt 90 ]; then
        return 1
    fi
    
    return 0
}

# 获取可用的GPU ID
get_available_gpu() {
    # 检查所有GPU，返回第一个可用的
    for gpu in 0 1 2 3; do
        if check_gpu_availability "$gpu"; then
            # 检查这个GPU是否已经在运行任务
            local gpu_running=$(get_gpu_running_jobs "$gpu")
            if [ "$gpu_running" -eq 0 ]; then
                echo "$gpu"
                return 0
            fi
        fi
    done
    
    echo ""  # 没有可用GPU
}

# 获取指定GPU上运行的任务数
get_gpu_running_jobs() {
    local target_gpu="$1"
    local count=0
    
    for task_id in "${!job_pids[@]}"; do
        if [[ "${job_status[$task_id]}" == "RUNNING" ]]; then
            local task_gpu="${job_params[${task_id}_gpu]}"
            if [[ "$task_gpu" == "$target_gpu" ]]; then
                ((count++))
            fi
        fi
    done
    
    echo "$count"
}

# 获取当前运行的任务数
get_running_jobs() {
    local count=0
    for task_id in "${!job_status[@]}"; do
        if [[ "${job_status[$task_id]}" == "RUNNING" ]]; then
            ((count++))
        fi
    done
    echo "$count"
}

# 检查任务状态
check_job_status() {
    local task_id="$1"
    local pid="${job_pids[$task_id]}"
    
    if [ -z "$pid" ]; then
        return
    fi
    
    # 检查进程是否仍在运行
    if kill -0 "$pid" 2>/dev/null; then
        job_status["$task_id"]="RUNNING"
    else
        # 进程已结束，检查退出状态
        wait "$pid"
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            job_status["$task_id"]="FINISHED"
            completed_tasks+=("$task_id")
            # 释放GPU
            local task_gpu="${job_params[${task_id}_gpu]}"
            if [ -n "$task_gpu" ]; then
                gpu_usage["$task_gpu"]="FREE"
            fi
            print_success "任务完成: ${job_params[$task_id]} (pid=$pid)"
        else
            job_status["$task_id"]="FAILED"
            print_error "任务失败: ${job_params[$task_id]} (pid=$pid, exit_code=$exit_code)"
        fi
    fi
}

# 提交任务到指定GPU
submit_task_to_gpu() {
    local task="$1"
    local task_id="$2"
    local target_gpu="$3"
    
    # 解析任务参数
    local lr=""
    local omega=""
    
    if [[ "$task" =~ lr:([^,]+) ]]; then
        lr="${BASH_REMATCH[1]}"
    fi
    
    if [[ "$task" =~ omega:([^,]+) ]]; then
        omega="${BASH_REMATCH[1]}"
    fi
    
    # 使用指定的GPU
    local gpu="$target_gpu"
    
    # 检查GPU是否可用
    if ! check_gpu_availability "$gpu"; then
        print_error "GPU $gpu 不可用"
        job_status["$task_id"]="FAILED"
        return 1
    fi
    
    # 标记GPU为已使用
    gpu_usage["$gpu"]="USED"
    
    # 生成保存路径（基于参数值）
    local save_path="results_lr${lr}_omega${omega}"
    
    # 构建命令
    local cmd="python workflow_miga.py --lr $lr --omega $omega --save_path $save_path"
    
    print_info "🚀 提交任务到GPU $gpu: $task"
    print_info "CUDA_VISIBLE_DEVICES: $gpu"
    print_info "执行命令: $cmd"
    print_info "保存路径: $save_path"
    
    # 在后台运行任务
    CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd" > "logs/miga_lr${lr}_omega${omega}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    local pid=$!
    
    job_pids["$task_id"]="$pid"
    job_status["$task_id"]="RUNNING"
    job_params["$task_id"]="$task"
    job_params["${task_id}_gpu"]="$gpu"
    print_success "提交成功: $task (pid=$pid, gpu=$gpu)"
    return 0
}

# 检查参数
if [ "$#" -lt 2 ]; then
    print_error "错误：参数不足"
    show_help
    exit 1
fi

param_type="$1"
shift

# 根据参数类型解析参数
case "$param_type" in
    "lr_omega")
        if [ "$#" -ne 2 ]; then
            print_error "lr_omega模式需要2个参数：学习率列表和omega列表"
            exit 1
        fi
        IFS=',' read -ra lrs <<< "$1"
        IFS=',' read -ra omegas <<< "$2"
        
        # 生成任务列表（笛卡尔积）
        tasks=()
        for lr in "${lrs[@]}"; do
            for omega in "${omegas[@]}"; do
                tasks+=("lr:$lr,omega:$omega")
            done
        done
        ;;
        
    "custom")
        if [ "$#" -ne 1 ]; then
            print_error "custom模式需要1个参数：自定义参数组合"
            exit 1
        fi
        IFS=',' read -ra custom_tasks <<< "$1"
        
        # 生成任务列表
        tasks=()
        for task in "${custom_tasks[@]}"; do
            tasks+=("$task")
        done
        ;;
        
    *)
        print_error "未知的参数类型: $param_type"
        show_help
        exit 1
        ;;
esac

# 初始化状态跟踪
declare -A job_pids
declare -A job_status
declare -A job_params
declare -A gpu_usage  # 跟踪GPU使用情况
pending_tasks=("${tasks[@]}")
submitted_tasks=()
completed_tasks=()

# 初始化GPU使用情况
available_gpus=(0 1 2 3)
for gpu in "${available_gpus[@]}"; do
    gpu_usage["$gpu"]="FREE"
done

print_info "📋 总任务数: ${#tasks[@]}"
print_info "📋 可用GPU: ${available_gpus[*]}"

# 创建日志和结果目录
mkdir -p logs
mkdir -p results

# 显示GPU信息
show_gpu_info() {
    print_info "系统GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | while IFS=',' read -r index name total used util; do
        # 清理名称中的空格
        name=$(echo "$name" | sed 's/^ *//;s/ *$//')
        echo "  GPU $index: $name, 内存: ${used}MB/${total}MB, 利用率: $util%"
    done
    echo ""
}

show_gpu_info

# 主循环
print_info "开始任务提交和监控..."
while true; do
    # 检查已完成的任务
    for task_id in "${!job_pids[@]}"; do
        if [[ "${job_status[$task_id]}" != "FINISHED" && "${job_status[$task_id]}" != "FAILED" ]]; then
            check_job_status "$task_id"
        fi
    done
    
    # 获取当前运行的任务数
    running_jobs=$(get_running_jobs)
    print_info "当前运行任务数: $running_jobs/$max_gpu_jobs"
    
    # 如果有待提交的任务且未达到最大GPU数，则提交新任务
    if [ ${#pending_tasks[@]} -gt 0 ] && [ "$running_jobs" -lt "$max_gpu_jobs" ]; then
        # 获取下一个待提交的任务
        task="${pending_tasks[0]}"
        task_id="task_$(date +%s)_$RANDOM"
        
        # 获取可用GPU
        gpu=$(get_available_gpu)
        
        if [ -n "$gpu" ]; then
            # 提交任务
            if submit_task_to_gpu "$task" "$task_id" "$gpu"; then
                # 从待提交列表中移除
                pending_tasks=("${pending_tasks[@]:1}")
                submitted_tasks+=("$task_id")
            else
                print_error "任务提交失败: $task"
                # 从待提交列表中移除失败的任务
                pending_tasks=("${pending_tasks[@]:1}")
            fi
        else
            print_warning "没有可用的GPU，等待资源释放..."
        fi
    elif [ ${#pending_tasks[@]} -eq 0 ] && [ "$running_jobs" -eq 0 ]; then
        print_success "所有任务已完成！"
        break
    else
        print_info "等待任务完成或资源释放..."
    fi
    
    # 显示统计信息
    echo "📊 统计信息:"
    echo "  待提交: ${#pending_tasks[@]}"
    echo "  运行中: $running_jobs"
    echo "  已完成: ${#completed_tasks[@]}"
    echo "  失败: $(echo "${job_status[@]}" | tr ' ' '\n' | grep -c "FAILED")"
    echo ""
    
    sleep $monitor_interval
done

# 显示最终结果
print_success "任务执行完成！"
echo "📊 最终统计:"
echo "  总任务数: ${#tasks[@]}"
echo "  成功完成: ${#completed_tasks[@]}"
echo "  失败任务: $(echo "${job_status[@]}" | tr ' ' '\n' | grep -c "FAILED")"

# 显示失败的任务
failed_tasks=()
for task_id in "${!job_status[@]}"; do
    if [[ "${job_status[$task_id]}" == "FAILED" ]]; then
        failed_tasks+=("${job_params[$task_id]}")
    fi
done

if [ ${#failed_tasks[@]} -gt 0 ]; then
    print_warning "失败的任务:"
    for task in "${failed_tasks[@]}"; do
        echo "  - $task"
    done
fi
