#!/bin/bash

# 配置参数
max_gpu_jobs=4              # 最大GPU任务数
submit_interval=180          # 提交检查间隔（秒）
monitor_interval=60          # 监控状态间隔（秒）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <参数类型> <参数列表>"
    echo ""
    echo "参数类型:"
    echo "  lr_only     - 仅学习率参数"
    echo "  lr_gpu      - 学习率列表和可用GPU ID列表（一一对应）"
    echo "  custom      - 自定义参数组合"
    echo ""
    echo "示例:"
    echo "  $0 lr_only \"0.001,0.005,0.01,0.05,0.1\""
    echo "  $0 lr_gpu \"0.001,0.005,0.01\" \"0,1,2\""
    echo "  $0 custom \"0.001:0,0.005:1,0.01:2\""
    echo ""
    echo "配置:"
    echo "  最大GPU任务数: $max_gpu_jobs"
    echo "  提交检查间隔: ${submit_interval}秒"
    echo "  监控状态间隔: ${monitor_interval}秒"
}

# 显示GPU信息的函数
show_gpu_info() {
    print_info "系统GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | while IFS=, read -r index name total used util; do
        # 清理名称中的空格
        name=$(echo "$name" | sed 's/^ *//;s/ *$//')
        echo "  GPU $index: $name, 内存: ${used}MB/${total}MB, 利用率: $util%"
    done
    echo ""
    
    # 显示可用性检查结果
    print_info "GPU可用性检查:"
    for gpu in 0 1 2 3; do
        if check_gpu_availability "$gpu"; then
            print_success "GPU $gpu: 可用"
        else
            print_warning "GPU $gpu: 不可用"
        fi
    done
    echo ""
}

# 检查参数
if [ "$#" -lt 2 ]; then
    print_error "错误：参数不足"
    show_help
    exit 1
fi

# 显示GPU信息
show_gpu_info

param_type="$1"
shift

# 根据参数类型解析参数
case "$param_type" in
    "lr_only")
        if [ "$#" -ne 1 ]; then
            print_error "lr_only模式需要1个参数：学习率列表"
            exit 1
        fi
        IFS=',' read -ra lrs <<< "$1"
        
        # 生成任务列表
        tasks=()
        for lr in "${lrs[@]}"; do
            tasks+=("lr:$lr")
        done
        ;;
        
    "lr_gpu")
        if [ "$#" -ne 2 ]; then
            print_error "lr_gpu模式需要2个参数：学习率列表和可用GPU ID列表"
            exit 1
        fi
        IFS=',' read -ra lrs <<< "$1"
        IFS=',' read -ra available_gpus <<< "$2"
        
        # 生成任务列表，GPU将在提交时动态分配
        tasks=()
        for lr in "${lrs[@]}"; do
            tasks+=("lr:$lr")
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

# 检查GPU可用性的函数
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

# 获取可用GPU的函数
get_available_gpu() {
    if [[ "$param_type" == "lr_gpu" ]]; then
        for gpu in "${available_gpus[@]}"; do
            if [[ "${gpu_usage[$gpu]}" == "FREE" ]] && check_gpu_availability "$gpu"; then
                echo "$gpu"
                return 0
            fi
        done
        echo ""  # 没有可用GPU
    else
        # 对于lr_only模式，检查GPU 0是否可用，如果不可用则尝试其他GPU
        if check_gpu_availability "0"; then
            echo "0"
        elif check_gpu_availability "1"; then
            echo "1"
        elif check_gpu_availability "2"; then
            echo "2"
        elif check_gpu_availability "3"; then
            echo "3"
        else
            echo ""
        fi
    fi
}

# 初始化GPU使用情况
if [[ "$param_type" == "lr_gpu" ]]; then
    print_info "检查GPU可用性..."
    available_gpus_checked=()
    for gpu in "${available_gpus[@]}"; do
        if check_gpu_availability "$gpu"; then
            gpu_usage["$gpu"]="FREE"
            available_gpus_checked+=("$gpu")
            print_success "GPU $gpu 可用"
        else
            print_warning "GPU $gpu 不可用，跳过"
        fi
    done
    available_gpus=("${available_gpus_checked[@]}")
    
    if [ ${#available_gpus[@]} -eq 0 ]; then
        print_error "没有可用的GPU！"
        exit 1
    fi
fi

print_info "📋 总任务数: ${#tasks[@]}"
print_info "📋 最大GPU任务数: $max_gpu_jobs"
if [[ "$param_type" == "lr_gpu" ]]; then
    print_info "📋 可用GPU: ${available_gpus[*]}"
fi

# 提交任务的函数
submit_task() {
    local task="$1"
    local task_id="$2"
    
    # 解析任务参数
    local lr=""
    
    if [[ "$task" =~ lr:([^,]+) ]]; then
        lr="${BASH_REMATCH[1]}"
    fi
    
    # 动态分配GPU
    local gpu=$(get_available_gpu)
    if [[ -z "$gpu" ]]; then
        print_error "没有可用的GPU"
        return 1
    fi
    
    # 标记GPU为使用中
    if [[ "$param_type" == "lr_gpu" ]]; then
        gpu_usage["$gpu"]="BUSY"
    fi
    
    # 根据学习率生成save_path，避免特殊字符
    local save_path="model_lr${lr//./_}"
    
    # 构建命令 - 修复CUDA_VISIBLE_DEVICES设置
    local cmd="CUDA_VISIBLE_DEVICES=$gpu python workflow_rank.py --lr $lr --gpu $gpu --save_path $save_path"
    
    print_info "🚀 提交任务: $task"
    print_info "分配GPU: $gpu"
    print_info "CUDA_VISIBLE_DEVICES: $gpu"
    print_info "执行命令: $cmd"
    print_info "保存路径: $save_path"
    
    # 在后台运行任务
    eval "$cmd" > "logs/grurank_lr${lr}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    local pid=$!
    
    job_pids["$task_id"]="$pid"
    job_status["$task_id"]="RUNNING"
    job_params["$task_id"]="$task"
    job_params["${task_id}_gpu"]="$gpu"
    print_success "提交成功: $task (pid=$pid, gpu=$gpu)"
    return 0
}

# 检查任务状态的函数
check_job_status() {
    local task_id="$1"
    local pid="${job_pids[$task_id]}"
    
    if [[ -z "$pid" ]]; then
        return 1
    fi
    
    # 检查进程是否还在运行
    if kill -0 "$pid" 2>/dev/null; then
        print_info "任务 $task_id (pid=$pid) 正在运行"
        return 1
    else
        print_warning "任务 $task_id (pid=$pid) 已完成"
        job_status["$task_id"]="FINISHED"
        completed_tasks+=("$task_id")
        
        # 释放GPU资源
        local gpu="${job_params[${task_id}_gpu]}"
        if [[ -n "$gpu" && "$param_type" == "lr_gpu" ]]; then
            gpu_usage["$gpu"]="FREE"
            print_info "释放GPU $gpu"
        fi
        
        return 0
    fi
}

# 获取当前运行的任务数
get_running_jobs() {
    local count=0
    for task_id in "${!job_pids[@]}"; do
        local pid="${job_pids[$task_id]}"
        if kill -0 "$pid" 2>/dev/null; then
            ((count++))
        fi
    done
    echo "$count"
}

# 创建日志目录
mkdir -p logs

# 主循环
print_info "开始任务提交和监控..."

while true; do
    # 检查已完成的任务
    for task_id in "${!job_pids[@]}"; do
        if [[ "${job_status[$task_id]}" != "FINISHED" ]]; then
            check_job_status "$task_id"
        fi
    done
    
    # 获取当前运行的任务数
    running_jobs=$(get_running_jobs)
    print_info "当前运行任务数: $running_jobs/$max_gpu_jobs"
    
    # 显示GPU使用情况
    if [[ "$param_type" == "lr_gpu" ]]; then
        echo "📊 GPU使用情况:"
        for gpu in "${available_gpus[@]}"; do
            local status="${gpu_usage[$gpu]}"
            if [[ "$status" == "FREE" ]]; then
                echo "  GPU $gpu: 🟢 空闲"
            else
                echo "  GPU $gpu: 🔴 使用中"
            fi
        done
    fi
    
    # 如果有待提交的任务且未达到最大GPU数，则提交新任务
    if [ ${#pending_tasks[@]} -gt 0 ] && [ "$running_jobs" -lt "$max_gpu_jobs" ]; then
        task="${pending_tasks[0]}"
        task_id="task_$(date +%s)_$RANDOM"
        
        if submit_task "$task" "$task_id"; then
            submitted_tasks+=("$task_id")
            pending_tasks=("${pending_tasks[@]:1}")  # 移除已提交的任务
            print_info "剩余待提交任务: ${#pending_tasks[@]}"
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
echo ""
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

# 显示日志文件
echo ""
print_info "日志文件:"
ls -la logs/grurank_*.log 2>/dev/null || echo "  暂无日志文件" 