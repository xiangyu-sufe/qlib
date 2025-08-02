#!/bin/bash

# 配置参数
max_gpu_jobs=3              # 最大GPU任务数
submit_interval=30           # 提交检查间隔（秒）
monitor_interval=60          # 监控状态间隔（秒）
command_file="bsub_template.txt"

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
    echo "  lr_sigma    - 学习率和sigma的笛卡尔积"
    echo "  lr_only     - 仅学习率参数"
    echo "  custom      - 自定义参数组合"
    echo ""
    echo "示例:"
    echo "  $0 lr_sigma \"0.001,0.005,0.01\" \"0.1,0.5,1,3.03\""
    echo "  $0 lr_only \"0.001,0.005,0.01,0.05,0.1\""
    echo "  $0 custom \"0.001:0.1,0.005:0.5,0.01:1.0\""
    echo ""
    echo "配置:"
    echo "  最大GPU任务数: $max_gpu_jobs"
    echo "  提交检查间隔: ${submit_interval}秒"
    echo "  监控状态间隔: ${monitor_interval}秒"
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
    "lr_sigma")
        if [ "$#" -ne 2 ]; then
            print_error "lr_sigma模式需要2个参数：学习率列表和sigma列表"
            exit 1
        fi
        IFS=',' read -ra lrs <<< "$1"
        IFS=',' read -ra sigmas <<< "$2"
        
        # 生成笛卡尔积
        tasks=()
        for lr in "${lrs[@]}"; do
            for sigma in "${sigmas[@]}"; do
                tasks+=("lr:$lr,sigma:$sigma")
            done
        done
        ;;
        
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

# 检查模板文件
if [ ! -f "$command_file" ]; then
    print_error "找不到命令模板文件 $command_file"
    exit 1
fi

# 读取模板文件
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

# 初始化状态跟踪
declare -A job_ids
declare -A job_status
declare -A job_params
pending_tasks=("${tasks[@]}")
submitted_tasks=()
completed_tasks=()

print_info "📋 总任务数: ${#tasks[@]}"
print_info "📋 最大GPU任务数: $max_gpu_jobs"

# 提交任务的函数
submit_task() {
    local task="$1"
    local task_id="$2"
    
    for template in "${templates[@]}"; do
        # 替换模板中的占位符
        cmd="$template"
        
        # 根据任务类型替换参数
        if [[ "$task" =~ lr:([^,]+) ]]; then
            lr="${BASH_REMATCH[1]}"
            cmd="${cmd//__LR__/$lr}"
        fi
        
        if [[ "$task" =~ sigma:([^,]+) ]]; then
            sigma="${BASH_REMATCH[1]}"
            cmd="${cmd//__SIGMA__/$sigma}"
        fi
        
        # 替换其他可能的占位符
        cmd="${cmd//__TASK_ID__/$task_id}"
        
        print_info "🚀 提交任务: $task"
        print_info "执行命令: $cmd"
        
        job_output=$(eval "$cmd" 2>&1)
        echo "$job_output"
        
        if [[ "$job_output" =~ \<([0-9]+)\> ]]; then
            job_id="${BASH_REMATCH[1]}"
            job_ids["$task_id"]="$job_id"
            job_status["$task_id"]="RUNNING"
            job_params["$task_id"]="$task"
            print_success "提交成功: $task (job_id=$job_id)"
            return 0
        else
            print_error "提交失败: $task"
            job_status["$task_id"]="FAILED"
            return 1
        fi
    done
}

# 检查任务状态的函数
check_job_status() {
    local task_id="$1"
    local job_id="${job_ids[$task_id]}"
    
    if [[ -z "$job_id" ]]; then
        return 1
    fi
    
    bjobs_out=$(bjobs "$job_id" 2>&1)
    
    if echo "$bjobs_out" | grep -q "not found"; then
        print_warning "任务 $task_id (job_id=$job_id) 已完成"
        job_status["$task_id"]="FINISHED"
        completed_tasks+=("$task_id")
        return 0
    else
        local state=$(echo "$bjobs_out" | awk 'NR==2 {print $3}')
        print_info "任务 $task_id 状态: $state"
        job_status["$task_id"]="$state"
        return 1
    fi
}

# 获取当前运行的任务数
get_running_jobs() {
    bjobs -u "$USER" 2>/dev/null | grep RUN | wc -l
}

# 主循环
print_info "开始任务提交和监控..."

while true; do
    # 检查已完成的任务
    for task_id in "${!job_ids[@]}"; do
        if [[ "${job_status[$task_id]}" != "FINISHED" && "${job_status[$task_id]}" != "FAILED" ]]; then
            check_job_status "$task_id"
        fi
    done
    
    # 获取当前运行的任务数
    running_jobs=$(get_running_jobs)
    print_info "当前运行任务数: $running_jobs/$max_gpu_jobs"
    
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