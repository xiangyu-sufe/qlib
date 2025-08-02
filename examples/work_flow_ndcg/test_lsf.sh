#!/bin/bash

echo "=== LSF 功能测试 ==="

# 测试 bjobs 命令
echo "1. 测试 bjobs 命令:"
if command -v bjobs >/dev/null 2>&1; then
    echo "✅ bjobs 命令可用"
    bjobs -u "$USER" 2>/dev/null | head -5
else
    echo "❌ bjobs 命令不可用"
fi

echo ""

# 测试 bsub 命令
echo "2. 测试 bsub 命令:"
if command -v bsub >/dev/null 2>&1; then
    echo "✅ bsub 命令可用"
    
    # 创建一个简单的测试作业
    test_script=$(mktemp)
    cat > "$test_script" << 'EOF'
#BSUB -J test_job
#BSUB -q gpu
#BSUB -n 1
#BSUB -o test_output.out
#BSUB -e test_error.err

echo "Hello from LSF job"
date
hostname
EOF

    echo "提交测试作业..."
    job_output=$(bsub < "$test_script" 2>&1)
    echo "$job_output"
    
    if [[ "$job_output" =~ \<([0-9]+)\> ]]; then
        job_id="${BASH_REMATCH[1]}"
        echo "✅ 测试作业提交成功，job_id=$job_id"
        
        # 等待几秒钟然后检查状态
        sleep 3
        echo "检查作业状态:"
        bjobs "$job_id" 2>&1
        
        # 清理测试作业
        bkill "$job_id" 2>/dev/null
    else
        echo "❌ 测试作业提交失败"
    fi
    
    rm -f "$test_script"
else
    echo "❌ bsub 命令不可用"
fi

echo ""

# 测试工作目录
echo "3. 测试工作目录:"
echo "当前目录: $(pwd)"
echo "目录内容:"
ls -la | head -10

echo ""

# 测试Python环境
echo "4. 测试Python环境:"
if command -v python >/dev/null 2>&1; then
    echo "✅ Python 可用: $(which python)"
    python --version
else
    echo "❌ Python 不可用"
fi

echo ""

# 测试workflow_ndcg.py
echo "5. 测试workflow_ndcg.py:"
if [ -f "workflow_ndcg.py" ]; then
    echo "✅ workflow_ndcg.py 文件存在"
    echo "文件大小: $(ls -lh workflow_ndcg.py | awk '{print $5}')"
else
    echo "❌ workflow_ndcg.py 文件不存在"
fi

echo ""
echo "=== 测试完成 ===" 