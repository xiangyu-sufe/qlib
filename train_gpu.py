import torch
import time

def get_gpu_available_memory(gpu_id):
    """获取指定GPU的实际可用显存，单位为MB"""
    try:
        torch.cuda.set_device(gpu_id)
        
        # 获取更详细的显存信息
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**2)
        allocated_memory = torch.cuda.memory_allocated(gpu_id) / (1024**2)
        cached_memory = torch.cuda.memory_reserved(gpu_id) / (1024**2)
        
        # 计算实际可用显存
        # 可用显存 = 总显存 - 已分配显存 - 缓存显存
        available_memory = total_memory - allocated_memory - cached_memory
        
        # 打印详细信息
        print(f"GPU {gpu_id} 显存详情:")
        print(f"  总显存: {total_memory:.2f} MB")
        print(f"  已分配显存: {allocated_memory:.2f} MB")
        print(f"  缓存显存: {cached_memory:.2f} MB")
        print(f"  实际可用显存: {available_memory:.2f} MB")
        
        return available_memory
    except Exception as e:
        print(f"获取GPU {gpu_id} 信息失败: {e}")
        return 0

def allocate_gpu_memory(gpu_id, safety_factor=0.8):
    """
    在指定GPU上分配内存
    safety_factor: 安全系数，实际分配的显存占可用显存的比例
    """
    try:
        torch.cuda.set_device(gpu_id)
        
        # 获取实际可用显存
        available_memory = get_gpu_available_memory(gpu_id)
        if available_memory <= 0:
            print(f"GPU {gpu_id} 没有可用显存")
            return None
        
        # 计算要分配的显存大小（考虑安全系数）
        alloc_memory = available_memory * safety_factor
        
        print(f"GPU {gpu_id}: 计划分配 {alloc_memory:.2f} MB ({safety_factor*100}%) 的可用显存")
        
        # 每个float32元素占用4字节
        # 转换为字节再计算可分配的元素数量
        alloc_memory_bytes = int(alloc_memory * 1024**2)
        num_elements = alloc_memory_bytes // 4
        
        if num_elements <= 0:
            print(f"GPU {gpu_id} 没有足够的显存可分配")
            return None
        
        # 为了避免创建过大的一维张量导致的问题，创建二维张量
        dim1 = int(num_elements**0.5)
        dim2 = num_elements // dim1
        
        # 确保有足够的元素
        if dim1 * dim2 < num_elements * 0.95:  # 允许5%的误差
            dim2 += 1
        
        # 尝试分配张量
        tensor = torch.randn(dim1, dim2, device=f'cuda:{gpu_id}')
        
        # 验证实际分配
        post_allocated = torch.cuda.memory_allocated(gpu_id) / (1024**2)
        print(f"GPU {gpu_id}: 成功分配，当前已分配显存: {post_allocated:.2f} MB")
        
        return tensor
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"GPU {gpu_id} 分配失败: 内存不足。尝试降低安全系数。")
            # 尝试使用更小的比例重新分配
            return allocate_gpu_memory(gpu_id, safety_factor * 0.8)
        else:
            print(f"GPU {gpu_id} 分配失败: {e}")
            return None
    except Exception as e:
        print(f"GPU {gpu_id} 分配过程中出错: {e}")
        return None

def main():
    if not torch.cuda.is_available():
        print("没有检测到可用的GPU (CUDA不可用)")
        return
        
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个可用的GPU")
    
    # 存储每个GPU上分配的张量
    allocated_tensors = []
    
    # 为每个GPU分配内存
    for gpu_id in range(num_gpus):
        print(f"\n处理GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        tensor = allocate_gpu_memory(gpu_id, safety_factor=0.7)  # 初始安全系数设为70%
        if tensor is not None:
            allocated_tensors.append((gpu_id, tensor))
    
    if allocated_tensors:
        print("\n所有GPU内存分配完成。")
        print(f"成功在 {len(allocated_tensors)} 个GPU上分配了内存")
        print("按Ctrl+C停止程序释放内存。")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n程序被中断，释放内存...")
    else:
        print("\n未能在任何GPU上分配内存")

if __name__ == "__main__":
    main()
