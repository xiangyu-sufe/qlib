from collections import defaultdict

def compute_grad_norm(model):
    """
        计算模型整体的梯度范数
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    return total_norm

def compute_layerwise_grad_norm(model):
    """
        计算每层模型的参数
    """
    layer_grad_norms = defaultdict(list)

    for name, param in model.named_parameters():
        if param.grad is not None:
            norm = param.grad.detach().data.norm(2).item()
            layer_grad_norms[name].append(norm)
            
    return layer_grad_norms