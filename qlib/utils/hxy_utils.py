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

def coverage_metric(y_pred, data):
    
    y_true = data.get_label()
    coverage = (y_true <= y_pred).astype(float).mean()
    return 'coverage', coverage, True

def get_label(dataset, segment = "test"):
    """
    获取与预测结果对齐的标签 DataFrame
    """
    handler = dataset.handler
    start, end = dataset.segments[segment]
    
    label_data = handler._infer.loc[slice(start, end), "label"]
        
    return label_data