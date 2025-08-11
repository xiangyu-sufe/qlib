import torch
import torch.nn as nn
from typing import Dict, Optional

class MIGALoss(nn.Module):  
    """
    Combined loss function with IC-based expert loss and router balance loss
    """
    def __init__(self, omega: float = 2e-3, epsilon: float = 1.0):
        super().__init__()
        self.omega = omega  # Router loss weight
        self.epsilon = epsilon  # Expert loss weight

    def pairwise_max_margin_loss(self, scores, labels, margin=1.0):
        # scores, labels: [N]
        scores = scores.view(-1, 1)  # [N, 1]
        labels = labels.view(-1, 1)  # [N, 1]
        # 构造成对的差
        score_diff = scores - scores.t()  # [N, N]
        label_diff = labels - labels.t()  # [N, N]
        # 只保留 y_i > y_j 的对
        mask = (label_diff > 0)
        # 计算 pairwise hinge loss
        loss_mat = torch.relu(margin - score_diff)
        # 只统计有效对
        loss = loss_mat[mask]
        if loss.numel() == 0:
            return torch.tensor(0.0, device=scores.device)
        return loss.mean()

    def information_coefficient_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative information coefficient as loss
        """
        # Flatten predictions and targets
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()
        
        # Calculate correlation coefficient (IC)
        pred_mean = torch.mean(pred_flat)
        target_mean = torch.mean(target_flat)
        
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        
        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2))
        
        # Add small epsilon to avoid division by zero
        ic = numerator / (denominator + 1e-8)
        
        # Return negative IC as loss (we want to maximize correlation)
        return -ic
        
    def router_balance_loss(self, hidden_representations: torch.Tensor) -> torch.Tensor:
        """
        Calculate router balance loss to prevent routing collapse
        """
        # Calculate distance from mean
        mean_hidden = torch.mean(hidden_representations, dim=1, keepdim=True)
        distance = torch.mean((hidden_representations - mean_hidden) ** 2)
        return distance
        
    def router_balance_loss_2(self, prob: torch.Tensor)->torch.Tensor:
        """
        Calculate router balance loss to prevent routing collapse
        """
        return torch.sum(prob ** 2) * len(prob)    
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                prob: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate combined MIGA loss
        """
        expert_loss = self.information_coefficient_loss(predictions, targets)
        router_loss = self.router_balance_loss_2(prob)
        
        total_loss = self.omega * router_loss + self.epsilon * expert_loss
        
        return {
            'total_loss': total_loss,
            'expert_loss': expert_loss * self.epsilon,
            'router_loss': router_loss * self.omega,
        }