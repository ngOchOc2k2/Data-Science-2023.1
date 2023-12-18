import torch
import torch.nn as nn
import torch.nn.functional as F



class CombineLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(CombineLoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, softmax_positive, label_positive, softmax_negative, label_negative, positive_similarities, negative_similarities, device=None):

        losses = []
        batch_size = positive_similarities.size(0)
        
        for i in range(batch_size):
            # Tính softmax loss
            logits = torch.cat([positive_similarities[i].unsqueeze(0), negative_similarities[i]], dim=0)
            softmax_logits = F.log_softmax(logits / self.temperature, dim=0)

            # Loss chỉ tính cho positive
            loss_i = -softmax_logits[0].mean()
            losses.append(loss_i)

        # Chuyển danh sách losses thành tensor
        loss_tensor = torch.stack(losses)

        # Tính trung bình loss trên toàn bộ batch
        loss = loss_tensor.mean()
        
        return loss

