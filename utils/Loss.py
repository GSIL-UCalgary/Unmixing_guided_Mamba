import warnings
import torch.nn.functional as F
import torch
import pdb
import torch.nn as nn
class SADLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(SADLoss, self).__init__()
        self.eps = eps

    def forward(self, y_true, y_pred):
        # Flatten spatial dimensions if they exist
        y_true = y_true.view(y_true.shape[0], y_true.shape[1], -1).transpose(1, 2)
        y_pred = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], -1).transpose(1, 2)
        if len(y_pred.shape) > 2:
            y_true = y_true.reshape(-1, y_true.shape[-1])
            y_pred = y_pred.reshape(-1, y_pred.shape[-1])

        # 1. Compute Dot Product
        dot_product = torch.sum(y_true * y_pred, dim=1)
        
        # 2. Compute Norms with epsilon to avoid sqrt(0) and division by 0
        y_true_norm = torch.norm(y_true, dim=1) + self.eps
        y_pred_norm = torch.norm(y_pred, dim=1) + self.eps
        
        # 3. Get Cosine Similarity
        # Use clamp to prevent values like 1.0000001 from crashing acos
        cosine_sim = dot_product / (y_true_norm * y_pred_norm)
        cosine_sim = torch.clamp(cosine_sim, -1.0 + self.eps, 1.0 - self.eps)
        
        # 4. Compute Angle
        angle = torch.acos(cosine_sim)
        return torch.mean(angle)



def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def head_loss(loss_func,logits,label,align_corners=True):
    # seg_logits = resize(
    #     input=logits,
    #     size=label.shape[1:],
    #     mode='bilinear',
    #     align_corners=align_corners)

    loss = loss_func(logits,label)
    return loss