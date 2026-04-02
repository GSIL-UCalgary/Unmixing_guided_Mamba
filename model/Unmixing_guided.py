import torch
import torch.nn as nn
from torch.nn import init
import pdb
import numpy as np
import math
import pdb
import warnings
import matplotlib.pyplot as plt
from mamba_ssm import Mamba
import torch.nn.functional as F
import numpy as np
import math


def batched_index_select(input, dim, index):
    """Select indices along specified dimension in batched manner"""
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms_x + self.eps))

        
class AttentionSparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_clusters=100,
                 sparsity_ratio=0.5, use_attention=True, num_heads=4, 
                 selection_mode='hybrid'):
        """
        Args:
            selection_mode: 'attention', 'cluster', or 'hybrid'
                - 'attention': pure attention-based selection
                - 'cluster': original cluster-based selection
                - 'hybrid': combine both for diversity + importance
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.num_clusters = num_clusters
        self.sparsity_ratio = sparsity_ratio
        self.use_attention = use_attention
        self.selection_mode = selection_mode

        # self.norm = RMSNorm(dim)
        # self.proj_in = nn.Linear(dim, self.expanded_dim)

        self.norm2 = RMSNorm(dim)
        self.proj_in1 = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        # self.proj_out = nn.Sequential(
        #     nn.GroupNorm(4, dim),
        #     nn.GELU()
        # )

        # self.proj_out_blocks = nn.ModuleList([
        #     nn.Linear(self.expanded_dim, dim)
        #     for _ in range(num_clusters)
        # ])

        # self.mu_proj = nn.Linear(self.expanded_dim, self.expanded_dim)
        # self.logvar_proj = nn.Linear(self.expanded_dim, 1)

        # self.conv_blocks = nn.ModuleList([
        #     nn.Conv1d(
        #         in_channels=self.expanded_dim,
        #         out_channels=self.expanded_dim,
        #         kernel_size=d_conv,
        #         padding=d_conv - 1,
        #         groups=self.expanded_dim,
        #         bias=False
        #     ) for _ in range(num_clusters)
        # ])
        # # Mamba SSM
        self.unselectedtoken_mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=dim,
                d_state=16,
                d_conv=4,
                expand=2,
            ) for _ in range(num_clusters)
        ])

    def select_pixels(self, x_proj, cluster_assignments, k_total, per_cluster, largest):
        """Select top-k pixels based on selection mode"""
        B, L, D = x_proj.shape
        CC = cluster_assignments.shape[1]
        cluster_assignments = cluster_assignments.permute(0, 2, 3, 1).reshape(B, L, CC)  # [B, K, L]
        if self.selection_mode == 'cluster':
            # Original cluster-based selection
            #k_per_cluster = max(1, int(k_total / self.num_clusters * self.sparsity_ratio))
            selected_indices = []
            for cluster_idx in range(len(per_cluster)):
                cluster_scores = cluster_assignments[:, :, cluster_idx]# 141 141 100
                # cluster_scores = cluster_scores + attention_scores
                # 13 * 128
                # 13 * 2
                topk_scores, topk_indices = torch.topk(cluster_scores,k=per_cluster[cluster_idx],dim=-1,largest=largest,sorted=True)
                selected_indices.append(topk_indices)
        return selected_indices
    
    def forward(self, x, per_cluster, cluster_assignments, return_cluster_assignments=False, largest=True):
        B, H, W, C = x.shape
        x_flat = x.view(1, -1, C)
        B, L, C = x_flat.shape
        k_total = max(1, int(L * self.sparsity_ratio))

        output_sparse = torch.zeros(B, L, C, device=x.device)
        selected_indices = self.select_pixels(x_flat, cluster_assignments, k_total, per_cluster, largest)
        
        # selected_indices: list, each element like [1, k_i] in your current code style

        all_selected = torch.cat([selected_indices[i][0] for i in range(len(selected_indices))], dim=0)
        unique_selected = torch.unique(all_selected)

        selected_mask = torch.zeros(L, dtype=torch.bool, device=x.device)
        selected_mask[unique_selected] = True

        output_unsel = torch.zeros(B, L, C, device=x.device, dtype=x.dtype)
        unselected_idx = (~selected_mask).nonzero(as_tuple=False).squeeze(1)   # [K_unsel]

        if unselected_idx.numel() > 0:
            # random subsample from unselected pixels
            sample_ratio = 0.3   # or 0.7
            max_unsel_tokens = 10000   # optional cap

            num_unsel = unselected_idx.numel()
            num_sample = max(1, int(num_unsel * sample_ratio))
            num_sample = min(num_sample, max_unsel_tokens)

            rand_perm = torch.randperm(num_unsel, device=x.device)[:num_sample]
            sampled_unselected_idx = unselected_idx[rand_perm]   # [K_sample]

            x_unsel = batched_index_select(
                x_flat, 1, sampled_unselected_idx.unsqueeze(0)
            )   # [1, K_sample, C]

            x_unsel_processed = self.unselectedtoken_mamba(x_unsel)   # [1, K_sample, C]

            output_unsel.scatter_add_(
                1,
                sampled_unselected_idx.unsqueeze(0).unsqueeze(-1).expand(-1, -1, C),
                x_unsel_processed
            )

        Nc = self.num_clusters
        batched_sparse = []

        # Gather all cluster features in one tensor
        for i in range(len(selected_indices)):
            # [B, k_per_cluster, C]
            xi = batched_index_select(x_flat, 1, selected_indices[i][0].unsqueeze(dim=0))
            batched_sparse.append(xi)

        x_processed = []
        x_processed.extend(
            self.mamba_blocks[i](batched_sparse[i])
            for i in range(len(selected_indices))
        )
        # Scatter results back into original positions
        for i in range(len(selected_indices)):
            output_sparse.scatter_add_(1,selected_indices[i][0].unsqueeze(dim=0).unsqueeze(-1).expand(-1, -1, C),x_processed[i])


        # 2) cluster-level summaries
        cluster_ctx = []
        for i in range(len(x_processed)):
            ctx_i = x_processed[i].mean(dim=1)   # [B, C]
            cluster_ctx.append(ctx_i)
        cluster_ctx = torch.stack(cluster_ctx, dim=1)   # [B, R, C]

        # 3) abundance broadcast to all pixels
        # cluster_assignments: [B, R, H, W]
        # dense_ctx = torch.einsum('brhw,brc->bhwc', cluster_assignments, cluster_ctx)
        # dense_ctx = dense_ctx.view(B, L, C)

        output = output_sparse + output_unsel + x_flat
        # output = output.reshape(B, H, W, C)
        # x_recon = output.permute(0, 3, 1, 2).contiguous()
        # output = self.proj_in1(self.norm2(output))
        # output = self.global_mamba(output)
        # output = self.proj_out(output)
        if return_cluster_assignments:
            #cluster_loss = self.compute_cluster_loss(x_proj, cluster_assignments)
            #cluster_loss = self.cluster_head.contrastive_center_loss()
            return output
        else:
            return output
        
class SpeMamba(nn.Module):
    def __init__(self,channels, num_clusters, token_num=8, use_residual=True, group_num=4):
        super(SpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual
        self.num_clusters = num_clusters

        self.group_channel_num = math.ceil(channels/token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba( # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=self.group_channel_num,  # Model dimension d_model
                            d_state=16,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                            )
        
        
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=self.group_channel_num,
                d_state=16,
                d_conv=4,
                expand=2,
            ) for _ in range(num_clusters)
        ])
        
        self.unselected_mamba =  Mamba(
                d_model=self.group_channel_num,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        self.proj = nn.Sequential(
            nn.BatchNorm2d(self.channel_num),
            nn.GELU()
        )

    def padding_feature(self,x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x
    def select_pixels(self, feat, cluster_assignments, per_cluster, largest):
        """Select top-k pixels based on selection mode"""
        B, L, D = feat.shape
        CC = cluster_assignments.shape[1]
        cluster_assignments = cluster_assignments.permute(0, 2, 3, 1).reshape(B, L, CC)  # [B, K, L]
        # Original cluster-based selection
        #k_per_cluster = max(1, int(k_total / self.num_clusters * self.sparsity_ratio))
        selected_indices = []
        for cluster_idx in range(len(per_cluster)):
            cluster_scores = cluster_assignments[:, :, cluster_idx]# 141 141 100
            # cluster_scores = cluster_scores + attention_scores
            # 13 * 128
            # 13 * 2
            topk_scores, topk_indices = torch.topk(cluster_scores,k=per_cluster[cluster_idx],dim=-1,largest=largest,sorted=True)
            selected_indices.append(topk_indices)
        return selected_indices
    
    def forward(self,x, per_cluster, abundance, largest):
        x_pad = self.padding_feature(x)
        x_pad = x_pad.permute(0, 2, 3, 1).contiguous()
        B, H, W, C_pad = x_pad.shape
        L = H*W
        x_flat = x_pad.reshape(B, L, C_pad)
        selected_indices = self.select_pixels(x_pad.reshape(B, -1, C_pad), abundance, per_cluster, largest)
        
        # --------------------------------------------------
        # 3.5) build true unselected set from unique selected pixels
        # --------------------------------------------------
        all_selected = torch.cat([selected_indices[i][0] for i in range(len(selected_indices))], dim=0)
        unique_selected = torch.unique(all_selected)

        selected_mask = torch.zeros(L, dtype=torch.bool, device=x.device)
        selected_mask[unique_selected] = True
        unselected_idx = (~selected_mask).nonzero(as_tuple=False).squeeze(1)   # [K_unsel]

        x_processed = []
        x_processed.extend(
            self.mamba_blocks[i](batched_index_select(x_pad.reshape(B, -1, C_pad), 1, 
                                                      selected_indices[i][0].unsqueeze(dim=0)).view(B * len(selected_indices[i][0]), self.token_num, self.group_channel_num))
            for i in range(len(selected_indices))
        )
        output_sparse = torch.zeros(B, H * W, C_pad, device=x.device)
        for i in range(len(selected_indices)):
            output_sparse.scatter_add_(1,selected_indices[i][0].unsqueeze(dim=0).unsqueeze(-1).expand(-1, -1, C_pad),x_processed[i].view(B, -1, C_pad))
        output_sparse = output_sparse.view(B, H, W, C_pad).permute(0, 3, 1, 2).contiguous()

        if unselected_idx.numel() > 0:
            output_unsel = torch.zeros(B, L, C_pad, device=x.device, dtype=x.dtype)

            # ---------------------------------------
            # random subsample from unselected pixels
            # ---------------------------------------
            sample_ratio = 0.3  # or 0.7
            num_unsel = unselected_idx.numel()
            num_sample = max(1, int(num_unsel * sample_ratio))
            max_unsel_tokens = 10000
            num_sample = min(num_sample, max_unsel_tokens)

            rand_perm = torch.randperm(num_unsel, device=x.device)[:num_sample]
            sampled_unselected_idx = unselected_idx[rand_perm]   # [K_sample]

            x_unsel = batched_index_select(
                x_flat, 1, sampled_unselected_idx.unsqueeze(0)
            )   # [B, K_sample, C_pad]

            x_unsel_processed = self.unselected_mamba(
                x_unsel.view(B * x_unsel.shape[1], self.token_num, self.group_channel_num)
            )
            x_unsel_processed = x_unsel_processed.view(B, -1, C_pad)

            output_unsel.scatter_add_(
                1,
                sampled_unselected_idx.unsqueeze(0).unsqueeze(-1).expand(-1, -1, C_pad),
                x_unsel_processed
            )

            output_unsel = output_unsel.view(B, H, W, C_pad).permute(0, 3, 1, 2).contiguous()
        else:
            output_unsel = torch.zeros(B, C_pad, H, W, device=x.device, dtype=x.dtype)
        # 2) cluster-level summaries
        cluster_ctx = []
        for i in range(len(x_processed)):
            ctx_i = x_processed[i].reshape(-1, 1, C_pad).mean(dim=0)   # [B, C]
            cluster_ctx.append(ctx_i)
        cluster_ctx = torch.stack(cluster_ctx, dim=1)   # [B, R, C]
        # 3) abundance broadcast to all pixels
        # cluster_assignments: [B, R, H, W]
        # dense_ctx = torch.einsum('brhw,brc->bhwc', abundance, cluster_ctx) dense_ctx.permute(0, 3, 1, 2)
        output = output_sparse + output_unsel + x
        # output_flat = output.view(B * H * W, self.token_num, self.group_channel_num)
        # x_flat = self.mamba(output_flat)
        # x_recon = x_flat.view(B, H, W, C_pad)
        # x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        x_proj = self.proj(output)
        if self.use_residual:
            return x_proj + x
        else:
            return x_proj

class SpaMamba(nn.Module):
    def __init__(self, channels, use_residual=True, group_num=4, use_proj=True,
                 num_clusters=19, sparsity_ratio=1.0, use_attention=True,
                 num_heads=4, selection_mode='hybrid'):
        super(SpaMamba, self).__init__()
        self.use_residual = use_residual
        self.use_proj = use_proj

        self.mamba = AttentionSparseDeformableMambaBlock(
            dim=channels,
            num_clusters=num_clusters,
            sparsity_ratio=sparsity_ratio,
            use_attention=use_attention,
            num_heads=num_heads,
            selection_mode=selection_mode
        )
        
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.GELU()
            )

    def forward(self, x, per_cluster, cluster_map, largest):
        x_re = x.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = x_re.shape
        x_recon = self.mamba(x_re, per_cluster, cluster_map, return_cluster_assignments=True, largest=largest)
        x_recon = x_recon.view(B, H, W, C)
        x_recon = x_recon.permute(0, 3, 1, 2).contiguous()
        if self.use_proj:
            x_recon = self.proj(x_recon)
        if self.use_residual:
            return x_recon + x
        else:
            return x_recon

            
class BothMamba(nn.Module):
    def __init__(self, channels, token_num, use_residual, group_num=4, use_att=True,
                 num_clusters=19, sparsity_ratio=1.0, attention_heads=4,
                 selection_mode='hybrid'):
        super(BothMamba, self).__init__()
        self.use_att = use_att
        self.use_residual = use_residual

        if self.use_att:
            self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
            self.softmax = nn.Softmax(dim=0)
            
        self.spa_mamba = SpaMamba(
            channels, 
            use_residual=use_residual, 
            group_num=group_num,
            num_clusters=num_clusters,
            sparsity_ratio=sparsity_ratio,
            use_attention=True,
            num_heads=attention_heads,
            selection_mode=selection_mode
        )
        self.spe_mamba = SpeMamba(channels, num_clusters, token_num=token_num, use_residual=use_residual, group_num=group_num)
        self.conv_x11 = nn.Conv2d(channels, channels, 1)
        

    def forward(self, x, per_cluster, cluster_map, largest):

        spa_x = self.spa_mamba(x, per_cluster, cluster_map, largest)
        # spa_x = self.spa_mamba(spa_x, per_cluster, cluster_map)
        spe_x = self.spe_mamba(x, per_cluster, cluster_map, largest)
        # spe_x = self.spe_mamba(spe_x, per_cluster, cluster_map)
        weights = self.softmax(self.fusion_weights)
        fusion_x = spa_x * weights[0] + spe_x * weights[1]
        # fusion_x = torch.cat((spa_x, spe_x), dim=1)
        fusion_x = self.conv_x11(fusion_x)
        # fusion_x = fusion_x + x        
        return fusion_x + x
    
    
    
class Conv_Classifier(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def __init__(self, input_channels, num_classes, patch_size=7, n_planes=64):
        super(Conv_Classifier, self).__init__()
        self.input_channels = input_channels
        self.n_planes = n_planes
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(input_channels, n_planes, (3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(n_planes, 100, (3, 3), stride=(1, 1), padding=1)
        self.relu = nn.ReLU()

        self.feature_size = self._get_final_flattened_size()
        self.fc1 = nn.Conv2d(100, 100, kernel_size=1, stride=1, padding=0)
        self.fc2 = nn.Conv2d(100, num_classes, kernel_size=1, stride=1, padding=0)
        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            _, c, w, h = x.size()
            return c * w * h

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.fc2(self.relu(self.fc1(x)))
        return x

def get_3d_sp_pos_encoding(B, H, W, device='cuda'):
        x, y = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )
        z = (x + y) / 2

        sin_x = torch.sin(torch.pi * 2* x)      # [H, W]
        cos_y = torch.cos(torch.pi *2* y)      # [H, W]
        lin_z = z                             # [H, W]

        pos = torch.stack([sin_x, cos_y, lin_z], dim=0)  # [3, H, W]
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)     # [B, 3, H, W]
        return pos

def get_3d_ca_pos_encoding(B, H, W, device='cuda'):
        x, y = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing='ij'
        )

        lin_z = torch.sin(torch.pi * 2*x)/2 +  torch.cos(torch.pi *2* y)/2                            # [H, W]

        pos = torch.stack([x, y, lin_z], dim=0)  # [3, H, W]
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)     # [B, 3, H, W]
        return pos

class ChannelLinearAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels+3, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)

    def forward(self, q, kv):
        B, C, H, W = q.shape
        coords = get_3d_ca_pos_encoding(B, H, W, device=q.device)
        q_in = torch.cat([q, coords], dim=1)
        Q = self.q_proj(q_in)  # [B,C,H,W]
        K = self.k_proj(kv).mean(1, keepdim=True)  # [B,1,H,W]
        V = self.v_proj(kv).mean(1, keepdim=True)  # [B,1,H,W]
        
        attn_scores = torch.tanh((Q * K))  # [B,C,1,1]
        out = attn_scores * V  # [B,C,H,W]

        return q + out
    
class SpatialLinearAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels+3, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, q, kv):

        B, C, H, W = q.shape
        coords = get_3d_sp_pos_encoding(B, H, W, device=q.device)
        q_in = torch.cat([q, coords], dim=1)
        
        Q = self.q_proj(q_in)  # [B,C,H,W]
        K = self.k_proj(kv).mean(-1, keepdim=True)  # [B,C,H,1]
        V = self.v_proj(kv).mean(-2, keepdim=True)  # [B,C,H,1]

        attn_scores = torch.tanh((Q * K))  # [B,1,H,W]
        out = attn_scores * V  # [B,C,H,W]

        return q + out

class TemporalAbundanceEMA:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.ema_abu = None   # [K, H, W]
        self.count = 0

    def get_target(self):
        """
        return:
            None, if not initialized
            or bias-corrected EMA abundance: [K, H, W]
        """
        if self.ema_abu is None or self.count == 0:
            return None

        bias_correction = 1.0 - self.alpha ** self.count
        target = self.ema_abu / max(bias_correction, 1e-8)
        return target

    @torch.no_grad()
    def update(self, abu):
        """
        abu: [1, K, H, W] or [B, K, H, W], but here B=1
        """
        abu = abu.detach().squeeze(0)   # -> [K, H, W]

        if self.ema_abu is None:
            self.ema_abu = abu.clone()
        else:
            self.ema_abu = self.alpha * self.ema_abu + (1 - self.alpha) * abu

        self.count += 1
        

def temporal_abundance_loss_with_mask(abu, abu_target, conf_thresh=0.6):
    """
    abu: [1, K, H, W]
    abu_target: [K, H, W]
    """
    abu_target = abu_target.unsqueeze(0)   # [1, K, H, W]

    # confidence from current abundance
    conf = abu.max(dim=1, keepdim=True)[0]   # [1,1,H,W]
    mask = (conf > conf_thresh).float()

    diff = (abu - abu_target.detach()) ** 2
    diff = diff.mean(dim=1, keepdim=True)    # [1,1,H,W]

    loss = (diff * mask).sum() / (mask.sum() + 1e-8)
    return loss
    # """
    # abu: [1, K, H, W]
    # abu_target: [K, H, W]
    # """
    # abu_target = abu_target.unsqueeze(0)   # [1, K, H, W]
    # return F.mse_loss(abu, abu_target.detach())

def smooth_noise_1d(x, kernel_size=9):
    """
    Smooth 1D noise along the spectral dimension.

    Args:
        x: Tensor of shape [N, B]
        kernel_size: odd integer, smoothing window size

    Returns:
        Tensor of shape [N, B]
    """
    if kernel_size <= 1:
        return x

    pad = kernel_size // 2
    x = x.unsqueeze(1)  # [N, 1, B]

    weight = torch.ones(
        1, 1, kernel_size,
        device=x.device,
        dtype=x.dtype
    ) / kernel_size

    x = F.pad(x, (pad, pad), mode='reflect')
    x = F.conv1d(x, weight)
    return x.squeeze(1)  # [N, B]


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = max(0.0, min(float(current), float(rampup_length)))
    phase = 1.0 - current / rampup_length
    return math.exp(-5.0 * phase * phase)

class UTKMamba(nn.Module):
    """
    dual-branch subpixel-guided network for hyperspectral image classification
    """
    def __init__(self, band, dim, sub_num, num_classes):
        super(UTKMamba, self).__init__()
        self.num_classes = num_classes
        self.num_queries_times = 10
        self.num_clusters = sub_num    
        self.bands = band
        self.dim = dim
        # unmixing module
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=band, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.um_ = ChannelLinearAttention(dim)
        self.cls_ = SpatialLinearAttention(dim)
        self.unmix_encoder = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(),
            nn.Conv2d(dim//2, dim//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(),
            nn.Conv2d(dim//4, sub_num, kernel_size=1, stride=1, padding=0)
        )
        self.num_queries = self.num_queries_times * self.num_clusters
        self.query_embed = nn.Embedding(self.num_queries, band)
        self.weights = nn.Parameter(torch.ones((self.num_clusters, self.num_queries_times)))
        self.var_head = nn.Conv2d(sub_num, self.num_queries, kernel_size=1)

        # basic classification backbone module
        self.mamba = BothMamba(
            channels=dim,
            token_num=4,
            use_residual=True,
            group_num=4,
            use_att=True,
            num_clusters=sub_num,
            sparsity_ratio=1,
            attention_heads=4,
            selection_mode='cluster',
        )
        # fusion module
        self.conv = nn.Sequential(
            nn.Conv2d(sub_num, sub_num,  kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(sub_num),
            nn.ReLU()
        )
        self.fc = nn.Conv2d(dim + sub_num, num_classes, kernel_size=3, stride=1, padding=1)
    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.num_classes, self.patch_size, self.patch_size))
            x = self.conv(x)
            _, c, w, h = x.size()
            return c * w * h + self.num_classes

    def init_query_embed_from_vca(self, vca_endm, return_init=True):
        device = self.query_embed.weight.device
        dtype = self.query_embed.weight.dtype

        vca_endm = torch.as_tensor(vca_endm, device=device, dtype=dtype)  # [K, band]

        K, band = vca_endm.shape
        assert K == self.num_clusters
        assert band == self.query_embed.embedding_dim

        num_per_cluster = self.num_queries // self.num_clusters
        assert self.num_queries % self.num_clusters == 0

        init_list = []
        for k in range(self.num_clusters):
            base = vca_endm[k].unsqueeze(0).repeat(num_per_cluster, 1)
            init_list.append(base)

        init_weight = torch.cat(init_list, dim=0)  # [num_queries, band]
        with torch.no_grad():
            self.query_embed.weight.copy_(init_weight)
        if return_init:
            return init_weight
    def forward(self, x, abu_teacher=None, token_beta=0.0):
        B, C, H, W = x.shape
        feat_x = self.patch_embedding(x)
        um_feat = self.um_(feat_x, feat_x)
        cls_feat = self.cls_(feat_x, feat_x)
        abu1 = self.unmix_encoder(um_feat)

        abu = abu1.abs()
        abu = abu / (abu.sum(1, keepdim=True) + 1e-8)

        endm_get, end_var = self.get_endmember()

        W_logits = self.var_head(abu1)          # [B, P*R, H, W]
        W_logits = W_logits.view(
            B,
            self.num_clusters,        # P first
            self.num_queries_times,   # R second
            H, W
        )
        W_logits_ = torch.softmax(W_logits, dim=2)         # over R
        recon_linear = torch.einsum(
            'bprhw,prc->bchw',
            W_logits_ * abu.unsqueeze(2),
            end_var
        )
        # ===== temporal abundance for token selection =====
        if abu_teacher is not None:
            if abu_teacher.dim() == 3:
                abu_teacher = abu_teacher.unsqueeze(0)   # [1,K,H,W]

            abu_teacher = abu_teacher.to(abu.device)

            # mix current abundance and temporal abundance
            abu_for_token = (1.0 - token_beta) * abu + token_beta * abu_teacher

            # normalize again just in case
            abu_for_token = abu_for_token / (abu_for_token.sum(1, keepdim=True) + 1e-8)
        else:
            abu_for_token = abu
            

        per_cluster_num1 = adaptive_tokens_per_cluster_batch(
            abu_for_token,
            lam=0.1,
            min_tokens=100,
            thresh_ratio=0.5,
            mass_weight=0.3,
            area_weight=0.7
        ).cpu().numpy()
        
        # print("per cluster num", per_cluster_num1)

        feature_cls = self.mamba(cls_feat, per_cluster_num1[0], abu_for_token, largest=True)
        feature_abu = self.conv(abu)

        feature_fuse = torch.cat((feature_abu, feature_cls), dim=1)
        output_cls = self.fc(feature_fuse)

        return recon_linear, output_cls, abu, endm_get, end_var, W_logits_
        
    def get_endmember(self):
        query_embed_weight_split = torch.chunk(self.query_embed.weight, self.num_clusters, dim=0)
        query_embed_weight_split = torch.stack(query_embed_weight_split)
        endmember_get = query_embed_weight_split 
        endmember_get1 = torch.mean(endmember_get, dim=1)
        return endmember_get1, endmember_get
        

def show_abu(data):
    import matplotlib.pyplot as plt
    abu = data[0].cpu().numpy()
    fig, axes = plt.subplots(2, 7, figsize=(3*7, 3))
    for j in range(7):
        axes[j].imshow(abu[j, :, :], cmap='jet')
    plt.tight_layout()
    plt.savefig('abu.png', bbox_inches='tight')
    plt.close(fig)
    
def show_images(data, cmap='jet', save_path="abu_vis.png"):
    """
    data: H x W x N
    Automatically adapts grid and margins.
    """
    abu = data[0].cpu().numpy()
    data = abu.transpose(1, 2, 0)
    H, W, N = data.shape

    # adaptive grid
    cols = math.ceil(math.sqrt(N))
    rows = math.ceil(N / cols)

    # adapt figure size to image aspect ratio
    aspect = W / H
    fig_w = cols * 3
    fig_h = rows * 3 / aspect

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    axes = np.array(axes).reshape(-1)

    for i in range(N):
        axes[i].imshow(data[:, :, i], cmap=cmap)
        axes[i].axis("off")

    # hide empty axes
    for i in range(N, len(axes)):
        axes[i].axis("off")

    # adaptive spacing
    margin = 0.02
    wspace = 0.02 if cols > 1 else 0
    hspace = 0.02 if rows > 1 else 0

    plt.subplots_adjust(
        left=margin,
        right=1 - margin,
        top=1 - margin,
        bottom=margin,
        wspace=wspace,
        hspace=hspace
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path
    
    
def show_abu_argmax(data):
    import matplotlib.pyplot as plt
    abu = data[0].detach().cpu().numpy()
    fig, axes = plt.subplots()
    plt.imshow(np.argmax(abu, axis=0), cmap='tab20')
    plt.tight_layout()
    plt.savefig('abu_argmax.png', bbox_inches='tight')
    plt.close(fig)

def show_endmember(data, n=None, m=None, index=0, save_path=None):
    """
    data: [N, L] or [N, L, C]
          N = number of endmembers
          L = spectral length
          C = number of curves per subplot (optional)

    n, m: optional grid size
          if not given, they will be determined automatically
    """

    num_items = data.shape[0]

    # auto grid if not provided
    if n is None or m is None:
        m = int(np.ceil(np.sqrt(num_items)))
        n = int(np.ceil(num_items / m))

    # adaptive figure size
    fig_w = 3.2 * m
    fig_h = 2.6 * n
    fig, axes = plt.subplots(n, m, figsize=(fig_w, fig_h), squeeze=False)

    axes = axes.reshape(-1)

    for i in range(num_items):
        ax = axes[i]

        if data.ndim == 3:
            ax.plot(data[i].T)
        elif data.ndim == 2:
            ax.plot(data[i])
        else:
            raise ValueError("data should have shape [N, L] or [N, L, C]")

        ax.set_title(f"E{i}", fontsize=9)

        # cleaner look
        ax.tick_params(axis='both', labelsize=7)
        ax.margins(x=0.02, y=0.08)

    # hide unused axes
    for i in range(num_items, len(axes)):
        axes[i].axis("off")

    # adaptive spacing
    plt.subplots_adjust(
        left=0.06,
        right=0.98,
        top=0.94,
        bottom=0.08,
        wspace=0.28,
        hspace=0.38
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    return save_path
    

def adaptive_tokens_per_cluster_batch(
    abu,
    lam=0.01,
    min_tokens=100,
    max_tokens=None,
    thresh_ratio=0.5,
    mass_weight=0.7,
    area_weight=0.3,
):
    """
    abu: [B, K, H, W]
    return: [B, K] adaptive token count per cluster
    """
    B, K, H, W = abu.shape

    # soft mass
    mass = abu.sum(dim=(2, 3))  # [B, K]

    # active area
    maxv = abu.amax(dim=(2, 3), keepdim=True)   # [B,K,1,1]
    thresh = thresh_ratio * maxv
    area = (abu > thresh).float().sum(dim=(2, 3))  # [B, K]

    # normalize
    mass_norm = mass / (mass.sum(dim=1, keepdim=True) + 1e-8)
    area_norm = area / (area.sum(dim=1, keepdim=True) + 1e-8)

    # combined score
    score = mass_weight * mass_norm + area_weight * area_norm   # [B, K]

    # adaptive token number
    num_tokens = torch.ceil(lam * H * W * score).long()

    if min_tokens is not None:
        num_tokens = torch.clamp(num_tokens, min=min_tokens)

    if max_tokens is not None:
        num_tokens = torch.clamp(num_tokens, max=max_tokens)

    return num_tokens


def show_W_variants(W, b=0, p=0, save_path=None):
    # W: [B, P, R, H, W]
    W_bp = W[b, p].detach().cpu()   # [R, H, W]
    R, H, W_ = W_bp.shape

    ncols = min(5, R)
    nrows = math.ceil(R / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for r in range(R):
        im = axes[r].imshow(W_bp[r], cmap='viridis')
        axes[r].set_title(f'variant {r}')
        axes[r].axis('off')
        fig.colorbar(im, ax=axes[r], fraction=0.046, pad=0.04)

    for r in range(R, len(axes)):
        axes[r].axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)