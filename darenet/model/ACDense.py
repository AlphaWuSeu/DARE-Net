import torch
import torch.nn as nn
import torch.nn.functional as F


# ============= MoE Components for DARE-Net ============= #

class TopKGate(nn.Module):
    """
    Top-K Gating mechanism for Mixture of Experts
    Routes input to top-K experts based on learned gating scores
    Optionally uses diagnosis probability to guide routing (Diagnosis-Aware Routing)
    """
    def __init__(self, in_dim, num_experts, k=2, temp=1.0, use_dx=False, dx_dim=3):
        super(TopKGate, self).__init__()
        self.use_dx = use_dx
        self.num_experts = num_experts
        self.k = min(k, num_experts)
        self.temp = temp
        
        hidden = max(64, in_dim)
        self.pre = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU()
        )
        
        if use_dx:
            # Project diagnosis probability to hidden dim (additive gating)
            self.dx_proj = nn.Linear(dx_dim, hidden, bias=False)
        
        self.proj = nn.Linear(hidden, num_experts)

    def forward(self, h, dx_prob=None):
        """
        Args:
            h: feature tensor [B, in_dim]
            dx_prob: diagnosis probability [B, 3] (detached), optional
        Returns:
            scores: gating scores [B, num_experts] (sparse, summing to 1)
            load: average load per expert [num_experts]
        """
        z = self.pre(h)  # [B, hidden]
        
        if self.use_dx and dx_prob is not None:
            # Additive fusion: h + dx_proj(dx_prob)
            z = z + self.dx_proj(dx_prob)
        
        logits = self.proj(z)  # [B, num_experts]
        scores = torch.softmax(logits / self.temp, dim=-1)  # [B, num_experts]
        
        # Top-K sparse routing
        if self.k < self.num_experts:
            topk_val, topk_idx = torch.topk(scores, self.k, dim=-1)  # [B, k]
            mask = torch.zeros_like(scores).scatter_(1, topk_idx, 1.0)  # [B, num_experts]
            scores = scores * mask
            # Renormalize to sum to 1
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute load balancing statistics
        load = scores.mean(dim=0)  # [num_experts]
        
        return scores, load


class AgeExpert(nn.Module):
    """
    Single expert network for age regression
    Simple 2-layer MLP: in_dim -> 32 -> 1
    """
    def __init__(self, in_dim=40):
        super(AgeExpert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, h):
        """
        Args:
            h: feature tensor [B, in_dim]
        Returns:
            age prediction [B, 1]
        """
        return self.mlp(h)


class AgeMoEHead(nn.Module):
    """
    Mixture of Experts head for age regression
    Combines multiple expert predictions via learned gating
    """
    def __init__(self, in_dim=40, num_experts=6, topk=2, temp=1.2, use_dx=False):
        super(AgeMoEHead, self).__init__()
        self.num_experts = num_experts
        self.gate = TopKGate(in_dim, num_experts, topk, temp, use_dx, dx_dim=3)
        self.experts = nn.ModuleList([AgeExpert(in_dim) for _ in range(num_experts)])
    
    def forward(self, h, dx_prob=None):
        """
        Args:
            h: feature tensor [B, in_dim]
            dx_prob: diagnosis probability [B, 3], optional
        Returns:
            age: weighted combination of expert predictions [B, 1]
            aux: auxiliary info dict {'load': [E], 'gate_scores': [B, E]}
        """
        gate_scores, load = self.gate(h, dx_prob)  # [B, E], [E]
        
        # Compute expert predictions
        expert_outs = torch.stack([expert(h) for expert in self.experts], dim=2)  # [B, 1, E]
        
        # Weighted combination
        age = torch.sum(expert_outs * gate_scores.unsqueeze(1), dim=2)  # [B, 1]
        
        aux = {
            'load': load,
            'gate_scores': gate_scores
        }
        
        return age, aux


# ============= ACDense Backbone Components ============= #

class SE_block(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, inchannels, reduction=16):
        super(SE_block, self).__init__()
        self.GAP = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.FC1 = nn.Linear(inchannels, inchannels // reduction)
        self.FC2 = nn.Linear(inchannels // reduction, inchannels)

    def forward(self, x):
        model_input = x
        x = self.GAP(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = self.FC1(x)
        x = nn.ReLU()(x)
        x = self.FC2(x)
        x = nn.Sigmoid()(x)
        x = x.view(x.size(0), x.size(1), 1, 1, 1)
        return model_input * x


class AC_layer(nn.Module):
    """
    Asymmetric Convolution layer for ACDense backbone
    Uses multi-scale convolution kernels (3x3x3, 1x1x3, 3x1x1, 1x3x1)
    """
    def __init__(self, inchannels, outchannels):
        super(AC_layer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (3, 3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv2 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (1, 1, 3), stride=1, padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv3 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (3, 1, 1), stride=1, padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outchannels))
        self.conv4 = nn.Sequential(
            nn.Conv3d(inchannels, outchannels, (1, 3, 1), stride=1, padding=(0, 1, 0), bias=False),
            nn.BatchNorm3d(outchannels))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return x1 + x2 + x3 + x4


class dense_layer(nn.Module):
    """Dense layer with AC convolution and SE attention"""
    def __init__(self, inchannels, outchannels):
        super(dense_layer, self).__init__()
        self.block = nn.Sequential(
            AC_layer(inchannels, outchannels),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            AC_layer(outchannels, outchannels),
            nn.BatchNorm3d(outchannels),
            nn.ELU(),
            SE_block(outchannels),
            nn.MaxPool3d(2, 2),
        )

    def forward(self, x):
        new_features = self.block(x)
        x = F.max_pool3d(x, 2)
        x = torch.cat([x, new_features], 1)
        return x


class ACDense(nn.Module):
    """
    ACDense backbone for brain age estimation
    Uses Asymmetric Convolution with Dense connections
    
    Args:
        nb_filter (int): number of initial convolutional layer filter. Default: 8
        nb_block (int): number of Dense block. Default: 5
        use_gender (bool, optional): if use gender input. Default: True
    """
    def __init__(self, nb_filter=8, nb_block=5, use_gender=True):
        super(ACDense, self).__init__()
        self.nb_block = nb_block
        self.use_gender = use_gender
        self.pre = nn.Sequential(
            nn.Conv3d(1, nb_filter, kernel_size=7, stride=1, padding=1, dilation=2),
            nn.ELU(),
        )
        self.block, last_channels = self._make_block(nb_filter, nb_block)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.deep_fc = nn.Sequential(
            nn.Linear(last_channels, 32, bias=True),
            nn.ELU(),
        )

        self.male_fc = nn.Sequential(
            nn.Linear(2, 16, bias=True),
            nn.Linear(16, 8, bias=True),
            nn.ELU(),
        )
        self.end_fc_with_gender = nn.Sequential(
            nn.Linear(40, 16),
            nn.Linear(16, 1),
            nn.ReLU()
        )
        self.end_fc_without_gender = nn.Sequential(
            nn.Linear(32, 16),
            nn.Linear(16, 1),
            nn.ReLU()
        )

    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = inchannels * 2
            blocks.append(dense_layer(inchannels, outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels

    def forward(self, x, male_input):
        x = self.pre(x)
        x = self.block(x)
        x = self.gap(x)
        x = torch.reshape(x, (x.size(0), -1))
        x = self.deep_fc(x)
        if self.use_gender:
            male = torch.reshape(male_input, (male_input.size(0), -1))
            male = self.male_fc(male)
            x = torch.cat([x, male.type_as(x)], 1)
            x = self.end_fc_with_gender(x)
        else:
            x = self.end_fc_without_gender(x)
        return x


# Alias for backward compatibility
ScaleDense = ACDense


class DARENet(nn.Module):
    """
    DARE-Net: Diagnosis-Aware Routing Mixture-of-Experts Network
    
    Multi-Task Learning model for:
    1. Brain Age Estimation (Regression) - with Diagnosis-Aware MoE routing
    2. Dementia Staging (3-class Classification: CN/MCI/AD)
    
    Key Features:
    - ACDense backbone with dense connections and asymmetric convolutions
    - Diagnosis-aware sparse routing using predicted CN/MCI/AD posteriors
    - Scheduled teacher forcing for stable self-conditioned routing
    - Heteroscedastic regression for calibrated uncertainty
    - Uncertainty weighting for multi-task learning (Kendall & Gal, 2018)
    
    Args:
        nb_filter (int): number of initial convolutional layer filter. Default: 8
        nb_block (int): number of Dense block. Default: 5
        use_gender (bool): if use gender input. Default: True
        num_classes (int): number of disease classes (CN/MCI/AD). Default: 3
        opt: config options (for MoE parameters)
    """
    def __init__(self, nb_filter=8, nb_block=5, use_gender=True, num_classes=3, opt=None):
        super(DARENet, self).__init__()
        self.nb_block = nb_block
        self.use_gender = use_gender
        self.num_classes = num_classes
        self.opt = opt
        
        # Shared ACDense backbone
        self.pre = nn.Sequential(
            nn.Conv3d(1, nb_filter, kernel_size=7, stride=1, padding=1, dilation=2),
            nn.ELU(),
        )
        self.block, last_channels = self._make_block(nb_filter, nb_block)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.deep_fc = nn.Sequential(
            nn.Linear(last_channels, 32, bias=True),
            nn.ELU(),
        )
        
        # Gender branch
        self.male_fc = nn.Sequential(
            nn.Linear(2, 16, bias=True),
            nn.Linear(16, 8, bias=True),
            nn.ELU(),
        )
        
        # Age regression head: Diagnosis-Aware MoE or standard MLP
        moe_in_dim = 40 if use_gender else 32
        self.use_moe = getattr(opt, 'use_moe', False) if opt is not None else False
        
        if self.use_moe:
            # Diagnosis-Aware Routing MoE head
            self.age_moe = AgeMoEHead(
                in_dim=moe_in_dim,
                num_experts=getattr(opt, 'moe_num_experts', 8),
                topk=getattr(opt, 'moe_topk', 3),
                temp=getattr(opt, 'moe_gate_temp', 1.5),
                use_dx=getattr(opt, 'moe_use_dx', True)
            )
            print(f'=> DARE-Net: Using Diagnosis-Aware Routing MoE')
            print(f'   - {getattr(opt, "moe_num_experts", 8)} experts, Top-{getattr(opt, "moe_topk", 3)} routing')
        else:
            # Standard age regression head
            self.age_head_with_gender = nn.Sequential(
                nn.Linear(40, 16),
                nn.Linear(16, 1)
            )
            self.age_head_without_gender = nn.Sequential(
                nn.Linear(32, 16),
                nn.Linear(16, 1)
            )
        
        # Heteroscedastic regression head for calibrated uncertainty
        self.age_hetero = bool(getattr(opt, 'age_hetero', False)) if opt is not None else False
        if self.age_hetero:
            self.sigma_head = nn.Linear(moe_in_dim, 1)
            print(f'=> DARE-Net: Using heteroscedastic regression for uncertainty')
        
        # Dementia staging head (3-class: CN/MCI/AD)
        self.cls_head_with_gender = nn.Sequential(
            nn.Linear(40, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        self.cls_head_without_gender = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        # Learnable uncertainty weights for multi-task learning
        # log_vars[0] for age task, log_vars[1] for classification task
        self.log_vars = nn.Parameter(torch.zeros(2))
    
    def _make_block(self, nb_filter, nb_block):
        blocks = []
        inchannels = nb_filter
        for i in range(nb_block):
            outchannels = inchannels * 2
            blocks.append(dense_layer(inchannels, outchannels))
            inchannels = outchannels + inchannels
        return nn.Sequential(*blocks), inchannels
    
    def forward(self, x, male_input, dx_true=None, epoch=None):
        """
        Forward pass with Diagnosis-Aware Routing and Scheduled Teacher Forcing
        
        Args:
            x: input MRI image [B, 1, H, W, D]
            male_input: gender one-hot [B, 2]
            dx_true: true diagnosis label [B] for teacher forcing (optional, training only)
            epoch: current epoch for teacher forcing schedule (optional)
        
        Returns:
            age_pred: age prediction [B, 1]
            cls_logits: diagnosis logits [B, 3]
        """
        # Shared ACDense feature extraction
        x = self.pre(x)
        x = self.block(x)
        x = self.gap(x)
        x = torch.reshape(x, (x.size(0), -1))
        feat32 = self.deep_fc(x)  # [B, 32]
        
        # Incorporate gender information
        if self.use_gender:
            male = torch.reshape(male_input, (male_input.size(0), -1))
            male = self.male_fc(male)  # [B, 8]
            h = torch.cat([feat32, male.type_as(feat32)], dim=1)  # [B, 40]
            cls_logits = self.cls_head_with_gender(h)
        else:
            h = feat32  # [B, 32]
            cls_logits = self.cls_head_without_gender(h)
        
        # Age regression with Diagnosis-Aware Routing
        if self.use_moe:
            # Prepare diagnosis probability for routing
            dx_prob = None
            if getattr(self.opt, 'moe_use_dx', False):
                if self.training and dx_true is not None and epoch is not None:
                    # Scheduled Teacher Forcing: blend true label and prediction
                    alpha = self._get_teacher_forcing_alpha(epoch)
                    dx_onehot = torch.nn.functional.one_hot(dx_true.long(), num_classes=3).float()
                    dx_pred_soft = torch.softmax(cls_logits.detach(), dim=1)
                    dx_prob = alpha * dx_onehot + (1 - alpha) * dx_pred_soft
                else:
                    # Inference: use predicted diagnosis posterior only
                    dx_prob = torch.softmax(cls_logits.detach(), dim=1)
            
            # Diagnosis-Aware MoE forward
            age_pred, aux = self.age_moe(h, dx_prob)
            
            # Store MoE auxiliary info for loss computation
            self.moe_aux = {
                'balance': torch.sum((aux['load'] - 1.0 / self.age_moe.num_experts) ** 2),
                'entropy': -(aux['gate_scores'] * (aux['gate_scores'] + 1e-8).log()).sum(dim=1).mean(),
                'gate_scores': aux['gate_scores'],
                'load': aux['load']
            }
        else:
            # Standard age head
            if self.use_gender:
                age_pred = self.age_head_with_gender(h)
            else:
                age_pred = self.age_head_without_gender(h)
            self.moe_aux = None
        
        # Heteroscedastic uncertainty prediction
        if self.age_hetero:
            lv = self.sigma_head(h).clamp_(-5, 5)  # log σ²
            self.age_logvar = lv
        else:
            self.age_logvar = None
        
        return age_pred, cls_logits
    
    def _get_teacher_forcing_alpha(self, epoch):
        """
        Scheduled Teacher Forcing: linearly decay from 1.0 to 0.0
        
        Args:
            epoch: current training epoch
        
        Returns:
            alpha: teacher forcing weight in [0, 1]
        """
        tf_epochs = getattr(self.opt, 'moe_tf_epochs', 16)
        if epoch < tf_epochs:
            return 1.0 - (epoch / tf_epochs)
        else:
            return 0.0


# Alias for backward compatibility
ScaleDenseMTL = DARENet


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num/1e6, 'Trainable': trainable_num/1e6}
