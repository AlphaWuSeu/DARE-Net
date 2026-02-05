import torch
import torch.nn as nn
import scipy.stats.stats as stats
import numpy as np

def get_rank(batch_score, dim=0):
    rank = torch.argsort(batch_score, dim=dim)  
    rank = torch.argsort(rank, dim=dim)         
    rank = (rank * -1) + batch_score.size(dim)  
    rank = rank.float()
    rank = rank / batch_score.size(dim)         

    return rank

def get_tiedrank(batch_score, dim=0):
    batch_score = batch_score.cpu()
    rank = stats.rankdata(batch_score)
    rank = stats.rankdata(rank) - 1    
    rank = (rank * -1) + batch_score.size(dim)
    rank = torch.from_numpy(rank).cuda()
    rank = rank.float()
    rank = rank / batch_score.size(dim)  
    return rank

def model_loader(model_type, seq_len, pretrained_state_dict=None):

    if model_type == "lstm":
        model = lstm_baseline(seq_len)
    elif model_type == "lstmla":
        model = lstm_large(seq_len)
    elif model_type == "lstme":
        model = lstm_end(seq_len)
    elif model_type == "mlp":
        model = mlp(seq_len)
    else:
        raise Exception("Model type unknown", model_type)

    if pretrained_state_dict is not None:
        model.load_state_dict(pretrained_state_dict)

    return model

class lstm_baseline(nn.Module):
    def __init__(self, seq_len):
        super(lstm_baseline, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv1d(seq_len, seq_len, 256)

    def forward(self, input_):
        # input_ shape: [batch, seq_len]
        # Ensure input has seq_len elements
        if input_.size(1) != self.seq_len:
            if input_.size(1) < self.seq_len:
                # Pad to seq_len
                padding = torch.zeros(input_.size(0), self.seq_len - input_.size(1), 
                                    device=input_.device, dtype=input_.dtype)
                input_ = torch.cat([input_, padding], dim=1)
            else:
                # Truncate to seq_len
                input_ = input_[:, :self.seq_len]
        
        input_ = input_.reshape(input_.size(0), -1, 1)
        out, _ = self.lstm(input_)
        # LSTM output: [batch, seq_len, 256]
        # Conv1d expects [batch, channels, length] where channels=seq_len
        # PyTorch Conv1d interprets second dimension as channels
        # So [batch, seq_len, 256] -> channels=seq_len, length=256 ✓
        out = self.conv1(out)
        return out.view(input_.size(0), -1)

class mlp(nn.Module):
    def __init__(self, seq_len):
        super(mlp, self).__init__()
        self.lin1 = nn.Linear(seq_len, 2048)
        self.lin2 = nn.Linear(2048, 2048)
        self.lin3 = nn.Linear(2048, seq_len)

        self.relu = nn.ReLU()

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1)
        out = self.lin1(input_)
        out = self.lin2(self.relu(out))
        out = self.lin3(self.relu(out))

        return out.view(input_.size(0), -1)

class lstm_end(nn.Module):
    def __init__(self, seq_len):
        super(lstm_end, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.GRU(self.seq_len, 5 * self.seq_len, batch_first=True, bidirectional=False)

    def forward(self, input_):
        input_ = input_.reshape(input_.size(0), -1, 1).repeat(1, input_.size(1), 1).view(input_.size(0), input_.size(1), -1)
        _, out = self.lstm(input_)

        out = out.view(input_.size(0), self.seq_len, -1)  # .view(input_.size(0), -1)[:,:self.seq_len]
        out = out.sum(dim=2)

        return out

class lstm_large(nn.Module):

    def __init__(self, seq_len):
        super(lstm_large, self).__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(1, 512, 2, batch_first=True, bidirectional=True)
        # Note: Conv1d(seq_len, seq_len, 1024) expects input [batch, seq_len, length]
        # But PyTorch Conv1d requires [batch, channels, length], so this is actually wrong
        # However, to match pretrained weights, we'll transpose the LSTM output
        self.conv1 = nn.Conv1d(seq_len, seq_len, 1024)

    def forward(self, input_):
        # input_ shape: [batch, seq_len] (e.g., [1, 4] for batch_size=4)
        # Ensure input has seq_len elements (pad or truncate if needed)
        if input_.size(1) != self.seq_len:
            if input_.size(1) < self.seq_len:
                # Pad to seq_len
                padding = torch.zeros(input_.size(0), self.seq_len - input_.size(1), 
                                    device=input_.device, dtype=input_.dtype)
                input_ = torch.cat([input_, padding], dim=1)
            else:
                # Truncate to seq_len
                input_ = input_[:, :self.seq_len]
        
        # Reshape to [batch, seq_len, 1] for LSTM input
        input_ = input_.reshape(input_.size(0), -1, 1)
        # LSTM output: [batch, seq_len, hidden_size=1024] (512*2 bidirectional)
        out, _ = self.lstm(input_)
        
        # Original code doesn't transpose, but PyTorch Conv1d requires [batch, channels, length]
        # The conv1d(seq_len, seq_len, 1024) expects in_channels=seq_len
        # LSTM output is [batch, seq_len, 1024]
        # We need to transpose to [batch, channels, length] where channels=seq_len
        # So: [batch, seq_len, 1024] -> transpose to make seq_len the channel dimension
        # But if we transpose [batch, seq_len, 1024] -> [batch, 1024, seq_len], channels become 1024
        # The solution: the conv1d definition is wrong, but to match pretrained weights,
        # we need to work around it. Let's try transposing the first two dimensions differently
        # Actually, we need [batch, seq_len, 1024] -> [batch, seq_len, 1024] but Conv1d sees it as [batch, channels, length]
        # So channels=seq_len, length=1024, which matches conv1d(seq_len, seq_len, 1024)
        # The issue is PyTorch Conv1d automatically interprets the second dimension as channels
        # So if we pass [batch, seq_len, 1024], Conv1d sees channels=seq_len, length=1024 ✓
        # This should work! Let's try without transpose (original behavior)
        out = self.conv1(out)

        return out.view(input_.size(0), -1)


def load_sorter(checkpoint_path):
    # PyTorch 2.6+ changed default weights_only=True, but checkpoint contains argparse.Namespace
    # Set weights_only=False for trusted checkpoint files
    try:
        sorter_checkpoint = torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        sorter_checkpoint = torch.load(checkpoint_path)

    model_type = sorter_checkpoint["args_dict"].model_type
    seq_len = sorter_checkpoint["args_dict"].seq_len
    state_dict = sorter_checkpoint["state_dict"]

    return model_type, seq_len, state_dict



class SpearmanLoss(torch.nn.Module):
    """ Loss function  inspired by spearmann correlation.self
    Required the trained model to have a good initlization.

    Set lbd to 1 for a few epoch to help with the initialization.
    """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, lbd=0):
        super(SpearmanLoss, self).__init__()
        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

        self.criterion_mse = torch.nn.MSELoss()
        self.criterionl1 = torch.nn.L1Loss()

        self.lbd = lbd

    def forward(self, mem_pred, mem_gt, pr=False):
        rank_gt = get_tiedrank(mem_gt)

        # mem_pred shape: [batch_size, 1] or [batch_size]
        # Flatten to [batch_size] if needed
        if mem_pred.dim() > 1:
            mem_pred_flat = mem_pred.view(-1)
        else:
            mem_pred_flat = mem_pred
        
        # Add batch dimension: [batch_size] -> [1, batch_size]
        # The sorter expects input shape [batch, seq_len] where seq_len should match the checkpoint's seq_len
        rank_pred = self.sorter(mem_pred_flat.unsqueeze(0)).view(-1)
        return self.criterion_mse(rank_pred, rank_gt) + self.lbd * self.criterionl1(mem_pred_flat, mem_gt)
