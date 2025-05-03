# Utility functions for the project

import torch
import math
import logging
import os
import csv
from sklearn.metrics import f1_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("results/log.txt", mode='w'),
    ]
)
logger = logging.getLogger(__name__)

def bp2matrix(L, base_pairs):
    """Convert base pairs to a contact matrix"""
    matrix = torch.zeros((L, L))
    # base pairs are 1-based
    bp = torch.tensor(base_pairs) - 1
    if len(bp.shape) == 2:
        matrix[bp[:, 0], bp[:, 1]] = 1
        matrix[bp[:, 1], bp[:, 0]] = 1
    return matrix

def mat2bp(x):
    """Get base-pairs from connection matrix [N, N]. It uses upper
    triangular matrix only, without the diagonal. Positions are 1-based."""
    ind = torch.triu_indices(x.shape[0], x.shape[1], offset=1)
    pairs_ind = torch.where(x[ind[0], ind[1]] > 0)[0]

    pairs_ind = ind[:, pairs_ind].T
    # remove multiplets pairs
    multiplets = []
    for i, j in pairs_ind:
        ind = torch.where(pairs_ind[:, 1]==i)[0]
        if len(ind)>0:
            pairs = [bp.tolist() for bp in pairs_ind[ind]] + [[i.item(), j.item()]]
            best_pair = torch.tensor([x[bp[0], bp[1]] for bp in pairs]).argmax()

            multiplets += [pairs[k] for k in range(len(pairs)) if k!=best_pair]

    pairs_ind = [[bp[0]+1, bp[1]+1] for bp in pairs_ind.tolist() if bp not in multiplets]
    return pairs_ind

def contact_f1(ref_batch, pred_batch, Ls, th=0.5, reduce=True, method="triangular"):
    """Compute F1 from base pairs. Input goes to sigmoid and then thresholded"""
    f1_list = []

    if type(ref_batch) == float or len(ref_batch.shape) < 3:
        ref_batch = [ref_batch]
        pred_batch = [pred_batch]
        L = [Ls]

    for ref, pred, l in zip(ref_batch, pred_batch, Ls):
        # ignore padding
        ind = torch.where(ref != -1)
        pred = pred[ind].view(l, l)
        ref = ref[ind].view(l, l)

        # pred goes from -inf to inf
        pred = torch.sigmoid(pred)
        pred[pred<=th] = 0

        if method == "triangular":
            f1 = f1_triangular(ref, pred>0)
        elif method == "f1_shift":
            ref_bp = mat2bp(ref)
            pred_bp = mat2bp(pred)
            f1 = f1_shift(ref_bp, pred_bp)
        else:
            raise NotImplementedError

        f1_list.append(f1)

    if reduce:
        return torch.tensor(f1_list).mean().item()
    else:
        return torch.tensor(f1_list)

def f1_triangular(ref, pred):
    """Compute F1 from the upper triangular connection matrix"""
    # get upper triangular matrix without diagonal
    ind = torch.triu_indices(ref.shape[0], ref.shape[1], offset=1)

    ref = ref[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    return f1_score(ref, pred, zero_division=0)

def linear_beta(beta_0, t, beta_max, T):
    """Calculate beta value using linear schedule"""
    return beta_0 + ((t-1)/(T-1))*(beta_max-beta_0)

def cosine_beta(t, T, s=0.008):
    """Calculate beta value using cosine schedule"""
    def f(t):
        return math.cos((t / T + s) / (1 + s) * (math.pi / 2)) ** 2
  
    alpha_bar_t = f(t) / f(0)
    alpha_bar_t_prev = f(t - 1) / f(0) if t > 1 else 1.0 # alpha_bar at first it is 1

    alpha_t = alpha_bar_t / alpha_bar_t_prev
    beta_t = 1 - alpha_t

    return beta_t

def exponential_beta(beta_min, t, beta_max, T):
    """Calculate beta value using exponential schedule"""
    if t == 1:
        return beta_min
    elif t == T:
        return beta_max
    else:
        return beta_min * (beta_max / beta_min) ** ((t - 1) / (T - 1))

def setup_csv_logger(filepath, fieldnames):
    """Set up CSV logger for metrics"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    return filepath

def log_metrics_to_csv(filepath, metrics):
    """Log metrics to CSV file"""
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writerow(metrics)

def get_embed_dim(loader):
    """Get embedding dimension from dataloader"""
    batch_elem = next(iter(loader))
    return batch_elem["seq_embs_pad"].shape[2]

def outer_concat(t1, t2):
    """Outer concatenation of tensors"""
    assert t1.shape == t2.shape, f"Shapes of input tensors must match! ({t1.shape} != {t2.shape})"
    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)
    return torch.concat((a, b), dim=-1)
