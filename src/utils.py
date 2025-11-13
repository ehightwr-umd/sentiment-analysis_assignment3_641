#!/usr/bin/env python3

# Import Libraries
import torch, numpy as np, random
from sklearn.metrics import accuracy_score, f1_score

# Set Randomness, as required
def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# Function to Compute Metrics (F1, Accuracy)
def evaluate_model(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu()
    preds_bin = (preds > 0.5).int().numpy()
    labels = labels.numpy()
    acc = accuracy_score(labels, preds_bin)
    f1 = f1_score(labels, preds_bin, average='macro')
    return acc, f1

