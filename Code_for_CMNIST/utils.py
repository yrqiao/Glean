import numpy as np
import torch
from torch import nn, autograd
import random

def mean_nll(logits, y):
    return nn.functional.cross_entropy(logits, y.long())

def mean_accuracy(logits, y):
    preds = torch.argmax(logits, 1)
    correct = preds.eq(y.long()).float()
    return correct.mean()

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def set_deterministic(seed):
    # seed by default is None
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("Non-deterministic")

def pretty_print(*values):
    col_width = 13
    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)
    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))