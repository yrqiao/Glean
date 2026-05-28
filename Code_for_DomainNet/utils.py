import numpy as np
import torch
from torch import nn, autograd
import random
import pickle
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def lambda_epoch(epoch):
    return 0.1 if epoch > 10 else 1.0

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

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    # dict = dict.reset_index()
    # dict.rename(columns={'index': 'filename'}, inplace=True)
    return dict

class DomainNet(Dataset):
    def __init__(self, anno_file, img_dir, transform=None):
        self.anno_file = anno_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.anno_file)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.anno_file['Filename'][idx].lstrip('/'))
        image = Image.open(img_path)
        # label = self.anno_file['labels'][idx][0][9]
        label = self.anno_file['labels'][idx][0]
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


NICO = DomainNet
    
    
