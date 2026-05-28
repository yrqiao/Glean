import argparse
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim, autograd
from utils import *
from architecture import *
import os
from torch.optim.lr_scheduler import LambdaLR

parser = argparse.ArgumentParser(description='DomainNet')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=30)
parser.add_argument('--penalty_weight', type=float, default=10000)
parser.add_argument('--epochs', type=int, default=201)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--noise_sd', type=float, default=0.12)
parser.add_argument('--train_batch_size', default=512, type=int, help='Batch size')
parser.add_argument('--test_batch_size', type=int, default=512)
parser.add_argument('--Lipschitz_model', type=bool, default=True, choices=[True, False])
parser.add_argument(
    '--dataset_path',
    type=str,
    default='/home/qyr/Certified_Robustness_Defense_NICO/DomainNet',
    help='Path to the DomainNet image root directory'
)
flags = parser.parse_args()

set_deterministic(0)
os.makedirs('./saved_model', exist_ok=True)
if flags.Lipschitz_model:
    model = CNN().cuda()
else:
    model = CNN1().cuda()

env1 = unpickle('train_env1_DomainNet.pickle')
env2 = unpickle('train_env2_DomainNet.pickle')
env_test = unpickle('test_env_DomainNet.pickle')
trans_f = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
])

train_set = [
    DomainNet(
        env1,
        flags.dataset_path,
        transform=trans_f
    ),
    DomainNet(
        env2,
        flags.dataset_path,
        transform=trans_f
    ),
]
test_set = [
    DomainNet(
        env_test,
        flags.dataset_path,
        transform=trans_f
    ),
]
train_loader = [
    DataLoader(
        dataset,
        batch_size=flags.train_batch_size,
        shuffle=True,
        drop_last=False
    )
    for dataset in train_set
]
test_loader = [
    DataLoader(
        dataset,
        batch_size=flags.test_batch_size,
        shuffle=True,
        drop_last=False
    )
    for dataset in test_set
]

optimizer = optim.Adam(model.parameters(), lr=flags.lr)
# optimizer = optim.SGD(model.parameters(), lr=flags.lr, momentum=0.9, weight_decay=2e-4)
scheduler = LambdaLR(optimizer, lr_lambda=lambda_epoch)
# optimizer = optim.SGD(model.parameters(), lr=flags.lr, weight_decay=2e-4, momentum=0.9)
pretty_print('epoch', 'train nll', 'train acc', 'train penalty', 'total_loss', 'test acc')

for epoch in range(flags.epochs):
    avg_nll = 0
    avg_acc = 0
    avg_penalty = 0
    avg_total = 0

    for (images1, labels1), (images2, labels2) in zip(train_loader[0], train_loader[1]):
        inputs1, targets1 = images1.cuda(), labels1.cuda()
        inputs2, targets2 = images2.cuda(), labels2.cuda()
        targets1 = targets1.long()
        targets2 = targets2.long()

        outputs1 = model(inputs1+torch.randn_like(inputs1, device='cuda')*flags.noise_sd)
        train_nll1 = mean_nll(outputs1, targets1)
        train_penalty1 = penalty(outputs1, targets1)
        train_acc1 = mean_accuracy(outputs1, targets1)
        
        outputs2 = model(inputs2+torch.randn_like(inputs2, device='cuda')*flags.noise_sd)
        train_nll2 = mean_nll(outputs2, targets2)
        train_penalty2 = penalty(outputs2, targets2)
        train_acc2 = mean_accuracy(outputs2, targets2)

        train_nll = torch.stack([train_nll1, train_nll2]).mean()
        train_acc = torch.stack([train_acc1, train_acc2]).mean()
        train_penalty = torch.stack([train_penalty1, train_penalty2]).mean()
        
        avg_acc += train_acc
        avg_nll += train_nll
        avg_penalty += train_penalty

        loss = train_nll.clone()
        penalty_weight = (flags.penalty_weight 
            if epoch >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight
        avg_total += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_nll /= len(train_loader[0])
    avg_acc /= len(train_loader[0])
    avg_penalty /= len(train_loader[0])
    avg_total /= len(train_loader[0])

    avg_test_acc = 0
    with torch.no_grad():
        for images, labels in test_loader[0]:
            images, labels = images.cuda(), labels.cuda()
            labels = labels.long()
            logits = model(images+torch.randn_like(images, device='cuda')*flags.noise_sd)
            test_acc = mean_accuracy(logits, labels)
            avg_test_acc += test_acc
        avg_test_acc /= len(test_loader[0])
    if epoch % 1 == 0:
        pretty_print(
        np.int32(epoch),
        avg_nll.detach().cpu().numpy(),
        avg_acc.detach().cpu().numpy(),
        avg_penalty.detach().cpu().numpy(),
        avg_total.detach().cpu().numpy(),
        avg_test_acc.detach().cpu().numpy(),
        )
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'./saved_model/model_{epoch}.pth')

