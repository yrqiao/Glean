import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from utils import *
from architecture import *
import os

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=1)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
parser.add_argument('--noise_sd', type=float, default=0.12)
parser.add_argument('--Lipschitz_model', type=bool, default=True, choices=[True, False])
flags = parser.parse_args()

set_deterministic(0)

os.makedirs('./saved_model', exist_ok=True)
final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
    print("Restart", restart)

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments

    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() 
        images = images.reshape((-1, 28, 28))
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        images = torch.stack([images, images, torch.zeros_like(images)], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        images = images.float() / 255.
        return {
        'images': images.cuda(),
        'labels': labels.cuda()
        }

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]

    if flags.Lipschitz_model:
        mlp = MLP().cuda()
    else:
        mlp = MLP1().cuda()
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)
    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'] + torch.randn_like(env['images'], device='cuda')*flags.noise_sd)
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        loss = train_nll.clone()
        penalty_weight = (flags.penalty_weight 
            if step >= flags.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        test_acc = envs[2]['acc']
        if step % 100 == 0:
            pretty_print(
            np.int32(step),
            train_nll.detach().cpu().numpy(),
            train_acc.detach().cpu().numpy(),
            train_penalty.detach().cpu().numpy(),
            test_acc.detach().cpu().numpy()
            )
            torch.save(mlp.state_dict(), f'./saved_model/model_{step}.pth')

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
