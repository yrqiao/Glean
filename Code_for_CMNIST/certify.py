import argparse
import os
from core import Smooth
from time import time
import torch
import datetime
from architecture import *
from torchvision import datasets
import random
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--base_classifier", type=str, default='./saved_model/model_500.pth')
parser.add_argument("--sigma", type=float, default=0.12, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, default='output_file', help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--num_classes", type=float, default=2)
parser.add_argument('--Lipschitz_model', type=bool, default=True, choices=[True, False])
args = parser.parse_args()

if __name__ == "__main__":

    set_deterministic(0)
    if args.Lipschitz_model:
        base_classifier = MLP().cuda()
    else:
        base_classifier = MLP1().cuda
    base_classifier.load_state_dict(torch.load(args.base_classifier)) 
    
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, args.num_classes, args.sigma)
    

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset

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

    dataset = envs[2]
    for i in range(1000):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        x, label = dataset['images'][i], dataset['labels'][i]

        before_time = time()
        # certify the prediction of g around x
        x, label = x.cuda(), label.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.5}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed))
        print("{}\t{}\t{}\t{:.5}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()