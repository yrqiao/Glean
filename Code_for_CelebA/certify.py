import argparse
import os
from core import Smooth
from time import time
import torch
import datetime
from architecture import *
from torchvision import datasets, transforms
import random
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--base_classifier", type=str, default='./saved_model/model_40.pth')
parser.add_argument("--sigma", type=float, default=0.12, help="noise hyperparameter")
parser.add_argument("--outfile", type=str, default='output_file', help="output file")
parser.add_argument("--batch", type=int, default=512, help="batch size")
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
    # load the base classifier
    if args.Lipschitz_model:
        base_classifier = CNN().cuda()
    else:
        base_classifier = CNN1().cuda()
    base_classifier.load_state_dict(torch.load(args.base_classifier)) 
    
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, args.num_classes, args.sigma)
    

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    env_test = unpickle('test_env_smile.pickle')
    trans_f = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    dataset = [
        CelebA(env_test, 
            '/home/qyr/Certified_Robustness_Defense_CelebA/CelebA/img_align_celeba', 
            transform=trans_f
            ),
    ]
    indices = list(range(len(dataset[0])))
    random.shuffle(indices)

    # iterate through the dataset

    for idx, i in enumerate(indices[:100]):

        # only certify every args.skip examples, and stop after args.max examples
        if idx % args.skip != 0:
            continue
        if idx == args.max:
            break

        x, label = dataset[0][i]

        before_time = time()
        # certify the prediction of g around x
        x, label = x.cuda(), label.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.5}\t{}\t{}".format(
            idx, label, prediction, radius, correct, time_elapsed))
        print("{}\t{}\t{}\t{:.5}\t{}\t{}".format(
            idx, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()