import os
import argparse
from torchinfo import summary
import time 
from src.model import fchmrf
import torch
import src.util as util
import numpy as np
import scipy.stats as stats
import random 
from src import config
import nibabel as nib
import sys

def run(args):
    """
    Runs desired simulation setting for # replications 
    """
    if not os.path.exists(args.savepath):
        os.makedirs(args.savepath)

    # Load data, groundtruth (unused in training), and beta
    y = np.load(args.labelpath).reshape((1,1,30,30,30))
    beta = np.load(args.betapath).reshape((1,1,30,30,30))
    X = np.load(args.datapath)[config.data_index:config.data_index+1].reshape((1,1,30,30,30))
    beta = torch.FloatTensor(beta) 
    X = torch.FloatTensor(X) 

    # Run fchmrf, returns P(h=1|x)
    max_iter = args.e
    net = fchmrf(X, beta, mask=None, pval=None, threshold=args.threshold, lr=args.lr)
    print(summary(net))

    for epoch in range(max_iter):
        # Run a step of EM algorithm
        h, loss_q1, loss_q2 = net.em_step()

    net.eval()
    h, _, _ = net.em_step()
    fdr, fnr, atp, signal_lis = util.p_lis(h.squeeze(), threshold=args.threshold, label=y.ravel())
    print(fdr, fnr, atp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FcHMRF-LIS for spatial multiple hypothesis testing.')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--e', default=5, type=int)
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--labelpath', default='./', type=str)
    parser.add_argument('--betapath', default='./', type=str)
    parser.add_argument('--savepath', default='./', type=str)
    parser.add_argument('--roipath', default='None', type=str)
    parser.add_argument('--ppath', default='None', type=str)
    args = parser.parse_args()

    # Example run:
    # python -m src.train --lr 1e-5 --e 20 --threshold 0.05 --datapath data.npy --labelpath gt.npy --betapath beta.npy
    run(args)
