import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from nn_analysis.dataset.fakefmri import FakeFmriDataSet, gen_rnn_from_adjacency, hrf_filt
from sklearn.model_selection import train_test_split
from nn_analysis.model.torch import GenericModel, Flatten
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
from functools import reduce
from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap, InputXGradient, GuidedBackprop, GuidedGradCam, Deconvolution, FeatureAblation, Occlusion, FeaturePermutation, ShapleyValueSampling, Lime, KernelShap, LRP
import sys
from pprint import pprint
from collections import OrderedDict
from torch import nn
import sklearn.metrics as met
from termcolor import colored
import networkx as nx

def plot_attr(path, class_attr, name, label = '', xlabel = 'Features', ylabel = '', c = '0.0', highlight = None):
    n = len(class_attr)
    n_feat = len(class_attr[0])
    f = plt.figure(figsize = (6, 2*n), layout = 'constrained')
    feat = np.arange(n_feat)
    for k, attr in class_attr.items():
        ax = f.add_subplot(n,1,k+1) if k == 0 else f.add_subplot(n,1,k+1, sharex = ax, sharey = ax)
        attr = attr.detach().numpy()
        attr = 100*attr/np.max(attr)
        if not highlight is None:
            for i, (idx,_c,_label) in enumerate(highlight):
                ax.bar(feat, attr*idx[k], label = f'Features/ROIs discovered for Class {1+k}', width = 1.0, color = _c, zorder = i+1)
                #ax.bar(feat, attr*idx[k], label = f'{label} for Class {k}', width = 1.0, color = _c, zorder = i+1)
        ax.bar(feat, attr, width = 1.0, color = c, zorder = 0)
        #ax.bar(feat, attr, label = f'{label} for Class {k}', width = 1.0, color = c, zorder = 0)
        if not highlight is None:
            ax.legend()
        if k == 0:
            ax.set_title(f'{name} Feature Attribution Scores', weight = 'bold')
        if k == n-1:
            ax.set_xlabel(xlabel)
        ax.set_ylim(0, 110)
        ax.set_ylabel(f'{ylabel} Score (%)')
        ax.set_xticks(feat[::20])
    f.savefig(path, dpi = 200)

def plot_f1_score(path, class_attr, hypothesis, num_thresholds, name, xlabel = 'Threshold Score (%)'): 
    n = len(class_attr)
    f = plt.figure(figsize = (6, 2*n), layout = 'constrained')

    for k, attr in class_attr.items():
        ax = f.add_subplot(n,1,k+1) if k == 0 else f.add_subplot(n,1,k+1, sharex = ax, sharey = ax)
        attr = attr.detach().numpy()
        attr = 100*attr/(np.max(attr))
        thresholds = np.linspace(0, 100, num_thresholds)
        for idx, c, label in hypothesis:
            f1 = np.zeros(shape = (num_thresholds))
            for i, threshold in enumerate(thresholds):
                y_pred = attr >= threshold
                f1[i] = met.f1_score(idx[k], y_pred)
            ax.plot(thresholds, 100*f1, label = label, c = c, marker = '.')
        if k == 0:
            ax.legend()
            ax.set_title(f'{name}', weight = 'bold')
        if k == n-1:
            ax.set_xlabel(xlabel)
        ax.set_ylabel(f'F1 Score (%)\nfor clinical\npopulation {k} ')
    f.savefig(path)

def main(args):

    seed = 0
    pl.seed_everything(seed)


    # -------------------------------
    # Parameters
    # -------------------------------

    n_class = 2
    n_sample = 400 # number of samples used in training per class
    n_feat = args.n[0]
    n_step = 100
    delta = args.delta[0]
    n_aff = args.a[0]
    As = [np.zeros((n_feat, n_feat)) for i in range(n_class)]
    G = nx.connected_watts_strogatz_graph(n_feat, 10, 0.1)
    A = nx.adjacency_matrix(G).todense()
    _radius = max(np.abs(np.linalg.eigvals(A)))
    A = A/_radius
    As[0][...] = A
    As[1][...] = A
    A_max = np.max(A)
    As[1][:n_aff] += delta*A_max
    hrf = np.float32(np.load(f'{os.environ.get("DATA_PATH")}/../matlab/hrf_dt1.0.npy')[:, 0])
    snr = args.snr[0] # in dB

    gen_data = gen_rnn_from_adjacency
    params = [{'n_sample': n_sample, 'A': A, 'n_step': n_step, 'noise': 0.1, 'radius': None, 'transform': lambda x: hrf_filt(x, hrf, snr = snr)} for A in As]
    n_epoch = 150

    affected = {k: np.arange(n_feat) < n_aff for k in range(n_class)}


    c_affected = '0.0'
    c_not_affected = '0.8'
    highlight = [(affected, c_affected, 'ROI affected')]
    hypothesis = [(affected, c_affected, None)]

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where model are saved
    model_name = f'simple_n{n_feat}_a{n_aff:d}_delta{delta:.2f}_snr{snr:.2f}'
    model_path = f'{os.environ.get("DATA_PATH")}/rnn_exc/model/{model_name}'
    # path where attribution are saved
    attr_path = f'{os.environ.get("DATA_PATH")}/rnn_exc/attr/ig_baseline'
    os.makedirs(f'{model_path}', exist_ok=True)
    # path where log of training are saved
    log_path = f'{os.environ.get("TMP_PATH")}/rnn_exc/log/test'
    # path where model are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/rnn_exc/{model_name}'
    os.makedirs(f'{figure_path}', exist_ok=True)

    # -------------------------------
    # Test dataset
    # -------------------------------

    dataset = FakeFmriDataSet(gen_data, params)
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.label)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4, pin_memory = True)
    test_loader = DataLoader(train_dataset, batch_size = 128, shuffle = False, num_workers = 4, pin_memory = True)

    #x, label = train_dataset[:len(train_dataset)]
    x, label = test_dataset[:len(test_dataset)]
    x.requires_grad = True

    # -------------------------------
	# Feature attribution methods
	# -------------------------------

    methods = []
    methods += [(f'IG Zero Baseline', f'ig_zerob', IntegratedGradients, {'baselines': 0})]
    methods += [(f'IG Random Baseline Global', f'ig_rb_global', IntegratedGradients, {'baselines': torch.normal(mean = float(torch.mean(x)), std = float(torch.std(x)), size = x.shape[1:])[None]})]
    methods += [(f'IG Random Baseline By ROI', f'ig_rb_roi', IntegratedGradients, {'baselines': torch.distributions.multivariate_normal.MultivariateNormal(torch.mean(x, dim = (0,2)), torch.diag(torch.std(x, dim = (0,2)))).sample(x.shape[2:]).T[None]})]
    methods += [(f'IG Random Baseline By Time & ROI', f'ig_rb_time', IntegratedGradients, {'baselines': torch.normal(torch.mean(x, dim = 0), torch.std(x, dim = 0))[None]})]
    methods += [(f'IG Median Baseline', f'ig_medb', IntegratedGradients, {'baselines': torch.median(x, dim = 0, keepdim = True).values})]
    methods += [(f'IG Mean Baseline', f'ig_meanb', IntegratedGradients, {'baselines': torch.mean(x, dim = 0, keepdim = True)})]

    _methods = []

    if not args.redo:
        for name, tag, method, kwargs in methods:
            files = [f'{attr_path}/{tag}/{model_name}_epoch{epoch:02}.pt' for epoch in args.epochs]
            if not np.all([os.path.exists(path) for path in files]):
                _methods.append((name, tag, method, kwargs))
        methods = _methods
    if len(methods)==0:
        print(colored(f'{model_name} at epochs {args.epochs} already tested', 'green'))
        sys.exit(0)
    else:
        _methods = ', '.join([m for _,m,_,_ in methods])
        print(colored(f'Testing {model_name} at epochs {args.epochs} ({_methods})', 'green'))

    # -------------------------------
    # Saving model
    # -------------------------------


    model  = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(n_feat, 32, 3)),
            ('nonlin1', nn.ReLU()),
            ('avgpool1', nn.AvgPool1d(5)),
            ('conv2', nn.Conv1d(32, 32, 3)),
            ('nonlin2', nn.ReLU()),
            ('avgpool2', nn.AdaptiveAvgPool1d(1)),
            ('flatten', Flatten()),
            ('lin', nn.Linear(32, n_class))
        ]))

    model = GenericModel(model)
    trainer = pl.Trainer(default_root_dir=log_path, deterministic=True, devices="auto", accelerator="auto")

    # -------------------------------
    # Testing model
    # -------------------------------

    
    # -------------------------------
    #  Computing feature attribution
    # -------------------------------

    for i, epoch in enumerate(args.epochs):
        checkpoint = torch.load(f'{model_path}/epoch{epoch:02}.ckpt')
        model.load_state_dict(checkpoint['state_dict'])
        metrics, = trainer.test(model, test_loader)
        print('--------------------')
        for name, tag, method, kwargs in methods:
            print(f'{name}')
            method = method(model.model)
            attr = method.attribute(x, target=label, **kwargs)
            class_attr = {}
            for k in range(n_class):
                idx = label == k
                class_attr[k] = torch.mean(torch.abs(torch.median(attr[idx], axis = 2).values), axis = 0)
                print(f'mean attribution for class {k}:')
                pprint(class_attr[k])
            plot_attr(f'{figure_path}/attr_{tag}.png', class_attr, name, c = c_not_affected, label = 'ROI not affected', highlight = highlight, xlabel = 'Features/ROIs', ylabel = 'IG')
            plot_f1_score(f'{figure_path}/f1_{tag}.png', class_attr, hypothesis, 101, 'Discrimination of Affected ROI using Score', xlabel = 'Threshold Score (%)')
            os.makedirs(f'{attr_path}/{tag}', exist_ok=True)
            torch.save(torch.stack(list(class_attr.values()), axis = 0), f'{attr_path}/{tag}/{model_name}_epoch{epoch:02}.pt')
            print('--------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test model at different epochs to discriminate simulated fMRI')
    parser.add_argument('--redo', action='store_true', help='If set, resave even if file exists')
    parser.add_argument('--epochs', metavar = 'E', type = int, nargs = "+", help = 'list of epochs to test')
    parser.add_argument('--n', metavar = 'N', type = int, nargs = 1, default = 246, help = 'number of nodes')
    parser.add_argument('--a', metavar = 'A', type = int, nargs = 1, help = 'number of nodes affected by local imbalance')
    parser.add_argument('--delta', metavar = 'D', type = float, nargs = 1, default = [0], help = 'difference between gain')
    parser.add_argument('--snr', metavar = 'N', type = float, nargs = 1, default = [0.0], help = 'SNR in dB')
    args = parser.parse_args()
    main(args)