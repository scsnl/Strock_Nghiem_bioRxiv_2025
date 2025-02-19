import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from nn_analysis.dataset.fakefmri import FakeFmriDataSet, gen_from_file, NormalizeDataset, add_measurement_noise
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
from matplotlib.colors import to_rgb, hsv_to_rgb, rgb_to_hsv
from matplotlib.patches import Patch
from torchmetrics import Accuracy


def plot_attr(path, class_attr, name, label = '', xlabel = 'ROIs', ylabel = '', c = '0.0', highlight = None, idx = None, clabel = ''):
    f = plt.figure(figsize = (4, 2), layout = 'constrained')
    group = ['ASD imbalance']
    ax = f.add_subplot(1,1,1)
    attr = class_attr.detach().numpy()
    attr = 100*attr/np.max(np.abs(attr))
    if idx is None:
        idx = np.abs(attr) >= 0
    #elif np.max(np.abs(attr[np.logical_not(idx)])) >= 1e1:
    #    print(f'Warning: You missed some ROIs ({np.max(np.abs(attr[np.logical_not(idx)]))}) - {np.where(np.abs(attr[np.logical_not(idx)])>=5e-1)}')
    attr = attr[idx]
    n_feat = len(attr)
    feat = np.arange(n_feat)
    if n_feat <= 40:
        ax.bar(feat, attr, width = 1.0, color = '0.9', edgecolor = '0.0', zorder = 0)
    else:
        ax.bar(feat, attr, width = 1.0, color = '0.9', zorder = 0)
    if not highlight is None:
        if n_feat <= 40:
            ax.bar(feat[highlight[idx]], attr[highlight[idx]], width = 1.0, color = 'C3', edgecolor = '0.0', zorder = 1)
        else:
            ax.bar(feat[highlight[idx]], attr[highlight[idx]], width = 1.0, color = 'C3', zorder = 1)
    max_y_index = np.argmax(attr)
    x_at_max = feat[max_y_index]
    if not highlight is None:
        if n_feat <= 40:
            legend_elements = [Patch(facecolor='C3',edgecolor = '0.0',label=clabel), Patch(facecolor='0.9', edgecolor = '0.0', label='Other ROIs (no imbalance)')]
        else:
            legend_elements = [Patch(facecolor='C3',label=clabel), Patch(facecolor='0.9', label='Other ROIs (no imbalance)')]
        ax.legend(handles=legend_elements, ncol = 1, columnspacing = 1.0)
    ax.set_xlabel(xlabel)
    #for x in [-1e2, -1e1, -1e0, 0, 1e0, 1e1, 1e2]:
    #    ax.axhline(x, 0, 1, color = '0.95', linestyle = '-', zorder = -1)
    #ax.set_yscale('symlog', linthresh=1e0)
    #ax.set_ylim([-10**2.5, 10**2.5])
    ax.set_ylim([-110, 110])
    ax.set_ylabel(f'{ylabel} Score (%)')
    if n_feat <= 40:
        xticks = np.arange(10*np.rint(len(feat)/10)+1)[::10]
    else:
        xticks = np.arange(100*np.rint(len(feat)/100)+1)[::100]
    ax.set_xticks(xticks)
    ax.set_xlim(-0.1*n_feat, n_feat*1.1)
    f.savefig(path, dpi = 600)
    return idx

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
    n_step = 100
    n_epoch = 150
    region = args.region[0]
    gen_data = gen_from_file
    Qi_nt = args.qint
    Qi_asd = args.qiasd
    noise = args.noise
    snr = args.snr
    nodes = {
        'PCC': [46,47],
        'Pcun': [50,51],
        'Ang': [14,15],
        'RSC': [155, 156, 157, 368, 369, 370],
        'Cg': [1,2,214,215],
        'PrL': [129,342]
    }

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where the tvb model is saved
    data_path = f'{os.environ.get("OAK")}/projects/tanghiem/2022_TVB_AdEX/feature_attribution'
    if region in ['RSC', 'RSC_Cg', 'RSC_Cg_PrL']:
        n_feat = 426
        #params = [{'files': f'{data_path}/sig_BOLD__b_0_Qi_RSC{Qi:.1f}_repeatedAIstim_0.0EtoEIratio1.4_coupling0.15seed*.npy'} for Qi in [Qi_asd,Qi_nt]]
        params = [{'files': f'{data_path}/sig_BOLD__b_0_Qi_{region}_nodes{Qi:.1f}_repeatedAId_rightstim_0.0EtoEIratio1.4_coupling0.15seed*_noise{noise:.1e}.npy', 'transform':lambda x: add_measurement_noise(x, snr = snr)} for Qi in [Qi_asd,Qi_nt]]
    elif region in ['PCC', 'PCC_Pcun', 'PCC_Pcun_Ang']:
        n_feat = 68
        #params = [{'files': f'{data_path}/sig_BOLD__b_0_Qi_PCC{Qi:.1f}EtoEIratio1.4_coupling0.15seed*.npy'} for Qi in [Qi_asd,Qi_nt]]
        params = [{'files': f'{data_path}/sig_BOLD__b_0_Qi_{region}{Qi:.1f}_repeatedinsulastim_0.0EtoEIratio1.4_coupling0.15seed*_noise{noise:.1e}.npy', 'transform':lambda x: add_measurement_noise(x, snr = snr)} for Qi in [Qi_asd,Qi_nt]]
    else:
        raise NameError(f'Not implemented when E/I imbalance is in {region}')
    print(params)
    affected = {k: np.isin(np.arange(n_feat), sum([nodes[r] for r in region.split('_')], start = [])) for k in range(n_class)}
    # path where model are saved
    model_name = f'tvb_{region}_asdQi_{Qi_asd:.1f}_ntQi_{Qi_nt:.1f}_noise{noise:.1e}_snr{snr:.2f}'
    model_path = f'{os.environ.get("DATA_PATH")}/tvb/model/{model_name}'

    # path where attribution are saved
    attr_path = f'{os.environ.get("DATA_PATH")}/tvb/attr/{model_name}'
    os.makedirs(f'{attr_path}', exist_ok=True)
    # path where log of training are saved
    log_path = f'{os.environ.get("TMP_PATH")}/tvb/log/test'
    # path where model are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/tvb/{model_name}'
    os.makedirs(f'{figure_path}', exist_ok=True)

    # -------------------------------
    # Figure Parameters
    # -------------------------------
    
    c_affected = '0.0'
    c_not_affected = '0.8'
    highlight = affected
    hypothesis = [(affected, c_affected, None)]
    group_name = ['ASD', 'NT']

    # -------------------------------
    # Feature attribution methods
    # -------------------------------

    methods = [
        ('Integrated Gradients', 'ig', IntegratedGradients, {}),
        #('Saliency', 'saliency', Saliency, {}),
        ('DeepLift', 'deeplift', DeepLift, {}),
        #('DeepLiftShap', 'deepliftshap', DeepLiftShap, {'baselines': torch.zeros((np.int64(n_sample*n_class*0.2), n_step, n_feat))}),
        #('GradientShap', 'gradshap', GradientShap, {'baselines': torch.zeros((np.int64(n_sample*n_class*0.2), n_step, n_feat))}),
        ('Input X Gradient', 'ixg', InputXGradient, {}),
        ('Guided Backprop', 'guidedbp', GuidedBackprop, {}),
        #('Guided GradCAM', 'guidedgc', GuidedGradCam, {}), # need layer to be defined
        ('Deconvolution', 'deconvolution', Deconvolution, {}),
        #('Feature Ablation', 'ablation', FeatureAblation, {}),
        #('Occlusion', 'occlusion', Occlusion, {}), need sliding_window_shapes to be defined
        #('Feature Permutation', 'permutation', FeaturePermutation, {}),
        #('Shapley Value Sampling', 'shapley', ShapleyValueSampling, {}), # SLOW
        #('Lime', 'lime', Lime, {}), # SLOW
        # ('KernelShap', 'kernelshap', KernelShap, {}), SLOW
        # ('LRP', 'lrp', LRP, {}),
    ]

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
    # Test dataset
    # -------------------------------

    dataset = NormalizeDataset(FakeFmriDataSet(gen_data, params))
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.label)
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False, num_workers = 4, pin_memory = True)

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

    model = GenericModel(model, metrics = {'acc': Accuracy(task="multiclass", num_classes=2, top_k = 1)})
    trainer = pl.Trainer(default_root_dir=log_path, deterministic=True, devices="auto", accelerator="auto")

    # -------------------------------
    # Testing model
    # -------------------------------

    #x, label = train_dataset[:len(train_dataset)]
    x, label = test_dataset[:len(test_dataset)]
    x.requires_grad = True

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
            #attr = method.attribute(x, target=label, **kwargs) # TODO Why not symmetric?
            attr = method.attribute(x, target=0*label, **kwargs) # TODO Why not symmetric?
            class_attr = {}
            
            for k in range(n_class):
                idx = label == k
                #class_attr[k] = torch.mean(torch.median(attr[idx], axis = 2).values, axis = 0)
                class_attr[k] = torch.mean(torch.abs(torch.median(attr[idx], axis = 2).values), axis = 0)
                print(f'prominent node for class {k} (top 20): {list(np.argsort(class_attr[k].detach().numpy())[::-1][:20])}')
                print(f'prominent node for class {k} (50%): {list(np.where(class_attr[k]>0.5*class_attr[k].max())[0])}')
                print(f'prominent node for class {k} (10%): {list(np.where(class_attr[k]>0.1*class_attr[k].max())[0])}')
                #print(f'mean attribution for class {k}:')
                #pprint(class_attr[k])
                """
                if k == 0:
                    _idx = np.argsort(class_attr[k].detach().numpy())[::-1]
                    _highlight = np.where(highlight[k])
                    _idx = np.concatenate([_idx[np.isin(_idx, _highlight)], _idx[np.logical_not(np.isin(_idx, _highlight))]])
                    idx_plot = plot_attr(f'{figure_path}/single_attr_{tag}_{group_name[k]}.png', class_attr[k][_idx], name, c = c_not_affected, label = 'ROI not affected', highlight = highlight[k][_idx], xlabel = 'ROIs', ylabel = 'IG', clabel = 'RSC (E/I imbalance)')
                    idx_plot = plot_attr(f'{figure_path}/single_attr_{tag}_{group_name[k]}.pdf', class_attr[k][_idx], name, c = c_not_affected, label = 'ROI not affected', highlight = highlight[k][_idx], xlabel = 'ROIs', ylabel = 'IG', clabel = 'RSC (E/I imbalance)')
                else:
                    plot_attr(f'{figure_path}/single_attr_{tag}_{group_name[k]}.png', class_attr[k][_idx], name, c = c_not_affected, label = 'ROI not affected', highlight = highlight[k][_idx], xlabel = 'ROIs', ylabel = 'IG', idx = idx_plot, clabel = 'RSC (no imbalance)')
                    plot_attr(f'{figure_path}/single_attr_{tag}_{group_name[k]}.pdf', class_attr[k][_idx], name, c = c_not_affected, label = 'ROI not affected', highlight = highlight[k][_idx], xlabel = 'ROIs', ylabel = 'IG', idx = idx_plot, clabel = 'RSC (no imbalance)')
                """
            os.makedirs(f'{attr_path}/{tag}', exist_ok=True)
            torch.save(torch.stack(list(class_attr.values()), axis = 0), f'{attr_path}/{tag}/{model_name}_epoch{epoch:02}.pt')
            print('--------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test model at different epochs to discriminate simulated fMRI')
    parser.add_argument('--redo', action='store_true', help='If set, resave even if file exists')
    parser.add_argument('--epochs', metavar = 'E', type = int, nargs = "+", help = 'list of epochs to test')
    parser.add_argument('--region', metavar = 'R', type = str, nargs = 1, default = ['RSC'], help = 'Region(s) affected')
    parser.add_argument('--noise', metavar = 'R', type = float, default = 0.001, help = 'Amplitude of noise')
    parser.add_argument('--qiasd', metavar = 'R', type = float, default = 4.5, help = 'Region(s) affected')
    parser.add_argument('--qint', metavar = 'R', type = float, default = 5.0, help = 'Region(s) affected')
    parser.add_argument('--snr', metavar = 'N', type = float, default = 0.0, help = 'SNR in dB')
    args = parser.parse_args()
    main(args)