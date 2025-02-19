import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, Dataset
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

from nn_modeling.utils.memory import print_gpumem, print_cpumem


from sklearn.preprocessing import OneHotEncoder


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(246, 256, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv1d(8, 8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1))
        self.drop_out = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(512 + 16, 2)
        self.sig = nn.Sigmoid()

    def forward(self, x, covdataenc):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.mean(axis=2)
        out = self.drop_out(out)
        out = torch.cat((out, covdataenc), axis=1)
        out = self.fc1(out)
        out = self.sig(out)
        return out

class ABIDE(Dataset):

    def __init__(self):
        
        dataSetName = 'abide'  # abide, stanford
        dataSetId = dataSetName + '_brainnnetome_6_wmcsf'
        USE_FD_THRESH = True
        FD_THRESH = 0.5  # movement threshold
        USE_NYU_ONLY = False  # option available only for ABIDE
        FIRSTVOLS_TO_EXCLUDE = 6
        
        # Load numpy file for the data
        data_dir = '/oak/stanford/groups/menon/projects/sryali/2019_DNN/scripts/daelsaid/output/group/'
        datao = np.load(data_dir + dataSetId + '.npz')

        data = datao['data'][:, FIRSTVOLS_TO_EXCLUDE:, :]
        labels = datao['labels'].astype('int64')
        labels_numpy = labels - 1
        subjids = datao['subjids'].astype('int32')

        ## Get covariate data
        MISSING_VAL = -9999
        covenc = OneHotEncoder()

        sitesdata = datao['sites']
        sitedataenc = covenc.fit_transform(sitesdata.reshape(-1, 1))
        A = sitedataenc
        A2 = A.A
        sitedataenc = np.repeat(A2[:, :, np.newaxis], np.shape(data)[1], axis=2)

        sexdata = datao['genders']
        sexdataenc = covenc.fit_transform(sexdata.reshape(-1, 1))
        A = sexdataenc
        A2 = A.A
        sexdataenc = np.repeat(A2[:, :, np.newaxis], np.shape(data)[1], axis=2)

        fddata = datao['mean_fds'].astype('float64')
        fddata1 = np.repeat(fddata[:, np.newaxis], 1, axis=1)
        fddataenc = np.repeat(fddata1[:, :, np.newaxis], np.shape(data)[1], axis=2)

        covdataenc = np.concatenate([sitedataenc, sexdataenc], axis=1)

        ## Get brain data
        no_subjs, no_ts, no_channels = data.shape
        data_reshape = np.empty((no_subjs, no_channels, no_ts))
        for subj in np.arange(no_subjs):
            x_subj = data[subj, :, :]
            x_subj = np.transpose(x_subj)
            data_reshape[subj, :, :] = x_subj

        if USE_FD_THRESH:
            selix = np.squeeze(np.argwhere(fddata < FD_THRESH))
            data_reshape = data_reshape[selix, :, :]
            covdataenc = covdataenc[selix, :, :]
            labels_numpy = labels_numpy[selix]
            labels = labels[selix]

        self.x = torch.from_numpy(data_reshape).to(torch.float32)
        self.cov = torch.from_numpy(covdataenc).to(torch.float32)
        self.y = torch.from_numpy(labels_numpy)
        self.size = len(self.x)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.x[idx], self.cov[idx,:,0]
        label = self.y[idx]
        return  x, label

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

def main(args):

    seed = 0
    pl.seed_everything(seed)


    # -------------------------------
    # Parameters
    # -------------------------------

    n_class = 2
    c = '0.8'

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where model are saved
    model_name = 'Whole_model_data_abide_brainnnetome_6_wmcsf'
    model_path = f'/oak/stanford/groups/menon/projects/ksupekar/2021_ASD_NN/scripts/restfmri/classify/CNN1dPyTorch/trained_models/Whole_model_data_abide_brainnnetome_6_wmcsf.pt'
    # path where attribution are saved
    attr_path = f'{os.environ.get("DATA_PATH")}/data/attr'
    # path where log of training are saved
    log_path = f'{os.environ.get("TMP_PATH")}/data/log/test'
    # path where model are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/data/{model_name}'
    os.makedirs(f'{figure_path}', exist_ok=True)
    

    # -------------------------------
    # Test dataset
    # -------------------------------

    test_dataset = ABIDE()
    test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False, num_workers = 4, pin_memory = True)

    # -------------------------------
    # Loading model
    # -------------------------------

    model = ConvNet()
    trainer = pl.Trainer(default_root_dir=log_path, deterministic=True, devices="auto", accelerator="auto")
    #checkpoint = torch.load(f'{model_path}', map_location = torch.device('cpu'))
    checkpoint = torch.load(f'{model_path}')
    model.load_state_dict(checkpoint)
    model = GenericModel(model)

    # -------------------------------
    # Testing model
    # -------------------------------

    #x, label = train_dataset[:len(train_dataset)]
    idx = np.random.permutation(len(test_dataset))[:len(test_dataset)//5]
    (x,cov), label = test_dataset[idx]
    print(colored(np.unique(label, return_counts = True), 'red'))
    x.requires_grad = True
    cov.requires_grad = True

    #print_cpumem()
    #print_gpumem()
    #breakpoint()

    # -------------------------------
	# Feature attribution methods
	# -------------------------------

    methods = [
        ('Integrated Gradients', 'ig', IntegratedGradients, {}),
        ('Saliency', 'saliency', Saliency, {}),
        ('DeepLift', 'deeplift', DeepLift, {}),
        #('DeepLiftShap', 'deepliftshap', DeepLiftShap, {'baselines': (torch.zeros_like(x), torch.zeros_like(cov))}), # Try again
        #('GradientShap', 'gradshap', GradientShap, {'baselines': (torch.zeros_like(x), torch.zeros_like(cov))}), # Try again
        ('Input X Gradient', 'ixg', InputXGradient, {}),
        ('Guided Backprop', 'guidedbp', GuidedBackprop, {}),
        #('Guided GradCAM', 'guidedgc', GuidedGradCam, {}), # need layer to be defined
        ('Deconvolution', 'deconvolution', Deconvolution, {}),
        #('Feature Ablation', 'ablation', FeatureAblation, {}),
        #('Occlusion', 'occlusion', Occlusion, {}), # need sliding_window_shapes to be defined
        #('Feature Permutation', 'permutation', FeaturePermutation, {}),
        #('Shapley Value Sampling', 'shapley', ShapleyValueSampling, {}), # SLOW
        #('Lime', 'lime', Lime, {}), # SLOW
        # ('KernelShap', 'kernelshap', KernelShap, {}), SLOW
        # ('LRP', 'lrp', LRP, {}),
    ]

    _methods = []

    if not args.redo:
        for name, tag, method, kwargs in methods:
            #files = [f'{attr_path}/{tag}/{model_name}.pt']
            #if not np.all([os.path.exists(path) for path in files]):
                _methods.append((name, tag, method, kwargs))
        methods = _methods
    if len(methods)==0:
        print(colored(f'{model_name} at epochs {args.epochs} already tested', 'green'))
        sys.exit(0)
    else:
        _methods = ', '.join([m for _,m,_,_ in methods])
        print(colored(f'Testing {model_name} at epochs {args.epochs} ({_methods})', 'green'))
    
    # -------------------------------
    #  Computing feature attribution
    # -------------------------------
    print('--------------------')
    for name, tag, method, kwargs in methods:
        print(f'{name}')
        method = method(model.model)
        attr = method.attribute((x,cov), target=label, **kwargs)
        class_attr = {}
        for k in range(n_class):
            idx = label == k
            class_attr[k] = torch.mean(torch.abs(torch.median(attr[0][idx], axis = 2).values), axis = 0)
            print(f'mean attribution for class {k}:')
            pprint(class_attr[k])
        plot_attr(f'{figure_path}/attr_{tag}.png', class_attr, name, c = c, xlabel = 'Features/ROIs', ylabel = '')
        os.makedirs(f'{attr_path}/{tag}', exist_ok=True)
        torch.save(torch.stack(list(class_attr.values()), axis = 0), f'{attr_path}/{tag}/{model_name}.pt')
        print('--------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Test model at different epochs to discriminate simulated fMRI')
    parser.add_argument('--redo', action='store_true', help='If set, resave even if file exists')
    args = parser.parse_args()
    main(args)