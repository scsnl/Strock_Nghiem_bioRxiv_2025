import os
import glob
from pathlib import PurePath
import numpy as np
from scipy.signal import convolve
import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm

# -------------------------------
# Filters
# -------------------------------

def filt(x, window = 'hamming', length = 10, mode = 'same', axis = -1):
    n_axis = len(x.shape)
    if axis == -1:
        axis = n_axis-1
    w = eval(f'np.{window}(length)[{",".join(["None"]*axis+[":"]+["None"]*(n_axis-axis-1))}]')
    return np.float32(convolve(x, w/w.sum(), mode=mode))

def whitten(x):
    pca = PCA(n_components=x.shape[1])
    for i in range(x.shape[2]):
        x[:,:,i] = pca.fit_transform(x[:,:,i])
    return x

def add_measurement_noise(x, snr):
    noise_std = np.sqrt(np.max(np.var(x, axis = 2)) * (10**(-snr/10)))
    return np.float32(x + np.random.normal(loc = 0.0, scale = noise_std, size = x.shape))

def hrf_filt(x, hrf, snr = 0.0):
    y = np.apply_along_axis(lambda m: np.convolve(m, hrf, mode='same'), axis=2, arr=x)
    return add_measurement_noise(y, snr = snr)

# -------------------------------
# Generation of data choices
# -------------------------------

def gen_rnn_from_adjacency(n_sample, n_step, A, radius = 1.0, noise = 0.0, warmup = 0, f = np.tanh, transform = None):
    n_feat = A.shape[0]
    if not radius is None:
        _radius = max(np.abs(np.linalg.eigvals(A)))
        A = radius*A/_radius
    x = np.empty(shape = (n_sample, n_feat, warmup + n_step), dtype = np.float32)
    x[:,:,0] = np.random.normal(loc = 0.0, scale = 1.0, size=(n_sample, n_feat))
    for i in range(1, n_step + warmup):
        w = np.random.normal(loc = 0.0, scale = noise, size = (n_feat, n_sample)) if np.isscalar(noise) else np.random.multivariate_normal(mean = np.zeros(n_feat), cov = noise, size = n_sample)
        x[:,:,i] = f(A@x[:, :, i-1].T + w).T
    return x[:,:,warmup:] if transform is None else transform(x[:,:,warmup:])

def gen_from_file(files, transform = None):
    print(files)
    files = sorted(glob.glob(files))
    _x = np.load(files[0])
    n_sample = len(files)
    n_feat, n_step = _x.shape
    x = np.zeros(shape = (n_sample, n_feat, n_step), dtype = np.float32)
    for i,f in enumerate(files):
        _x = np.load(f)
        _n_step = min(n_step, _x.shape[1])
        x[i,:,:_n_step] = _x[:,:_n_step]
    return x if transform is None else transform(x)

# -------------------------------
# Main generation of data
# -------------------------------

def gen_all_data(gen_data, params):
    x = []
    label = []
    for i, param in enumerate(params):
        _x = gen_data(**param)
        x.append(_x)
        label.append(np.full((len(_x),), i, dtype = np.int64))
    return np.concatenate(x), np.concatenate(label)


# -------------------------------
# Interface with pytorch
# -------------------------------

no_transform = lambda x: x

class FakeFmriDataSet(Dataset):

    def __init__(self, gen_data, params, transform = no_transform, transformlabel = no_transform):
        self.transform = transform
        self.transformlabel = transformlabel
        self.params = params
        x, label = gen_all_data(gen_data, params)
        self.x, self.label = torch.from_numpy(x), torch.from_numpy(label)
        self.size = len(self.label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.transform(self.x[idx])
        label = self.transformlabel(self.label[idx])
        return  x, label

class FmriDataSet(Dataset):

    def __init__(self, path, convert = no_transform, transform = no_transform, transformlabel = no_transform):
        self.transform = transform
        self.transformlabel = transformlabel
        x, label = convert(np.load(path))
        self.x, self.label = torch.from_numpy(x), torch.from_numpy(label)
        self.size = len(self.label)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.transform(self.x[idx])
        label = self.transformlabel(self.label[idx])
        return  x, label