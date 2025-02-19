import os, sys, argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from figures.plots import letter, get_figsize, plot, plot_multiple, imshow
import numpy as np
import torch
import sklearn.metrics as met
import glob
import h5py
import pandas as pd

def plot_bar(f, gs, class_attr, highlight = None, legend = False, title = '', test_threshold = None, ylim = None, showlegend = False, max_roi = 14, roi_names = None, path = None, hline = None, tick_only_highlight = False, name_score = ''):
    n_feat = len(class_attr)
    feat = np.arange(n_feat)
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title, fontsize = 10, fontweight = 'bold')
    class_attr = np.abs(class_attr)
    attr = 100*class_attr/np.max(class_attr)
    _idx = np.argsort(attr)[::-1]
    attr = attr[_idx]
    ax.bar(feat[:max_roi], attr[:max_roi], width = 1.0, color = '0.8', zorder = 0)
    if not highlight is None:
        for i, (idx,_c,_label) in enumerate(highlight):
            n_highlight = np.sum(idx)
            idx = np.where(idx[_idx])[0]
            good_idx = idx[idx<max_roi]
            bad_idx = idx[idx>=max_roi]
            ax.bar(feat[good_idx][:max_roi], attr[good_idx][:max_roi], label = _label, width = 1.0, color = [_c]*n_highlight, zorder = i+1)
    ax.set_ylabel(f'{name_score} Score (%)')
    if not ylim is None:
        ax.set_ylim(ylim)
    if not roi_names is None:
        ax.set_xticks(np.arange(len(feat[:max_roi])))
        labels = roi_names[_idx][:max_roi]
        idx = np.where(labels == 'None')[0]
        labels[idx] = [f'' for i in range(len(idx))] if tick_only_highlight else [f'{i:d}' for i in range(len(idx))]
        ax.set_xticklabels(labels, rotation = 45, ha = 'right')
    else:
        ax.set_xlabel('ROIs')
    if not hline is None:
        ax.axhline(hline, 0, 1, linestyle = '--', c = '0.0', zorder = i+2)
    return ax

def f1_score(class_attr, thresholds, class_rois):
    f1_score = []
    for attr, rois_true in zip(class_attr, class_rois):
        f1_score.append(np.zeros_like(thresholds))
        attr = 100*attr/np.max(attr)
        for i, threshold in enumerate(thresholds):
            rois_pred = attr >= threshold
            f1_score[-1][i] = met.f1_score(rois_true, rois_pred)
    return f1_score

def find_attr(file):
    l = glob.glob(file)
    return torch.load(l[0]).detach().numpy()

def rename(a, d):
    f = np.vectorize(lambda k: d[k] if k in d.keys() else 'None')
    return f(a)

def main(args):

    seed = 0
    np.random.seed(seed)

    # -------------------------------
    # Data parameters
    # -------------------------------

    n_class = 2
    n_sample = 400 # number of samples used in training per class
    n_feat = 100
    n_step = 100
    n_epoch = 150
    epoch = 10

    # -------------------------------
    # Figure parameters
    # -------------------------------

    zoom = 10
    ws = np.array([2,2])
    wspace = 1
    hs = np.array([1,1])
    hspace = 0.75

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where attribution are saved
    attr_path = f'{os.environ.get("DATA_PATH")}/tvb/attr'
    attr_path_data = f'{os.environ.get("DATA_PATH")}/data/attr'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper'
    os.makedirs(figure_path, exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------

    methods = ['ig', 'deeplift', 'ixg', 'guidedbp']
    class_attr_data = {method: find_attr(f'{attr_path_data}/{method}/Whole_model_data_abide_brainnnetome_6_wmcsf.pt') for method in methods}
    threshold_data = {method: 0.5*np.max(np.abs(class_attr_data[method][1])) for method in methods}
    roi_name_data = pd.read_csv(f'{os.environ.get("DATA_PATH")}/data/subregion_func_network_Yeo_yz.csv')
    roi_name_data = (roi_name_data['subregion_name']+' '+roi_name_data['region']).to_numpy()
    roi_name_data = rename(roi_name_data, {"A23d CG_L_7_1": "PCC-L", "A23d CG_R_7_1": "PCC-R", "A31 PCun_R_4_4": "PCun-R"})
    highlight_data = {method:[((np.abs(class_attr_data[method][1])>=threshold_data[method]), 'C3', 'higher attribution')] for method in methods}

    # -------------------------------
    # Display
    # -------------------------------

    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))

    ylim = 0,110
    ax_A = letter('A',plot_bar)(f, gs[0,0], class_attr_data['ig'][1], highlight_data['ig'], legend = True, title = f'Integrated Gradient (IG)', ylim = ylim, showlegend=True, roi_names = roi_name_data, hline = 50, name_score = 'IG')
    ax_B = letter('B',plot_bar)(f, gs[0,1], class_attr_data['deeplift'][1], highlight_data['deeplift'], legend = True, title = f'DeepLIFT (DLIFT)', ylim = ylim, showlegend=True, roi_names = roi_name_data, hline = 50, name_score = 'DLIFT')
    ax_C = letter('C',plot_bar)(f, gs[1,0], class_attr_data['ixg'][1], highlight_data['ixg'], legend = True, title = f'Input X Gradient (IXG)', ylim = ylim, showlegend=True, roi_names = roi_name_data, hline = 50, name_score = 'IXG')
    ax_D = letter('D',plot_bar)(f, gs[1,1], class_attr_data['guidedbp'][1], highlight_data['ixg'], legend = True, title = f'Guided Backprop (GB)', ylim = ylim, showlegend=True, roi_names = roi_name_data, hline = 50, name_score = 'GB', max_roi = 200, tick_only_highlight = True)
    f.savefig(f'{figure_path}/figure7.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 7')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--permutation', action='store_true')
    args = parser.parse_args()
    main(args)
