import os, sys, argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from figures.plots import letter, get_figsize, plot, plot_multiple, imshow
import numpy as np
import torch
import sklearn.metrics as met
import glob
from matplotlib import rc
#rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
import pandas as pd
import re

def plot_bar(f, gs, class_attr, highlight = None, legend = False, title = '', test_threshold = None, ylim = None, showlegend = False, max_roi = 10, roi_names = None, path = None, hline = None, titlebrain = '', subtitle = None):
    n_feat = len(class_attr)
    feat = np.arange(n_feat)
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_title(title + ('' if subtitle is None else '\n\n'), usetex = True)
    if not subtitle is None:
        ax.text(0.5, 1.1, subtitle, color = '0.6', transform = ax.transAxes, ha = 'center', va = 'bottom', fontstyle='italic')
    class_attr = np.abs(class_attr)
    attr = 100*class_attr/np.max(class_attr)
    _idx = np.argsort(attr)[::-1]
    attr = attr[_idx]
    ax.bar(feat[:max_roi], attr[:max_roi], width = 1.0, color = '0.8', zorder = 0)
    if not highlight is None:
        for i, (idx,_c,_label) in enumerate(highlight):
            n_highlight = np.sum(idx)
            idx = idx[_idx]
            ax.bar(feat[idx][:max_roi], attr[idx][:max_roi], label = _label, width = 1.0, color = [_c]*n_highlight, zorder = i+1)
    ax.set_ylabel('IG Score (%)')
    if not ylim is None:
        ax.set_ylim(ylim)
    if not roi_names is None:
        ax.set_xticks(np.arange(len(feat[:max_roi])))
        labels = roi_names[_idx][:max_roi]
        idx = np.where(labels == 'None')[0]
        labels[idx] = [f'{i:d}' for i in range(len(idx))]
        ax.set_xticklabels(labels, rotation = 45, ha = 'right')
    else:
        ax.set_xlabel('ROIs')
    if not hline is None:
        ax.axhline(hline, 0, 1, linestyle = '--', c = '0.0', zorder = i+2)
    if not path is None:
        ax2 = ax.inset_axes([0.5, 0.4, 0.5, 0.5])
        ax2.patch.set_alpha(1.0)
        img = plt.imread(path)
        ax2.imshow(img)
        ax2.text(0.5, 1, titlebrain, va = 'bottom', ha = 'center', transform = ax2.transAxes, color = 'C3', usetex = True)
        ax2.axis('off')
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
    if len(l) == 0:
        print(file)
    return torch.load(l[0]).detach().numpy()

def find_accuracy(model, acc_path, attr_path):
    l = glob.glob(f'{attr_path}/{model}/ig/{model}_epoch*.pt')
    epoch = int(re.findall("epoch([0-9]+)", l[0])[0])
    acc = torch.load(f'{acc_path}/{model}/test_accuracy.npy').numpy()
    return acc[epoch+1]

def scores(attr, groundtruth):
    return met.f1_score(groundtruth, attr), met.precision_score(groundtruth, attr), met.recall_score(groundtruth, attr)

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
    nodes = {
        'PCC': [46,47],
        'Pcun': [50,51],
        'Ang': [14,15],
        'RSC': [155, 156, 157, 368, 369, 370]
    }
    qint = 5.0

    # -------------------------------
    # Figure parameters
    # -------------------------------

    zoom = 15
    ws = np.array([2,2,2])
    wspace = 1
    hs = np.array([1,1,1])
    hspace = 1

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, left = 1.5)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where accuracies are saved
    acc_path = f'{os.environ.get("DATA_PATH")}/tvb/model'
    # path where attribution are saved
    attr_path = f'{os.environ.get("DATA_PATH")}/tvb/attr'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper'
    os.makedirs(figure_path, exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------

    method = args.m[0]
    region1 = 'PCC'
    qiasd1 = 4.5
    snr1 = [10, 0, -10]
    class_attr_snr = [find_attr(f'{attr_path}/tvb_{region1}_asdQi_{qiasd1:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr:.2f}/ig/tvb_{region1}_asdQi_{qiasd1:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr:.2f}_epoch*.pt') for snr in snr1]
    acc_snr = [find_accuracy(f'tvb_{region1}_asdQi_{qiasd1:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr:.2f}', acc_path, attr_path) for snr in snr1]

    snr2 = 0
    qiasd2 = 4.5
    region2 = ['PCC', 'PCC_Pcun', 'PCC_Pcun_Ang']
    class_attr_aff = [find_attr(f'{attr_path}/tvb_{region}_asdQi_{qiasd2:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr2:.2f}/ig/tvb_{region}_asdQi_{qiasd2:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr2:.2f}_epoch*.pt') for region in region2]
    acc_aff = [find_accuracy(f'tvb_{region}_asdQi_{qiasd2:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr2:.2f}', acc_path, attr_path) for region in region2]

    snr3 = 0
    region3 = 'PCC'
    qiasd3 = [4.5, 4.6, 4.7]
    class_attr_delta = [find_attr(f'{attr_path}/tvb_{region3}_asdQi_{qiasd:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr3:.2f}/ig/tvb_{region3}_asdQi_{qiasd:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr3:.2f}_epoch*.pt') for qiasd in  qiasd3]
    acc_delta = [find_accuracy(f'tvb_{region3}_asdQi_{qiasd:.1f}_ntQi_{qint:.1f}_noise1.0e-04_snr{snr3:.2f}', acc_path, attr_path) for qiasd in  qiasd3]

    highlight_snr = [[((np.isin(np.arange(68), sum([nodes[r] for r in region.split('_')], start = []))), 'C3', 'affected ROIs')] for region in [region1]*3]
    highlight_aff = [[((np.isin(np.arange(68), sum([nodes[r] for r in region.split('_')], start = []))), 'C3', 'affected ROIs')] for region in region2]
    highlight_delta = [[((np.isin(np.arange(68), sum([nodes[r] for r in region.split('_')], start = []))), 'C3', 'affected ROIs')] for region in [region3]*3]

    performance_snr = [scores(attr[1]>=0.5*attr[1].max(),groundtruth[0][0]) for attr, groundtruth in zip(class_attr_snr, highlight_snr)]
    performance_aff = [scores(attr[1]>=0.5*attr[1].max(),groundtruth[0][0]) for attr, groundtruth in zip(class_attr_aff, highlight_aff)]
    performance_delta = [scores(attr[1]>=0.5*attr[1].max(),groundtruth[0][0]) for attr, groundtruth in zip(class_attr_delta, highlight_delta)]

    roi_names = np.array(list(pd.read_csv(f'{os.environ.get("DATA_PATH")}/tvb/centres.txt', index_col = 0, sep = '\t', header = None).index))
    roi_names = rename(roi_names, {"posteriorcingulate_R":"PCC-R", "posteriorcingulate_L":"PCC-L", "precuneus_R": "PCun-R", "precuneus_L": "PCun-L", "inferiorparietal_R": "AnG-R", "inferiorparietal_L": "AnG-R"})
    region2 = [r.replace('Pcun', 'PCun') for r in region2]

    # -------------------------------
    # Display
    # -------------------------------

    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))

    ylim = 0,110
    ax_A = letter('A',plot_bar)(f, gs[0,0], class_attr_snr[0][1], highlight_snr[0], legend = True, path = f'{figure_path}/tvb/tvb_pcc.png', titlebrain = region1.replace('_', ' + '), roi_names = roi_names, title = f'$\Delta Q_i$ = {qint-qiasd1:.1f}nS, \\textbf{{SNR = {snr1[0]:d}dB}}', subtitle = f'acc = {acc_snr[0]:.0%}, f1 = {performance_snr[0][0]:.0%}, p = {performance_snr[0][1]:.0%}, r = {performance_snr[0][2]:.0%}')
    ax_B = letter('B',plot_bar)(f, gs[0,1], class_attr_snr[1][1], highlight_snr[1], legend = True, path = f'{figure_path}/tvb/tvb_pcc.png', titlebrain = region1.replace('_', ' + '), roi_names = roi_names, title = f'$\Delta Q_i$ = {qint-qiasd1:.1f}nS, \\textbf{{SNR = {snr1[1]:d}dB}}', subtitle = f'acc = {acc_snr[1]:.0%}, f1 = {performance_snr[1][0]:.0%}, p = {performance_snr[1][1]:.0%}, r = {performance_snr[1][2]:.0%}')
    ax_C = letter('C',plot_bar)(f, gs[0,2], class_attr_snr[2][1], highlight_snr[2], legend = True, path = f'{figure_path}/tvb/tvb_pcc.png', titlebrain = region1.replace('_', ' + '), roi_names = roi_names, title = f'$\Delta Q_i$ = {qint-qiasd1:.1f}nS, \\textbf{{SNR = {snr1[2]:d}dB}}', subtitle = f'acc = {acc_snr[2]:.0%}, f1 = {performance_snr[2][0]:.0%}, p = {performance_snr[2][1]:.0%}, r = {performance_snr[2][2]:.0%}')

    ax_D = letter('D',plot_bar)(f, gs[1,0], class_attr_aff[0][1], highlight_aff[0], path = f'{figure_path}/tvb/tvb_pcc.png', titlebrain = f'\\textbf{{{region2[0].replace("_", " + ")}}}', roi_names = roi_names, title = f'SNR = {snr2:d}dB, $\Delta Q_i$ = 0.5nS', subtitle = f'acc = {acc_aff[0]:.0%}, f1 = {performance_aff[0][0]:.0%}, p = {performance_aff[0][1]:.0%}, r = {performance_aff[0][2]:.0%}')
    ax_E = letter('E',plot_bar)(f, gs[1,1], class_attr_aff[1][1], highlight_aff[1], path = f'{figure_path}/tvb/tvb_pcc_pcun.png', titlebrain = f'\\textbf{{{region2[1].replace("_", " + ")}}}', roi_names = roi_names, title = f'SNR = {snr2:d}dB, $\Delta Q_i$ = 0.5nS', subtitle = f'acc = {acc_aff[1]:.0%}, f1 = {performance_aff[1][0]:.0%}, p = {performance_aff[1][1]:.0%}, r = {performance_aff[1][2]:.0%}')
    ax_F = letter('F',plot_bar)(f, gs[1,2], class_attr_aff[2][1], highlight_aff[2], path = f'{figure_path}/tvb/tvb_pcc_pcun_ang.png', titlebrain = f'\\textbf{{{region2[2].replace("_", " + ")}}}', roi_names = roi_names, title = f'SNR = {snr2:d}dB, $\Delta Q_i$ = 0.5nS', subtitle = f'acc = {acc_aff[2]:.0%}, f1 = {performance_aff[2][0]:.0%}, p = {performance_aff[2][1]:.0%}, r = {performance_aff[2][2]:.0%}')

    ax_G = letter('G',plot_bar)(f, gs[2,0], class_attr_delta[0][1], highlight_delta[0], path = f'{figure_path}/tvb/tvb_pcc.png', titlebrain = region3.replace('_', ' + '), roi_names = roi_names, title = f'SNR = {snr3:d}dB, $\\boldsymbol{{\Delta Q_i}}$\\textbf{{ = {qint-qiasd3[0]:.1f}nS}}', subtitle = f'acc = {acc_delta[0]:.0%}, f1 = {performance_delta[0][0]:.0%}, p = {performance_delta[0][1]:.0%}, r = {performance_delta[0][2]:.0%}')
    ax_H = letter('H',plot_bar)(f, gs[2,1], class_attr_delta[1][1], highlight_delta[1], path = f'{figure_path}/tvb/tvb_pcc.png', titlebrain = region3.replace('_', ' + '), roi_names = roi_names, title = f'SNR = {snr3:d}dB, $\\boldsymbol{{\Delta Q_i}}$\\textbf{{ = {qint-qiasd3[1]:.1f}nS}}', subtitle = f'acc = {acc_delta[1]:.0%}, f1 = {performance_delta[1][0]:.0%}, p = {performance_delta[1][1]:.0%}, r = {performance_delta[1][2]:.0%}')
    ax_I = letter('I',plot_bar)(f, gs[2,2], class_attr_delta[2][1], highlight_delta[2], path = f'{figure_path}/tvb/tvb_pcc.png', titlebrain = region3.replace('_', ' + '), roi_names = roi_names, title = f'SNR = {snr3:d}dB, $\\boldsymbol{{\Delta Q_i}}$\\textbf{{ = {qint-qiasd3[2]:.1f}nS}}', subtitle = f'acc = {acc_delta[2]:.0%}, f1 = {performance_delta[2][0]:.0%}, p = {performance_delta[2][1]:.0%}, r = {performance_delta[2][2]:.0%}')

    if method == 'ig':
        f.savefig(f'{figure_path}/figure3.png', dpi = 600)
    else:
        f.savefig(f'{figure_path}/figure3_{method}.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 3 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--permutation', action='store_true')
    parser.add_argument('--m', metavar = 'M', type = str, nargs = 1, default = ['ig'])
    args = parser.parse_args()
    main(args)
