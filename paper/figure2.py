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
import re

def plot_bar(f, gs, class_attr, highlight = None, legend = False, title = '', test_threshold = None, ylim = None, showlegend = False, subtitle = None):
    n_feat = len(class_attr)
    feat = np.arange(n_feat)
    ax = f.add_subplot(gs)
    ax.spines[['right', 'top']].set_visible(False)
    print(title)
    ax.set_title(title + ('' if subtitle is None else '\n\n'), usetex = True)
    if not subtitle is None:
        ax.text(0.5, 1.1, subtitle, color = '0.6', transform = ax.transAxes, ha = 'center', va = 'bottom', fontstyle='italic')
    attr = 100*class_attr/np.max(class_attr)
    ax.bar(feat, attr, width = 1.0, color = '0.8', zorder = 0)
    if not highlight is None:
        for i, (idx,_c,_label) in enumerate(highlight):
            n_highlight = np.sum(idx)
            ax.bar(feat[idx], attr[idx], label = _label, width = 1.0, color = _c, zorder = i+1)
    ax.set_xlabel('ROIs')
    ax.set_ylabel('IG Score (%)')
    if not ylim is None:
        ax.set_ylim(ylim)
    ax2 = ax.inset_axes([0.6, 0.2, 0.35, 0.7])
    n = 16
    theta0 = np.pi/2
    theta = np.linspace(theta0, theta0+2*np.pi, n+1)[:-2]
    ax2.scatter(np.cos(theta), np.sin(theta), clip_on = False, color = ['C3']*n_highlight+['0.8']*(n-1-n_highlight), s = 150)
    for i,t in enumerate(theta):
        ax2.text(np.cos(t), np.sin(t), f'{i:d}', color = '1.0', va = 'center', ha = 'center', size = 8)
        for k in range(1,min(6, n-i-1)):
            t2 = theta[i+k]
            ax2.plot(np.cos([t,t2]), np.sin([t,t2]), color = '0.9', zorder = -1, lw = 0.5, alpha = 0.5)
    n = 3*n
    theta = np.linspace(theta0, theta0+2*np.pi, n+1)[-5:-2]
    ax2.scatter(np.cos(theta), np.sin(theta), clip_on = False, color = '0.8', marker = '.', s = 0.5)
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
    return torch.load(l[0]).detach().numpy()

def find_accuracy(model, acc_path, attr_path):
    l = glob.glob(f'{attr_path}/ig/{model}_epoch*.pt')
    epoch = int(re.findall("epoch([0-9]+)", l[0])[0])
    acc = torch.load(f'{acc_path}/{model}/test_accuracy.npy').numpy()
    return acc[epoch+1]

def scores(attr, groundtruth):
    return met.f1_score(groundtruth, attr), met.precision_score(groundtruth, attr), met.recall_score(groundtruth, attr)

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
    acc_path = f'{os.environ.get("DATA_PATH")}/rnn_exc/model'
    # path where attribution are saved
    attr_path = f'{os.environ.get("DATA_PATH")}/rnn_exc/attr'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper'
    os.makedirs(figure_path, exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------

    method = args.m[0]
    a1 = 5
    d1 = 0.3
    snr1 = [10, 0, -10]
    class_attr_snr = [find_attr(f'{attr_path}/ig/simple_n100_a{a1:d}_delta{d1:.2f}_snr{snr:.2f}_epoch*.pt') for snr in snr1]
    acc_snr = [find_accuracy(f'simple_n100_a{a1:d}_delta{d1:.2f}_snr{snr:.2f}', acc_path, attr_path) for snr in snr1]

    snr2 = 0
    d2 = 0.3
    a2 = [1, 5, 10]
    class_attr_aff = [find_attr(f'{attr_path}/ig/simple_n100_a{a:d}_delta{d2:.2f}_snr{snr2:.2f}_epoch*.pt') for a in a2]
    acc_aff = [find_accuracy(f'simple_n100_a{a:d}_delta{d2:.2f}_snr{snr2:.2f}', acc_path, attr_path) for a in a2]

    snr3 = 0
    a3 = 5
    d3 = [0.5, 0.3, 0.1]
    class_attr_delta = [find_attr(f'{attr_path}/ig/simple_n100_a{a3:d}_delta{d:.2f}_snr{snr3:.2f}_epoch*.pt') for d in d3]
    acc_delta = [find_accuracy(f'simple_n100_a{a3:d}_delta{d:.2f}_snr{snr3:.2f}', acc_path, attr_path) for d in d3]

    highlight_snr = [[((np.arange(100)<a1), 'C3', 'affected ROIs')]]*3
    highlight_aff = [[((np.arange(100)<a2[0]), 'C3', 'affected ROIs')], [((np.arange(100)<a2[1]), 'C3', 'affected ROIs')], [((np.arange(100)<a2[2]), 'C3', 'affected ROIs')]]
    highlight_delta = [[((np.arange(100)<a3), 'C3', 'affected ROIs')]]*3
    
    threshold = 0.5
    performance_snr = [scores(attr[1]>=0.5*attr[1].max(),groundtruth[0][0]) for attr, groundtruth in zip(class_attr_snr, highlight_snr)]
    performance_aff = [scores(attr[1]>=0.5*attr[1].max(),groundtruth[0][0]) for attr, groundtruth in zip(class_attr_aff, highlight_aff)]
    performance_delta = [scores(attr[1]>=0.5*attr[1].max(),groundtruth[0][0]) for attr, groundtruth in zip(class_attr_delta, highlight_delta)]

    # -------------------------------
    # Display
    # -------------------------------

    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))

    ylim = 0,110
    ax_A = letter('A',plot_bar)(f, gs[0,0], class_attr_snr[0][1], highlight_snr[0], legend = True, title = f'prevalence = {a1:d}\%, $\delta$ = {d1:.1f}, \\textbf{{SNR = {snr1[0]:d}dB}}', ylim = ylim, showlegend=True, subtitle = f'acc = {acc_snr[0]:.0%}, f1 = {performance_snr[0][0]:.0%}, p = {performance_snr[0][1]:.0%}, r = {performance_snr[0][2]:.0%}')
    ax_B = letter('B',plot_bar)(f, gs[0,1], class_attr_snr[1][1], highlight_snr[1], legend = True, title = f'prevalence = {a1:d}\%, $\delta$ = {d1:.1f}, \\textbf{{SNR = {snr1[1]:d}dB}}', ylim = ylim, showlegend=True, subtitle = f'acc = {acc_snr[1]:.0%}, f1 = {performance_snr[1][0]:.0%}, p = {performance_snr[1][1]:.0%}, r = {performance_snr[1][2]:.0%}')
    ax_C = letter('C',plot_bar)(f, gs[0,2], class_attr_snr[2][1], highlight_snr[2], legend = True, title = f'prevalence = {a1:d}\%, $\delta$ = {d1:.1f}, \\textbf{{SNR = {snr1[2]:d}dB}}', ylim = ylim, showlegend=True, subtitle = f'acc = {acc_snr[2]:.0%}, f1 = {performance_snr[2][0]:.0%}, p = {performance_snr[2][1]:.0%}, r = {performance_snr[2][2]:.0%}')
    
    ax_D = letter('D',plot_bar)(f, gs[1,0], class_attr_aff[0][1], highlight_aff[0], title = f'\\textbf{{prevalence = {a2[0]:d}\%}}, $\delta$ = {d2:.1f}, SNR = {snr2:d}dB', ylim = ylim, subtitle = f'acc = {acc_aff[0]:.0%}, f1 = {performance_aff[0][0]:.0%}, p = {performance_aff[0][1]:.0%}, r = {performance_aff[0][2]:.0%}')
    ax_E = letter('E',plot_bar)(f, gs[1,1], class_attr_aff[1][1], highlight_aff[1], title = f'\\textbf{{prevalence = {a2[1]:d}\%}}, $\delta$ = {d2:.1f}, SNR = {snr2:d}dB', ylim = ylim, subtitle = f'acc = {acc_aff[1]:.0%}, f1 = {performance_aff[1][0]:.0%}, p = {performance_aff[1][1]:.0%}, r = {performance_aff[1][2]:.0%}')
    ax_F = letter('F',plot_bar)(f, gs[1,2], class_attr_aff[2][1], highlight_aff[2], title = f'\\textbf{{prevalence = {a2[2]:d}\%}}, $\delta$ = {d2:.1f}, SNR = {snr2:d}dB', ylim = ylim, subtitle = f'acc = {acc_aff[2]:.0%}, f1 = {performance_aff[2][0]:.0%}, p = {performance_aff[2][1]:.0%}, r = {performance_aff[2][2]:.0%}')

    ax_G = letter('G',plot_bar)(f, gs[2,0], class_attr_delta[0][1], highlight_delta[0], title = f'prevalence = {a3:d}\%, $\\boldsymbol{{\delta}}$\\textbf{{ = {d3[0]:.1f}}}, SNR = {snr3:d}dB', ylim = ylim, subtitle = f'acc = {acc_delta[0]:.0%}, f1 = {performance_delta[0][0]:.0%}, p = {performance_delta[0][1]:.0%}, r = {performance_delta[0][2]:.0%}')
    ax_H = letter('H',plot_bar)(f, gs[2,1], class_attr_delta[1][1], highlight_delta[1], title = f'prevalence = {a3:d}\%, $\\boldsymbol{{\delta}}$\\textbf{{ = {d3[1]:.1f}}}, SNR = {snr3:d}dB', ylim = ylim, subtitle = f'acc = {acc_delta[1]:.0%}, f1 = {performance_delta[1][0]:.0%}, p = {performance_delta[1][1]:.0%}, r = {performance_delta[1][2]:.0%}')
    ax_I = letter('I',plot_bar)(f, gs[2,2], class_attr_delta[2][1], highlight_delta[2], title = f'prevalence = {a3:d}\%, $\\boldsymbol{{\delta}}$\\textbf{{ = {d3[2]:.1f}}}, SNR = {snr3:d}dB', ylim = ylim, subtitle = f'acc = {acc_delta[2]:.0%}, f1 = {performance_delta[2][0]:.0%}, p = {performance_delta[2][1]:.0%}, r = {performance_delta[2][2]:.0%}')

    if method == 'ig':
        f.savefig(f'{figure_path}/figure2.png', dpi = 600)
    else:
        f.savefig(f'{figure_path}/figure2_{method}.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 2 of manuscript')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--permutation', action='store_true')
    parser.add_argument('--m', metavar = 'M', type = str, nargs = 1, default = ['ig'])
    args = parser.parse_args()
    main(args)
