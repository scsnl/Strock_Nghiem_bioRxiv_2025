import os, sys, argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from figures.plots import letter, get_figsize, plot, plot_multiple, imshow
import numpy as np
from matplotlib import patches
import networkx as nx
from nn_analysis.dataset.fakefmri import gen_rnn_from_adjacency, gen_from_file, hrf_filt, add_measurement_noise
import h5py
import pandas as pd
import glob
import torch
from matplotlib.colors import to_rgba
from scipy.stats import pearsonr
import matplotlib.patheffects as PathEffects

def rename(a, d):
    f = np.vectorize(lambda k: d[k] if k in d.keys() else 'None')
    return f(a)

def empty(f, gs, title):
    ax = f.add_subplot(gs)
    ax.set_title(title, fontsize = 12, weight = 'bold')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.0)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def model(f, gs, n = 16, n_highlight = 5, path_tvb_human = None, path_tvb_mouse = None, title_brain_human = '', title_brain_mouse = ''):
    ax = empty(f, gs, 'Model brain dynamics')
    _gs = gs.subgridspec(3,1,wspace=0,hspace=0)
    ax_rnn = f.add_subplot(_gs[0,0])
    scale = 0.5
    theta0 = np.pi/2
    theta = np.linspace(theta0, theta0+2*np.pi, n+1)[:-2]
    ax_rnn.scatter(np.cos(theta), np.sin(theta), clip_on = False, color = ['C3']*n_highlight+['0.8']*(n-1-n_highlight), s = scale**2*150)
    for i,t in enumerate(theta):
        ax_rnn.text(np.cos(t), np.sin(t), f'{i:d}', color = '1.0', va = 'center', ha = 'center', size = scale*10)
        for k in range(1,min(6, n-i-1)):
            t2 = theta[i+k]
            ax_rnn.plot(np.cos([t,t2]), np.sin([t,t2]), color = '0.9', zorder = -1, lw = 0.5, alpha = 0.5)
    n = 3*n
    theta = np.linspace(theta0, theta0+2*np.pi, n+1)[-5:-2]
    ax_rnn.scatter(np.cos(theta), np.sin(theta), clip_on = False, color = '0.8', marker = '.', s = scale**2*0.5)
    bbox = ax_rnn.get_window_extent()
    a,b = bbox.height, bbox.width
    ax_rnn.set_xlim(-(1+(b-a)/a)/scale, (1+(b-a)/a)/scale)
    ax_rnn.set_ylim(-1/scale, 1/scale)
    ax_rnn.text(0.5, 0.85, 'RNN', va = 'bottom', ha = 'center', transform = ax_rnn.transAxes, color = '0.0', weight = 'bold')
    ax_rnn.axis('off')
    ax_tvb_human = f.add_subplot(_gs[1,0])
    ax_tvb_human.patch.set_alpha(1.0)
    img = plt.imread(path_tvb_human)
    ax_tvb_human.imshow(img)
    xlim, ylim = ax_tvb_human.get_xlim(), ax_tvb_human.get_ylim()
    alpha = 0.45
    xlim = xlim[0]-alpha*(xlim[1]-xlim[0]),xlim[1]+alpha*(xlim[1]-xlim[0])
    ylim = ylim[0]-alpha*(ylim[1]-ylim[0]),ylim[1]+alpha*(ylim[1]-ylim[0])
    ax_tvb_human.set_xlim(xlim)
    ax_tvb_human.set_ylim(ylim)
    ax_tvb_human.text(0.5, 0.85, 'TVB HUMAN', va = 'bottom', ha = 'center', transform = ax_tvb_human.transAxes, color = '0.0', weight = 'bold')
    ax_tvb_human.axis('off')
    ax_tvb_mouse = f.add_subplot(_gs[2,0])
    ax_tvb_mouse.patch.set_alpha(1.0)
    img = plt.imread(path_tvb_mouse)
    ax_tvb_mouse.imshow(img)
    xlim, ylim = ax_tvb_mouse.get_xlim(), ax_tvb_mouse.get_ylim()
    alpha = 0.45
    xlim = xlim[0]-alpha*(xlim[1]-xlim[0]),xlim[1]+alpha*(xlim[1]-xlim[0])
    ylim = ylim[0]-alpha*(ylim[1]-ylim[0]),ylim[1]+alpha*(ylim[1]-ylim[0])
    ax_tvb_mouse.set_xlim(xlim)
    ax_tvb_mouse.set_ylim(ylim)
    ax_tvb_mouse.text(0.5, 0.85, 'TVB MOUSE', va = 'bottom', ha = 'center', transform = ax_tvb_mouse.transAxes, color = '0.0', weight = 'bold')
    ax_tvb_mouse.axis('off')
    return ax

def add_layer(ax, c, w, h, name = '', wa = None, backward = False):
    rect = patches.Rectangle(c, w, h, linewidth=1, edgecolor='0.0', facecolor='none')
    ax.add_patch(rect)
    ax.text(c[0]+w/2, c[1]+h/2, name, va = 'center', ha = 'center', rotation = 90)
    if not wa is None:
        ax.add_patch(patches.FancyArrowPatch((c[0]-wa, c[1]+h/2), (c[0], c[1]+h/2),  shrinkA=0,  shrinkB=0, color="0.0", arrowstyle="-|>",  mutation_scale=15, linewidth=1.0))
        if backward:
            ax.add_patch(patches.FancyArrowPatch( (c[0]+w/4, c[1]),(c[0]-wa-w/4, c[1]),  shrinkA=0,  shrinkB=0, color="C3", arrowstyle="-|>",  mutation_scale=15, linewidth=2.0, connectionstyle="angle3,angleA=75,angleB=-75", clip_on = False, capstyle = 'butt', zorder = -1))

def add_output(ax, c, w, h, names = [], wa = None, highlight_output = [], y_bw = None):
    r = w/2
    for name, y in zip(names[::-1], np.linspace(c[1]+r/2, c[1]+h-r/2, len(names))):
        circle = patches.Circle((c[0]+r, y), r, linewidth= 2 if name in highlight_output else 1, edgecolor='C3' if name in highlight_output else '0.0', facecolor='none')
        ax.add_patch(circle)
        ax.text(c[0]+2.5*r, y, name, va = 'center', ha = 'left', color= 'C3' if name in highlight_output else '0.0', weight = 'bold' if name in highlight_output else 'normal')
        if not wa is None:
            ax.add_patch(patches.FancyArrowPatch((c[0]-wa, c[1]+h/2), (c[0], y),  shrinkA=0,  shrinkB=0, color= "0.0", arrowstyle="-|>",  mutation_scale=15, linewidth=1.0))
        if name in highlight_output:
            ax.add_patch(patches.FancyArrowPatch( (c[0]+w/2, y-r),(c[0]-wa-w/4, y_bw),  shrinkA=0,  shrinkB=0, color="C3", arrowstyle="-|>",  mutation_scale=15, linewidth=2.0, connectionstyle="angle3,angleA=75,angleB=-75", clip_on = False, capstyle = 'butt'))


def plot_classifier(f, gs, ax = None, showname = True, highlight_output = [], title = ''):
    if ax is None:
        ax = f.add_subplot(gs)
        ax.axis('off')
    ax.text(0.5, 0.9, 'stDNN', va = 'center', ha = 'center', transform = ax.transAxes, weight = 'bold')
    width, height = 0.15, 0.75
    n_layer = 8
    space = (3-width*n_layer)/(n_layer+1)
    jump = space+width
    add_layer(ax, (space, (1-height)/2), width, height, name = 'Simulated fMRI' if showname else '', backward = len(highlight_output)>0)
    add_layer(ax, (space+jump, (1-height)/2), width, height, name = 'Conv1D' if showname else '', wa = space, backward = len(highlight_output)>0)
    add_layer(ax, (space+2*jump, (1-height)/2), width, height, name = 'ReLU' if showname else '', wa = space, backward = len(highlight_output)>0)
    add_layer(ax, (space+3*jump, (1-height)/2), width, height, name = 'AvgPool1D' if showname else '', wa = space, backward = len(highlight_output)>0)
    add_layer(ax, (space+4*jump, (1-height)/2), width, height, name = 'Conv1D' if showname else '', wa = space, backward = len(highlight_output)>0)
    add_layer(ax, (space+5*jump, (1-height)/2), width, height, name = 'ReLU' if showname else '', wa = space, backward = len(highlight_output)>0)
    add_layer(ax, (space+6*jump, (1-height)/2), width, height, name = 'AvgPool1D' if showname else '', wa = space, backward = len(highlight_output)>0)
    y_bw = (1-height)/2
    height = 0.5
    add_output(ax, (space+7*jump, (1-height)/2), width, height, names = ['Control', 'Autism'], wa = space, highlight_output = highlight_output, y_bw = y_bw)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-0.3, 3.3)
    return ax

def classifier(f, gs):
    ax = empty(f, gs, 'Classify groups')
    plot_classifier(f, gs, ax)
    return ax

def plot_roi(f, gs, x, affected, title = '', scale = 1, n_affected = 2, n_not_affected = 4, show_affected = False):
    n_feat = x.shape[0]
    ax = f.add_subplot(gs)
    ax.set_title(title, weight = 'bold', fontsize = 10)
    roi_affected = np.where(affected)[0]
    roi_not_affected =  np.where(np.logical_not(affected))[0]
    x -= np.mean(x, axis = 1, keepdims = True)
    x *= scale
    for i,k in enumerate(roi_affected[:n_affected]):
        ax.plot(i+x[k], color = 'C3')
    for i,k in enumerate(roi_not_affected[:n_not_affected]):
        ax.plot(i+x[k]+3, color = '0.8')
    ax.text(x.shape[1]/2, n_affected, '⋮', va = 'center', ha = 'center')
    if show_affected:
        ax.text(x.shape[1]*1.1, (n_affected-1)/2, '}', va = 'center', ha = 'right', color = 'C3', size = 20)
        ax.text(x.shape[1]*1.1, (n_affected-1)/2, 'affected', va = 'center', ha = 'left', color = 'C3', weight = 'bold')
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_affected+n_not_affected+1))
    ax.set_yticklabels([])
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_ylim(n_affected+n_not_affected+1, -1)
    ax.set_ylabel('ROIs')
    ax.set_xlabel('Time (s)')
    return ax

def plot_rnn_fmri(f, gs, title = '', n_feat = 100, n_aff = 1, delta = 0.0, snr = 10.0, seed = 0):
    np.random.seed(seed)
    G = nx.connected_watts_strogatz_graph(n_feat, 10, 0.1) if n_feat > 10 else nx.connected_watts_strogatz_graph(n_feat, 2, 0.0)
    A = nx.adjacency_matrix(G).todense()
    _radius = max(np.abs(np.linalg.eigvals(A)))
    A = A/_radius
    A[:n_aff] += delta*np.max(A)
    hrf = np.float32(np.load(f'{os.environ.get("DATA_PATH")}/../matlab/hrf_dt1.0.npy')[:, 0])
    x = gen_rnn_from_adjacency(1, 100, A, radius = None, noise = 0.1, warmup = 0, f = np.tanh, transform = lambda x: hrf_filt(x, hrf, snr = snr))[0]
    return plot_roi(f, gs, x, np.arange(n_feat)<n_aff, title)

def plot_tvb_fmri(f, gs, title, region = 'PCC', noise =  0.0001, Qi = 5.0, snr = 0.0, seed = 0, show_affected = False):
    np.random.seed(seed)
    data_path =  f'{os.environ.get("OAK")}/projects/tanghiem/2022_TVB_AdEX/feature_attribution'
    if region in ['RSC', 'RSC_Cg', 'RSC_Cg_PrL']:
        params = {'files': f'{data_path}/sig_BOLD__b_0_Qi_{region}_nodes{Qi:.1f}_repeatedAId_rightstim_0.0EtoEIratio1.4_coupling0.15seed{seed:d}_noise{noise:.1e}.npy', 'transform':lambda x: add_measurement_noise(x, snr = snr)}
        roi_names = np.char.decode(h5py.File(f'{os.environ.get("DATA_PATH")}/tvb/Connectivity_Allen_Oh2014.h5','r')['region_labels'][...], 'utf-8')
        roi_names = rename(roi_names, {"RSPv_left": "RSC-VL", "RSPv_right": "RSC-VR", "RSPagl_left": "RSC-AL", "RSPagl_right": "RSC-AR", "RSPd_left": "RSC-DL", "RSPd_right": "RSC-DR", "ACAd_left": "Cg-DL", "ACAv_left": "Cg-VL", "ACAd_right": "Cg-DR", "ACAv_right": "Cg-VR", "PL_left": "PrL-L", "PL_right": "PrL-R"})
    elif region in ['PCC', 'PCC_Pcun', 'PCC_Pcun_Ang']:
        params = {'files': f'{data_path}/sig_BOLD__b_0_Qi_{region}{Qi:.1f}_repeatedinsulastim_0.0EtoEIratio1.4_coupling0.15seed{seed:d}_noise{noise:.1e}.npy', 'transform':lambda x: add_measurement_noise(x, snr = snr)}
        roi_names = np.array(list(pd.read_csv(f'{os.environ.get("DATA_PATH")}/tvb/centres.txt', index_col = 0, sep = '\t', header = None).index))
        roi_names = rename(roi_names, {"posteriorcingulate_R":"PCC-R", "posteriorcingulate_L":"PCC-L", "precuneus_R": "PCun-R", "precuneus_L": "PCun-L", "inferiorparietal_R": "AnG-R", "inferiorparietal_L": "AnG-R"})
    else:
        raise NameError(f'Not implemented when E/I imbalance is in {region}')
    nodes = {'PCC': [46,47], 'Pcun': [50,51], 'Ang': [14,15], 'RSC': [155, 156, 157, 368, 369, 370], 'Cg': [1,2,214,215], 'PrL': [129,342]}
    x = gen_from_file(**params)[0]
    n_feat = x.shape[0]
    affected = np.isin(np.arange(n_feat), sum([nodes[r] for r in region.split('_')], start = []))
    #return plot_roi(f, gs, x, affected, title, roi_names, scale = 10)
    return plot_roi(f, gs, x, affected, title, scale = 7.5, show_affected = show_affected)

def simulated_fmri(f, gs):
    ax = empty(f, gs, 'Simulate fMRI for different groups')
    wspace, hspace = 0.5, 0.5
    n_w, n_h = 2, 1
    _gs = gs.subgridspec(2*n_h+1,2*n_w+1,wspace=0,hspace=0,width_ratios=[wspace]+[1,wspace]*n_w,height_ratios=[hspace]+[1,hspace]*n_h)
    plot_tvb_fmri(f, _gs[1,1], title = 'Control')
    plot_tvb_fmri(f, _gs[1,3], title = 'Autism', Qi = 4.5, show_affected = True)
    return ax

def find_attr(file):
    l = glob.glob(file)
    if len(l) == 0:
        print(file)
    return torch.load(l[0]).detach().numpy()

def plot_attr(f, gs, attr = None, affected = None, title = '', n_affected = 2, n_not_affected = 4, noise =  0.001, Qi = 5.0, snr = 0.0, seed = 0, threshold = None):
    ax = f.add_subplot(gs)
    if not attr is None and not affected is None:
        ax.bar(np.arange(n_affected), np.sort(attr[affected])[::-1][:n_affected], width = 1.0, color = 'C3', zorder = 0)
        ax.bar(np.arange(n_affected+1, n_affected+n_not_affected+1), np.sort(attr[np.logical_not(affected)])[::-1][:n_not_affected], width = 1.0, color = '0.8', zorder = 0)
        ax.text(n_affected,25, '⋯', va = 'center', ha = 'center')
        ax.text((n_affected-1)/2, 105, 'affected', va = 'bottom', ha = 'center', color = 'C3', weight = 'bold')
        if not threshold is None:
            ax.axhline(threshold, 0, 1, linestyle = '--', c = '0.0')
            ax.text(n_affected+n_not_affected+1, threshold+5, 'detection\nthreshold', va = 'bottom', ha = 'center', weight = 'bold')
    ax.set_title(title, weight = 'bold', fontsize = 10)
    ax.set_xticks(np.arange(n_affected+n_not_affected+1))
    ax.set_xticklabels([])
    ax.set_yticks([0,50,100])
    ax.set_xlim(-2, n_affected+n_not_affected+1)
    ax.set_ylim(0, 130)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel('ROIs')
    ax.set_ylabel('Attribution (%)')
    return ax

def attribution(f, gs):
    attr_path = f'{os.environ.get("DATA_PATH")}/tvb/attr'
    attr = find_attr(f'{attr_path}/tvb_PCC_asdQi_4.5_ntQi_5.0_noise1.0e-04_snr0.00/ig/tvb_PCC_asdQi_4.5_ntQi_5.0_noise1.0e-04_snr0.00_epoch*.pt')[1]
    attr *= 100/attr.max()
    ax = empty(f, gs,  'Determine ROI contribution')
    wspace, hspace = 0.5, 0.5
    n_w, n_h = 2, 1
    _gs = gs.subgridspec(2*n_h+1,2*n_w+3,wspace=0,hspace=0,width_ratios=[wspace,1,0,(n_w*(1+wspace)+wspace)/2,wspace,1,wspace],height_ratios=[hspace]+[1,hspace]*n_h)
    groundtruth = np.isin(np.arange(68), [46,47])
    plot_attr(f, _gs[1,1], 100.0*groundtruth, groundtruth, title = 'Ground truth')
    plot_attr(f, _gs[1,-2], attr, groundtruth, title = 'Actual attribution', threshold = 50)
    _ax = plot_classifier(f, _gs[1,3], showname = False, highlight_output = ['ASD'])
    _ax.text(0.5, -0.2, 'which ROIs contributed to the decision?', transform=_ax.transAxes, color = 'C3', weight = 'bold', va = 'top', ha = 'center')
    return ax

def groundtruth_comparison(f, gs):
    ax = empty(f, gs, 'Compare to ground truth')
    r = 0.625
    scale = 0.25*0.625/0.5

    baseline = 0.75
    wedge_not_affected = patches.Wedge((1, baseline), r, 0, 180, linewidth=2, edgecolor='0.8', facecolor='none', zorder = 0)
    wedge_affected = patches.Wedge((1, baseline), r, 180, 360, linewidth=2, edgecolor='C3', facecolor='none', zorder = 1)
    wedge_detected_not_affected = patches.Wedge((1, baseline), r/2, 0, 180, linewidth=2, edgecolor='0.8', facecolor=to_rgba('0.8', 0.5), zorder = 0)
    wedge_detected_affected = patches.Wedge((1, baseline), r/2, 180, 360, linewidth=2, edgecolor='C3', facecolor=to_rgba('C3', 0.5), zorder = 1)
    for p in [wedge_not_affected, wedge_affected, wedge_detected_not_affected, wedge_detected_affected]:
        ax.add_patch(p)
    
    baseline = 1.125
    wedge_detected_affected = patches.Wedge((2.5, baseline+3*scale*r/4), scale*r/2, 180, 360, linewidth=1, edgecolor='C3', facecolor=to_rgba('C3', 0.5), zorder = 1)
    wedge_detected_not_affected2 = patches.Wedge((2.5, baseline-3*scale*r/4), scale*r/2, 0, 180, linewidth=1, edgecolor='0.8', facecolor=to_rgba('0.8', 0.5), zorder = 0)
    wedge_detected_affected2 = patches.Wedge((2.5, baseline-3*scale*r/4), scale*r/2, 180, 360, linewidth=1, edgecolor='C3', facecolor=to_rgba('C3', 0.5), zorder = 1)
    for p in [wedge_detected_affected, wedge_detected_not_affected2, wedge_detected_affected2]:
        ax.add_patch(p)
    ax.plot([2.5-1.1*scale*r,2.5+1.1*scale*r], [baseline,baseline], linewidth = 1, color = '0.0', zorder = 1, solid_capstyle = 'butt')
    ax.text(2.5, baseline-0.5, 'precision', va = 'bottom', ha = 'center')

    wedge_detected_affected = patches.Wedge((3.5, baseline+3*scale*r/4), scale*r/2, 180, 360, linewidth=1, edgecolor='C3', facecolor=to_rgba('C3', 0.5), zorder = 1)
    wedge_affected2 = patches.Wedge((3.5, baseline-scale*r/4), scale*r, 180, 360, linewidth=1, edgecolor='C3', facecolor='none', zorder = 1)
    wedge_detected_affected2 = patches.Wedge((3.5, baseline-scale*r/4), scale*r/2, 180, 360, linewidth=1, edgecolor='C3', facecolor=to_rgba('C3', 0.5), zorder = 1)
    for p in [wedge_detected_affected, wedge_affected2, wedge_detected_affected2]:
        ax.add_patch(p)
    ax.plot([3.5-1.1*scale*r,3.5+1.1*scale*r], [baseline,baseline], linewidth = 1, color = '0.0', zorder = 1, solid_capstyle = 'butt')
    ax.text(3.5, baseline-0.5, 'recall', va = 'bottom', ha = 'center')
    ax.text(3, 0.2, 'F1', va = 'center', ha = 'center')
            
    ax.add_patch(patches.FancyArrowPatch((2.5, baseline-0.5), (2.9, 0.25),  shrinkA=0,  shrinkB=0, color="black", arrowstyle="-|>",  mutation_scale=15, linewidth=1.0))
    ax.add_patch(patches.FancyArrowPatch((3.5, baseline-0.5), (3.1, 0.25),  shrinkA=0,  shrinkB=0, color="black", arrowstyle="-|>",  mutation_scale=15, linewidth=1.0))

    legend_elements = [patches.Patch(edgecolor='C3', facecolor = 'none', label='affected'), patches.Patch(edgecolor='0.0', facecolor = to_rgba('0.0', 0.5), label='detected'), patches.Patch(edgecolor='0.8', facecolor = 'none', label='not affected'), patches.Patch(edgecolor='0.0', facecolor = 'none', label='not detected')]
    leg = ax.legend(handles=legend_elements, loc='upper left', ncols = 2, columnspacing=0.8, frameon=False)
    for patch, text in zip(leg.get_patches(), leg.get_texts()):
        text.set_color(patch.get_edgecolor())
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 2)
    return ax

def scatter_attribution(f, gs, attr1, attr2, groundtruth, attr1_name = '1', attr2_name = '2'):
    ax = f.add_subplot(gs)
    ax.set_xticks([0,50,100])
    ax.set_yticks([0,50,100])
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel(f'Attribution {attr1_name} (%)')
    ax.set_ylabel(f'Attribution {attr2_name} (%)')
    ax.scatter(attr1[groundtruth], attr2[groundtruth], marker = '.', color = 'C3', label = 'affected')
    ax.scatter(attr1[np.logical_not(groundtruth)], attr2[np.logical_not(groundtruth)], marker = '.', color = '0.8', label = 'not affected', zorder = -1)
    x,y = attr1, attr2
    a,b = np.polyfit(x, y, 1)
    r,p = pearsonr(x, y)
    x_m = np.mean([x.min(), x.max()])
    theta = ax.transData.transform_angles(np.array([180*np.arctan(a)/np.pi]), np.array([[x_m, a*x_m+b]]))[0]
    ax.plot(x, a*x+b, color = '0.5', zorder = -2)
    #ax.text(0.95, 0.1, f'r = {r:.2f}', color = 'C3', va = 'bottom', ha = 'right', transform = ax.transAxes, path_effects = [PathEffects.withStroke(linewidth=2, foreground='1.0')])
    leg = ax.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1.2), ncols = 2, columnspacing=0.8, frameon=False, handletextpad=0.1)
    delta = 10
    ax.text(x_m-delta, a*x_m+b+delta, 'correlation?', va = 'bottom', ha = 'center', color = '0.5', rotation = theta, rotation_mode='anchor', path_effects = [PathEffects.withStroke(linewidth=1, foreground='1.0')])
    return ax

def attribution_desc(f, gs):
    ax = f.add_subplot(gs)
    ax.axis('off')
    ax.text(0.5, 1.0, 'Across methods', weight = 'bold', va = 'center', ha = 'center', transform = ax.transAxes)
    ax.text(0.5, 0.75, 'IG, DLIFT,\nIXG, GB', va = 'center', ha = 'center', transform = ax.transAxes)
    ax.text(0.5, 0.5, 'Across baselines', weight = 'bold', va = 'center', ha = 'center', transform = ax.transAxes)
    ax.text(0.5, 0.25, 'Zero, Mean,\nMedian, Random', va = 'center', ha = 'center', transform = ax.transAxes)
    return ax

def correlation(f, gs):
    ax = empty(f, gs, 'Compare attribution')
    attr_path = f'{os.environ.get("DATA_PATH")}/tvb/attr'
    attr1 = find_attr(f'{attr_path}/tvb_PCC_asdQi_4.5_ntQi_5.0_noise1.0e-04_snr0.00/ig/tvb_PCC_asdQi_4.5_ntQi_5.0_noise1.0e-04_snr0.00_epoch*.pt')[1]
    attr2 = find_attr(f'{attr_path}/tvb_PCC_asdQi_4.5_ntQi_5.0_noise1.0e-04_snr0.00/deeplift/tvb_PCC_asdQi_4.5_ntQi_5.0_noise1.0e-04_snr0.00_epoch*.pt')[1]
    attr1 = attr1*100/attr1.max()
    attr2 = attr2*100/attr2.max()
    groundtruth = np.isin(np.arange(68), [46,47])
    wspace, hspace = 0.75, 0.75
    n_w, n_h = 2, 1
    _gs = gs.subgridspec(2*n_h+1,2*n_w+1,wspace=0,hspace=0,width_ratios=[wspace]+[1,wspace]*n_w,height_ratios=[hspace]+[1,hspace]*n_h)
    scatter_attribution(f, _gs[1,1], attr1, attr2, groundtruth)
    attribution_desc(f, _gs[1:,3])
    return ax

def connect(f, ax1, ax2, x1 = None, y1 = None, x2 = None, y2 = None):
    transFigure = f.transFigure.inverted()
    if x1 is None:
        assert not y1 is None and not x2 is None and not y2 is None
        coord2 = transFigure.transform(ax2.transAxes.transform((x2,y2)))
        coord1 = transFigure.transform(ax1.transAxes.transform((0,y1)))
        coord1[0] = coord2[0]
    if x2 is None:
        assert not y2 is None and not x1 is None and not y1 is None
        coord1 = transFigure.transform(ax1.transAxes.transform((x1,y1)))
        coord2 = transFigure.transform(ax2.transAxes.transform((0,y2)))
        coord2[0] = coord1[0]
    if y1 is None:
        assert not x1 is None and not x2 is None and not y2 is None
        coord2 = transFigure.transform(ax2.transAxes.transform((x2,y2)))
        coord1 = transFigure.transform(ax1.transAxes.transform((x1,0)))
        coord1[1] = coord2[1]
    if y2 is None:
        assert not x2 is None and not x1 is None and not y1 is None
        coord1 = transFigure.transform(ax1.transAxes.transform((x1,y1)))
        coord2 = transFigure.transform(ax2.transAxes.transform((x2,0)))
        coord2[1] = coord1[1]
    f.patches.append( patches.FancyArrowPatch(coord1, coord2,  shrinkA=0,  shrinkB=0,  transform=f.transFigure, color="black", arrowstyle="-|>",  mutation_scale=30, linewidth=2.0))

def main(args):

    seed = 0
    np.random.seed(seed)

    # -------------------------------
    # Figure parameters
    # -------------------------------

    zoom = 10
    ws = np.array([1,0.5,0.5,1])
    wspace = 0.5
    hs = np.array([1,1,1,1])
    hspace = 0.25

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, left = 1.5)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper'
    os.makedirs(figure_path, exist_ok=True)
    
    # -------------------------------
    # Display
    # -------------------------------

    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    ax_A = letter('A',model)(f, gs[:2,0], path_tvb_human = f'{figure_path}/tvb/tvb_pcc.png', title_brain_human = 'PCC', path_tvb_mouse = f'{figure_path}/tvb/tvb_rsc.png', title_brain_mouse = 'RSC')
    ax_B = letter('B',simulated_fmri)(f, gs[0,1:])
    ax_C = letter('C',classifier)(f, gs[1,1:])
    ax_D = letter('D',attribution)(f, gs[2,:])
    ax_E = letter('E',groundtruth_comparison)(f, gs[3,:2])
    ax_F = letter('F',correlation)(f, gs[3,2:])

    connect(f, ax_A, ax_B, x1 = 1, x2 = 0, y2 = 0.5)
    connect(f, ax_B, ax_C, x2 = 0.9, y1 = 0, y2 = 1)
    connect(f, ax_C, ax_D, x1 = 0.9, y1 = 0, y2 = 1)
    connect(f, ax_D, ax_E, x2 = 0.15, y1 = 0, y2 = 1)
    connect(f, ax_D, ax_F, x2 = 0.85, y1 = 0, y2 = 1)

    f.savefig(f'{figure_path}/figure1.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 1 of manuscript')
    
    args = parser.parse_args()
    main(args)
