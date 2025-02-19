import os, sys, argparse, re
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from figures.plots import letter, get_figsize, plot, plot_multiple, imshow, violinplot
import numpy as np
import torch
import sklearn.metrics as met
from scipy.stats import pearsonr, ttest_ind, ttest_1samp
import glob

def correlation_attr(class_attrs, methods):
    n_method = len(methods)
    c = np.zeros((n_method, n_method))
    p = np.zeros((n_method, n_method))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            attr1 = class_attrs[m1].flatten()
            attr2 = class_attrs[m2].flatten()
            c[i,j], p[i,j] = pearsonr(attr1, attr2)
    return c, p

def correlation_attr_list(class_attrs, methods):
    n_method = len(methods)
    c = np.zeros((n_method, n_method))
    p = np.zeros((n_method, n_method))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            attr1 = class_attrs[m1].flatten()
            attr2 = class_attrs[m2].flatten()
            c[i,j], p[i,j] = pearsonr(attr1, attr2)
    return c, p

def f1_score(class_attr, thresholds, class_rois):
    f1_score = []
    for attr, rois_true in zip(class_attr, class_rois):
        f1_score.append(np.zeros(thresholds.shape))
        attr = 100*attr/np.max(attr)
        for i, threshold in enumerate(thresholds):
            rois_pred = attr >= threshold
            f1_score[-1][i] = met.f1_score(rois_true, rois_pred)
    return f1_score

def imshow_class(f, gs, correlation, p, methods, methods_short, stats = [], title = None):
    for idx1, idx2 in stats:
        print(f'Min correlation {[methods_short[i] for i in idx1]} vs {[methods_short[i] for i in idx2]}: (r > {np.min(correlation[idx1][:, idx2]):.3f}, p < {np.max(p[idx1][:, idx2]):.3e})')
    #ax = imshow(f, gs, correlation, methods, methods_short, clim = (-1,1), ylabel = f'', rotatex = True, clabel = 'correlation', cmap = 'coolwarm', title = title)
    ax = imshow(f, gs, correlation, methods, methods_short, clim = (0,1), ylabel = f'', rotatex = True, clabel = 'correlation', cmap = 'viridis', title = title)
    cmap = cm.get_cmap("viridis")
    norm  = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(i,j,f'{correlation[i,j]:.2f}', va = 'center', ha = 'center', color =  sm.to_rgba(0) if correlation[i,j]>0.5 else sm.to_rgba(1), weight = 'bold')
    return ax

def violinplot_class(f, gs, x, methods, methods_short, ylabel, pd = 'less'):
    ax = violinplot(f, gs, [100*x[m] for m in methods], names = methods_short, xlabel = '', ylabel = f'{ylabel}', ylim = (0, 100), c = ['0.5']*len(methods), pd = pd, sort = True, order = -1, showsign = False)
    return ax

def find_attr(file):
    l = glob.glob(file)
    if len(l) == 0:
        print(file)
    return torch.load(l[0]).detach().numpy()

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
    n_affs = np.arange(1,11)
    deltas = np.arange(6)/10
    snrs = np.array([-10, 0, 10])

    params = [ (snr,a,d) for snr in snrs for a in n_affs for d in deltas if d != 0]
    affected = [[(np.arange(n_feat)<i), (np.arange(n_feat)<i)] for (snr,i,o) in params]
    thresholds = np.arange(0,101)

    regions_human = np.array(['PCC', 'PCC_Pcun', 'PCC_Pcun_Ang'])
    regions_mouse = np.array(['RSC', 'RSC_Cg', 'RSC_Cg_PrL'])
    qiasds = np.arange(45,50)/10
    params_tvb_human = [ (snr, region, qiasd) for snr in snrs for region in regions_human for qiasd in qiasds]
    params_tvb_mouse = [ (snr, region, qiasd) for snr in snrs for region in regions_mouse for qiasd in qiasds]

    # -------------------------------
    # Figure parameters
    # -------------------------------

    zoom = 10
    ws = np.array([1,1])
    wspace = 1
    hs = np.array([1,1])
    hspace = 0.75

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, bottom = 1.25, top = 4, right = 2.5)



    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

    # path where attribution are saved
    attr_path = f'{os.environ.get("DATA_PATH")}/rnn_exc/attr'
    attr_path_data = f'{os.environ.get("DATA_PATH")}/data/attr'
    attr_path_tvb = f'{os.environ.get("DATA_PATH")}/tvb/attr'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper'
    os.makedirs(figure_path, exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------
    
    #methods = ['ig', 'deeplift', 'ixg', 'guidedbp', 'deconvolution']
    methods = ['ig', 'deeplift', 'ixg', 'guidedbp']
    #method_names = ['Integrated Gradients (IG)', 'DeepLIFT (DLIFT)', 'Input X Gradient (IXG)', 'Guided Backprop (GB)', 'Deconvolution (D)']
    method_names = ['Integrated Gradients (IG)', 'DeepLIFT (DLIFT)', 'Input X Gradient (IXG)', 'Guided Backprop (GB)']
    #method_names_short = ['IG', 'DLIFT', 'IXG', 'GB', 'D']
    method_names_short = ['IG', 'DLIFT', 'IXG', 'GB']

    class_attrs_data = {m: find_attr(f'{attr_path_data}/{m}/Whole_model_data_abide_brainnnetome_6_wmcsf.pt') for m in methods}
    correlation_data, p_correlation_data = correlation_attr(class_attrs_data, methods)

    class_attrs_tvb_human = {m: np.concatenate([find_attr(f'{attr_path_tvb}/tvb_{region}_asdQi_{qiasd:.1f}_ntQi_5.0_noise1.0e-04_snr{snr:.2f}/{m}/tvb_{region}_asdQi_{qiasd:.1f}_ntQi_5.0_noise1.0e-04_snr{snr:.2f}_epoch*.pt') for snr,region,qiasd in params_tvb_human], axis = 1) for m in methods}
    correlation_tvb_human, p_correlation_tvb_human = correlation_attr(class_attrs_tvb_human, methods)
 
    class_attrs_tvb_mouse = {m: np.concatenate([find_attr(f'{attr_path_tvb}/tvb_{region}_asdQi_{qiasd:.1f}_ntQi_5.0_noise1.0e-04_snr{snr:.2f}/{m}/tvb_{region}_asdQi_{qiasd:.1f}_ntQi_5.0_noise1.0e-04_snr{snr:.2f}_epoch*.pt') for snr,region,qiasd in params_tvb_mouse], axis = 1) for m in methods}
    correlation_tvb_mouse, p_correlation_tvb_mouse = correlation_attr(class_attrs_tvb_mouse, methods)
 
    class_attrs = {m: [find_attr(f'{attr_path}/{m}/simple_n100_a{a:d}_delta{d:.2f}_snr{snr:.2f}_epoch*.pt') for (snr,a,d) in params] for m in methods}
    class_attrs = {m: np.stack(class_attr, axis = 0) for m, class_attr in class_attrs.items()}    
    correlation, p_correlation = correlation_attr(class_attrs, methods)

    # -------------------------------
    # Stats
    # -------------------------------

    print(f'Min correlations (RNN): {np.min(correlation)}')
    print(f'Min correlations (TVB-Human): {np.min(correlation_tvb_human)}')
    print(f'Min correlations (TVB-Mouse): {np.min(correlation_tvb_mouse)}')
    print(f'Min correlations (ABIDE): {np.min(correlation_data)}')
    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    ax_A = letter('A',imshow_class)(f, gs[0,0], correlation, p_correlation, method_names, method_names_short, title = 'Simulation (RNN)')
    ax_B = letter('B',imshow_class)(f, gs[0,1], correlation_tvb_human, p_correlation_tvb_human, method_names, method_names_short, title = 'Simulation (TVB-Human)')
    ax_C = letter('C',imshow_class)(f, gs[1,0], correlation_tvb_mouse, p_correlation_tvb_mouse, method_names, method_names_short, title = 'Simulation (TVB-Mouse)')
    ax_D = letter('D',imshow_class)(f, gs[1,1], correlation_data, p_correlation_data, method_names, method_names_short, stats = [(range(3), range(3)), (range(3), [3])], title = 'Data (ABIDE)')
    f.savefig(f'{figure_path}/figure5.png', dpi = 200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure 5 of manuscript')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)
