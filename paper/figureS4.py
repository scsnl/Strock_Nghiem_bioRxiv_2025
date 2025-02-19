import os, sys, argparse, re
import matplotlib.cm as cm
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
            print(rois_true.shape, rois_pred.shape, attr.shape)
            f1_score[-1][i] = met.f1_score(rois_true, rois_pred)
    return f1_score

def precision_score(class_attr, thresholds, class_rois):
    precision_score = []
    for attr, rois_true in zip(class_attr, class_rois):
        precision_score.append(np.zeros(thresholds.shape))
        attr = 100*attr/np.max(attr)
        for i, threshold in enumerate(thresholds):
            rois_pred = attr >= threshold
            precision_score[-1][i] = met.precision_score(rois_true, rois_pred)
    return precision_score

def recall_score(class_attr, thresholds, class_rois):
    recall_score = []
    for attr, rois_true in zip(class_attr, class_rois):
        recall_score.append(np.zeros(thresholds.shape))
        attr = 100*attr/np.max(attr)
        for i, threshold in enumerate(thresholds):
            rois_pred = attr >= threshold
            recall_score[-1][i] = met.recall_score(rois_true, rois_pred)
    return recall_score

def violinplot_class(f, gs, x, methods, methods_short, ylabel, pd = 'less', title = ''):
    ax = violinplot(f, gs, [100*x[m] for m in methods], names = methods_short, xlabel = '', ylabel = f'{ylabel}', ylim = (-10, 160), c = ['0.0']*len(methods), pd = pd, sort = True, order = -1, showsign = False, title = title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, ha = 'right')
    return ax

def find_attr(file):
    l = glob.glob(file)
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
    n_affs = np.arange(1,11) #np.arange(10, 31, 10)
    deltas = np.arange(6)/10 # np.arange(-8, 9, 2)/10
    snrs = np.array([-10, 0, 10]) # np.array([-10, -5, 0, 5, 10])

    params = [ (snr,a,d) for snr in snrs for a in n_affs for d in deltas if d != 0]
    affected = [(np.arange(n_feat)<i,)*2 for (snr,i,o) in params]

    regions_human = np.array(['PCC', 'PCC_Pcun', 'PCC_Pcun_Ang'])
    regions_mouse = np.array(['RSC', 'RSC_Cg', 'RSC_Cg_PrL'])
    qiasds = np.arange(45,50)/10
    params_tvb_human = [ (snr, region, qiasd) for snr in snrs for region in regions_human for qiasd in qiasds]
    params_tvb_mouse = [ (snr, region, qiasd) for snr in snrs for region in regions_mouse for qiasd in qiasds]

    nodes = {
        'PCC': [46,47],
        'Pcun': [50,51],
        'Ang': [14,15],
        'RSC': [155, 156, 157, 368, 369, 370],
        'Cg': [1,2,214,215],
        'PrL': [129,342]
    }
    affected_tvb_human = [(np.isin(np.arange(68), sum([nodes[r] for r in region.split('_')], start = [])),)*2 for noise, region, qiasd in params_tvb_human]
    affected_tvb_mouse = [(np.isin(np.arange(426), sum([nodes[r] for r in region.split('_')], start = [])),)*2 for noise, region, qiasd in params_tvb_mouse]

    thresholds = np.array([50])

    # -------------------------------
    # Figure parameters
    # -------------------------------

    zoom = 15
    ws = np.array([2,2,2])
    wspace = 2
    hs = np.array([1,1,1])
    hspace = 1.5

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, bottom = 1, top = 2)



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


    class_attrs = {m: [find_attr(f'{attr_path}/{m}/simple_n100_a{a:d}_delta{d:.2f}_snr{snr:.2f}_epoch*.pt') for (snr,a,d) in params] for m in methods}
    class_attrs = {m: np.stack(class_attr, axis = 0) for m, class_attr in class_attrs.items()}    
    
    f1 = {m: np.array([f1_score(class_attrs[m][i], thresholds, affected[i]) for i in range(len(class_attrs['ig']))]) for m in methods}
    precision = {m: np.array([precision_score(class_attrs[m][i], thresholds, affected[i]) for i in range(len(class_attrs['ig']))]) for m in methods}
    recall = {m: np.array([recall_score(class_attrs[m][i], thresholds, affected[i]) for i in range(len(class_attrs['ig']))]) for m in methods}
    best_f1 = {m: np.max(f1[m], axis = (1,2)) for m in methods}
    best_precision = {m: np.max(precision[m], axis = (1,2)) for m in methods}
    best_recall = {m: np.max(recall[m], axis = (1,2)) for m in methods}

    class_attrs_tvb_human = {m: np.stack([find_attr(f'{attr_path_tvb}/tvb_{region}_asdQi_{qiasd:.1f}_ntQi_5.0_noise1.0e-04_snr{snr:.2f}/{m}/tvb_{region}_asdQi_{qiasd:.1f}_ntQi_5.0_noise1.0e-04_snr{snr:.2f}_epoch*.pt') for snr,region,qiasd in params_tvb_human], axis = 0) for m in methods}
    correlation_tvb_human, p_correlation_tvb_human = correlation_attr(class_attrs_tvb_human, methods)
    f1_tvb_human = {m: np.array([f1_score(class_attrs_tvb_human[m][i], thresholds, affected_tvb_human[i]) for i in range(len(class_attrs_tvb_human['ig']))]) for m in methods}
    precision_tvb_human = {m: np.array([precision_score(class_attrs_tvb_human[m][i], thresholds, affected_tvb_human[i]) for i in range(len(class_attrs_tvb_human['ig']))]) for m in methods}
    recall_tvb_human = {m: np.array([recall_score(class_attrs_tvb_human[m][i], thresholds, affected_tvb_human[i]) for i in range(len(class_attrs_tvb_human['ig']))]) for m in methods}
    best_f1_tvb_human = {m: np.max(f1_tvb_human[m], axis = (1,2)) for m in methods}
    best_precision_tvb_human = {m: np.max(precision_tvb_human[m], axis = (1,2)) for m in methods}
    best_recall_tvb_human = {m: np.max(recall_tvb_human[m], axis = (1,2)) for m in methods}
    
    class_attrs_tvb_mouse = {m: np.stack([find_attr(f'{attr_path_tvb}/tvb_{region}_asdQi_{qiasd:.1f}_ntQi_5.0_noise1.0e-04_snr{snr:.2f}/{m}/tvb_{region}_asdQi_{qiasd:.1f}_ntQi_5.0_noise1.0e-04_snr{snr:.2f}_epoch*.pt') for snr,region,qiasd in params_tvb_mouse], axis = 0) for m in methods}
    correlation_tvb_mouse, p_correlation_tvb_mouse = correlation_attr(class_attrs_tvb_mouse, methods)
    f1_tvb_mouse = {m: np.array([f1_score(class_attrs_tvb_mouse[m][i], thresholds, affected_tvb_mouse[i]) for i in range(len(class_attrs_tvb_mouse['ig']))]) for m in methods}
    precision_tvb_mouse = {m: np.array([precision_score(class_attrs_tvb_mouse[m][i], thresholds, affected_tvb_mouse[i]) for i in range(len(class_attrs_tvb_mouse['ig']))]) for m in methods}
    recall_tvb_mouse = {m: np.array([recall_score(class_attrs_tvb_mouse[m][i], thresholds, affected_tvb_mouse[i]) for i in range(len(class_attrs_tvb_mouse['ig']))]) for m in methods}
    best_f1_tvb_mouse = {m: np.max(f1_tvb_mouse[m], axis = (1,2)) for m in methods}
    best_precision_tvb_mouse = {m: np.max(precision_tvb_mouse[m], axis = (1,2)) for m in methods}
    best_recall_tvb_mouse = {m: np.max(recall_tvb_mouse[m], axis = (1,2)) for m in methods}




    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))

    ax_A = letter('A',violinplot_class)(f, gs[0,0], best_f1, methods, method_names, ylabel = f'best F1\nscore (%)', pd = [(0,1,2), [(k,3) for k in (0,1,2)]], title = 'RNN')
    ax_B = letter('B',violinplot_class)(f, gs[0,1], best_precision, methods, method_names, ylabel = f'best precision\nscore (%)', pd = [(0,1,2), [(k,3) for k in (0,1,2)]], title = 'RNN')
    ax_C = letter('C',violinplot_class)(f, gs[0,2], best_recall, methods, method_names, ylabel = f'best recall\nscore (%)', pd =  [(0,1,2,3)], title = 'RNN')

    ax_D = letter('D',violinplot_class)(f, gs[1,0], best_f1_tvb_human, methods, method_names, ylabel = f'best F1\nscore (%)', pd = [(0,1,2), [(k,3) for k in (0,1,2)]], title = 'TVB (Human)')
    ax_E = letter('E',violinplot_class)(f, gs[1,1], best_precision_tvb_human, methods, method_names, ylabel = f'best precision\nscore (%)', pd = [(0,1,2), [(k,3) for k in (0,1,2)]], title = 'TVB (Human)')
    ax_F = letter('F',violinplot_class)(f, gs[1,2], best_recall_tvb_human, methods, method_names, ylabel = f'best recall\nscore (%)', pd =  [(1,2,3), [(0,k) for k in (1,2,3)]], title = 'TVB (Human)')

    ax_G = letter('G',violinplot_class)(f, gs[2,0], best_f1_tvb_mouse, methods, method_names, ylabel = f'best F1\nscore (%)', pd = [(0,1,2), [(k,3) for k in (0,1,2)]], title = 'TVB (Mouse)')
    ax_H = letter('H',violinplot_class)(f, gs[2,1], best_precision_tvb_mouse, methods, method_names, ylabel = f'best precision\nscore (%)', pd = [(0,1,2), [(k,3) for k in (0,1,2)]], title = 'TVB (Mouse)')
    ax_I = letter('I',violinplot_class)(f, gs[2,2], best_recall_tvb_mouse, methods, method_names, ylabel = f'best recall\nscore (%)', pd =  [(0,1,2,3)], title = 'TVB (Mouse)')

    #ax_C = letter('C',violinplot_class)(f, gs[2,0], best_f1_tvb_mouse, methods, method_names, ylabel = f'best F1\nscore (%)', pd = [(0,1,2,3)], title = 'TVB')


    f.savefig(f'{figure_path}/figureS4.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure S4 of manuscript')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)
