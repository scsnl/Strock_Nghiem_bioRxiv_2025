import os, sys, argparse
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from figures.plots import letter, get_figsize, plot, plot_multiple, imshow, violinplot
import numpy as np
import torch
import glob

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
    reaching_threshold = 0.90
    n_consecutive = 3
    convergence_threshold = 0.05
    Qi_nt = 5.0

    # -------------------------------
    # Figure parameters
    # -------------------------------

    zoom = 15
    ws = np.array([2,2,2])
    wspace = 2
    hs = np.array([1,1])
    hspace = 1

    figsize, _ws, _hs = get_figsize(ws, wspace, hs, hspace, zoom, right = 2)

    # -------------------------------
    # Paths where to load/save data
    # -------------------------------

	# path where model are saved
    model_path = f'{os.environ.get("DATA_PATH")}/tvb/model'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper'
    # path where attribution 
    attr_path = f'{os.environ.get("DATA_PATH")}/tvb/attr'

    os.makedirs(figure_path, exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------

    regions = np.array(['PCC', 'PCC_Pcun', 'PCC_Pcun_Ang'])
    deltas = (np.arange(6)/10)[::-1]
    snrs = np.array([-10, 0, 10])
    accuracy = np.full((len(regions),len(deltas),len(snrs),1+n_epoch), float('nan'))
    reaching_time = np.full((len(regions),len(deltas),len(snrs)), float('nan'))
    convergence_time = np.full((len(regions),len(deltas),len(snrs)), float('nan'))
    for k, snr in enumerate(snrs):
        for i, region in enumerate(regions):
            for j, delta in enumerate(deltas):
                Qi_asd = Qi_nt-delta
                model_name = f'tvb_{region}_asdQi_{Qi_asd:.1f}_ntQi_{Qi_nt:.1f}_noise1.0e-04_snr{snr:.2f}'
                try:
                    a = torch.load(f'{model_path}/{model_name}/test_accuracy.npy').numpy()
                    if len(a) == 1+n_epoch:
                        accuracy[i,j,k] = a
                    elif len(a) == n_epoch:
                        print(f'Rerun train {model_name}, old model')
                        accuracy[i,j,k,1:] = a
                        accuracy[i,j,k,0] = 0.5
                    else:
                        raise NameError("Wrong number of epoch")
                except:
                    print(f'Rerun train {model_name}')
                try:
                #if delta>0:
                    reaching_time[i,j,k] = np.where(np.logical_and.reduce([accuracy[i,j,k,l:-n_consecutive+l+1]>reaching_threshold for l in range(n_consecutive-1)]+[accuracy[i,j,k,n_consecutive-1:]>reaching_threshold]))[0][0]
                    delta_acc = np.array([np.max(np.abs(accuracy[i,j,k,l]-accuracy[i,j,k,l+1:])) for l in range(accuracy.shape[3]-1)])
                    convergence_time[i,j,k] = np.where(delta_acc<convergence_threshold)[0][0]
                except:
                    pass
                try:
                    find_attr(f'{attr_path}/{model_name}/ig/{model_name}_epoch*.pt')
                except:
                    print(f'Rerun test {model_name}')

    regions = np.char.replace(regions, '_', '\n+')
    _snrs = np.tile(snrs[None,None,:], (len(regions),len(deltas),1))
    _regions = np.tile(regions[:,None,None], (1,len(deltas),len(snrs)))
    _deltas = np.tile(deltas[None,:,None], (len(regions),1,len(snrs)))
    idx = _deltas > 0.0
    idx2 = _deltas == 0.0

    best_accuracy = accuracy[:,:,:,-1] #np.max(accuracy, axis = -1)
    idx_a, idx_delta, idx_snr = np.where(best_accuracy<0.6)

    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))

    i,j,k = 0,0,-1 # affected = RSC, ΔQi = 0.5, SNR = 10dB
    print(f'SNR = {snrs[k]}dB, regions = {regions[i]}, ΔQi = {deltas[j]}nS')
    xlim = (-1,51)
    ax_A = letter('A',plot_multiple)(f, gs[0,0], np.arange(-1, n_epoch), 100*accuracy[i,j], snrs, xlabel = 'epoch', ylabel = 'accuracy (%)', clabel = 'SNR (dB)', ylim = (40, 110), title = f'affected = {regions[i]}, $\mathbf{{\Delta Q_i}}$ = {deltas[j]}nS', xlim = xlim, order = lambda i: -i, highlightx = [], highlighty = [(50, '0.5', '--'), (100*reaching_threshold, 'C3', '--')])
    ax_A.text(xlim[1], 50, 'chance level', va = 'bottom', ha = 'right', color = '0.5')
    ax_A.text(xlim[1], 100*reaching_threshold, 'threshold', va = 'top', ha = 'right', color = 'C3')
    ax_B = letter('B',plot_multiple)(f, gs[0,1], np.arange(-1, n_epoch), 100*accuracy[:,j,k], regions, xlabel = 'epoch', ylabel = 'accuracy (%)', clabel = 'regions', ylim = (40, 110), clim = (1, 10), title = f'SNR = 10dB, $\mathbf{{\Delta Q_i}}$ = {deltas[j]}nS', xlim = xlim, highlightx = [], highlighty = [(50, '0.5', '--'), (100*reaching_threshold, 'C3', '--')])
    ax_C = letter('C',plot_multiple)(f, gs[0,2], np.arange(-1, n_epoch), 100*accuracy[i,:,k], deltas, xlabel = 'epoch', ylabel = 'accuracy (%)', clabel = '$\Delta Q_i$', ylim = (40, 110), clim = (0.0, 0.5), title = f'SNR = 10dB, affected = {regions[i]}', xlim = xlim, order = lambda i: -i, highlightx = [], highlighty = [(50, '0.5', '--'), (100*reaching_threshold, 'C3', '--')])

    print(np.nanmax(reaching_time[:,:,k]), np.nanmax(convergence_time[:,:,k]))
    ax_D = letter('D',violinplot)(f, gs[1,0], [100*accuracy[:,:,:,-1][idx&np.isfinite(accuracy[:,:,:,-1])], 100*accuracy[:,:,:,-1][idx2&np.isfinite(accuracy[:,:,:,-1])]], names = [f'ΔQi>0', f'ΔQi=0'], xlabel = '', ylabel = f'accuracy (%)\nat epoch {n_epoch-1:d}', ylim = (0, 110), c = ['0.5']*4, sort = True, order = 1, pd = 'all')
    ax_E = letter('E',imshow)(f, gs[1,1], reaching_time[:,:,k], regions, deltas, xlabel = '', ylabel = 'ΔQi', clabel = 'threshold\nepoch', title = f'SNR = 10 dB', clim = (0,65))
    ax_F = letter('F',imshow)(f, gs[1,2], convergence_time[:,:,k], regions, deltas, xlabel = '', ylabel = 'ΔQi', clabel = 'convergeance\nepoch', title = f'SNR = 10dB', clim = (0,65))
    
    f.savefig(f'{figure_path}/figureS2.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure S2 of manuscript')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)
