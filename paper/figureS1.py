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
    model_path = f'{os.environ.get("DATA_PATH")}/rnn_exc/model'
    # path where figures are saved
    figure_path = f'{os.environ.get("FIG_PATH")}/paper/rnn_exc'
    # path where attribution 
    attr_path = f'{os.environ.get("DATA_PATH")}/rnn_exc/attr'

    os.makedirs(figure_path, exist_ok=True)

    # -------------------------------
    # Prepare data
    # -------------------------------
    n_affs = np.arange(1, 11) # np.arange(10, 51, 10)
    deltas = (np.arange(6)/10)[::-1] # np.arange(-8, 9, 2)/10
    snrs = np.array([10, 0, -10])# np.array([-10, -5, 0, 5, 10])
    accuracy = np.full((len(n_affs),len(deltas),len(snrs),1+n_epoch), float('nan'))
    reaching_time = np.full((len(n_affs),len(deltas),len(snrs)), float('nan'))
    convergence_time = np.full((len(n_affs),len(deltas),len(snrs)), float('nan'))
    for k, snr in enumerate(snrs):
        for i, n_aff in enumerate(n_affs):
            for j, delta in enumerate(deltas):
                model_name = f'simple_n{n_feat}_a{n_aff:d}_delta{delta:.2f}_snr{snr:.2f}'
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
                    reaching_time[i,j,k] = np.where(np.logical_and.reduce([accuracy[i,j,k,l:-n_consecutive+l+1]>reaching_threshold for l in range(n_consecutive-1)]+[accuracy[i,j,k,n_consecutive-1:]>reaching_threshold]))[0][0]
                    delta_acc = np.array([np.max(np.abs(accuracy[i,j,k,l]-accuracy[i,j,k,l+1:])) for l in range(accuracy.shape[3]-1)])
                    convergence_time[i,j,k] = np.where(delta_acc<convergence_threshold)[0][0]
                except:
                    pass
                try:
                    find_attr(f'{attr_path}/ig/{model_name}_epoch*.pt')
                except:
                    print(f'Rerun test {model_name}')

    n_overlaps = np.arange(0, n_feat+1, 10)
    _snrs = np.tile(snrs[None,None,:], (len(n_affs),len(deltas),1))
    _n_affs = np.tile(n_affs[:,None,None], (1,len(deltas),len(snrs)))
    _deltas = np.tile(deltas[None,:,None], (len(n_affs),1,len(snrs)))
    idx = _deltas > 0.0
    idx2 = _deltas == 0.0

    best_accuracy = accuracy[:,:,:,-1] #np.max(accuracy, axis = -1)
    idx_a, idx_delta, idx_snr = np.where(best_accuracy<0.6)

    # -------------------------------
    # Display
    # -------------------------------
    
    f = plt.figure(figsize = figsize)
    gs = f.add_gridspec(len(hs), len(ws), height_ratios = hs, hspace = hspace/np.mean(hs), width_ratios = ws, wspace = wspace/np.mean(ws), left = _ws[0]/np.sum(_ws), right = 1.0-_ws[-1]/np.sum(_ws), bottom = _hs[-1]/np.sum(_hs), top = 1.0-_hs[0]/np.sum(_hs))
    #i,j,k = -1,0,0 # SNR = 10dB, prevalence = 10%, δ = 0.5
    i,j,k = 4,-2,0 # SNR = 10dB, prevalence = 5%, δ = 0.1
    print(f'SNR = {snrs[k]}dB, prevalence = {n_affs[i]}%, δ = {deltas[j]}')
    xlim = (-1.5,20.5) #(-0.5,15.5)
    ax_A = letter('A',plot_multiple)(f, gs[0,0], np.arange(-1, n_epoch), 100*accuracy[i,j], snrs, xlabel = 'epoch', ylabel = 'accuracy (%)', clabel = 'SNR (in dB)', ylim = (40, 110), title = f'prevalence = {n_affs[i]:d}%, $\mathbf{{\delta}}$ = {deltas[j]}', xlim = xlim, order = lambda i: -i, highlightx = [], highlighty = [(50, '0.5', '--'), (100*reaching_threshold, 'C3', '--')])
    ax_A.text(xlim[1], 50, 'chance level', va = 'bottom', ha = 'right', color = '0.5')
    ax_A.text(xlim[1], 100*reaching_threshold, 'threshold', va = 'top', ha = 'right', color = 'C3')
    ax_B = letter('B',plot_multiple)(f, gs[0,1], np.arange(-1, n_epoch), 100*accuracy[:,j,k], n_affs, xlabel = 'epoch', ylabel = 'accuracy (%)', clabel = 'prevalence (%)', ylim = (40, 110), clim = (1, 10), title = f'SNR = {snrs[k]:d}dB, $\mathbf{{\delta}}$ = {deltas[j]}', xlim = xlim, highlightx = [], highlighty = [(50, '0.5', '--'), (100*reaching_threshold, 'C3', '--')])
    ax_C = letter('C',plot_multiple)(f, gs[0,2], np.arange(-1, n_epoch), 100*accuracy[i,:,k], deltas, xlabel = 'epoch', ylabel = 'accuracy (%)', clabel = '$\delta$', ylim = (40, 110), clim = (0.0, 0.5), title = f'SNR = {snrs[k]:d}dB, prevalence = {n_affs[i]:d}%', xlim = xlim, order = lambda i: -i, highlightx = [], highlighty = [(50, '0.5', '--'), (100*reaching_threshold, 'C3', '--')])

    ax_D = letter('D',violinplot)(f, gs[1,0], [100*accuracy[:,:,:,-1][idx&np.isfinite(accuracy[:,:,:,-1])], 100*accuracy[:,:,:,-1][idx2&np.isfinite(accuracy[:,:,:,-1])]], names = [f'δ>0', f'δ=0'], xlabel = '', ylabel = f'accuracy (%)\nat epoch {n_epoch-1:d}', ylim = (40, 110), c = ['0.5']*4, sort = True, order = 1, pd = 'all')
    ax_E = letter('E',imshow)(f, gs[1,1], reaching_time[:,:,k], n_affs[::2], deltas, xlabel = 'prevalence (%)', ylabel = 'δ', clabel = 'threshold\nepoch', title = f'SNR = {snrs[k]}dB', clim = (0,50))
    ax_F = letter('F',imshow)(f, gs[1,2], convergence_time[:,:,k], n_affs[::2], deltas, xlabel = 'prevalence (%)', ylabel = 'δ', clabel = 'convergeance\nepoch', title = f'SNR = {snrs[k]}dB', clim = (0,50))
    f.savefig(f'{figure_path}/figureS1.png', dpi = 600)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Figure S1 of manuscript')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    main(args)
