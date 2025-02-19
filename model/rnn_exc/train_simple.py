import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from nn_analysis.dataset.fakefmri import FakeFmriDataSet, gen_rnn_from_adjacency, hrf_filt
from sklearn.model_selection import train_test_split
from nn_analysis.model.torch import GenericModel, Flatten
import numpy as np
from functools import reduce
import sys
from pprint import pprint
from collections import OrderedDict
from torch import nn
import argparse
from termcolor import colored
import networkx as nx

def main(args):

	# -------------------------------
	# Parameters
	# -------------------------------

	seed = 0
	n_class = 2
	n_sample = 400 # number of samples used in training per class
	n_feat = args.n[0]
	n_step = 100
	delta = args.delta[0]
	n_aff = args.a[0]
	n_epoch = 150
	snr = args.snr[0] # in dB

	# -------------------------------
	# Paths where to load/save data
	# -------------------------------

	# path where model are saved
	model_name = f'simple_n{n_feat}_a{n_aff:d}_delta{delta:.2f}_snr{snr:.2f}'
	model_path = f'{os.environ.get("DATA_PATH")}/rnn_exc/model/{model_name}'
	os.makedirs(f'{model_path}', exist_ok=True)
	# path where log of training are saved
	log_path = f'{os.environ.get("TMP_PATH")}/rnn_exc/log/train'

	# -------------------------------
	# Speed up
	# -------------------------------

	files = [f'{model_path}/{name}.npy' for name in ['train_loss','test_loss','test_accuracy']]
	if not args.redo and np.all([os.path.exists(path) for path in files]):
		print(colored(f'{model_name} already trained', 'green'))
		sys.exit(0)
	else:
		print(colored(f'Training {model_name}', 'green'))
		
	# -------------------------------
	# Training/Test dataset
	# -------------------------------

	pl.seed_everything(seed)

	As = [np.zeros((n_feat, n_feat)) for i in range(n_class)]
	G = nx.connected_watts_strogatz_graph(n_feat, 10, 0.1) if n_feat > 10 else nx.connected_watts_strogatz_graph(n_feat, 2, 0.1)
	A = nx.adjacency_matrix(G).todense()
	_radius = max(np.abs(np.linalg.eigvals(A)))
	A = A/_radius
	As[0][...] = A
	As[1][...] = A
	A_max = np.max(A)
	As[1][:n_aff] += delta*A_max
	hrf = np.float32(np.load(f'{os.environ.get("DATA_PATH")}/../matlab/hrf_dt1.0.npy')[:, 0])

	gen_data = gen_rnn_from_adjacency
	params = [{'n_sample': n_sample, 'A': A, 'n_step': n_step, 'noise': 0.1, 'radius': None, 'transform': lambda x: hrf_filt(x, hrf, snr = snr)} for A in As]

	dataset = FakeFmriDataSet(gen_data, params)
	train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.label)
	train_dataset = Subset(dataset, train_idx)
	test_dataset = Subset(dataset, test_idx)

	train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True, num_workers = 4, pin_memory = True)
	test_loader = DataLoader(test_dataset, batch_size = 128, shuffle = False, num_workers = 4, pin_memory = True)

	# -------------------------------
	# Initializing model
	# -------------------------------

	model  = nn.Sequential(OrderedDict([
			('conv1', nn.Conv1d(n_feat, 32, 3)),
			('nonlin1', nn.ReLU()),
			('avgpool1', nn.AvgPool1d(5)),
			('conv2', nn.Conv1d(32, 32, 3)),
			('nonlin2', nn.ReLU()),
			('avgpool2', nn.AdaptiveAvgPool1d(1)),
			('flatten', Flatten()),
			('lin', nn.Linear(32, n_class))
		]))

	model = GenericModel(model)

	# -------------------------------
	# Saving model
	# -------------------------------

	train_loss = torch.zeros((n_epoch,))
	test_loss = torch.zeros((1+n_epoch,))
	test_accuracy = torch.zeros((1+n_epoch,))
	class MetricsCallback(pl.Callback):
		def on_train_epoch_end(self, *args, **kwargs):
			metrics = trainer.callback_metrics
			train_loss[trainer.current_epoch] = metrics['loss']
			test_loss[trainer.current_epoch+1] = metrics['val_loss']
			test_accuracy[trainer.current_epoch+1] = metrics['val_acc']

	# saving initial model
	torch.save({
		"epoch": -1,
		"global_step": 0,
		"pytorch-lightning_version": pl.__version__,
		"state_dict": model.state_dict()
	}, f'{model_path}/epoch-1.ckpt')
	# using checkpoint to save models after each epoch
	checkpoint = pl.callbacks.ModelCheckpoint(dirpath = model_path, filename = 'epoch{epoch:02d}', auto_insert_metric_name = False, save_on_train_epoch_end = True, save_top_k = 1, monitor = 'val_loss')
	# saving gpu stats
	gpu_stats = pl.callbacks.DeviceStatsMonitor()
	# print
	metricscb = MetricsCallback()

	# -------------------------------
	# Training model
	# -------------------------------
	
	trainer = pl.Trainer(default_root_dir = log_path, callbacks = [gpu_stats, checkpoint, metricscb], deterministic = True, accelerator='gpu', devices=1, strategy = "ddp", num_nodes = 1, max_epochs=n_epoch)	
	metrics, = trainer.validate(model, test_loader)
	test_loss[0] = metrics['val_loss_epoch']
	test_accuracy[0] = metrics['val_acc_epoch']
	trainer.fit(model, train_loader, test_loader)
	torch.save(train_loss, f'{model_path}/train_loss.npy')
	torch.save(test_loss, f'{model_path}/test_loss.npy')
	torch.save(test_accuracy, f'{model_path}/test_accuracy.npy')
	

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train 1D CNN to discriminate simulated fMRI')
    parser.add_argument('--redo', action='store_true', help='If set, resave even if file exists')
    parser.add_argument('--n', metavar = 'I', type = int, nargs = 1, default = 246, help = 'number of nodes')
    parser.add_argument('--a', metavar = 'A', type = int, nargs = 1, help = 'number of nodes affected by local imbalance')
    parser.add_argument('--delta', metavar = 'D', type = float, nargs = 1, default = [0], help = 'difference between gain')
    parser.add_argument('--snr', metavar = 'N', type = float, nargs = 1, default = [0.0], help = 'SNR in dB')
    args = parser.parse_args()
    main(args)