import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from nn_analysis.dataset.fakefmri import FakeFmriDataSet, gen_from_file, NormalizeDataset, add_measurement_noise
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
from torchmetrics import Accuracy

def main(args):

	# -------------------------------
	# Parameters
	# -------------------------------

	seed = 0
	n_class = 2
	n_sample = 400 # number of samples used in training per class
	n_step = 100
	n_epoch = 150
	region = args.region
	gen_data = gen_from_file
	Qi_nt = args.qint
	Qi_asd = args.qiasd
	noise = args.noise
	snr = args.snr

	# -------------------------------
	# Paths where to load/save data
	# -------------------------------

	# path where the tvb model is saved
	data_path = f'{os.environ.get("OAK")}/projects/tanghiem/2022_TVB_AdEX/feature_attribution'
	if region in ['RSC', 'RSC_Cg', 'RSC_Cg_PrL']:
		n_feat = 426
		#params = [{'files': f'{data_path}/sig_BOLD__b_0_Qi_RSC{Qi:.1f}_repeatedAIstim_0.0EtoEIratio1.4_coupling0.15seed*.npy'} for Qi in [Qi_asd,Qi_nt]]
		params = [{'files': f'{data_path}/sig_BOLD__b_0_Qi_{region}_nodes{Qi:.1f}_repeatedAId_rightstim_0.0EtoEIratio1.4_coupling0.15seed*_noise{noise:.1e}.npy', 'transform':lambda x: add_measurement_noise(x, snr = snr)} for Qi in [Qi_asd,Qi_nt]]
	elif region in ['PCC', 'PCC_Pcun', 'PCC_Pcun_Ang']:
		n_feat = 68
		#params = [{'files': f'{data_path}/sig_BOLD__b_0_Qi_PCC{Qi:.1f}EtoEIratio1.4_coupling0.15seed*.npy'} for Qi in [Qi_asd,Qi_nt]]
		params = [{'files': f'{data_path}/sig_BOLD__b_0_Qi_{region}{Qi:.1f}_repeatedinsulastim_0.0EtoEIratio1.4_coupling0.15seed*_noise{noise:.1e}.npy', 'transform':lambda x: add_measurement_noise(x, snr = snr)} for Qi in [Qi_asd,Qi_nt]]
	else:
		raise NameError(f'Not implemented when E/I imbalance is in {region}')
	# path where model are saved
	model_name = f'tvb_{region}_asdQi_{Qi_asd:.1f}_ntQi_{Qi_nt:.1f}_noise{noise:.1e}_snr{snr:.2f}'
	model_path = f'{os.environ.get("DATA_PATH")}/tvb/model/{model_name}'
	os.makedirs(f'{model_path}', exist_ok=True)
	# path where log of training are saved
	log_path = f'{os.environ.get("TMP_PATH")}/tvb/log/train'

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

	gen_data = gen_from_file
	dataset = NormalizeDataset(FakeFmriDataSet(gen_data, params))
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

	model = GenericModel(model, metrics = {'acc': Accuracy(task="multiclass", num_classes=2, top_k = 1)})

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
	test_loss[0] = metrics['val_loss']
	test_accuracy[0] = metrics['val_acc']
	trainer.fit(model, train_loader, test_loader)
	torch.save(train_loss, f'{model_path}/train_loss.npy')
	torch.save(test_loss, f'{model_path}/test_loss.npy')
	torch.save(test_accuracy, f'{model_path}/test_accuracy.npy')
	

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train 1D CNN to discriminate simulated fMRI')
    parser.add_argument('--redo', action='store_true', help='If set, resave even if file exists')
    parser.add_argument('--region', metavar = 'R', type = str, default = 'PCC', help = 'Region(s) affected (PCC, PCC_Pcun, PCC_Pcun_Ang)')
    parser.add_argument('--noise', metavar = 'R', type = float, default = 0.001, help = 'Amplitude of noise')
    parser.add_argument('--qiasd', metavar = 'R', type = float, default = 4.5, help = 'Region(s) affected')
    parser.add_argument('--qint', metavar = 'R', type = float, default = 5.0, help = 'Region(s) affected')
    parser.add_argument('--snr', metavar = 'N', type = float, default = 0.0, help = 'SNR in dB')
    args = parser.parse_args()
    main(args)