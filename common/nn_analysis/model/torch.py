import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import functional as FM
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    def forward(self, x):
        return x

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert type(lambd) is types.LambdaType
        self.lambd = lambd

    def forward(self, *x):
        return self.lambd(*x)

class GenericModel(pl.LightningModule):
    def __init__(self, model, loss = nn.CrossEntropyLoss(), loss_params = {}, optimizer = 'adam', optimizer_params = {}, map_location=None, metrics = {}, show = None, average = True):
        super(GenericModel, self).__init__()
        self.model = model
        self.loss = loss
        self.loss_params = loss_params
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.metrics = nn.ModuleDict(metrics)
        self.show = {m: False for m in self.metrics}
        self.show['loss'] = True
        if isinstance(show, list):
            self.show.update({k:True for k in show})
        elif isinstance(show, dict):
            self.show.update(show)
        self.average = average

    def forward(self, x):
        if type(x) is list or type(x) is tuple:
            return self.model(*x)
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, pre = 'val_')
        return metrics

    def test_step(self, batch, batch_idx):
        loss, metrics = self._shared_eval_step(batch, batch_idx, pre = 'test_')
        return metrics

    def _shared_eval_step(self, batch, batch_idx, pre = ''):
        x = batch[0]
        y = batch[1]
        y_hat = self(x)
        _metrics = {f'{pre}{name}': m(y_hat, y) for name,m in self.metrics.items()}
        loss = _metrics[f'{pre}loss'] = self.loss(y_hat, y, **self.loss_params)
        if self.average:
            metrics = {name: torch.mean(m) for name,m in _metrics.items()}
        else:
            metrics = {name: m for name,m in _metrics.items() if len(m.shape)==0}
            for name,m in _metrics.items():
                if len(m.shape)==1:
                    for i in range(len(m)):
                        metrics[f'{name}_{i}'] = m[i]
        self.log_dict({k:v for k,v in metrics.items() if self.show[k[len(pre):]]}, on_step=False, on_epoch=True, sync_dist=True, prog_bar = True)
        self.log_dict({k:v for k,v in metrics.items() if not self.show[k[len(pre):]]}, on_step=False, on_epoch=True, sync_dist=True, prog_bar = False)
        return loss, metrics

    def configure_optimizers(self):
        if not 'params' in self.optimizer_params.keys():
            self.optimizer_params['params'] = self.parameters()
        if isinstance(self.optimizer, str):
            if self.optimizer == 'adam':
                optimizer = torch.optim.Adam(**self.optimizer_params)
            elif self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(**self.optimizer_params)
            elif self.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(**self.optimizer_params)
            else:
                raise NameError(f'Unknown optimizer {self.optimizer}')
        else:
            optimizer = self.optimizer(**self.optimizer_params)
        return optimizer

class ZeroMask(nn.Module):
    def __init__(self, mask):
        super(ZeroMask, self).__init__()
        self.mask = mask
    
    def forward(self, x):
        x[:,self.mask] = 0
        return x

class Noise(nn.Module):
    def __init__(self, std):
        super(Noise, self).__init__()
        self.std = std
    
    def forward(self, x):
        return x + torch.normal(0, self.std, size = x.shape, dtype = x.dtype).to(x.device)