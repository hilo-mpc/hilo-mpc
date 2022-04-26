#   
#   This file is part of HILO-MPC
#
#   HILO-MPC is a toolbox for easy, flexible and fast development of machine-learning-supported
#   optimal control and estimation problems
#
#   Copyright (c) 2021 Johannes Pohlodek, Bruno Morabito, Rolf Findeisen
#                      All rights reserved
#
#   HILO-MPC is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   HILO-MPC is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with HILO-MPC. If not, see <http://www.gnu.org/licenses/>.
#

from typing import Optional
import warnings

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, Dropout, MSELoss, L1Loss, CosineSimilarity, SmoothL1Loss
from torch.nn import functional as F
from torch.optim import Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, RMSprop, Rprop, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from ...util.machine_learning import net_to_casadi_graph


class _MLP(Module):
    """Multilayer perceptron"""
    def __init__(self, n_features, n_labels, layers):
        """Constructor method"""
        super().__init__()

        hidden, activation = _process_layers(n_features, n_labels, layers)
        self._hidden = hidden
        self._activation = activation
        self._n_layers = len(layers)
        self._nodes = (n_features, ) + tuple(layer.nodes for layer in layers if layer.type != 'dropout') + (n_labels, )

    @property
    def shape(self):
        """

        :return:
        """
        return self._nodes

    def forward(self, x):
        """

        :param x:
        :return:
        """
        for k in range(self._n_layers):
            x = self._hidden[k](x)
            if self._activation[k] is not None:
                x = self._activation[k](x)
        # NOTE: Right now the output layer is assumed to be linear and cannot be changed
        x = self._hidden[-1](x)

        return x


class _DataSet(Dataset):
    """"""
    def __init__(self, x_data, y_data):
        """Constructor method"""
        self.features = x_data
        self.labels = y_data

    def __len__(self):
        """Length method"""
        return len(self.features)

    def __getitem__(self, item):
        """Item getter method"""
        return self.features[item, :], self.labels[item, :]


class _EarlyStopping:
    """"""
    def __init__(self, folder='.', patience=7, verbose=False, delta=0, debug=False):
        """Constructor method"""
        self._patience = patience
        self._verbose = verbose
        self._counter = 0
        self._best_score = None
        self.early_stop = False
        self._val_loss_min = np.inf
        self._delta = delta
        self._folder = folder
        self._debug = debug

    def __call__(self, val_loss, model):
        """Calling method"""
        score = -val_loss

        if self._best_score is None:
            self._best_score = score
            if not self._debug:
                self._save_checkpoint(val_loss, model)
        elif score < self._best_score + self._delta:
            self._counter += 1
            if self._verbose:
                print(f"Early stopping counter: {self._counter} out of {self._patience}")
            if self._counter >= self._patience:
                self.early_stop = True
        else:
            self._best_score = score
            if not self._debug:
                self._save_checkpoint(val_loss, model)
            self._counter = 0

    def _save_checkpoint(self, val_loss, model):
        """
        Saves model when validation loss decreases

        :param val_loss:
        :param model:
        :return:
        """
        if self._verbose:
            print(f"Test loss decreased ({self._val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), f'{self._folder}/checkpoint.pt')
        self._val_loss_min = val_loss


class MAPELoss(Module):
    """"""
    def __init__(self, dimension: int) -> None:
        """Constructor method"""
        super().__init__()

        self._ones = torch.ones(1, dimension)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """

        :param input:
        :param target:
        :return:
        """
        return mape_loss(input, target, ones=self._ones)


class MSLELoss(Module):
    """"""
    def __init__(self, dimension: int) -> None:
        """Constructor method"""
        super().__init__()

        self._ones = torch.ones(1, dimension)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """

        :param input:
        :param target:
        :return:
        """
        return msle_loss(input, target, ones=self._ones)


class LogCoshLoss(Module):
    """"""
    def __init__(self, eps: float = 1e-8) -> None:
        """Constructor method"""
        super().__init__()

        self._eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """

        :param input:
        :param target:
        :return:
        """
        return logcosh_loss(input, target, eps=self._eps)


def mape_loss(input: Tensor, target: Tensor, ones: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
    """

    :param input:
    :param target:
    :param ones:
    :param reduction:
    :return:
    """
    if ones is None:
        dimension = target.numel()
        ones = torch.ones(1, dimension)
    return torch.mul(F.l1_loss(ones, input / target, reduction=reduction), 100)


def msle_loss(input: Tensor, target: Tensor, ones: Optional[Tensor] = None, reduction: str = 'mean') -> Tensor:
    """

    :param input:
    :param target:
    :param ones:
    :param reduction:
    :return:
    """
    if ones is None:
        dimension = target.numel()
        ones = torch.ones(1, dimension)
    return F.mse_loss(torch.log(input + ones), torch.log(target + ones), reduction=reduction)


def logcosh_loss(input: Tensor, target: Tensor, eps: float = 1e-8, reduction: str = 'mean'):
    """

    :param input:
    :param target:
    :param eps:
    :param reduction:
    :return:
    """
    return torch.mean(torch.log(torch.cosh(target - input + eps)))


def rmse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """

    :param input:
    :param target:
    :param reduction:
    :return:
    """
    return torch.sqrt(F.mse_loss(input, target, reduction=reduction))


LOSS = {
    'mse': MSELoss,
    'mae': L1Loss,
    'mape': MAPELoss,
    'msle': MSLELoss,
    'cs': CosineSimilarity,
    'huber': SmoothL1Loss,
    'log-cosh': LogCoshLoss
}
OPTIMIZER = {
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'adam': Adam,
    'adamw': AdamW,
    'sparse_adam': SparseAdam,
    'adamax': Adamax,
    'asgd': ASGD,
    # 'ftrl': NotImplemented,
    'lbfgs': LBFGS,
    # 'nadam': NotImplemented,
    'rmsprop': RMSprop,
    'rprop': Rprop,
    'sgd': SGD
}
METRIC = {
    'mse': F.mse_loss,
    'rmse': rmse_loss,
    'mae': F.l1_loss,
    'mape': mape_loss,
    'msle': msle_loss,
    'cs': F.cosine_similarity,
    'log-cosh': logcosh_loss
}


class _PyTorchWrapper:
    """"""
    def __init__(self, module: Module, **kwargs) -> None:
        """Constructor method"""
        self._module = module

        self._seed = kwargs.get('seed')

        learning_rate = kwargs.get('learning_rate')
        if learning_rate is None:
            learning_rate = .001

        loss = kwargs.get('loss')
        if loss is None:
            loss = 'mse'
        loss = loss.lower()
        if loss in ['mape', 'msle']:
            self._loss = LOSS[loss.lower()](self._module.shape[-1])
        else:
            self._loss = LOSS[loss.lower()]()

        optimizer = kwargs.get('optimizer')
        if optimizer is None:
            optimizer = 'adam'
        self._optimizer = OPTIMIZER[optimizer.lower()](self._module.parameters(), lr=learning_rate)

        metric = kwargs.get('metric')
        if metric is None:
            metric = []
        self._metrics = [METRIC[k.lower()] for k in metric]
        self._metric_names = metric

        device = kwargs.get('device')
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device = torch.device(device)
        self._dtype = torch.float
        self._module.to(device=self._device, dtype=self._dtype)

        self._tensorboard = kwargs.get('tensorboard')
        self._tensorboard_comment = {
            'batch_size': 0,
            'learning_rate': learning_rate,
            'n_layers': len(self._module.shape) - 2,
            'nodes': self._module.shape[1:-1],
            'seed': self._seed
        }
        self._browser = kwargs.get('browser')

        self._train_loader = None
        self._validate_loader = None
        self._test_loader = None
        self._early_stopping = None
        self._scheduler = None
        self._writer = None

    def _get_data_loader(self, data, batch_size, shuffle):
        """

        :param data:
        :param batch_size:
        :param shuffle:
        :return:
        """
        x_data = torch.tensor(data[0], device=self._device, dtype=self._dtype)
        y_data = torch.tensor(data[1], device=self._device, dtype=self._dtype)
        data_set = _DataSet(x_data, y_data)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    def _preprocessing(
            self,
            data,
            validation_data,
            batch_size,
            verbose,
            patience,
            shuffle
    ):
        """

        :param data:
        :param validation_data:
        :param batch_size:
        :param verbose:
        :param patience:
        :param shuffle:
        :return:
        """
        self._train_loader = self._get_data_loader(data, batch_size, shuffle)
        # TODO: Does it make sense to shuffle validation data
        self._test_loader = self._get_data_loader(validation_data, batch_size, False)

        if patience is not None and len(self._test_loader) > 0:
            if verbose > 1:
                self._early_stopping = _EarlyStopping(patience=patience, verbose=True)
            else:
                self._early_stopping = _EarlyStopping(patience=patience)
            self._scheduler = ReduceLROnPlateau(optimizer=self._optimizer, patience=int(patience / 2))
        elif patience is not None and len(self._test_loader) == 0:
            warnings.warn("Early stopping is only available when validation data is supplied for the training of the "
                          "neural network")

        if self._tensorboard is not None:
            self._tensorboard_comment['batch_size'] = batch_size
            self._writer = self._tensorboard.get_summary_writer(**self._tensorboard_comment)
            if self._browser is not None:
                self.show_tensorboard()

    def _train(self, epoch, train_losses, train_metrics, verbose):
        """

        :param epoch:
        :param train_losses:
        :param train_metrics:
        :param verbose:
        :return:
        """
        running_loss = 0.
        running_metric = torch.zeros(len(self._metrics), device=self._device, dtype=self._dtype)

        # TODO: Need a better variable name
        batches = len(self._train_loader)

        self._module.train()

        for idx, (x, y) in enumerate(self._train_loader):
            self._optimizer.zero_grad()
            pred = self._module(x)
            # TODO: Constraint rates to zero
            loss = self._loss(pred, y)
            metric = [metric(pred, y) for metric in self._metrics]
            loss.backward()
            self._optimizer.step()
            running_loss += loss
            running_metric += torch.tensor(metric, device=self._device, dtype=self._dtype)
            if verbose == 1:
                _progress_bar(idx + 1, batches, running_loss / (idx + 1), self._metric_names,
                              running_metric / (idx + 1), 30, "=", "\r")
        if verbose == 1:
            print()

        train_loss = running_loss / batches
        train_metric = running_metric / batches
        train_losses.append(train_loss.detach().cpu().numpy())
        train_metrics.append(train_metric.detach().cpu().numpy())

        if verbose == 2:
            _one_line_per_epoch(batches, train_loss, self._metric_names, train_metric)

        if self._writer is not None:
            self._writer.add_scalar('train_loss', train_loss, epoch)
            for k in range(len(self._metrics)):
                self._writer.add_scalar('train_metric_' + str(k + 1), train_metric[k], epoch)
            for name, param in self._module.named_parameters():
                self._writer.add_histogram(name, param, epoch)
                self._writer.add_histogram(f'{name}.grad', param.grad, epoch)

    def _test(self, data_loader, epoch, test_losses, test_metrics, verbose):
        """

        :param data_loader:
        :param epoch:
        :param test_losses:
        :param test_metrics:
        :param verbose:
        :return:
        """
        running_loss = 0.
        running_metric = torch.zeros(len(self._metrics), device=self._device, dtype=self._dtype)

        # TODO: Need a better variable name
        batches = len(data_loader)

        self._module.eval()

        with torch.no_grad():
            for idx, (x, y) in enumerate(data_loader):
                self._optimizer.zero_grad()
                pred = self._module(x)
                # TODO: Constraint rates to zero
                metric = [metric(pred, y) for metric in self._metrics]
                running_loss += self._loss(pred, y)
                running_metric += torch.tensor(metric, device=self._device, dtype=self._dtype)
                if verbose == 1:
                    _progress_bar(idx + 1, batches, running_loss / (idx + 1), self._metric_names,
                                  running_metric / (idx + 1), 30, "=", "\r")
            if verbose == 1:
                print()

            test_loss = running_loss / batches
            test_metric = running_metric / batches
            test_losses.append(test_loss.detach().cpu().numpy())
            test_metrics.append(test_metric.detach().cpu().numpy())

            if verbose == 2:
                _one_line_per_epoch(batches, test_loss, self._metric_names, test_metric)

            if self._early_stopping is not None:
                self._early_stopping(test_loss, self._module)
                self._scheduler.step(test_loss)

            if self._writer is not None:
                self._writer.add_scalar('test_loss', test_loss, epoch)
                for k in range(len(self._metrics)):
                    self._writer.add_scalar('test_metric_' + str(k + 1), test_metric[k], epoch)

    def train(
            self,
            data,
            validation_data,
            batch_size,
            epochs,
            verbose,
            patience,
            shuffle
    ):
        """

        :param data:
        :param validation_data:
        :param batch_size:
        :param epochs:
        :param verbose:
        :param patience:
        :param shuffle:
        :return:
        """
        self._preprocessing(data, validation_data, batch_size, verbose, patience, shuffle)

        train_losses = []
        train_metrics = []
        test_losses = []
        test_metrics = []

        for epoch in range(epochs):
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")

            self._train(epoch, train_losses, train_metrics, verbose)

            if len(self._test_loader) > 1:
                if verbose > 0:
                    print("Evaluate on validation data")
                loss, metrics = self.evaluate(epoch=epoch, verbose=verbose)
                test_losses.extend(loss)
                test_metrics.extend(metrics)

            if self._early_stopping is not None and self._early_stopping.early_stop:
                if verbose > 0:
                    print("Early stopping")
                break

        if self._writer is not None:
            self._writer.flush()
            self._writer.close()

    def evaluate(
            self,
            data=None,
            batch_size=None,
            epoch=1,
            verbose=1
    ):
        """

        :param data:
        :param batch_size:
        :param epoch:
        :param verbose:
        :return:
        """
        if data is not None:
            data_loader = self._get_data_loader(data, batch_size, False)
        else:
            data_loader = self._test_loader

        test_losses = []
        test_metrics = []

        self._test(data_loader, epoch, test_losses, test_metrics, verbose)

        return test_losses, test_metrics

    def build_graph(self, x, layers, input_scaling=None, output_scaling=None):
        """

        :param x:
        :param layers:
        :param input_scaling:
        :param output_scaling:
        :return:
        """
        return net_to_casadi_graph(self, x, layers, input_scaling=input_scaling, output_scaling=output_scaling)

    def get_weights_and_bias(self):
        """

        :return:
        """
        weights = []
        bias = []

        for name, param in self._module.named_parameters():
            if name.endswith('weight'):
                weights.append(param.detach().cpu().numpy())
            elif name.endswith('bias'):
                bias.append(param.detach().cpu().numpy())

        return weights, bias

    def save_net(self, path_to_file):
        """

        :param path_to_file:
        :return:
        """
        if not path_to_file.endswith('.pt'):
            path_to_file += '.pt'
        torch.save(self._module.state_dict(), path_to_file)

    def show_tensorboard(self, browser: Optional[str] = None) -> None:
        """

        :param browser:
        :return:
        """
        if browser is None:
            browser = self._browser
        browser = browser.lower()

        if self._tensorboard is not None:
            self._tensorboard.show(browser)
        else:
            warnings.warn("No TensorBoard was saved. If you want to see a TensorBoard set the corresponding flag "
                          "during setup and train again.")

    def close_tensorboard(self):
        """

        :return:
        """
        if self._tensorboard is not None:
            self._tensorboard.close()


LAYER = {
    'dense': Linear,
    'dropout': Dropout
}
INITIALIZER = {
    'uniform': torch.nn.init.uniform_,
    'normal': torch.nn.init.normal_,
    'constant': torch.nn.init.constant_,
    'ones': torch.nn.init.ones_,
    'zeros': torch.nn.init.zeros_,
    'eye': torch.nn.init.eye_,
    'xavier_uniform': torch.nn.init.xavier_uniform_,
    'xavier_normal': torch.nn.init.xavier_normal_,
    'kaiming_uniform': torch.nn.init.kaiming_uniform_,
    'kaiming_normal': torch.nn.init.kaiming_normal_,
    'orthogonal': torch.nn.init.orthogonal_,
    'sparse': torch.nn.init.sparse_
}
ACTIVATION = {
    'relu': torch.relu,
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': torch.softmax
}


def _process_layers(n_features: int, n_labels: int, layers: list) -> (ModuleList, list):
    """

    :param n_features:
    :param n_labels:
    :param layers:
    :return:
    """
    # NOTE: Right now the output layer is assumed to be linear and cannot be changed
    n_inputs = n_features
    hidden = []
    activation = []
    for layer in layers:
        type_ = layer.type.lower()
        if type_ == 'dropout':
            hidden.append(LAYER[type_](p=layer.rate))
            activation.append(None)
        else:
            layer_ = LAYER[type_](n_inputs, layer.nodes)
            initializer_ = layer.initializer
            if isinstance(initializer_, str):
                initializer = INITIALIZER[initializer_.lower()]
                initializer(layer_.weight)
                initializer(layer_.bias)
            elif isinstance(initializer_, dict):
                key = list(initializer_)[0]
                initializer = INITIALIZER[key]
                val = initializer_[key]
                if isinstance(val, dict):
                    initializer(layer_.weight, **val)
                    initializer(layer_.bias, **val)
                else:
                    initializer(layer_.weight, val)
                    initializer(layer_.bias, val)
            hidden.append(layer_)
            activation.append(ACTIVATION[layer.activation.lower()])
            n_inputs = layer.nodes
    hidden.append(LAYER['dense'](n_inputs, n_labels))
    return ModuleList(hidden), activation


def _progress_bar(iteration, total, loss, metric_names, metrics, length, fill, print_end):
    """

    :param iteration:
    :param total:
    :param loss:
    :param metric_names:
    :param metrics:
    :param length:
    :param fill:
    :param print_end:
    :return:
    """
    metrics_string = ""
    if metric_names:
        for k, metric in enumerate(metrics):
            metrics_string += f" - {metric_names[k]}: {metric:.4f}"
    filled_length = int(length * iteration // total)
    unfilled_length = length - filled_length
    bar = fill * filled_length + ">" * bool(unfilled_length) + "." * (unfilled_length - 1)
    print(f"\r{iteration}/{total} [{bar}] - loss: {loss:.4f}{metrics_string}", end=print_end)


def _one_line_per_epoch(total, loss, metric_names, metrics):
    """

    :param total:
    :param loss:
    :param metric_names:
    :param metrics:
    :return:
    """
    metrics_string = ""
    if metric_names:
        for k, metric in enumerate(metrics):
            metrics_string += f" - {metric_names[k]}: {metric:.4f}"
    print(f"{total}/{total} - loss: {loss:.4f}{metrics_string}")


MODULES = {
    'mlp': _MLP
}


def get_wrapper(kind, *args, **kwargs):
    """

    :param kind:
    :param args:
    :param kwargs:
    :return:
    """
    seed = kwargs.get('seed')
    if seed is not None:
        torch.manual_seed(seed)

    module = MODULES[kind.lower()](*args)

    return _PyTorchWrapper(module, **kwargs)


__all__ = [
    'get_wrapper'
]
