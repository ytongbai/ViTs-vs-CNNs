import typing
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
# import global_val

class GhostBN2D_ADV(nn.Module):
    def __init__(self,
                 num_features,
                 *args,
                 virtual2actual_batch_size_ratio=2,
                 affine=False,
                 sync_stats=False,
                 **kwargs):
        super().__init__()
        self.virtual2actual_batch_size_ratio = virtual2actual_batch_size_ratio
        self.affine = affine
        self.num_features = num_features
        self.sync_stats = sync_stats
        self.proxy_bn = nn.BatchNorm2d(num_features *
                                       virtual2actual_batch_size_ratio,
                                       *args,
                                       **kwargs,
                                       affine=False)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

        # for mimic the behavior that different GPUs use different stats when eval
        self.eval_use_different_stats = False

    def reset_parameters(self) -> None:
        self.proxy_bn.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def get_actual_running_stats(self) -> typing.Tuple[Tensor, Tensor]:
        if not self.proxy_bn.track_running_stats:
            return None, None
        else:
            select_fun = {
                False: lambda x: x[0],
                True: lambda x: torch.mean(x, dim=0)
            }[self.sync_stats]
            return tuple(
                select_fun(
                    var.reshape(self.virtual2actual_batch_size_ratio,
                                self.num_features)) for var in
                [self.proxy_bn.running_mean, self.proxy_bn.running_var])

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            bn_training = True
        else:
            bn_training = (self.proxy_bn.running_mean is
                           None) and (self.proxy_bn.running_var is None)

        if bn_training or self.eval_use_different_stats:
            n, c, h, w = input.shape
            if n % self.virtual2actual_batch_size_ratio != 0:
                raise RuntimeError()
            proxy_input = input.reshape(
                int(n / self.virtual2actual_batch_size_ratio),
                self.virtual2actual_batch_size_ratio * c, h, w)
            proxy_output = self.proxy_bn(proxy_input)
            proxy_output = proxy_output.reshape(n, c, h, w)

            if self.affine:
                weight = self.weight
                bias = self.bias
                weight = weight.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
#                 print('proxy_output', proxy_output.shape)
                return proxy_output * weight + bias
            else:
                return proxy_output
        else:
#             print('running_mean', running_mean.shape)
            running_mean, running_var = self.get_actual_running_stats()
            
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                # won't update running_mean & running_var
                0.0,
                self.proxy_bn.eps)



class GhostBN2D(nn.Module):

    def __init__(self,
                 num_features,
                 *args,
                 virtual2actual_batch_size_ratio=1,
                 affine=True,
                 sync_stats=False,
                 **kwargs):
        super().__init__()
        self.virtual2actual_batch_size_ratio = virtual2actual_batch_size_ratio
        self.affine = affine
        self.num_features = num_features
        self.sync_stats = sync_stats
        self.proxy_bn = nn.BatchNorm2d(num_features *
                                       virtual2actual_batch_size_ratio,
                                       *args,
                                       **kwargs,
                                       affine=False)
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        del self.proxy_bn.weight
        del self.proxy_bn.bias
        self.reset_parameters()

        # for mimic the behavior that different GPUs use different stats when eval
        self.eval_use_different_stats = False

    def reset_parameters(self) -> None:
        self.proxy_bn.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def get_actual_running_stats(self) -> typing.Tuple[Tensor, Tensor]:
        if not self.proxy_bn.track_running_stats:
            return None, None
        else:
            select_fun = {
                False: lambda x: x[0],
                True: lambda x: torch.mean(x, dim=0)
            }[self.sync_stats]
            return tuple(
                select_fun(
                    var.reshape(self.virtual2actual_batch_size_ratio,
                                self.num_features)) for var in
                [self.proxy_bn.running_mean, self.proxy_bn.running_var])

    def _prepare_fake_weight_bias(self):
        if not self.affine:
            self.proxy_bn.weight = None
            self.proxy_bn.bias = None
        _fake_weight, _fake_bias = [
            var.unsqueeze(0).expand(self.virtual2actual_batch_size_ratio,
                                    self.num_features).reshape(-1)
            for var in [self.weight, self.bias]
        ]
        self.proxy_bn.weight = _fake_weight
        self.proxy_bn.bias = _fake_bias

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            bn_training = True
        else:
            bn_training = (self.proxy_bn.running_mean is
                           None) and (self.proxy_bn.running_var is None)

        if bn_training or self.eval_use_different_stats:
            n, c, h, w = input.shape
            if n % self.virtual2actual_batch_size_ratio != 0:
                raise RuntimeError()
            proxy_input = input.reshape(
                int(n / self.virtual2actual_batch_size_ratio),
                self.virtual2actual_batch_size_ratio * c, h, w)
            self._prepare_fake_weight_bias()
            proxy_output = self.proxy_bn(proxy_input)
            proxy_output = proxy_output.reshape(n, c, h, w)
            return proxy_output
        else:
            running_mean, running_var = self.get_actual_running_stats()
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                # won't update running_mean & running_var
                0.0,
                self.proxy_bn.eps)