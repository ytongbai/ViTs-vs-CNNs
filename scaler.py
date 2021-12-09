""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

try:
    from apex import amp
    has_apex = True
except ImportError:
    amp = None
    has_apex = False

class AccumulativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, forward_time=8):
        self._scaler = torch.cuda.amp.GradScaler()
        self.forward_time = forward_time
    
    def set_iters(self, iteration):
        self.iteration = iteration

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False):
        
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
        if (self.iteration+1) % self.forward_time == 0:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
