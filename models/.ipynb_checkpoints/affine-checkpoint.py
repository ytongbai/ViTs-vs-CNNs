import torch
import torch.nn as nn

class Affine(nn.Module):

    def __init__(self, width, *args, k=1, **kwargs):
        super(Affine, self).__init__()
        self.bnconv = nn.Conv2d(width,
                              width,
                              k,
                              padding=(k - 1) // 2,
                              groups=width,
                              bias=True)

    def forward(self, x):
        return self.bnconv(x)