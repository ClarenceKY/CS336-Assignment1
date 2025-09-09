import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from myclasses import *

class myself_posit_ff(nn.Module):

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):

        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = myself_linear(d_model, d_ff)
        self.w2 = myself_linear(d_ff, d_model)
        self.w3 = myself_linear(d_model, d_ff)

    def forward(self, in_features: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        SiLU = self.w1.forward(in_features) * torch.sigmoid(self.w1.forward(in_features))
        GLU = SiLU * self.w3.forward(in_features)
        FFN = self.w2.forward(GLU)

        return FFN