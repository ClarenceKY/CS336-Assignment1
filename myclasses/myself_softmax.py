import torch
from jaxtyping import Float, Int
from torch import Tensor

def myself_softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x  = x - x.max(dim=dim, keepdim=True).values
    softmax = torch.exp(x) / torch.sum(torch.exp(x), dim=dim, keepdim=True)

    return softmax