import math
import torch
from jaxtyping import Float, Int
from torch import Tensor

from myclasses import *

def myself_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"],
) -> Float[Tensor, " ... queries d_v"]:

    #print("Q stats:", Q.min().item(), Q.max().item(), torch.isnan(Q).any())
    #print("K stats:", K.min().item(), K.max().item(), torch.isnan(K).any())
    #print("V stats:", V.min().item(), V.max().item(), torch.isnan(V).any())

    d_k = Q.shape[-1]
    pre_softmax = torch.einsum("...qd,...kd->...qk", Q, K)/math.sqrt(d_k)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float("-1e9"))
    pre_softmax = pre_softmax.clamp(-1e9, 1e9)
    softmax = myself_softmax(pre_softmax, dim=-1)
    #print(f"The softmax is {softmax}.")

    dot_product = torch.einsum("...qk,...kd->...qd", softmax, V)
    #print("dot stats:", dot_product.min().item(), dot_product.max().item(), torch.isnan(dot_product).any())
    return dot_product