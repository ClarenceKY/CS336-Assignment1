import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from myclasses import *

'''
Five steps:
1.Linear projections: for Q, K, and V using the given projection weight W_Q, W_K, W_V

2.Split into multiple heads

3.Scaled dot-product attention per head

4.Concatenate heads

5.Final projection with W_O
'''
class myself_multihead_self_attention(nn.Module):

    def __init__(self, d_model: int, num_heads: int,
                 theta=None, max_seq_len=None, apply_rope=None, device=None, dtype=None):
        '''
        d_model: int | Dimensionality of the Transformer block inputs
        num_heads: int | Number of heads to use in multi-head self-attention
        '''
        super().__init__()

        self.d_moel = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.apply_rope = apply_rope

        # according to the 'attention is all you need',
        # you need a projection matrix, which has learnable weights used later for optimizing,
        # before splitting into multiple heads.
        self.q_proj = myself_linear(d_model, d_model)
        self.k_proj = myself_linear(d_model, d_model)
        self.v_proj = myself_linear(d_model, d_model)
        self.output_proj = myself_linear(d_model, d_model)

        if self.apply_rope:
            # Notice! myself_rope class will use GPU to compute while other function uses cpu
            # so we need to use .to(device=torch.device("cpu")) to tackle with issue
            self.rope = myself_rope(d_k=self.d_k, theta=theta, max_seq_len=max_seq_len).to(device=torch.device("cpu"))
        else:
            self.rope = None

    def forward(self, in_features: Float[Tensor, " ... sequence_length d_in"],token_positions=None
        ) -> Float[Tensor, " ... sequence_length d_out"]:

        batch, seq_len, d_in = in_features.shape
        # generate the positions yourself since there are not given in transformer block
        if token_positions is None:
            token_positions = torch.arange(seq_len).view(1, 1, seq_len)
            #print(token_positions)

        Q = self.q_proj.forward(in_features)
        K = self.k_proj.forward(in_features)
        V = self.v_proj.forward(in_features)

        # num_heads should be treated as a "batch" kind and made as a dimension before seq_len
        def split_heads(tensor):
            return tensor.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)

        # RoPE should be applied to the query and key vectors, but not the value vectors.
        if self.apply_rope:
            Q = self.rope.forward(Q, token_positions)
            K = self.rope.forward(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch, self.num_heads, seq_len, seq_len)

        concat = myself_scaled_dot_product_attention(Q, K, V, mask)
        print("concat stats:", torch.isnan(concat).any())
        concat = concat.transpose(1, 2).contiguous().view(batch, seq_len, d_in)

        return self.output_proj.forward(concat)