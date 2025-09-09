import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from myclasses import *

# input tensor with shape(batch_size, seq_len, d_model)
class myself_transformer_block(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None,
                 device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len

        # initialize the functions in the block
        # (you do the initialize because you need the learnable weights to optimize)
        '''
        In order to load the weight dictionary, you need to match the keys both in names and orders.
        
        The given weight dictionary is: dict_keys(['attn.q_proj.weight', 'attn.k_proj.weight', 'attn.v_proj.weight', 
        'attn.output_proj.weight', 'ffn.w1.weight', 'ffn.w2.weight', 'ffn.w3.weight', 'ln1.weight', 'ln2.weight'])
        
        So we need to initialize the blocks as the following order.
        '''
        self.attn = myself_multihead_self_attention(d_model=d_model, num_heads=num_heads,
                                                    theta=theta, max_seq_len=max_seq_len, apply_rope=True)
        self.ffn = myself_posit_ff(self.d_model, self.d_ff)
        self.ln1 = myself_RMSNorm(self.d_model) #1st normalization
        self.ln2 = myself_RMSNorm(self.d_model) #2nd normalization

    def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"])-> Float[Tensor,
                                                                    " batch sequence_length d_model"]:

        step_first_norm = self.ln1.forward(in_features)
        step_multihead = self.attn.forward(step_first_norm)
        step_first_add = in_features + step_multihead

        step_second_norm = self.ln2.forward(step_first_add)
        step_ffn = self.ffn.forward(step_second_norm)
        step_second_add = step_first_add + step_ffn

        return step_second_add