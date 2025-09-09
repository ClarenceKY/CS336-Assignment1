import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from myclasses import *

# input tensor with shape(batch_size, seq_len, d_model)
class myself_transformer_lm(nn.Module):

    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
                 d_model, num_heads, d_ff, rope_theta=None, device=None, dtype=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = rope_theta

        # initialize the blocks (you do the initialize because you need the learnable weights to optimize)
        self.token_embeddings = myself_embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([
            myself_transformer_block(d_model=self.d_model, num_heads=self.num_heads,
                                     d_ff=d_ff, theta=self.theta, max_seq_len=self.max_seq_len)
            for _ in range(self.num_layers)
        ])
        self.ln_final = myself_RMSNorm(self.d_model)
        self.lm_head = myself_linear(self.d_model, self.vocab_size)

    def forward(self, in_features: Float[Tensor, " batch sequence_length d_model"])-> Float[Tensor,
                                                                    " batch sequence_length d_model"]:
        x = self.token_embeddings.forward(in_features)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final.forward(x)
        x = self.lm_head.forward(x)

        return x