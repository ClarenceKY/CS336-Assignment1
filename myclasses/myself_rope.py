import torch
import torch.nn as nn

class myself_rope(nn.Module):

    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        # This is a convenient way to store and pass common tensor creation settings—specifically
        # the device (e.g., CPU or GPU) and dtype (data type like torch.float32, torch.float16, etc.).
        super().__init__()

        '''
        theta: float | Θ value for the RoPE
        d_k: int | dimension of query and key vectors
        max_seq_len: int | Maximum sequence length that will be inputted
        '''
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # for each token position i(total number is max_seq_len),
        # there should be one block-diagonal matrix of size d_k × d_k
        half_dim = self.d_k // 2
        k = torch.arange(half_dim, device=device)  # k: 0 to d/2 - 1
        angle_rate = 1.0 / (self.theta ** (2 * k / self.d_k))
        # Create a [max_seq_len, dim/2] matrix of theta angles
        # 2d pre-computed buffer of sin and cos values
        positions = torch.arange(self.max_seq_len, device=device).unsqueeze(1)  # [max_pos, 1]
        angles = positions * angle_rate.unsqueeze(0)  # [max_pos, dim/2]
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)  # [max_pos, dim/2]
        #print(self.cos_cached.size()) #check the cos_cached matched with token_positions
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)  # [max_pos, dim/2]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape
        '''
        *leading_dims, seq_len, dim = x.shape
        half_dim = dim // 2
        x_ = x.view(*leading_dims, seq_len, half_dim, 2)
        x_even = x_[..., 0]
        x_odd = x_[..., 1]

        # Look up sin/cos from precomputed buffer
        cos = self.cos_cached[token_positions]  # [..., seq_len, dim/2]
        sin = self.sin_cached[token_positions]  # [..., seq_len, dim/2]
        # Apply rotation
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        x_rotated = torch.stack([x_rot_even, x_rot_odd], dim=-1).view(*leading_dims, seq_len, dim)
        return x_rotated