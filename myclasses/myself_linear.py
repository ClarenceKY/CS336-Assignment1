import torch
import torch.nn as nn
import math

# class myself_linear(nn.Module):
#
#     def __init__(self, in_features, out_features, device=None, dtype=None):
#         # This is a convenient way to store and pass common tensor creation settings—specifically
#         # the device (e.g., CPU or GPU) and dtype (data type like torch.float32, torch.float16, etc.).
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#
#         '''
#         in_features: int | final dimension of the input, i.e. d_model
#         out_features: int | final dimension of the output, i.e. d_ff
#         '''
#         self.in_features = in_features
#         self.out_features = out_features
#         sigma = math.sqrt( 2/(in_features+out_features) )
#         weight = torch.empty(out_features, in_features, **factory_kwargs)
#         torch.nn.init.trunc_normal_(weight, std=sigma, a=-3 * sigma, b=3 * sigma)
#         self.weight = nn.Parameter(weight)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         #print(self.weight.shape)
#         #print(x.shape)
#         return torch.einsum("fm,...m->...f", self.weight, x)

class myself_linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features

        # match PyTorch nn.Linear initialization (Xavier uniform)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # init like nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, in_features) @ (out_features, in_features)^T → (B, T, out_features)
        return torch.matmul(x, self.weight.T) + (self.bias if self.bias is not None else 0.0)