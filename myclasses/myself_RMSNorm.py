import torch
import torch.nn as nn

class myself_RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        d_model: int | Hidden dimension of the model
        eps: float = 1e-5 | Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        '''
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        weight = torch.empty(self.d_model, **factory_kwargs)
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process an input tensor of shape (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # when we're doing row-wise (per-vector) normalization, use torch.rsqrt and mean(-1,keepdim=True)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # the expression below is totally wrong because the output is a float, not a tensor
        # RMS = math.sqrt( torch.sum(x ** 2) / self.d_model + self.eps )
        result = torch.einsum("...d,d->...d", x, self.weight) * rms
        return result.to(in_dtype)