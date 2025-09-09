import torch
from jaxtyping import Float, Int
from torch import Tensor
from myclasses import *

def myself_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    """
    Given a tensor of inputs and targets, compute the average cross-entropy loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
            Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    logits = inputs - inputs.max(dim=-1, keepdim=True).values
    log_probs  = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    loss = -log_probs[torch.arange(inputs.size(0)), targets]

    return loss.mean()
