from collections.abc import Callable, Iterable
from typing import Optional
from jaxtyping import Float, Int
from torch import Tensor
import torch
import math
import numpy as np

"""
This file defines all the training methods used.
"""


# def myself_cross_entropy(inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]):
#     """
#     Given a tensor of inputs and targets, compute the average cross-entropy loss across examples.
#
#     Args:
#         inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
#             unnormalized logit of jth class for the ith example.
#         targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
#             Each value must be between 0 and `num_classes - 1`.
#
#     Returns:
#             Float[Tensor, ""]: The average cross-entropy loss across examples.
#     """
#     logits = inputs - inputs.max(dim=-1, keepdim=True).values
#     log_probs = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
#     loss = -log_probs[torch.arange(inputs.size(0)), targets]
#
#     return loss.mean()

def myself_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: (batch_size, context_length, vocab_size)
    The predicted token ids after fitting the training data into LM

    targets: (batch_size, context_length)
    The true token ids

    returns: scalar mean loss over B*T tokens
    """
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    targets = targets.reshape(B * T)

    # numerically stable log-softmax
    logits = logits - logits.max(dim=1, keepdim=True).values
    #print(f"The logits is {logits}.")
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    #print(f"The log_probs is {log_probs}.")

    loss = -log_probs[torch.arange(B * T, device=logits.device), targets]
    return loss.mean()

class myself_AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-2, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                state = self.state[p]  # Get state associated with p.

                # Init state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)# update the first moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)# update the second moment estimate

                # Bias correction
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Apply weight decay (AdamW decouples weight decay)
                if group['weight_decay'] != 0:
                    p.data.add_(p, alpha=-group['lr'] * group['weight_decay'])

        return loss



def myself_gradient_clipping(parameters, max_l2_norm):

    '''
    In PyTorch training loops, parameters is usually an
       iterator of tensors (the model’s weights or their gradients).
    clipping should act on .grad, not directly on the parameter values
    '''
    grads = [p.grad for p in parameters if p.grad is not None]
    l2_norm = torch.sqrt(sum(torch.sum(g**2) for g in grads))

    clip_coef = max_l2_norm / (l2_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.mul_(clip_coef)



def myself_learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        lr = it/warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        lr = (min_learning_rate + 1/2 * (1 +
              math.cos(math.pi * (it-warmup_iters) / (cosine_cycle_iters-warmup_iters)))
              *(max_learning_rate-min_learning_rate))
    else:
        lr = min_learning_rate

    return lr


def myself_get_batch(dataset, batch_size, device) -> tuple[torch.Tensor, torch.Tensor]:

    # Random starting indices
    ix = np.random.randint(0, len(dataset), size=batch_size)

    # Each dataset[i] already returns (x,y) tensors
    batch = [dataset[i] for i in ix]
    xs, ys = zip(*batch)

    # Stack into batch tensors
    inputs = torch.stack(xs).to(device)
    targets = torch.stack(ys).to(device)

    return inputs, targets