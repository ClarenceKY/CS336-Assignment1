from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

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