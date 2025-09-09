import torch

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