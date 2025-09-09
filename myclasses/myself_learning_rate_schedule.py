import math


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