import torch
import numpy as np

def myself_get_batch(dataset, batch_size, context_length, device) -> tuple[torch.Tensor, torch.Tensor]:
    # Random starting indices
    ix = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    # Gather sequences
    inputs = [dataset[i: i + context_length] for i in ix]
    targets = [dataset[i + 1: i + 1 + context_length] for i in ix]

    # Convert to tensors
    inputs = torch.tensor(np.array(inputs), dtype=torch.long, device=device)
    targets = torch.tensor(np.array(targets), dtype=torch.long, device=device)

    return inputs, targets