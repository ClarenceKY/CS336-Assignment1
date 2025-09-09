import torch
from jaxtyping import Int

def myself_save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def myself_load_checkpoint(src: str,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    # Load checkpoint dictionary
    checkpoint = torch.load(src)

    # Restore model and optimizer states
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint['iteration']