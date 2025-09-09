import torch
import argparse
from types import SimpleNamespace
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import wandb
import json
import time
from myclasses import *
from myclasses.myself_BPE_tokenizer import MySelfTokenizer
from myclasses.myself_checkpoint import *
from myclasses.myself_training import *

seed = 3
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class MemmapDataset(Dataset):
    def __init__(self, data_path, shape, dtype=np.int64, context_length=128):
        # Assume dataset is tokenized integers (e.g. language modeling)
        self.data = np.memmap(data_path, mode="r", dtype=dtype, shape=shape)
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_length]
        y = self.data[idx + 1 : idx + 1 + self.context_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train(args):
    device = torch.device(args.device)

    # 1. Dataset
    train_dataset = MemmapDataset(args.train_data, shape=args.train_shape, context_length=args.context_length)
    val_dataset = MemmapDataset(args.val_data, shape=args.val_shape, context_length=args.context_length)

    # 2. Model + optimizer
    model = myself_transformer_lm(vocab_size=args.vocab_size, context_length=args.context_length,
                                  num_layers=args.num_layers, d_model=args.d_model,
                                  num_heads=args.num_heads, d_ff=args.d_ff, rope_theta=args.rope_theta)
    optimizer = myself_AdamW(model.parameters(),
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay,
                            eps=args.epsilon,
                            betas=(args.beta1, args.beta2))

    # 3. Optionally init W&B
    if args.use_wandb:
        wandb.init(project=args.project, config=vars(args))
        wandb.watch(model)

    # 4. Training loop
    torch.set_default_dtype(torch.float32)
    model = model.float()
    iteration = 0
    for epoch in range(args.epochs):
        model.train()
        print(f'There are total {len(train_dataset) // args.batch_size} steps.')
        for step in range(len(train_dataset) // args.batch_size):
            x, y = myself_get_batch(train_dataset, args.batch_size, args.device)
            print(f"The input shape is {x.shape} and the target shape is {y.shape}.")
            optimizer.zero_grad()
            logits = model(x)
            #print(f"The logits shape is {logits.shape}.")
            #print(f"The logits after model fitted is {logits}.")
            loss = myself_cross_entropy(logits, y)
            print(f"The loss of Iter {iteration} is {loss}.")
            loss.backward()
            myself_gradient_clipping(model.parameters(), args.max_grad_norm)
            optimizer.step()

            if iteration % args.log_interval == 0:
                print(f"[Epoch {epoch} Iter {iteration}] Train Loss: {loss.item():.4f}")
                if args.use_wandb:
                    wandb.log({"train_loss": loss.item(), "iteration": iteration})

            if iteration % args.val_interval == 0:
                val_loss = evaluate(model, val_dataset, args)
                print(f"[Epoch {epoch} Iter {iteration}] Val Loss: {val_loss:.4f}")
                if args.use_wandb:
                    wandb.log({"val_loss": val_loss, "iteration": iteration})

            if iteration % args.ckpt_interval == 0 and args.ckpt_dir:
                ckpt_path = f"{args.ckpt_dir}/ckpt_{iteration}.pt"
                myself_save_checkpoint(model, optimizer, iteration, ckpt_path)

            iteration += 1
    print(iteration)

def evaluate(model, dataset, args):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.eval_batches):  # add eval_batches to args
            x, y = myself_get_batch(dataset, args.batch_size, next(model.parameters()).device)
            logits = model(x)
            loss = myself_cross_entropy(logits, y)
            losses.append(loss.item())
    return sum(losses) / len(losses)

"""
=====================================================================================
"""

def get_tokenizer_path(vocab_path, merges_path, special_tokens):
    """
    Use the vocab and merges returned by trained BPE tokenizer on the dataset
    as the input to the Tokenizer for encoding and decoding.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = {int(k): v.encode("utf-8") for k, v in json.load(f).items()}

    next_id = max(vocab.keys()) + 1
    for b in range(256):
        bt = bytes([b])
        if bt not in vocab.values():
            vocab[next_id] = bt
            next_id += 1

    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines or comments
            parts = line.split()
            if len(parts) == 2:  # only valid pairs
                merges.append(tuple(parts))

    # Return your tokenizer class instance
    return MySelfTokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def prepare_dataset(tokenizer, input_path, output_path):
    """
    Use the Tokenizer to encode the text, which is the input data to LM
    """
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded {len(text)} characters from {input_path}")
    # Encode into tokens (list[int])
    tokens = tokenizer.encode(text)
    print(f"Encoded {len(tokens)} tokens, example: {tokens[:50]}")

    # Save as binary int64 file
    arr = np.array(tokens, dtype=np.int64)
    print(f"Array shape: {arr.shape}")
    arr.tofile(output_path)

    return len(arr)


def generate(model, tokenizer, prompt, max_new_tokens=100, device="cuda"):
    """
    Use the trained LM to generate the token ids given a prompt,
    and use our tokenizer to decode to text
    """
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]  # (1, seq)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.context_length:]
        with torch.no_grad():
            logits = model(x_cond)[:, -1, :]  # logits for last position
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    # Convert token IDs → text
    output_text = tokenizer.decode(x[0].tolist())
    return output_text


"""
=================================================================================================
Generation Code
"""

if __name__ == "__main__":
    # Generate the tokenizer
    vocab_path = "/Users/clarence_deng/PycharmProjects/assignment1-basics/data/vocab.json"
    merges_path = "/Users/clarence_deng/PycharmProjects/assignment1-basics/data/merges.txt"
    special_tokens = ["<|endoftext|>"]
    Tokenizer = get_tokenizer_path(vocab_path, merges_path, special_tokens)
    vocab_size = len(Tokenizer.vocab)

    # Generate the train and valid encoded data
    # train_len = prepare_dataset(Tokenizer, "/Users/clarence_deng/PycharmProjects/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", "/Users/clarence_deng/PycharmProjects/assignment1-basics/data/train.bin")
    # val_len = prepare_dataset(Tokenizer, "/Users/clarence_deng/PycharmProjects/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", "/Users/clarence_deng/PycharmProjects/assignment1-basics/data/val.bin")

    train_len = 2195140774 #given bt the array shape printed by prepare_dataset function
    val_len = 22171041

    # Train the LM
    args = SimpleNamespace(
        train_data="/Users/clarence_deng/PycharmProjects/assignment1-basics/data/train.bin",
        train_shape=(train_len,),
        val_data="/Users/clarence_deng/PycharmProjects/assignment1-basics/data/val.bin",
        val_shape=(val_len,),
        context_length=256,
        vocab_size=vocab_size,
        rope_theta=10000,
        batch_size=320,
        num_layers=4,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        learning_rate=3e-4,
        weight_decay=1e-2,
        epsilon=1e-8,
        beta1=0.9,
        beta2=0.95,
        epochs=1,
        max_grad_norm=1.0,
        eval_batches=50,
        device="cpu",
        log_interval=100,
        val_interval=1000,
        ckpt_interval=50,
        ckpt_dir="/Users/clarence_deng/PycharmProjects/assignment1-basics/data",
        use_wandb=False
    )

    #train(args)

    # Use the trained LM to generate text
    # Recreate model (must match architecture used in training)
    trained_model = myself_transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    )

    # Optimizer can be dummy if you don’t care about resuming training
    optimizer = myself_AdamW(trained_model.parameters(),
                             lr=args.learning_rate,
                             weight_decay=args.weight_decay,
                             eps=args.epsilon,
                             betas=(args.beta1, args.beta2))

    # Load checkpoint
    iteration = myself_load_checkpoint("/Users/clarence_deng/PycharmProjects/assignment1-basics/data/ckpt_100.pt", model=trained_model, optimizer=optimizer)

    # Put model on device + eval mode
    trained_model.to(args.device)
    trained_model.eval()

    # Use a prompt to generate the text
    # a = Tokenizer.encode("Once upon a time")
    # b = Tokenizer.encode("Bananas are blue?")
    # print(a[:40], b[:40], a == b)
    prompt = "Once upon a time."
    input_ids = Tokenizer.encode(prompt)  # e.g. [12, 523, 44, ...]
    #print(input_ids)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(args.device)

    with torch.no_grad():
        # ids_a = torch.tensor([Tokenizer.encode("Once upon a time")], dtype=torch.long).to(args.device)
        # ids_b = torch.tensor([Tokenizer.encode("Bananas are blue?")], dtype=torch.long).to(args.device)
        # la = trained_model(ids_a)[:, -1, :]  # (1, vocab)
        # lb = trained_model(ids_b)[:, -1, :]
        # print("logits close?", torch.allclose(la, lb, atol=1e-6))
        for _ in range(50):
            logits = trained_model(input_ids)  # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature to smooth/sharpen distribution
            temperature = 0.7
            probs = torch.softmax(logits / temperature, dim=-1)

            # Sample instead of argmax
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode to text
    output_text = Tokenizer.decode(input_ids[0].tolist())
    print(output_text)