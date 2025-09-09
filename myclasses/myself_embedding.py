import torch
import torch.nn as nn
import math

seed = 1
torch.manual_seed(seed)

class myself_embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # num_embeddings: Size of the vocabulary
        # embedding_dim: Dimension of the embedding vectors, i.e., d_model
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Each row of embedding.weight is the embedding vector for one index
        # embedding(torch.tensor([4])) == embedding.weight[4]
        sigma = math.sqrt(2 / (num_embeddings + embedding_dim))
        weight = torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        # !!!This will cause randomness so we have to set the random seed
        torch.nn.init.trunc_normal_(weight, std=sigma, a=-3 * sigma, b=3 * sigma)
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # print to show if the tokenizer is generating larger IDs than the embedding table allows
        print(f"The biggest token IDs generated is {token_ids.max().item()}, and the vocab size is {self.weight.shape[0]}")
        embedding_lookup = self.weight[token_ids]
        return embedding_lookup