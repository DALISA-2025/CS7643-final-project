from torch import nn
from torch.nn import Module, ModuleList

from Attention import Attention
from FeedForward import FeedForward


class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for i in range(depth):
            self.layers.append(ModuleList([Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),FeedForward(dim, mlp_dim, dropout = dropout)]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
