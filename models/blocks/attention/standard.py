import torch
import torch.nn.functional as F
import torch.nn as nn
from ..normalization.normalization import RMSNorm


class StandardAttention(nn.Module):
    def __init__(self, args):
        super(StandardAttention).__init__()

        self.n_heads = args.heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.w_q = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.w_k = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.w_k = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.w_o = nn.Linear(self.n_heads*self.head_dim, self.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        bs, seq_len, dim = x.shape

        xq = self.w_q(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.w_k(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.w_o(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        xq = rope.apply_rotary_emb(xq)
        xk = rope.apply_rotary_emb(xk)

