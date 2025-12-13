import math

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
        # GQA 分组因子
        self.n_rep = self.n_heads // self.n_kv_heads

        self.w_q = nn.Linear(args.dim, args.n_heads*self.head_dim, bias=False)
        self.w_k = nn.Linear(args.dim, args.n_kv_heads*self.head_dim, bias=False)
        self.w_k = nn.Linear(args.dim, args.n_kv_heads*self.head_dim, bias=False)
        self.w_o = nn.Linear(self.n_heads*self.head_dim, self.dim, bias=False)
        self.resid_dropout = nn.Dropout(args.dropout)
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x, rope, layer_idx, kv_cache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        # layer_idx kv_cache start_pos 都用于生成式推理
        if paged_attention_inputs:
            return self._forward_paged(x, rope, layer_idx, paged_attention_inputs)

        bs, seq_len, dim = x.shape

        xq = self.w_q(x).view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = self.w_k(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = self.w_o(x).view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        xq = rope.apply_rotary_emb(xq)
        xk = rope.apply_rotary_emb(xk)

        if kv_cache:
            keys, values = kv_cache.update(layer_idx, start_pos, xk, xv)
        else:
            keys, values = xk, xv

        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # seq_len == 1 推理时
        if seq_len > 1:
            current_seq_len = seq_len + start_pos
            scores = scores + self.mask[:, :, start_pos:current_seq_len, start_pos:current_seq_len]

        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, values).type_as(xq)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, dim)
        return self.resid_dropout(self.w_o(output))

    def _forward_paged(self, x, rope, layer_idx, paged_inputs):
        # 用于推理时处理长序列，通过分块KV缓存节省内存    分页cache [num_blocks, n_layers, n_kv_heads, block_size, head_dim] 块表 [bs, max_blocks_per_seq] 记录每个序列用了哪些内存块
        # x [total_tokens, dim]
        # paged_inputs包含: positions, tokens_per_seq, context_lengths, k_cache, v_cache, block_tables
        # 每个token位置 新token数量 序列总长度 kv cache 块表（每个序列的块映射）
        positions, tokens_per_seq, context_lengths, k_cache, v_cache, block_tables = paged_inputs
        # （total_tokens, n_heads, head_dim)
        xq = self.w_q(x).view(-1, self.n_heads, self.head_dim)
        xk = self.w_k(x).view(-1, self.n_heads, self.head_dim)
        xv = self.w_o(x).view(-1, self.n_heads, self.head_dim)

        xq = rope.apply_rotary_emb_paged(xq, positions)
        xk = rope.apply_rotary_emb_paged(xk, positions)

        block_size = k_cache.shape[3]
        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            current_ctx_len = context_lengths[seq_idx].item()
            start_pos = current_ctx_len - num_tokens

            for i in range(num_tokens):
                # 在序列中的绝对位置
                pos = start_pos + i
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size

                k_cache[block_idx, layer_idx, :, offset, : ] = xk[token_idx]
                v_cache[block_idx, layer_idx, :, offset, : ] = xv[token_idx]
                token_idx += 1

        output = torch.zeros_like(xq)
        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            seq_len = context_lengths[seq_idx].item()

            gathered_k = torch.zeros(self.n_kv_heads, seq_len, self.head_dim, device=x.device, dtype=x.dtype)
            gathered_v = torch.zeros(self.n_kv_heads, seq_len, self.head_dim, device=x.device, dtype=x.dtype)

            for pos in range(seq_len):
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size
                gathered_k[:, pos, :] = k_cache[block_idx, layer_idx, :, offset, : ]
                gathered_v[:, pos, :] = v_cache[block_idx, layer_idx, :, offset, : ]
            if self.n_rep > 1:
                gathered_k = gathered_k.repeat_interleave(self.n_rep, dim=0)
                gathered_v = gathered_v.repeat_interleave(self.n_rep, dim=0)

            q_curr = xq[token_idx: token_idx + num_tokens].transpose(0, 1)

            scores = torch.matmul(q_curr, gathered_k.transpose(1, 2)) / math.sqrt(self.head_dim)

            if num_tokens > 1:
                q_pos = positions[token_idx: token_idx + num_tokens]
                k_pos = torch.arange(seq_len, device=x.device)
                mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
                scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

            probs = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, gathered_v)

            output[token_idx:token_idx + num_tokens] = attn_out.transpose(0, 1)
            token_idx += num_tokens

        output_flat = output.view(-1, self.n_heads * self.head_dim)
        return self.resid_dropout(self.w_o(output_flat))













