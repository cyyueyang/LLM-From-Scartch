import math

import torch
import torch.nn.functional as F
import torch.nn as nn

from inference.engine.kv_cache import LatentKVCache
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
        # 每个token的位置索引 每个序列的token数 每个序列的上下文长度 分块的键值缓存 块表（记录每个序列使用了哪些块）
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


class MultiHeadLatentAttention(torch.nn.Module):
    def __init__(self, args):
        super(MultiHeadLatentAttention, self).__init__()

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.nope_head_dim = args.nope_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.v_head_dim = args.v_head_dim

        # q_lora_rank == 0  禁用 q 压缩
        if self.q_lora_rank > 0:
            self.wq_down = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.wq_up = nn.Linear(self.q_lora_rank, self.n_heads*self.nope_head_dim, bias=False)
            self.wq_rope= nn.Linear(self.q_lora_rank, self.n_heads*self.rope_head_dim, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=args.norm_eps)
        else:
            self.wq_up = nn.Linear(self.dim, self.n_heads*self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(self.dim, self.n_heads*self.rope_head_dim, bias=False)

        # kv down
        self.wkv_down = nn.Linear(self.dim, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)

        # kv up
        self.wkv_up = nn.Linear(self.kv_lora_rank, self.n_heads*(self.nope_head_dim+self.v_head_dim), bias=False)
        #k rope
        self.wk_rope = nn.Linear(self.dim, self.rope_head_dim, bias=False)

        self.wo = nn.Linear(self.n_heads*self.v_head_dim, args.dim, bias=False)

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, x, rope, layer_idx, kv_chache=None, start_pos=0, paged_attention_inputs=None, **kwargs):
        bs, seq_len, dim = x.shape

        if kv_chache is None and seq_len == 1:
            assert isinstance(kv_chache, LatentKVCache) "MLA requires kv_chache to be a LatentKVCache"
            return self._forward_inference_optimized(x, rope, layer_idx, kv_chache, start_pos)

        if paged_attention_inputs is not None:
            raise NotImplementedError("Paged attention inputs not yet implemented")

        if self.q_lora_rank > 0:
            q_compressed = self.wq_down(x)
            q_compressed = self.q_norm(q_compressed)
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        kv_compressed = self.wkv_down(x)
        kv_compressed = self.kv_norm(kv_compressed)

        kv_up = self.wkv_up(kv_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
        k_nope, v = torch.split(kv_up, [self.nope_head_dim, self.v_head_dim], dim=-1)

        k_rope_shared = self.wk_rope(x).view(bs, seq_len, 1, self.rope_head_dim)

        if kv_chache is not None:
            k_rope_for_cache = k_rope_shared.squeeze(2)
            kv_chache.update(layer_idx, start_pos, kv_compressed, k_rope_for_cache)

        # [bs, seq_len, n_heads, head_dim] -> [bs, n_heads, seq_len, head_dim]
        q_pe = q_pe.permute(0, 2, 1, 3)
        k_rope_shared = k_rope_shared.permute(0, 2, 1, 3)

        q_pe = rope.apply_rotary_emb(q_pe)
        k_rope_shared = rope.apply_rotary_emb(k_rope_shared)

        # [bs, n_heads, seq_len, head_dim] -> [bs, seq_len, n_heads, head_dim]
        q_pe = q_pe.permute(0, 2, 1, 3)
        k_rope_shared = k_rope_shared.permute(0, 2, 1, 3)

        k_rope = k_rope_shared.expand(-1, -1, self.n_heads, -1)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        if seq_len > 1:
            local_mask = self.mask[:, :seq_len, :seq_len]
            scores = scores + local_mask

        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)
        output = output.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)
        output = self.wo(output)
        return output

    def _forward_inference_optimized(self, x, rope, layer_idx, kv_cache: LatentKVCache, start_pos: int):
        pass










