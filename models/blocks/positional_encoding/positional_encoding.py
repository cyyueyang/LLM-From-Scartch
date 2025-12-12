import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super(LearnedPositionalEncoding, self).__init__()

        self.pe = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(seq_len, dtype=torch.long, device=x.device)
        x = x + self.pe(position)
        x = self.dropout(x)
        return x

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super(SinusoidalPositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        x = self.dropout(x)

        return x

@dataclass
class RoPEConfig:
    head_dim: int = 64
    max_seq_len: int = 512
    base: int = 10000

class RoPE(nn.Module):
    def __init__(self, config: RoPEConfig):
        super(RoPE, self).__init__()
        self.head_dim = config.head_dim
        self.base = config.base
        self.max_seq_len = config.max_seq_len

        theta = 1 / (self.base ** (torch.arange(0, self.head_dim, 2)).float() / self.head_dim)
        freqs = torch.arange(self.max_seq_len)
        freqs = torch.outer(freqs, theta).float()
        self.register_buffer('cos_cached', torch.cos(freqs))
        self.register_buffer('sin_cached', torch.sin(freqs))

    def _rotate_half(self, t: torch.Tensor) -> torch.Tensor:
        t_even = t[..., ::2]
        t_odd = t[..., 1::2]
        t_shifted = torch.stack([-t_odd, t_even], dim=-1)
        return t_shifted.flatten(-2)

    def apply_rotary_emb(self, x: torch.Tensor) -> torch.Tensor:
        # x size [bs, num_heads, seq_len, head_dim]
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len, :].to(x.device)
        sin = self.sin_cached[:seq_len, :].to(x.device)
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
        cos_emb = cos.repeat_interleave(repeats=2, dim=-1)
        sin_emb = sin.repeat_interleave(repeats=2, dim=-1)
        x_shifted = self._rotate_half(x)
        return (x * cos_emb + x_shifted * sin_emb).type_as(x)

    def apply_rotary_emb_paged(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        kv cached中的不连续 将旋转位置编码应用到特定位置的toekn上
        :param x: size [num_tokens, n_heads, head_dim]
        :param positions: [num_tokens]
        :return: 旋转后的tensor
        """

        cos = self.cos_cached[positions].to(x.device)
        sin = self.sin_cached[positions].to(x.device)
        cos = cos.unsqueeze(1) # [num_tokens, dim/2] -> [num_tokens, 1, dim/2]
        sin = sin.unsqueeze(1)
        cos_emb = cos.repeat_interleave(repeats=2, dim=-1)
        sin_emb = sin.repeat_interleave(repeats=2, dim=-1)
        x_shifted = self._rotate_half(x)
        return (x * cos_emb + x_shifted * sin_emb).type_as(x)

def get_alibi_bias(n_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    生成 ALiBi偏置矩阵
    :param n_heads: 头数
    :param seq_len: 序列长度
    :param device: 设备号
    :return: size[1, n_heads, seq_len, seq_len]
    """

    def get_slopes(n):
        def get_next_power_of_2(n):
            return 2 ** math.ceil(math.log2(n))

        m = torch.tensor(get_next_power_of_2(n)).to(torch.float)
        r = torch.arange(n)

        return (m ** (-2.0 ** (-math.log(m) * (r + 1)))).tolist()

    slopes = torch.tensor(get_slopes(n_heads)).to(device)
    relative_positions = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
    alibi = -torch.abs(relative_positions)
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    return alibi.unsqueeze(0)

if __name__ == '__main__':
    d_model, max_seq_len, batch_size = 128, 16, 4
    n_heads, head_dim = 4, d_model // 4

    rope_config = RoPEConfig(head_dim=head_dim, max_seq_len=max_seq_len)
    rope = RoPE(rope_config)
    num_tokens = 8
    q_paged = torch.randn(num_tokens, n_heads, head_dim)
    positions = torch.tensor([0, 3, 4, 7, 1, 9, 2, 10], dtype=torch.long)

    output_rope_paged = rope.apply_rotary_emb_paged(q_paged, positions)
    assert output_rope_paged.shape == q_paged.shape

    print("形状验证成功")


