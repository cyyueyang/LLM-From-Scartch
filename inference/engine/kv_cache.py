import torch
from typing import Tuple, List

class KVCacheBase:
    def update(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class StandardKVCache(KVCacheBase):
    def __init__(self,
                 max_batch_size: int,
                 max_seq_len: int,
                 n_layers: int,
                 n_kv_heads: int,
                 head_dim: int,
                 device: torch.device,
                 dtype: torch.dtype) -> None:
        """

        :param max_batch_size: 最大批数量
        :param max_seq_len: 最大序列长度
        :param n_layers: 层数 每层需要单独存储
        :param n_kv_heads: KV头数
        :param head_dim: 每个头维度
        :param device: 存放设备
        :param dtype: 存放类型
        """
        super().__init__()

        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        self.cache_k = torch.zeros((n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim), dtype=dtype, device=device)
        self.cache_v = torch.zeros((n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim), dtype=dtype, device=device)

    def update(self, layer_idx: int, start_pos: int, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新kv cache
        :param layer_idx: 层索引
        :param start_pos: 新kv 的起始位置
        :param xk: key 向量
        :param xv: value 向量
        :return: cache_k, cache_v
        """
        assert xk.shape == xv.shape
        bs, n_heads, seq_len, head_dim = xk.shape
        self.cache_k[layer_idx, :bs, :, start_pos:start_pos + seq_len, :] = xk
        self.cache_v[layer_idx, :bs, :, start_pos:start_pos + seq_len, :] = xv
        keys = self.cache_k[layer_idx, :bs, :, :start_pos + seq_len, :]
        values = self.cache_v[layer_idx, :bs, :, :start_pos + seq_len, :]
        return keys, values

class LatentKVCache(KVCacheBase):
    def __init__(self,
                 max_batch_size: int,
                 max_seq_len: int,
                 n_layers: int,
                 kv_lora_rank: int,
                 rope_head_dim: int,
                 device: torch.device,
                 dtype: torch.dtype
                 ) -> None:
        super().__init__()

        self.cache_latent = torch.zero(
            (n_layers, max_batch_size, max_seq_len, kv_lora_rank),
            dtype=dtype,
            device=device
        )

        self.cache_k_rope = torch.zeros(
            (n_layers, max_batch_size, max_seq_len, rope_head_dim),
            dtype=dtype,
            device=device
        )

    def update(self, layer_idx: int, start_pos: int, c_kv: torch.Tensor, k_rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新潜变量缓存
        :param layer_idx: 层索引
        :param start_pos: 开始位置
        :param c_kv: 潜变量
        :param k_rope: rope后的k
        :return: 完整 潜变量缓存 和 k_rope缓存
        """

        bs, seq_len, kv_lora_rank = c_kv.shape

        self.cache_latent[layer_idx, :bs, start_pos:start_pos + seq_len, :] = c_kv
        self.cache_k_rope[layer_idx, :bs, start_pos:start_pos + seq_len, :] = k_rope

        full_c_kv = self.cache_latent[layer_idx, :bs, :start_pos + seq_len, :]
        full_k_rope = self.cache_k_rope[layer_idx, :bs, :start_pos + seq_len, :]

        return full_c_kv, full_k_rope