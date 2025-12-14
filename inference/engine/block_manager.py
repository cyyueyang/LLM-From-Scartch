import torch
from collections import defaultdict
from typing import Tuple, List, Dict, Optional
import warnings

class BlockManager:
    def __init__(self,
                 num_blocks: int,
                 n_layers: int,
                 n_kv_heads: int,
                 block_size: int,
                 head_dim: int,
                 dtype: torch.dtype,
                 device: torch.device,) -> None:
        self.num_blocks = num_blocks
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # 1. 创建物理KVCache Pool
        self.k_cache_pool = torch.zeros(
            (self.num_blocks, self.n_layers, self.n_kv_heads, self.block_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_cache_pool = torch.zeros(
            (self.num_blocks, self.n_layers, self.n_kv_heads, self.block_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # 2. 创建空闲块列表
        # free_blocks 栈 用于快速分配和回收块的物理索引
        self.free_blocks = list(range(self.num_blocks))

        self.block_tables: defaultdict[int, List[int]] = defaultdict(list)

    def can_allocate(self, num_required_blocks: int) -> bool:
        """
        判断能否分配
        :param num_required_blocks: 需要的块数
        :return: 是否能分配
        """
        return len(self.free_blocks) >= num_required_blocks

    def allocate(self, seq_id: int, num_required_blocks: int) -> None:
        """
        对seq_id 分配 num_required_blocks 个 block
        :param seq_id: 序列编号
        :param num_required_blocks: 需要的块数
        :return: None
        """
        if not self.can_allocate(num_required_blocks):
            raise ValueError(f"Cannot allocate {num_required_blocks} blocks. Only {len(self.free_blocks)} blocks can be allowed.")

        block_indices = [self.free_blocks.pop() for _ in range(num_required_blocks)]
        self.block_tables[seq_id] = block_indices

    def free(self, seq_id: int) -> None:
        """
        释放 seq_id 占用的block
        :param seq_id: 序列编号
        :return: None
        """
        if seq_id not in self.block_tables:
            return

        block_indices = self.block_tables.pop(seq_id)
        self.free_blocks.extend(block_indices)

    def append_block(self, seq_id: int):
        """
        为 seq_id 添加一个块
        :param seq_id: 序列编号
        :return:
        """
        if not self.can_allocate(num_required_blocks=1):
            raise ValueError("内存不足，无法追加新块")

        if seq_id not in self.block_tables:
            warnings.warn(f"{seq_id} not in block table")
            block_index = self.free_blocks.pop()
            self.block_tables[seq_id].append(block_index)

        block_index = self.free_blocks.pop()
        self.block_tables[seq_id].append(block_index)

    def get_block_table(self, seq_id: int) -> List[int]:
        """
        获取序列的块表
        :param seq_id: 序列编号
        :return: seq_id 对应分配的块表
        """
        return self.block_tables[seq_id]

    def get_num_free_blocks(self) -> int:
        """
        获取空闲块的数量
        :return: 空闲块的数量
        """
        return len(self.free_blocks)