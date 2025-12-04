"""
此处实现为SwiGLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    SwiGLU Block
    FFN(x) = (Silu((x @ w_gate) * x @ w_up)) @ w_down
    silu(x) = x * sigmoid(x)
    """
    def __init__(self, dim: int, hidden_dim: int):
        """
        :param dim: input dimension and output dimension
        :param hidden_dim: hidden layer dimension 如果可以被256整除 会加速
        如果想与传统FFN 保持参数量一致 hidden_dim 应该为 以前的 2/3
        """
        super(FeedForward, self).__init__()
        self.w_up = nn.Linear(dim, hidden_dim)
        self.w_down = nn.Linear(hidden_dim, dim)
        self.w_gate = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        return self.w_down(F.relu(self.w_gate(x)) * self.w_up(x))

if __name__ == '__main__':
    x = torch.randn(1, 3, 64, 64)
    model = FeedForward(64, 64)
    out = model(x)
    assert out.shape == x.shape
    print(out.shape)


