import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        return self.gamma * (x - mean) / std + self.beta

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms_x = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.alpha * x * rms_x

class Qwen2RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(Qwen2RMSNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms_x = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (1 + self.alpha) * x * rms_x

if __name__ == '__main__':
    x = torch.randn(4, 128, 64)
    print("testing LayerNorm")
    my_ln = LayerNorm(64, eps=1e-6)
    pytorch_ln = nn.LayerNorm(64, eps=1e-6)
    with torch.no_grad():
        pytorch_ln.weight.copy_(my_ln.gamma)
        pytorch_ln.bias.copy_(my_ln.beta)
    my_ln_output = my_ln(x)
    pytorch_ln_output = pytorch_ln(x)
    if torch.allclose(my_ln_output, pytorch_ln_output, rtol=1e-3, atol=1e-3):
        print("LayerNorm Test passed")

    print("testing RMSNorm")
    my_rmsn = RMSNorm(64, eps=1e-6)
    pytorch_rmsn = nn.RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        pytorch_rmsn.weight.copy_(my_rmsn.alpha)
    my_rmsn_output = my_rmsn(x)
    pytorch_rmsn_output = pytorch_rmsn(x)
    if torch.allclose(my_rmsn_output, pytorch_rmsn_output, rtol=1e-3, atol=1e-3):
        print("RMSNorm Test passed")