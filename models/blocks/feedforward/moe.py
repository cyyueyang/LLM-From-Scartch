import torch
import torch.nn.functional as F
import torch.nn as nn
from models.blocks.feedforward.feedforward import FeedForward

class MoELayer(nn.Module):
    """
    MoE layer
    """
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 num_experts: int,
                 num_experts_per_tok: int,
                 num_shared_experts: int,
                 use_aux_free_lb: bool,
                 ) -> None:
        """
        MoE layer
        :param dim: 输入输出维度
        :param hidden_dim: 隐藏层维度
        :param num_experts: 专家数量
        :param num_experts_per_tok: 每个token激活的专家数量
        :param num_shared_experts: 共享专家数量
        :param use_aux_free_lb: 是否使用无辅助损失的负载均衡
        """
        super(MoELayer, self).__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.use_aux_free_lb = use_aux_free_lb

        # 无偏置 后面增加动态偏置
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)

        if self.use_aux_free_lb:
            # 不需要梯度更新 根据统计来更新
            self.experts_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)

        self.experts = nn.ModuleList(
            [FeedForward(dim=self.dim, hidden_dim=self.hidden_dim) for _ in range(self.num_experts)]
        )
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [FeedForward(dim=self.dim, hidden_dim=self.hidden_dim) for _ in range(self.num_shared_experts)]
            )
        self.register_buffer('last_expert_counts', torch.zeros(self.num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, dim = x.shape
        # MoE是token来选择专家 与 bs seq_len 无关
        x_flat= x.view(-1, dim)

        # 通过共享专家
        shared_output = 0.0
        if self.num_shared_experts > 0:
            for expert in self.shared_experts:
                shared_output += expert(x_flat)

        # 专家路由
        router_logits = self.router(x_flat)
        if self.use_aux_free_lb:
            router_logits = self.experts_bias + router_logits

        # selected_experts shape [num_tokens, topk]
        routing_weights, selected_experts = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        final_output = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(selected_experts, self.num_experts).permute(2, 0, 1) # [num_tokens, topk, num_experts] -> [num_experts, num_tokens, topk]

        # 负载统计
        if self.training and self.use_aux_free_lb:
            current_counts = expert_mask.sum(dim=(1, 2)).detach().float()
            self.last_expert_counts = current_counts

        # 往专家上面分配
        for expert_idx in range(self.num_experts):
            idx_in_topk = torch.where(expert_mask[expert_idx] > 0)

            # 没有选择这个专家
            if idx_in_topk[0].numel() == 0:
                continue

            token_indices = idx_in_topk[0]
            expert_output = self.experts[expert_idx](x_flat[token_indices])

            weights = routing_weights[token_indices, idx_in_topk[1]].unsqueeze(-1)
            final_output.index_add_(0, token_indices, expert_output * weights)

        if self.num_shared_experts > 0:
            final_output = final_output + shared_output

        return final_output.view(bs, seq_len, dim)

    def update_bias(self, lr: float = 0.5) -> None:
        """
        [DeepSeek-V3] Aux-free Load Balancing 更新逻辑
        :param lr: 更新率
        :return: None
        """
        if not self.use_aux_free_lb:
            return

        counts = self.last_expert_counts
        mean_counts = torch.mean(counts) + 1e-6

        # 如果某个专家的负载远超平均值，bias 应该降低
        # 如果某个专家没人选，bias 应该升高
        # sign(count - mean) > 0 => count > mean => bias decreases

        error = mean_counts - counts
        update_step = lr * torch.sign(error)

        self.experts_bias = self.experts_bias - update_step

        self.experts_bias = self.experts_bias.clamp(min=-10.0, max=10.0)

        self.last_expert_counts.zero_()

if __name__ == '__main__':
    x = torch.randn(16, 128, 512)
    model = MoELayer(dim=512, hidden_dim=2048, num_experts=16, num_experts_per_tok=4, num_shared_experts=2, use_aux_free_lb=False)
    y = model(x)
    print(y.shape)












