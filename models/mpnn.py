"""
消息传递神经网络(MPNN)
基于论文第III节A部分，公式(9)-(11)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from typing import Dict, Tuple


class MessagePassingNN(nn.Module):
    """
    消息传递神经网络 - 直接输出Q值
    基于论文：行为网络和目标网络都用MPNN构建，输出Q(S, a; θ)
    """
    
    def __init__(self, 
                 node_feature_dim: int,
                 hidden_dim: int,
                 action_dim: int,  # 动作维度（k条路径）
                 num_message_passing: int = 3,
                 aggregation: str = 'sum'):
        """
        初始化MPNN
        
        Args:
            node_feature_dim: ISL节点特征维度
            hidden_dim: 隐藏层维度
            action_dim: 动作维度（k条候选路径）
            num_message_passing: 消息传递迭代次数 T
            aggregation: 聚合函数 ('sum', 'mean', 'max') - 论文使用求和
        """
        super(MessagePassingNN, self).__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_message_passing = num_message_passing
        self.aggregation = aggregation
        
        # 状态预处理层（论文提到：状态在通过全连接层之前进行预处理）
        self.preprocess = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 全连接层（负责消息传递，论文提到）
        self.message_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # RNN用于更新历史消息（论文提到：使用RNN更新历史消息）
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 读出函数输出Q值（论文：Q(S, a; θ)）
        # 需要为每个动作输出Q值
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # 输出每个动作的Q值
        )
    
    def forward(self, node_features: torch.Tensor, 
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 输出Q值
        
        Args:
            node_features: ISL节点特征 (batch_size, num_isl, node_feature_dim)
            adjacency: ISL邻接矩阵 (batch_size, num_isl, num_isl)
            
        Returns:
            q_values: Q值 (batch_size, action_dim)
        """
        batch_size, num_isl, _ = node_features.shape
        
        # 状态预处理
        h = self.preprocess(node_features)  # (batch_size, num_isl, hidden_dim)
        
        # T次消息传递迭代
        for t in range(self.num_message_passing):
            # 计算并聚合消息（使用求和函数，论文提到）
            messages = self._compute_messages(h, adjacency)
            
            # 使用RNN更新历史消息（论文提到）
            h = self._update_with_rnn(h, messages)
        
        # 全局聚合所有ISL特征（用于输出Q值）
        # 使用求和聚合（论文提到使用求和函数）
        graph_feature = torch.sum(h, dim=1)  # (batch_size, hidden_dim)
        
        # 读出函数输出Q值
        q_values = self.readout(graph_feature)  # (batch_size, action_dim)
        
        return q_values
    
    def _compute_messages(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        计算并聚合消息（使用求和函数）
        公式(9): m_v^(t+1) = A({M(h_u^(t), h_v^(t)) | u ∈ B(v)})
        
        Args:
            h: ISL节点隐藏状态 (batch_size, num_isl, hidden_dim)
            adjacency: ISL邻接矩阵 (batch_size, num_isl, num_isl)
            
        Returns:
            aggregated_messages: 聚合后的消息 (batch_size, num_isl, hidden_dim)
        """
        batch_size, num_isl, hidden_dim = h.shape
        
        # 计算所有ISL对之间的消息
        h_i = h.unsqueeze(2).expand(batch_size, num_isl, num_isl, hidden_dim)
        h_j = h.unsqueeze(1).expand(batch_size, num_isl, num_isl, hidden_dim)
        edge_features = torch.cat([h_i, h_j], dim=-1)
        
        # 消息函数
        messages = self.message_fn(edge_features)
        
        # 应用邻接矩阵掩码
        adjacency_mask = adjacency.unsqueeze(-1)
        messages = messages * adjacency_mask
        
        # 求和聚合（论文明确提到使用求和函数）
        aggregated = torch.sum(messages, dim=2)
        
        return aggregated
    
    def _update_with_rnn(self, h: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        使用RNN更新历史消息
        
        Args:
            h: 当前ISL节点状态 (batch_size, num_isl, hidden_dim)
            messages: 聚合消息 (batch_size, num_isl, hidden_dim)
            
        Returns:
            h_new: 新的ISL节点状态 (batch_size, num_isl, hidden_dim)
        """
        batch_size, num_isl, hidden_dim = h.shape
        
        h_flat = h.reshape(-1, hidden_dim)
        messages_flat = messages.reshape(-1, hidden_dim)
        
        # RNN更新
        h_new_flat = self.rnn(messages_flat, h_flat)
        
        h_new = h_new_flat.reshape(batch_size, num_isl, hidden_dim)
        return h_new


class StateEncoder:
    """状态编码器 - 将ISL作为节点进行编码"""
    
    def __init__(self, config):
        """
        初始化状态编码器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.DEVICE
    
    def encode_state(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码状态为MPNN输入
        将ISL作为节点，构建ISL图
        
        Args:
            state: 环境状态，包含isl_states
            
        Returns:
            isl_features: ISL节点特征 (batch_size, num_isl, feature_dim)
            isl_adjacency: ISL邻接矩阵 (batch_size, num_isl, num_isl)
        """
        graph = state['graph']
        isl_states = state['isl_states']  # 使用ISL状态
        
        # 构建ISL特征向量
        isl_features = []
        for isl_state in isl_states:
            # ISL特征向量包含：
            # 1. 接收的业务需求（load）
            # 2. 剩余容量（residual_capacity）
            # 3. 容量（capacity）- 用于归一化
            # 4. 距离（distance）- 归一化
            # 5. 零元素 - 用于存储相邻ISL聚合信息
            
            load = isl_state['load']
            residual_capacity = isl_state['residual_capacity']
            capacity = isl_state['capacity']
            distance = isl_state['distance']
            
            # 归一化
            normalized_load = load / (capacity + 1e-6)
            normalized_residual = residual_capacity / (capacity + 1e-6)
            # 假设最大距离约10000km进行归一化
            max_distance = 10000.0
            normalized_distance = distance / max_distance
            
            # 构建特征向量
            # 前几个元素是ISL自身特征
            feature_vector = [
                normalized_load,
                normalized_residual,
                normalized_distance,
            ]
            
            # 添加零元素用于存储相邻ISL聚合信息
            # 根据论文描述，在特定位置增加零元素
            num_aggregation_slots = self.config.NODE_FEATURE_DIM - len(feature_vector)
            feature_vector.extend([0.0] * num_aggregation_slots)
            
            # 确保维度正确
            feature_vector = feature_vector[:self.config.NODE_FEATURE_DIM]
            isl_features.append(feature_vector)
        
        # 构建ISL邻接矩阵
        # 两个ISL相邻的条件：它们共享一个卫星节点
        num_isl = len(isl_states)
        isl_adjacency = np.zeros((num_isl, num_isl))
        
        # 创建ISL到节点的映射
        isl_to_nodes = {}
        for idx, isl_state in enumerate(isl_states):
            nodes = isl_state['nodes']
            isl_to_nodes[idx] = nodes
        
        # 如果两个ISL共享节点，则它们相邻
        for i in range(num_isl):
            for j in range(i + 1, num_isl):
                nodes_i = set(isl_to_nodes[i])
                nodes_j = set(isl_to_nodes[j])
                if nodes_i & nodes_j:  # 有交集
                    isl_adjacency[i, j] = 1
                    isl_adjacency[j, i] = 1
        
        # 转换为张量并添加batch维度
        isl_features = torch.FloatTensor(isl_features).unsqueeze(0).to(self.device)
        isl_adjacency = torch.FloatTensor(isl_adjacency).unsqueeze(0).to(self.device)
        
        return isl_features, isl_adjacency


if __name__ == "__main__":
    # 测试MPNN
    batch_size = 2
    num_isl = 20  # ISL数量
    node_feature_dim = 8  # ISL特征维度
    hidden_dim = 64
    action_dim = 3  # k条路径
    
    # 创建随机输入
    isl_features = torch.randn(batch_size, num_isl, node_feature_dim)
    isl_adjacency = torch.randint(0, 2, (batch_size, num_isl, num_isl)).float()
    
    # 创建模型
    mpnn = MessagePassingNN(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,  # 直接输出Q值
        num_message_passing=3
    )
    
    # 前向传播
    q_values = mpnn(isl_features, isl_adjacency)
    
    print(f"ISL特征形状: {isl_features.shape}")
    print(f"ISL邻接矩阵形状: {isl_adjacency.shape}")
    print(f"Q值输出形状: {q_values.shape}")
    print(f"MPNN参数量: {sum(p.numel() for p in mpnn.parameters())}")

