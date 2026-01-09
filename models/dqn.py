"""
深度Q网络(DQN)
基于论文第III节B部分，公式(12)-(14)
使用Experience Replay和Fixed Q-Targets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import copy

# 导入MPNN
from .mpnn import MessagePassingNN


class DQNAgent:
    """
    DQN智能体 - 使用MPNN作为Q网络
    行为网络和目标网络都用MPNN结构构建
    """
    
    def __init__(self, 
                 state_dim: int,  # ISL特征维度
                 action_dim: int,  # k条候选路径
                 config,
                 replay_buffer):
        """
        初始化DQN智能体
        
        Args:
            state_dim: ISL特征维度
            action_dim: 动作维度（k条候选路径）
            config: 配置对象
            replay_buffer: 经验回放缓冲区
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = config.DEVICE
        
        # Behavior Network - 使用MPNN结构（论文要求）
        self.behavior_net = MessagePassingNN(
            node_feature_dim=state_dim,
            hidden_dim=config.HIDDEN_DIM,
            action_dim=action_dim,
            num_message_passing=config.NUM_MESSAGE_PASSING,
            aggregation='sum'  # 论文提到使用求和函数
        ).to(self.device)
        
        # Target Network - 使用MPNN结构（论文要求）
        self.target_net = MessagePassingNN(
            node_feature_dim=state_dim,
            hidden_dim=config.HIDDEN_DIM,
            action_dim=action_dim,
            num_message_passing=config.NUM_MESSAGE_PASSING,
            aggregation='sum'
        ).to(self.device)
        
        # 初始化Target Network与Behavior Network相同
        self.target_net.load_state_dict(self.behavior_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.behavior_net.parameters(), 
            lr=config.LEARNING_RATE
        )
        
        # 经验回放
        self.replay_buffer = replay_buffer
        
        # 超参数
        self.gamma = config.GAMMA
        self.batch_size = config.BATCH_SIZE
        self.target_update_freq = config.TARGET_UPDATE_FREQ
        
        # ε-greedy参数
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY
        
        # 训练步数
        self.training_steps = 0
    
    def select_action(self, isl_features: torch.Tensor, 
                     isl_adjacency: torch.Tensor,
                     available_actions: List[int] = None,
                     explore: bool = True) -> int:
        """
        选择动作 - 使用MPNN输出Q值 (ε-greedy策略)
        
        Args:
            isl_features: ISL特征 (batch_size, num_isl, feature_dim)
            isl_adjacency: ISL邻接矩阵 (batch_size, num_isl, num_isl)
            available_actions: 可用动作列表
            explore: 是否探索
            
        Returns:
            action: 选择的动作索引
        """
        # ε-greedy探索
        if explore and np.random.random() < self.epsilon:
            if available_actions:
                return np.random.choice(available_actions)
            else:
                return np.random.randint(0, self.action_dim)
        
        # 利用：使用MPNN输出Q值
        with torch.no_grad():
            q_values = self.behavior_net(isl_features, isl_adjacency)
            
            if available_actions:
                mask = torch.full_like(q_values, float('-inf'))
                mask[0, available_actions] = 0
                q_values = q_values + mask
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def compute_q_values(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """
        计算Q值
        公式(12): Q(s,a) = E[Σ γ^k * r_{t+k+1} | s_t = s, a_t = a]
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            q_value: Q值
        """
        q_values = self.behavior_net(state)
        q_value = q_values[:, action]
        return q_value
    
    def train_step(self, isl_features_batch, isl_adjacency_batch, 
                   actions_batch, rewards_batch, 
                   next_isl_features_batch, next_isl_adjacency_batch, 
                   dones_batch) -> float:
        """
        训练一步
        
        Args:
            isl_features_batch: ISL特征批次 (batch_size, num_isl, feature_dim)
            isl_adjacency_batch: ISL邻接矩阵批次 (batch_size, num_isl, num_isl)
            actions_batch: 动作批次
            rewards_batch: 奖励批次
            next_isl_features_batch: 下一状态ISL特征批次
            next_isl_adjacency_batch: 下一状态ISL邻接矩阵批次
            dones_batch: 结束标志批次
            
        Returns:
            loss: 损失值
        """
        # 转换为张量（如果还不是）
        if not isinstance(isl_features_batch, torch.Tensor):
            isl_features_batch = torch.FloatTensor(isl_features_batch).to(self.device)
        if not isinstance(isl_adjacency_batch, torch.Tensor):
            isl_adjacency_batch = torch.FloatTensor(isl_adjacency_batch).to(self.device)
        if not isinstance(next_isl_features_batch, torch.Tensor):
            next_isl_features_batch = torch.FloatTensor(next_isl_features_batch).to(self.device)
        if not isinstance(next_isl_adjacency_batch, torch.Tensor):
            next_isl_adjacency_batch = torch.FloatTensor(next_isl_adjacency_batch).to(self.device)
        
        actions_batch = torch.LongTensor(actions_batch).to(self.device)
        rewards_batch = torch.FloatTensor(rewards_batch).to(self.device)
        dones_batch = torch.FloatTensor(dones_batch).to(self.device)
        
        # 计算当前Q值 - Behavior Network (MPNN)
        current_q_values = self.behavior_net(isl_features_batch, isl_adjacency_batch)
        current_q = current_q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值 - Target Network (MPNN)
        with torch.no_grad():
            next_q_target = self.target_net(next_isl_features_batch, next_isl_adjacency_batch)
            max_next_q = next_q_target.max(dim=1)[0]
            target_q = rewards_batch + self.gamma * max_next_q * (1 - dones_batch)
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.behavior_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        
        # 更新Target Network
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()
        
        # 衰减ε
        if self.training_steps % 5 == 0:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """
        更新Target Network
        每N步将Behavior Network的参数复制到Target Network
        """
        self.target_net.load_state_dict(self.behavior_net.state_dict())
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'behavior_net': self.behavior_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.behavior_net.load_state_dict(checkpoint['behavior_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config import Config
    from utils.replay_buffer import ReplayBuffer
    
    # 测试DQN
    state_dim = 8  # ISL特征维度
    action_dim = 3  # k条路径
    num_isl = 20  # ISL数量
    
    buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)
    agent = DQNAgent(state_dim, action_dim, Config, buffer)
    
    print(f"DQN智能体初始化完成")
    print(f"  ISL特征维度: {state_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  Behavior Network参数量: {sum(p.numel() for p in agent.behavior_net.parameters())}")
    print(f"  初始ε: {agent.epsilon}")
    
    # 测试选择动作（需要ISL特征和邻接矩阵）
    test_isl_features = torch.randn(1, num_isl, state_dim).to(Config.DEVICE)
    test_isl_adjacency = torch.randint(0, 2, (1, num_isl, num_isl)).float().to(Config.DEVICE)
    action = agent.select_action(test_isl_features, test_isl_adjacency)
    print(f"\n测试动作选择: {action}")
    
    # 添加一些经验并测试训练
    for i in range(100):
        state_dict = {
            'isl_features': np.random.randn(num_isl, state_dim),
            'isl_adjacency': np.random.randint(0, 2, (num_isl, num_isl)).astype(float)
        }
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        next_state_dict = {
            'isl_features': np.random.randn(num_isl, state_dim),
            'isl_adjacency': np.random.randint(0, 2, (num_isl, num_isl)).astype(float)
        }
        done = False
        
        buffer.push(state_dict, action, reward, next_state_dict, done)
    
    # 训练
    if len(buffer) >= Config.BATCH_SIZE:
        batch = buffer.sample(Config.BATCH_SIZE)
        if isinstance(batch, tuple) and len(batch) == 7:
            (isl_features_batch, isl_adjacency_batch, actions_batch, 
             rewards_batch, next_isl_features_batch, next_isl_adjacency_batch, 
             dones_batch) = batch
            loss = agent.train_step(
                isl_features_batch, isl_adjacency_batch,
                actions_batch, rewards_batch,
                next_isl_features_batch, next_isl_adjacency_batch,
                dones_batch
            )
            print(f"\n训练损失: {loss:.6f}")
            print(f"训练后ε: {agent.epsilon:.6f}")

