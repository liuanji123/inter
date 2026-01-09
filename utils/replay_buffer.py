"""
经验回放缓冲区
用于DQN的Experience Replay机制
支持存储ISL特征和邻接矩阵
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """
        初始化缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        添加经验
        
        Args:
            state: 当前状态（可以是ISL特征和邻接矩阵的字典，或numpy数组）
            action: 动作
            reward: 奖励
            next_state: 下一状态（格式同state）
            done: 是否结束
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 经验批次
            如果状态是字典格式（包含isl_features和isl_adjacency），
            则返回分离的特征和邻接矩阵批次
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 检查第一个状态是否是字典格式（ISL特征和邻接矩阵）
        if isinstance(states[0], dict) and 'isl_features' in states[0]:
            # 提取ISL特征和邻接矩阵
            isl_features_batch = np.array([s['isl_features'] for s in states])
            isl_adjacency_batch = np.array([s['isl_adjacency'] for s in states])
            next_isl_features_batch = np.array([s['isl_features'] for s in next_states])
            next_isl_adjacency_batch = np.array([s['isl_adjacency'] for s in next_states])
            
            return (isl_features_batch, isl_adjacency_batch, actions, rewards, 
                   next_isl_features_batch, next_isl_adjacency_batch, dones)
        else:
            # 传统格式：展平的状态向量
            return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def is_ready(self, batch_size: int) -> bool:
        """
        检查是否有足够的经验进行采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            ready: 是否就绪
        """
        return len(self.buffer) >= batch_size


if __name__ == "__main__":
    # 测试回放缓冲区
    buffer = ReplayBuffer(capacity=100)
    
    # 添加一些经验
    for i in range(10):
        state = {'data': i}
        action = i % 3
        reward = float(i)
        next_state = {'data': i + 1}
        done = (i == 9)
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {len(buffer)}")
    print(f"是否就绪(batch_size=5): {buffer.is_ready(5)}")
    
    # 采样
    if buffer.is_ready(5):
        batch = buffer.sample(5)
        states, actions, rewards, next_states, dones = batch
        print(f"采样批次大小: {len(states)}")
        print(f"奖励: {rewards}")

