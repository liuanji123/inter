"""
GNN和DRL集成模型(GQN)
基于论文第III节C部分和算法1
结合MPNN和DQN实现智能路由决策
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import networkx as nx

from .mpnn import MessagePassingNN, StateEncoder
from .dqn import DQNAgent
from algorithms.k_shortest_path import YenKShortestPaths


class GQNAgent:
    """
    GNN+DQN集成智能体
    使用MPNN提取网络特征，DQN进行路由决策
    """
    
    def __init__(self, config, replay_buffer):
        """
        初始化GQN智能体
        
        Args:
            config: 配置对象
            replay_buffer: 经验回放缓冲区
        """
        self.config = config
        self.device = config.DEVICE
        self.replay_buffer = replay_buffer
        
        # 状态编码器 - ISL作为节点
        self.state_encoder = StateEncoder(config)
        
        # DQN智能体 - 使用MPNN结构
        # 状态维度 = ISL特征维度
        self.state_dim = config.NODE_FEATURE_DIM
        # 动作维度 = k条候选路径
        self.action_dim = config.K_SHORTEST_PATHS
        
        self.dqn = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            replay_buffer=replay_buffer
        )
        
        # k最短路径算法
        self.k = config.K_SHORTEST_PATHS
        
        # 缓存路径
        self.candidate_paths_cache = {}
    
    def select_actions(self, state: Dict, explore: bool = True) -> Dict[int, Tuple[int, ...]]:
        """
        为所有流量需求选择路径
        
        Args:
            state: 环境状态
            explore: 是否探索
            
        Returns:
            actions: {demand_idx: path}
        """
        graph = state['graph']
        demands = state['demands']
        actions = {}
        
        # 编码状态为ISL特征和邻接矩阵
        isl_features, isl_adjacency = self.state_encoder.encode_state(state)
        
        # 为每个需求选择路径
        for idx, demand in enumerate(demands):
            # 生成k条候选路径
            candidate_paths = self._get_candidate_paths(
                graph, 
                demand.origin, 
                demand.destination
            )
            
            if not candidate_paths:
                actions[idx] = (demand.origin,)
                continue
            
            # 使用MPNN输出Q值选择动作
            available_actions = list(range(len(candidate_paths)))
            action_idx = self.dqn.select_action(
                isl_features,
                isl_adjacency,
                available_actions=available_actions,
                explore=explore
            )
            
            if action_idx < len(candidate_paths):
                actions[idx] = tuple(candidate_paths[action_idx])
            else:
                actions[idx] = tuple(candidate_paths[0])
        
        return actions
    
    def _get_candidate_paths(self, graph: nx.Graph, 
                           source: int, destination: int) -> List[List[int]]:
        """
        获取k条候选路径
        使用Yen's k-shortest路径算法
        
        Args:
            graph: 网络图
            source: 源节点
            destination: 目标节点
            
        Returns:
            paths: 候选路径列表
        """
        cache_key = (source, destination)
        
        # 检查缓存
        if cache_key in self.candidate_paths_cache:
            return self.candidate_paths_cache[cache_key]
        
        # 使用Yen算法生成k条最短路径
        yen = YenKShortestPaths(graph, weight='distance')
        paths = yen.find_k_shortest_paths(source, destination, k=self.k)
        
        # 缓存路径
        self.candidate_paths_cache[cache_key] = paths
        
        return paths
    
    def train_step(self, state: Dict, action: Dict, reward: float, 
                   next_state: Dict, done: bool) -> float:
        """
        训练一步
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            
        Returns:
            loss: 损失值
        """
        # 提取状态特征
        state_feature = self._extract_state_feature(state)
        next_state_feature = self._extract_state_feature(next_state)
        
        # 将动作转换为整数（使用第一个需求的动作作为代表）
        # 在实际应用中，可以考虑更复杂的动作表示
        action_idx = 0
        if action:
            first_demand_action = list(action.values())[0]
            # 找到这个路径在候选路径中的索引
            demand = state['demands'][0]
            candidate_paths = self._get_candidate_paths(
                state['graph'], 
                demand.origin, 
                demand.destination
            )
            for i, path in enumerate(candidate_paths):
                if tuple(path) == first_demand_action:
                    action_idx = i
                    break
        
        # 存储经验
        self.replay_buffer.push(
            state_feature.cpu().numpy().flatten(),
            action_idx,
            reward,
            next_state_feature.cpu().numpy().flatten(),
            done
        )
        
        # 训练DQN
        loss = self.dqn.train_step()
        
        return loss
    
    def _extract_state_feature(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取状态特征（ISL特征和邻接矩阵）
        
        Args:
            state: 环境状态
            
        Returns:
            isl_features: ISL特征
            isl_adjacency: ISL邻接矩阵
        """
        return self.state_encoder.encode_state(state)
    
    def save(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'dqn_behavior': self.dqn.behavior_net.state_dict(),
            'dqn_target': self.dqn.target_net.state_dict(),
            'dqn_optimizer': self.dqn.optimizer.state_dict(),
            'epsilon': self.dqn.epsilon,
            'training_steps': self.dqn.training_steps
        }, path)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.dqn.behavior_net.load_state_dict(checkpoint['dqn_behavior'])
        self.dqn.target_net.load_state_dict(checkpoint['dqn_target'])
        self.dqn.optimizer.load_state_dict(checkpoint['dqn_optimizer'])
        self.dqn.epsilon = checkpoint['epsilon']
        self.dqn.training_steps = checkpoint['training_steps']
        print(f"模型已从 {path} 加载")
    
    def clear_path_cache(self):
        """清除路径缓存（拓扑变化时调用）"""
        self.candidate_paths_cache.clear()


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config import Config
    from utils.replay_buffer import ReplayBuffer
    from environment.satellite_network import SatelliteNetwork
    
    # 测试GQN
    buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)
    agent = GQNAgent(Config, buffer)
    
    print("=" * 60)
    print("GQN智能体测试")
    print("=" * 60)
    print(f"DQN (MPNN) 参数量: {sum(p.numel() for p in agent.dqn.behavior_net.parameters())}")
    print(f"总参数量: {sum(p.numel() for p in agent.dqn.behavior_net.parameters())}")
    
    # 测试环境交互
    env = SatelliteNetwork(Config)
    state = env.reset()
    
    print(f"\n测试动作选择:")
    print(f"流量需求数: {len(state['demands'])}")
    print(f"ISL数量: {len(state['isl_states'])}")
    
    # 选择动作
    actions = agent.select_actions(state, explore=True)
    print(f"选择的动作数: {len(actions)}")
    
    # 执行动作
    next_state, reward, done, info = env.step(actions)
    print(f"奖励: {reward:.4f}")
    print(f"成功率: {info['success_rate']:.2%}")
    
    # 训练一步（需要手动处理，因为train_step方法已改变）
    print(f"\n注意: train_step方法需要从main.py中调用")

