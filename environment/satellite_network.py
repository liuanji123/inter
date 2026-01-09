"""
卫星网络环境 - 仿真LEO卫星网络的MDP环境
基于论文第II节问题建模和第III节
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .topology import TopologyGenerator


class TrafficDemand:
    """流量需求类"""
    
    def __init__(self, origin: int, destination: int, data_rate: float):
        """
        初始化流量需求
        
        Args:
            origin: 源节点
            destination: 目的节点
            data_rate: 数据传输率 f_{o,d} (Mbps)
        """
        self.origin = origin
        self.destination = destination
        self.data_rate = data_rate
        self.selected_path = None
        self.allocated_flow = 0.0


class SatelliteNetwork:
    """LEO卫星网络环境 - MDP建模"""
    
    def __init__(self, config):
        """
        初始化卫星网络环境
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 拓扑生成器
        self.topo_gen = TopologyGenerator(
            num_orbits=config.NUM_ORBITS,
            sats_per_orbit=config.SATS_PER_ORBIT,
            altitude=config.ALTITUDE,
            inclination=config.INCLINATION,
            earth_radius=config.EARTH_RADIUS
        )
        
        # 网络状态
        self.graph = None
        self.current_time = 0.0
        self.time_slot = 0
        
        # 流量需求
        self.traffic_demands: List[TrafficDemand] = []
        self.num_demands = 0
        
        # ISL参数
        self.isl_capacity = config.ISL_CAPACITY
        self.speed_of_light = config.SPEED_OF_LIGHT
        
        # 初始化网络
        self.reset()
    
    def reset(self, time: float = 0.0) -> Dict:
        """
        重置环境
        
        Args:
            time: 初始时间
            
        Returns:
            state: 初始状态
        """
        self.current_time = time
        self.time_slot = 0
        
        # 生成网络拓扑
        self.graph = self.topo_gen.build_topology_graph(time)
        
        # 设置ISL容量
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['capacity'] = self.isl_capacity
            self.graph[edge[0]][edge[1]]['load'] = 0.0
            self.graph[edge[0]][edge[1]]['residual_capacity'] = self.isl_capacity
        
        # 生成流量需求
        self._generate_traffic_demands()
        
        return self.get_state()
    
    def _generate_traffic_demands(self, num_demands: Optional[int] = None):
        """
        生成随机流量需求
        
        Args:
            num_demands: 流量需求数量，默认为配置中的NUM_DEMANDS
        """
        self.traffic_demands = []
        
        if num_demands is None:
            # 使用配置中的流量需求数量（大幅减少）
            num_demands = getattr(self.config, 'NUM_DEMANDS', 10)
        
        self.num_demands = num_demands
        nodes = list(self.graph.nodes())
        
        for _ in range(num_demands):
            # 随机选择源和目的节点
            origin, destination = np.random.choice(nodes, size=2, replace=False)
            
            # 随机生成数据传输率
            data_rate = np.random.uniform(
                self.config.MIN_TRAFFIC_DEMAND,
                self.config.MAX_TRAFFIC_DEMAND
            )
            
            demand = TrafficDemand(origin, destination, data_rate)
            self.traffic_demands.append(demand)
    
    def get_state(self) -> Dict:
        """
        获取当前环境状态
        状态包含所有ISL的流量和容量信息
        
        Returns:
            state: 状态字典
        """
        state = {
            'graph': self.graph.copy(),
            'time': self.current_time,
            'time_slot': self.time_slot,
            'demands': self.traffic_demands.copy(),
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
        }
        
        # ISL状态向量
        isl_states = []
        for edge in self.graph.edges():
            i, j = edge
            edge_data = self.graph[i][j]
            isl_state = {
                'nodes': (i, j),
                'capacity': edge_data['capacity'],
                'load': edge_data['load'],
                'residual_capacity': edge_data['residual_capacity'],
                'distance': edge_data['distance'],
            }
            isl_states.append(isl_state)
        
        state['isl_states'] = isl_states
        
        return state
    
    def step(self, actions: Dict[int, Tuple[int, ...]]) -> Tuple[Dict, float, bool, Dict]:
        """
        执行动作并返回新状态
        
        Args:
            actions: 动作字典 {demand_idx: path}
                    path是节点序列元组，如 (0, 5, 10, 15)
        
        Returns:
            next_state: 下一状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 重置ISL负载
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['load'] = 0.0
        
        # 应用路由决策
        successful_demands = 0
        total_demands = len(self.traffic_demands)
        
        for demand_idx, path in actions.items():
            if demand_idx >= len(self.traffic_demands):
                continue
            
            demand = self.traffic_demands[demand_idx]
            demand.selected_path = path
            
            # 检查路径有效性并分配流量
            if self._allocate_flow_on_path(demand, path):
                successful_demands += 1
        
        # 更新ISL剩余容量
        for edge in self.graph.edges():
            i, j = edge
            load = self.graph[i][j]['load']
            capacity = self.graph[i][j]['capacity']
            self.graph[i][j]['residual_capacity'] = max(0, capacity - load)
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 更新时间
        self.time_slot += 1
        done = self.time_slot >= self.config.MAX_STEPS_PER_EPISODE
        
        # 额外信息
        info = {
            'successful_demands': successful_demands,
            'total_demands': total_demands,
            'success_rate': successful_demands / total_demands if total_demands > 0 else 0,
        }
        
        next_state = self.get_state()
        
        return next_state, reward, done, info
    
    def _allocate_flow_on_path(self, demand: TrafficDemand, path: Tuple[int, ...]) -> bool:
        """
        在路径上分配流量
        支持部分分配：当容量不足时，分配可用的最大流量
        
        Args:
            demand: 流量需求
            path: 路径（节点序列）
        
        Returns:
            success: 是否成功分配（包括部分分配）
        """
        if len(path) < 2:
            return False
        
        # 检查路径是否有效
        if path[0] != demand.origin or path[-1] != demand.destination:
            return False
        
        # 检查路径中所有链路的容量，找到瓶颈
        min_available = float('inf')
        for i in range(len(path) - 1):
            node_i, node_j = path[i], path[i + 1]
            
            if not self.graph.has_edge(node_i, node_j):
                return False
            
            residual = self.graph[node_i][node_j].get('residual_capacity', 
                                                       self.graph[node_i][node_j]['capacity'])
            min_available = min(min_available, residual)
        
        # 如果没有任何可用容量，分配失败
        if min_available <= 0:
            return False
        
        # 分配流量：取需求和可用容量的最小值
        allocated_flow = min(demand.data_rate, min_available)
        
        # 在路径上分配流量
        for i in range(len(path) - 1):
            node_i, node_j = path[i], path[i + 1]
            self.graph[node_i][node_j]['load'] += allocated_flow
        
        demand.allocated_flow = allocated_flow
        
        # 即使是部分分配也算成功
        return allocated_flow > 0
    
    def _calculate_reward(self) -> float:
        """
        计算奖励值 (改进版)
        基于论文公式(15): Reward = α * (L_standard / L_action) + β * (R_action / R_standard)
        
        改进:
        1. 移除tanh压缩,提供更清晰的训练信号
        2. 增加成功率奖励
        3. 更明确的惩罚机制
        
        Returns:
            reward: 奖励值
        """
        alpha = self.config.ALPHA
        beta = self.config.BETA
        
        # 计算当前路径的平均传播延迟
        total_delay = 0.0
        total_residual_capacity = 0.0
        num_valid_demands = 0
        num_satisfied_demands = 0
        
        for demand in self.traffic_demands:
            if demand.selected_path is not None and len(demand.selected_path) >= 2:
                # 计算传播延迟
                path_delay = self._calculate_path_propagation_delay(demand.selected_path)
                total_delay += path_delay
                
                # 计算路径剩余容量
                path_residual = self._calculate_path_residual_capacity(demand.selected_path)
                total_residual_capacity += path_residual
                
                num_valid_demands += 1
                
                # 检查需求是否被充分满足
                if demand.allocated_flow >= demand.data_rate * 0.9:  # 满足90%以上
                    num_satisfied_demands += 1
        
        if num_valid_demands == 0:
            return -10.0  # 严重惩罚无效动作
        
        avg_delay = total_delay / num_valid_demands
        avg_residual = total_residual_capacity / num_valid_demands
        
        # 参考值（使用最短路径作为标准）
        reference_delay = self._calculate_reference_delay()
        reference_residual = self.isl_capacity * 0.3  # 参考剩余容量
        
        # 延迟奖励: reference/actual (值越大越好)
        delay_reward = reference_delay / (avg_delay + 1e-6) if avg_delay > 0 else 1.0
        delay_reward = np.clip(delay_reward, 0.5, 3.0)  # 限制在合理范围
        
        # 容量奖励: actual/reference (值越大越好)
        capacity_reward = avg_residual / (reference_residual + 1e-6) if reference_residual > 0 else 1.0
        capacity_reward = np.clip(capacity_reward, 0.5, 3.0)
        
        # 成功率奖励
        success_rate = num_satisfied_demands / len(self.traffic_demands) if len(self.traffic_demands) > 0 else 0
        success_reward = success_rate * 2.0  # [0, 2]
        
        # 组合奖励 (不再使用tanh压缩,保持原始信号强度)
        reward = alpha * delay_reward + beta * capacity_reward + 0.3 * success_reward
        
        # 惩罚丢包
        packet_loss = self._calculate_packet_loss_rate()
        if packet_loss > 0.1:  # 丢包率超过10%
            reward -= packet_loss * 5.0  # 丢包惩罚
        
        return reward
    
    def _calculate_path_propagation_delay(self, path: Tuple[int, ...]) -> float:
        """
        计算路径传播延迟 - 公式(4)
        
        Args:
            path: 路径
            
        Returns:
            delay: 传播延迟（秒）
        """
        total_delay = 0.0
        
        for i in range(len(path) - 1):
            node_i, node_j = path[i], path[i + 1]
            if self.graph.has_edge(node_i, node_j):
                distance = self.graph[node_i][node_j]['distance']
                delay = distance / self.speed_of_light  # 秒
                total_delay += delay
        
        return total_delay
    
    def _calculate_path_residual_capacity(self, path: Tuple[int, ...]) -> float:
        """
        计算路径剩余容量 - 公式(3)
        路径剩余容量 = min(各边剩余容量)
        
        Args:
            path: 路径
            
        Returns:
            residual: 剩余容量（Mbps）
        """
        min_residual = float('inf')
        
        for i in range(len(path) - 1):
            node_i, node_j = path[i], path[i + 1]
            if self.graph.has_edge(node_i, node_j):
                capacity = self.graph[node_i][node_j]['capacity']
                load = self.graph[node_i][node_j]['load']
                residual = capacity - load
                min_residual = min(min_residual, residual)
        
        return max(0, min_residual) if min_residual != float('inf') else 0
    
    def _calculate_reference_delay(self) -> float:
        """
        计算参考延迟（使用最短路径）
        
        Returns:
            reference_delay: 参考延迟
        """
        total_delay = 0.0
        num_demands = 0
        
        for demand in self.traffic_demands:
            try:
                shortest_path = nx.shortest_path(
                    self.graph, 
                    demand.origin, 
                    demand.destination,
                    weight='distance'
                )
                delay = self._calculate_path_propagation_delay(tuple(shortest_path))
                total_delay += delay
                num_demands += 1
            except nx.NetworkXNoPath:
                continue
        
        return total_delay / num_demands if num_demands > 0 else 1.0
    
    def calculate_metrics(self) -> Dict:
        """
        计算性能指标
        基于论文公式(16)-(21)
        
        Returns:
            metrics: 性能指标字典
        """
        metrics = {}
        
        # 1. 平均端到端延迟 - 公式(16)-(19)
        metrics['avg_e2e_delay'] = self._calculate_avg_e2e_delay()
        
        # 2. 丢包率 - 公式(20)-(21)
        metrics['packet_loss_rate'] = self._calculate_packet_loss_rate()
        
        # 3. 链路利用率方差
        metrics['link_utilization_variance'] = self._calculate_link_utilization_variance()
        
        # 4. 平均链路利用率
        metrics['avg_link_utilization'] = self._calculate_avg_link_utilization()
        
        return metrics
    
    def _calculate_avg_e2e_delay(self) -> float:
        """
        计算平均端到端延迟
        L_E2E = L_prop + L_que + L_tran (公式16)
        """
        total_delay = 0.0
        num_valid = 0
        
        for demand in self.traffic_demands:
            # 只处理有效且有实际数据速率的需求
            if demand.data_rate > 0 and demand.selected_path and len(demand.selected_path) >= 2:
                # 传播延迟
                L_prop = self._calculate_path_propagation_delay(demand.selected_path)
                
                # 排队延迟（简化）
                L_que = self._calculate_queuing_delay(demand.selected_path)
                
                # 传输延迟
                L_tran = self._calculate_transmission_delay(demand)
                
                total_delay += (L_prop + L_que + L_tran)
                num_valid += 1
        
        # 如果没有有效需求，返回一个基准延迟值而不是0
        if num_valid == 0:
            return self._calculate_reference_delay()
        
        return total_delay / num_valid
    
    def _calculate_queuing_delay(self, path: Tuple[int, ...]) -> float:
        """计算排队延迟 - 公式(18)"""
        total_que_delay = 0.0
        
        for i in range(len(path) - 1):
            node_i, node_j = path[i], path[i + 1]
            if self.graph.has_edge(node_i, node_j):
                load = self.graph[node_i][node_j]['load']
                capacity = self.graph[node_i][node_j]['capacity']
                
                # M/M/1排队模型
                utilization = load / capacity if capacity > 0 else 0
                if utilization < 1.0:
                    que_delay = utilization / (capacity * (1 - utilization) + 1e-6)
                else:
                    que_delay = 1.0  # 过载情况
                
                total_que_delay += que_delay
        
        return total_que_delay
    
    def _calculate_transmission_delay(self, demand: TrafficDemand) -> float:
        """计算传输延迟"""
        if demand.selected_path and len(demand.selected_path) >= 2:
            num_hops = len(demand.selected_path) - 1
            packet_size = self.config.PACKET_SIZE  # Kbs
            avg_capacity = self.isl_capacity  # Mbps
            
            # 传输延迟 = 包大小 / 传输速率
            tran_delay = (packet_size / 1000.0) / avg_capacity * num_hops
            return tran_delay
        
        return 0.0
    
    def _calculate_packet_loss_rate(self) -> float:
        """计算丢包率 - 公式(20)-(21)"""
        total_loss = 0.0
        total_flow = 0.0
        
        for demand in self.traffic_demands:
            # 只处理有实际数据速率的需求
            if demand.data_rate > 0 and demand.selected_path and len(demand.selected_path) >= 2:
                path_loss = 0.0
                
                for i in range(len(demand.selected_path) - 1):
                    node_i, node_j = demand.selected_path[i], demand.selected_path[i + 1]
                    if self.graph.has_edge(node_i, node_j):
                        load = self.graph[node_i][node_j]['load']
                        capacity = self.graph[node_i][node_j]['capacity']
                        
                        # 丢包率 = max(0, (load - capacity) / load)
                        if load > capacity:
                            link_loss = (load - capacity) / load
                            path_loss = max(path_loss, link_loss)
                
                total_loss += path_loss * demand.data_rate
                total_flow += demand.data_rate
            elif demand.data_rate > 0 and (not demand.selected_path or len(demand.selected_path) < 2):
                # 如果需求存在但没有分配路径，视为100%丢包
                total_loss += demand.data_rate
                total_flow += demand.data_rate
        
        return total_loss / total_flow if total_flow > 0 else 0.0
    
    def _calculate_link_utilization_variance(self) -> float:
        """计算链路利用率方差"""
        utilizations = []
        
        for edge in self.graph.edges():
            i, j = edge
            load = self.graph[i][j]['load']
            capacity = self.graph[i][j]['capacity']
            utilization = load / capacity if capacity > 0 else 0
            utilizations.append(utilization)
        
        if len(utilizations) > 0:
            return np.var(utilizations)
        
        return 0.0
    
    def _calculate_avg_link_utilization(self) -> float:
        """计算平均链路利用率"""
        utilizations = []
        
        for edge in self.graph.edges():
            i, j = edge
            load = self.graph[i][j]['load']
            capacity = self.graph[i][j]['capacity']
            utilization = load / capacity if capacity > 0 else 0
            utilizations.append(utilization)
        
        if len(utilizations) > 0:
            return np.mean(utilizations)
        
        return 0.0


if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from config import Config
    
    # 测试环境
    env = SatelliteNetwork(Config)
    state = env.reset()
    
    print("=" * 60)
    print("卫星网络环境测试")
    print("=" * 60)
    print(f"节点数: {state['num_nodes']}")
    print(f"边数: {state['num_edges']}")
    print(f"流量需求数: {len(state['demands'])}")
    print(f"时间槽: {state['time_slot']}")
    
    # 测试随机动作
    actions = {}
    for idx, demand in enumerate(env.traffic_demands[:5]):
        try:
            path = nx.shortest_path(env.graph, demand.origin, demand.destination)
            actions[idx] = tuple(path)
        except:
            pass
    
    next_state, reward, done, info = env.step(actions)
    print(f"\n执行动作后:")
    print(f"  奖励: {reward:.4f}")
    print(f"  成功率: {info['success_rate']:.2%}")
    
    metrics = env.calculate_metrics()
    print(f"\n性能指标:")
    print(f"  平均端到端延迟: {metrics['avg_e2e_delay']:.6f} s")
    print(f"  丢包率: {metrics['packet_loss_rate']:.4f}")
    print(f"  链路利用率方差: {metrics['link_utilization_variance']:.4f}")
    print(f"  平均链路利用率: {metrics['avg_link_utilization']:.4f}")

