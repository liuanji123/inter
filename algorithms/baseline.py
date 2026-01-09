"""
基准路由算法
包括：最短路径(SP)、负载均衡(LB)
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple


class ShortestPathRouting:
    """最短路径路由(SP)"""
    
    def __init__(self, weight: str = 'distance'):
        """
        初始化
        
        Args:
            weight: 权重属性
        """
        self.weight = weight
    
    def route(self, graph: nx.Graph, demands: list) -> Dict[int, Tuple[int, ...]]:
        """
        为所有流量需求计算最短路径
        
        Args:
            graph: 网络图
            demands: 流量需求列表
            
        Returns:
            actions: {demand_idx: path}
        """
        actions = {}
        
        for idx, demand in enumerate(demands):
            try:
                path = nx.shortest_path(
                    graph,
                    demand.origin,
                    demand.destination,
                    weight=self.weight
                )
                actions[idx] = tuple(path)
            except nx.NetworkXNoPath:
                # 如果没有路径，使用空路径
                actions[idx] = (demand.origin,)
        
        return actions


class LoadBalancingRouting:
    """负载均衡路由(LB)"""
    
    def __init__(self, k: int = 3, weight: str = 'distance'):
        """
        初始化
        
        Args:
            k: 候选路径数量
            weight: 权重属性
        """
        self.k = k
        self.weight = weight
    
    def route(self, graph: nx.Graph, demands: list) -> Dict[int, Tuple[int, ...]]:
        """
        使用负载均衡策略路由
        选择剩余容量最大的路径
        
        Args:
            graph: 网络图
            demands: 流量需求列表
            
        Returns:
            actions: {demand_idx: path}
        """
        from .k_shortest_path import YenKShortestPaths
        
        actions = {}
        
        # 为每个需求找k条候选路径
        for idx, demand in enumerate(demands):
            yen = YenKShortestPaths(graph, weight=self.weight)
            candidate_paths = yen.find_k_shortest_paths(
                demand.origin,
                demand.destination,
                k=self.k
            )
            
            if not candidate_paths:
                actions[idx] = (demand.origin,)
                continue
            
            # 选择剩余容量最大的路径
            best_path = None
            max_residual = -1
            
            for path in candidate_paths:
                residual = self._calculate_path_residual_capacity(graph, path)
                if residual > max_residual:
                    max_residual = residual
                    best_path = path
            
            actions[idx] = tuple(best_path) if best_path else (demand.origin,)
        
        return actions
    
    def _calculate_path_residual_capacity(self, graph: nx.Graph, path: list) -> float:
        """
        计算路径剩余容量
        
        Args:
            graph: 网络图
            path: 路径
            
        Returns:
            residual: 剩余容量
        """
        min_residual = float('inf')
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if graph.has_edge(u, v):
                capacity = graph[u][v].get('capacity', 0)
                load = graph[u][v].get('load', 0)
                residual = capacity - load
                min_residual = min(min_residual, residual)
        
        return max(0, min_residual) if min_residual != float('inf') else 0


if __name__ == "__main__":
    # 简单测试
    G = nx.Graph()
    G.add_edge(0, 1, distance=1, capacity=10, load=2)
    G.add_edge(1, 2, distance=1, capacity=10, load=5)
    G.add_edge(0, 2, distance=2, capacity=10, load=1)
    
    # 模拟流量需求
    class MockDemand:
        def __init__(self, o, d):
            self.origin = o
            self.destination = d
    
    demands = [MockDemand(0, 2)]
    
    # 测试SP
    sp = ShortestPathRouting()
    actions = sp.route(G, demands)
    print(f"SP路由: {actions}")
    
    # 测试LB
    lb = LoadBalancingRouting(k=2)
    actions = lb.route(G, demands)
    print(f"LB路由: {actions}")

