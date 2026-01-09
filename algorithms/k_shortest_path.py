"""
Yen's K最短路径算法
基于论文第III节C部分，用于生成候选路径作为动作空间
参考: J. Y. Yen, "An Algorithm for Finding Shortest Routes from All Source 
      Nodes to A Given Destination in General Networks," 1970
"""

import networkx as nx
from typing import List, Tuple, Optional
import heapq


class YenKShortestPaths:
    """Yen's K最短路径算法实现"""
    
    def __init__(self, graph: nx.Graph, weight: str = 'distance'):
        """
        初始化算法
        
        Args:
            graph: NetworkX图对象
            weight: 边权重属性名
        """
        self.graph = graph
        self.weight = weight
    
    def find_k_shortest_paths(self, source: int, target: int, k: int = 3) -> List[List[int]]:
        """
        寻找从source到target的k条最短路径
        
        Args:
            source: 源节点
            target: 目标节点
            k: 路径数量
            
        Returns:
            paths: k条最短路径列表，每条路径是节点列表
        """
        if source == target:
            return [[source]]
        
        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return []
        
        # A存储已找到的k条最短路径
        A = []
        
        # B是候选路径的优先队列 (路径长度, 路径)
        B = []
        
        try:
            # 找第一条最短路径
            first_path = nx.shortest_path(
                self.graph, 
                source, 
                target, 
                weight=self.weight
            )
            first_length = self._path_length(first_path)
            A.append(first_path)
        except nx.NetworkXNoPath:
            return []
        
        # 迭代寻找第2到第k条路径
        for k_iter in range(1, k):
            # 上一条路径
            prev_path = A[k_iter - 1]
            
            # 遍历上一条路径的所有节点（除了目标节点）
            for i in range(len(prev_path) - 1):
                # Spur node: 分支节点
                spur_node = prev_path[i]
                # Root path: 从源到spur node的路径
                root_path = prev_path[:i + 1]
                
                # 临时移除的边
                edges_removed = []
                
                # 对于A中所有与root_path相同的路径
                for path in A:
                    if len(path) > i and root_path == path[:i + 1]:
                        # 移除与该路径相同的下一条边
                        if i + 1 < len(path):
                            u, v = path[i], path[i + 1]
                            if self.graph.has_edge(u, v):
                                edge_data = self.graph[u][v].copy()
                                self.graph.remove_edge(u, v)
                                edges_removed.append((u, v, edge_data))
                
                # 临时移除root path中的节点（除了spur node）
                nodes_removed = []
                for node in root_path[:-1]:
                    if node != spur_node and self.graph.has_node(node):
                        # 保存节点的所有边
                        node_edges = []
                        for neighbor in list(self.graph.neighbors(node)):
                            edge_data = self.graph[node][neighbor].copy()
                            node_edges.append((node, neighbor, edge_data))
                        
                        self.graph.remove_node(node)
                        nodes_removed.append((node, node_edges))
                
                # 寻找从spur node到target的最短路径
                try:
                    spur_path = nx.shortest_path(
                        self.graph,
                        spur_node,
                        target,
                        weight=self.weight
                    )
                    
                    # 组合路径
                    total_path = root_path[:-1] + spur_path
                    total_length = self._path_length(total_path)
                    
                    # 添加到候选路径（如果不重复）
                    if total_path not in [p for _, p in B] and total_path not in A:
                        heapq.heappush(B, (total_length, total_path))
                
                except nx.NetworkXNoPath:
                    pass
                
                # 恢复移除的节点
                for node, node_edges in nodes_removed:
                    self.graph.add_node(node)
                    for u, v, data in node_edges:
                        self.graph.add_edge(u, v, **data)
                
                # 恢复移除的边
                for u, v, data in edges_removed:
                    self.graph.add_edge(u, v, **data)
            
            # 如果没有候选路径了，结束
            if len(B) == 0:
                break
            
            # 选择候选路径中最短的加入A
            _, shortest_candidate = heapq.heappop(B)
            A.append(shortest_candidate)
        
        return A
    
    def _path_length(self, path: List[int]) -> float:
        """
        计算路径长度
        
        Args:
            path: 节点路径
            
        Returns:
            length: 路径总长度
        """
        length = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.graph.has_edge(u, v):
                length += self.graph[u][v].get(self.weight, 1.0)
            else:
                return float('inf')
        return length
    
    def find_k_shortest_paths_for_all_pairs(self, k: int = 3) -> dict:
        """
        为所有节点对找k条最短路径
        
        Args:
            k: 路径数量
            
        Returns:
            all_paths: {(source, target): [path1, path2, ...]}
        """
        all_paths = {}
        nodes = list(self.graph.nodes())
        
        for source in nodes:
            for target in nodes:
                if source != target:
                    paths = self.find_k_shortest_paths(source, target, k)
                    if paths:
                        all_paths[(source, target)] = paths
        
        return all_paths


def test_yen_algorithm():
    """测试Yen算法"""
    # 创建测试图
    G = nx.Graph()
    edges = [
        (0, 1, 1), (0, 2, 2), (1, 2, 1), (1, 3, 3),
        (2, 3, 1), (2, 4, 2), (3, 4, 1), (3, 5, 2),
        (4, 5, 1)
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, distance=w)
    
    # 测试算法
    yen = YenKShortestPaths(G, weight='distance')
    paths = yen.find_k_shortest_paths(0, 5, k=3)
    
    print("从节点0到节点5的3条最短路径:")
    for i, path in enumerate(paths, 1):
        length = yen._path_length(path)
        print(f"  路径{i}: {path}, 长度: {length}")


if __name__ == "__main__":
    test_yen_algorithm()

