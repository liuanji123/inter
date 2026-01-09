"""
拓扑生成器 - LEO卫星网络拓扑构建
基于论文第II节问题建模
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict
import math


class TopologyGenerator:
    """LEO卫星网络拓扑生成器"""
    
    def __init__(self, num_orbits: int, sats_per_orbit: int, 
                 altitude: float, inclination: float, earth_radius: float):
        """
        初始化拓扑生成器
        
        Args:
            num_orbits: 轨道数量
            sats_per_orbit: 每轨道卫星数
            altitude: 轨道高度 (km)
            inclination: 轨道倾角 (度)
            earth_radius: 地球半径 (km)
        """
        self.num_orbits = num_orbits
        self.sats_per_orbit = sats_per_orbit
        self.total_satellites = num_orbits * sats_per_orbit
        self.altitude = altitude
        self.inclination = inclination
        self.earth_radius = earth_radius
        self.orbital_radius = earth_radius + altitude
        
        # 卫星位置缓存
        self.satellite_positions = None
        
    def generate_satellite_positions(self, time: float = 0.0) -> np.ndarray:
        """
        生成卫星3D位置坐标
        
        Args:
            time: 当前时间 (秒)，用于动态拓扑
            
        Returns:
            positions: (N, 3) 卫星位置数组 [x, y, z]
        """
        positions = []
        
        # 轨道周期 (秒) - 简化计算
        orbital_period = 2 * np.pi * np.sqrt(self.orbital_radius**3 / 398600.4418)
        
        for orbit_idx in range(self.num_orbits):
            # 轨道平面的相位
            orbit_phase = 2 * np.pi * orbit_idx / self.num_orbits
            
            for sat_idx in range(self.sats_per_orbit):
                # 卫星在轨道内的相位角
                sat_phase = 2 * np.pi * sat_idx / self.sats_per_orbit
                
                # 考虑时间演化
                current_phase = sat_phase + (2 * np.pi * time / orbital_period)
                
                # 3D笛卡尔坐标
                x = self.orbital_radius * np.cos(current_phase) * np.cos(orbit_phase)
                y = self.orbital_radius * np.cos(current_phase) * np.sin(orbit_phase)
                z = self.orbital_radius * np.sin(current_phase) * np.sin(np.radians(self.inclination))
                
                positions.append([x, y, z])
        
        self.satellite_positions = np.array(positions)
        return self.satellite_positions
    
    def calculate_distance(self, sat_i: int, sat_j: int) -> float:
        """
        计算两颗卫星之间的欧氏距离
        
        Args:
            sat_i: 卫星i索引
            sat_j: 卫星j索引
            
        Returns:
            distance: 距离 (km)
        """
        if self.satellite_positions is None:
            self.generate_satellite_positions()
        
        pos_i = self.satellite_positions[sat_i]
        pos_j = self.satellite_positions[sat_j]
        
        return np.linalg.norm(pos_i - pos_j)
    
    def build_topology_graph(self, time: float = 0.0) -> nx.Graph:
        """
        构建卫星网络拓扑图
        使用四链路算法：每颗卫星连接同轨道前后卫星和相邻轨道卫星
        
        Args:
            time: 当前时间
            
        Returns:
            G: NetworkX图对象
        """
        # 更新卫星位置
        self.generate_satellite_positions(time)
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for sat_id in range(self.total_satellites):
            G.add_node(sat_id, position=self.satellite_positions[sat_id])
        
        # 添加ISL边 - 四链路算法
        for orbit_idx in range(self.num_orbits):
            for sat_idx in range(self.sats_per_orbit):
                current_sat = orbit_idx * self.sats_per_orbit + sat_idx
                
                # 1. 同轨道内前向链路
                next_sat_in_orbit = orbit_idx * self.sats_per_orbit + (sat_idx + 1) % self.sats_per_orbit
                distance = self.calculate_distance(current_sat, next_sat_in_orbit)
                G.add_edge(current_sat, next_sat_in_orbit, 
                          distance=distance, 
                          capacity=0.0,  # 将在环境中设置
                          load=0.0)
                
                # 2. 相邻轨道链路 (两条)
                next_orbit = (orbit_idx + 1) % self.num_orbits
                
                # 连接到相邻轨道的同位置卫星
                neighbor_sat_1 = next_orbit * self.sats_per_orbit + sat_idx
                distance = self.calculate_distance(current_sat, neighbor_sat_1)
                G.add_edge(current_sat, neighbor_sat_1,
                          distance=distance,
                          capacity=0.0,
                          load=0.0)
                
                # 连接到相邻轨道的前一位置卫星
                neighbor_sat_2 = next_orbit * self.sats_per_orbit + (sat_idx - 1) % self.sats_per_orbit
                if not G.has_edge(current_sat, neighbor_sat_2):
                    distance = self.calculate_distance(current_sat, neighbor_sat_2)
                    G.add_edge(current_sat, neighbor_sat_2,
                              distance=distance,
                              capacity=0.0,
                              load=0.0)
        
        return G
    
    def get_isl_list(self, graph: nx.Graph) -> List[Tuple[int, int]]:
        """
        获取所有ISL列表
        
        Args:
            graph: 网络图
            
        Returns:
            isl_list: ISL边列表 [(i, j), ...]
        """
        return list(graph.edges())
    
    def visualize_topology(self, graph: nx.Graph, save_path: str = None):
        """
        可视化网络拓扑
        
        Args:
            graph: 网络图
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制卫星节点
        positions = self.satellite_positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
                  c='red', marker='o', s=50, label='Satellites')
        
        # 绘制ISL
        for edge in graph.edges():
            sat_i, sat_j = edge
            pos_i = positions[sat_i]
            pos_j = positions[sat_j]
            ax.plot([pos_i[0], pos_j[0]], 
                   [pos_i[1], pos_j[1]], 
                   [pos_i[2], pos_j[2]], 
                   'b-', alpha=0.3, linewidth=0.5)
        
        # 绘制地球
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x = self.earth_radius * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='blue', alpha=0.3)
        
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'LEO Satellite Network Topology\n{self.total_satellites} Satellites')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    # 测试拓扑生成
    from config import Config
    
    topo_gen = TopologyGenerator(
        num_orbits=Config.NUM_ORBITS,
        sats_per_orbit=Config.SATS_PER_ORBIT,
        altitude=Config.ALTITUDE,
        inclination=Config.INCLINATION,
        earth_radius=Config.EARTH_RADIUS
    )
    
    # 生成拓扑
    G = topo_gen.build_topology_graph()
    print(f"生成的网络拓扑:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  平均度: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # 可视化
    topo_gen.visualize_topology(G, save_path='topology_visualization.png')
    print("拓扑可视化已保存")

