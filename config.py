"""
配置文件 - LEO卫星网络星间路由参数
基于论文 "Inter-Satellite Routing for LEO Satellite Networks: A GNN and DRL Integrated Approach"
"""

import torch

class Config:
    """系统配置参数"""
    
    # ==================== 卫星网络参数 ====================
    # 卫星星座配置（快速训练模式：减小规模）
    NUM_ORBITS = 6                    # 轨道数量
    SATS_PER_ORBIT = 8                # 每轨道卫星数
    TOTAL_SATELLITES = NUM_ORBITS * SATS_PER_ORBIT  # 总卫星数: 48
    INCLINATION = 50.0                # 轨道倾角 (度)
    ALTITUDE = 550.0                  # 轨道高度 (km)
    EARTH_RADIUS = 6371.0             # 地球半径 (km)
    
    # ISL配置
    NUM_ISL_PER_SAT = 4               # 每颗卫星的ISL数量
    ISL_CAPACITY = 15.0               # ISL容量 (Mbps) - 增加以支持高负载
    SPEED_OF_LIGHT = 299792.458       # 光速 (km/s)
    
    # 流量配置
    MIN_TRAFFIC_DEMAND = 0.5          # 最小流量需求 (Mbps) - 降低以适应小网络
    MAX_TRAFFIC_DEMAND = 1.5          # 最大流量需求 (Mbps)
    PACKET_SIZE = 10.0                # 数据包大小 (Kbs)
    
    # ==================== GNN模型参数 ====================
    # MPNN配置（优化版：减少复杂度）
    NODE_FEATURE_DIM = 8              # 节点特征维度
    HIDDEN_DIM = 64                   # 隐藏层维度
    OUTPUT_DIM = 32                   # 输出维度
    NUM_MESSAGE_PASSING = 3           # 消息传递迭代次数 T（减少）
    AGGREGATION = 'sum'               # 聚合函数: sum, mean, max
    
    # ==================== DRL参数 ====================
    # DQN配置 (优化版)
    LEARNING_RATE = 0.0005            # 学习率 (提高以加快学习)
    GAMMA = 0.99                      # 折扣因子 γ (提高以重视长期奖励)
    EPSILON_START = 1.0               # ε-greedy初始值
    EPSILON_END = 0.01                # ε-greedy最终值 (降低以允许更多探索)
    EPSILON_DECAY = 0.997             # ε衰减率 (更慢衰减，增加探索)
    
    # Experience Replay
    REPLAY_BUFFER_SIZE = 20000        # 回放缓冲区大小 (增加以存储更多经验)
    BATCH_SIZE = 128                  # 批次大小 (增加以提高训练稳定性)
    
    # Target Network更新
    TARGET_UPDATE_FREQ = 100          # Target Network更新频率 N (降低更新频率，提高稳定性)
    
    # ==================== 路由算法参数 ====================
    K_SHORTEST_PATHS = 3              # k最短路径数量
    
    # 奖励函数权重 (公式15)
    ALPHA = 0.5                       # 传播延迟权重 α
    BETA = 0.5                        # 剩余容量权重 β
    
    # ==================== 训练参数 ====================
    NUM_EPISODES = 500                # 训练episode数量 (增加以充分训练)
    MAX_STEPS_PER_EPISODE = 20        # 每个episode最大步数 (增加以获得更多经验)
    EVAL_INTERVAL = 25                # 评估间隔
    SAVE_INTERVAL = 100               # 保存间隔
    NUM_DEMANDS = 30                  # 流量需求数量 (增加以支持高负载场景)
    
    # ==================== 实验参数 ====================
    # 时间槽配置
    TIME_SLOTS = 5                    # 时间槽数量
    SLOT_DURATION = 60.0              # 时间槽持续时间 (秒)
    
    # 数据传输速率测试范围
    DATA_RATES = [10, 20, 30, 40, 50, 60, 70, 80, 90]  # Mbps
    
    # ==================== 系统参数 ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42                         # 随机种子
    
    # 路径配置
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    RESULT_DIR = './results'
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print("LEO卫星网络星间路由 - 配置参数")
        print("=" * 60)
        print(f"卫星网络:")
        print(f"  - 轨道数: {cls.NUM_ORBITS}")
        print(f"  - 每轨道卫星数: {cls.SATS_PER_ORBIT}")
        print(f"  - 总卫星数: {cls.TOTAL_SATELLITES}")
        print(f"  - 轨道高度: {cls.ALTITUDE} km")
        print(f"  - ISL容量: {cls.ISL_CAPACITY} Mbps")
        print(f"\nGNN模型:")
        print(f"  - 隐藏层维度: {cls.HIDDEN_DIM}")
        print(f"  - 消息传递次数: {cls.NUM_MESSAGE_PASSING}")
        print(f"\nDRL参数:")
        print(f"  - 学习率: {cls.LEARNING_RATE}")
        print(f"  - Gamma: {cls.GAMMA}")
        print(f"  - 批次大小: {cls.BATCH_SIZE}")
        print(f"\n设备: {cls.DEVICE}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()

