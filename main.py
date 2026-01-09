"""
主训练脚本
训练GQN模型进行LEO卫星网络星间路由
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from config import Config
from environment.satellite_network import SatelliteNetwork
from models.gqn import GQNAgent
from utils.replay_buffer import ReplayBuffer
from utils.metrics import MetricsCalculator


def setup_directories():
    """创建必要的目录"""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train():
    """训练GQN模型"""
    print("=" * 80)
    print("LEO卫星网络星间路由 - GQN训练")
    print("=" * 80)
    
    # 设置
    setup_directories()
    set_seed(Config.SEED)
    Config.print_config()
    
    # 初始化环境
    env = SatelliteNetwork(Config)
    
    # 初始化智能体
    replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_SIZE)
    agent = GQNAgent(Config, replay_buffer)
    
    # 训练历史
    episode_rewards = []
    episode_losses = []
    episode_metrics = []
    
    # 用于奖励归一化的统计量
    reward_mean = 0.0
    reward_std = 1.0
    
    best_reward = -float('inf')
    
    print("\n开始训练...")
    print("-" * 80)
    
    # 训练循环
    for episode in tqdm(range(Config.NUM_EPISODES), desc="训练进度"):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        # Episode循环
        for step in range(Config.MAX_STEPS_PER_EPISODE):
            # 选择动作
            actions = agent.select_actions(state, explore=True)
            
            # 执行动作
            next_state, reward, done, info = env.step(actions)
            
            # 提取状态特征（ISL特征和邻接矩阵）
            isl_features, isl_adjacency = agent._extract_state_feature(state)
            next_isl_features, next_isl_adjacency = agent._extract_state_feature(next_state)
            
            # 计算动作索引（为每个需求记录其选择的路径索引）
            # 使用第一个需求的动作索引作为代表
            action_idx = 0
            if actions and len(state['demands']) > 0:
                first_demand = state['demands'][0]
                if 0 in actions:
                    # 使用Yen算法生成候选路径
                    from algorithms.k_shortest_path import YenKShortestPaths
                    yen = YenKShortestPaths(state['graph'], weight='distance')
                    candidate_paths = yen.find_k_shortest_paths(
                        first_demand.origin,
                        first_demand.destination,
                        k=Config.K_SHORTEST_PATHS
                    )
                    selected_path = actions[0]
                    for i, path in enumerate(candidate_paths):
                        if tuple(path) == selected_path:
                            action_idx = i
                            break
            
            # 存储经验（存储ISL特征和邻接矩阵）
            # 移除batch维度，因为存储时不需要
            state_dict = {
                'isl_features': isl_features.cpu().numpy().squeeze(0),  # (num_isl, feature_dim)
                'isl_adjacency': isl_adjacency.cpu().numpy().squeeze(0)  # (num_isl, num_isl)
            }
            next_state_dict = {
                'isl_features': next_isl_features.cpu().numpy().squeeze(0),
                'isl_adjacency': next_isl_adjacency.cpu().numpy().squeeze(0)
            }
            
            replay_buffer.push(
                state_dict,
                action_idx,
                reward,
                next_state_dict,
                done
            )
            
            # 训练（每步都尝试训练）
            loss = 0.0
            if len(replay_buffer) >= Config.BATCH_SIZE:
                # 从回放缓冲区采样并训练
                batch = replay_buffer.sample(Config.BATCH_SIZE)
                if isinstance(batch, tuple) and len(batch) == 7:
                    # 新格式：ISL特征和邻接矩阵
                    (isl_features_batch, isl_adjacency_batch, actions_batch, 
                     rewards_batch, next_isl_features_batch, next_isl_adjacency_batch, 
                     dones_batch) = batch
                    loss = agent.dqn.train_step(
                        isl_features_batch, isl_adjacency_batch,
                        actions_batch, rewards_batch,
                        next_isl_features_batch, next_isl_adjacency_batch,
                        dones_batch
                    )
            
            # 记录
            episode_reward += reward
            episode_loss += loss
            step_count += 1
            
            state = next_state
            
            if done:
                break
        
        # 记录episode统计
        avg_loss = episode_loss / step_count if step_count > 0 else 0
        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)
        
        # 评估
        if (episode + 1) % Config.EVAL_INTERVAL == 0:
            eval_metrics = evaluate(agent, env, num_episodes=5)
            episode_metrics.append(eval_metrics)
            
            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth'))
            
            # 打印进度
            print(f"\nEpisode {episode + 1}/{Config.NUM_EPISODES}")
            print(f"  平均奖励: {episode_reward:.4f}")
            print(f"  平均损失: {avg_loss:.6f}")
            print(f"  Epsilon: {agent.dqn.epsilon:.4f}")
            print(f"  端到端延迟: {eval_metrics['avg_e2e_delay']:.6f} s")
            print(f"  丢包率: {eval_metrics['packet_loss_rate']:.4f}")
            print(f"  链路利用率: {eval_metrics['avg_link_utilization']:.4f}")
            print("-" * 80)
        
        # 定期保存
        if (episode + 1) % Config.SAVE_INTERVAL == 0:
            agent.save(os.path.join(Config.CHECKPOINT_DIR, f'model_ep{episode+1}.pth'))
    
    # 保存最终模型
    agent.save(os.path.join(Config.CHECKPOINT_DIR, 'final_model.pth'))
    
    # 绘制训练曲线
    plot_training_curves(episode_rewards, episode_losses)
    
    print("\n训练完成！")
    print(f"最佳奖励: {best_reward:.4f}")
    print(f"模型保存在: {Config.CHECKPOINT_DIR}")


def evaluate(agent: GQNAgent, env: SatelliteNetwork, num_episodes: int = 10) -> dict:
    """
    评估智能体性能
    
    Args:
        agent: GQN智能体
        env: 环境
        num_episodes: 评估episode数
        
    Returns:
        metrics: 性能指标
    """
    all_metrics = []
    
    for _ in range(num_episodes):
        state = env.reset()
        
        for _ in range(Config.MAX_STEPS_PER_EPISODE):
            # 不探索，使用贪婪策略
            actions = agent.select_actions(state, explore=False)
            next_state, reward, done, info = env.step(actions)
            state = next_state
            
            if done:
                break
        
        # 计算指标
        metrics = env.calculate_metrics()
        all_metrics.append(metrics)
    
    # 平均指标
    calc = MetricsCalculator()
    avg_metrics = calc.calculate_all_metrics(all_metrics)
    
    return avg_metrics


def plot_training_curves(rewards, losses):
    """
    绘制训练曲线
    
    Args:
        rewards: 奖励列表
        losses: 损失列表
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 奖励曲线
    ax1.plot(rewards, label='Episode Reward', alpha=0.6)
    if len(rewards) > 10:
        # 平滑曲线
        window = min(50, len(rewards) // 10)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), smoothed, 
                label=f'Smoothed (window={window})', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    ax2.plot(losses, label='Training Loss', alpha=0.6, color='orange')
    if len(losses) > 10:
        window = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(losses)), smoothed,
                label=f'Smoothed (window={window})', linewidth=2, color='red')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(Config.RESULT_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n训练曲线已保存到: {save_path}")
    plt.close()


if __name__ == "__main__":
    train()

