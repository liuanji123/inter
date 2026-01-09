"""
运行完整实验流程
包括训练和评估
"""

import os
import sys
import argparse
from datetime import datetime

from config import Config


def run_training():
    """运行训练"""
    print("\n" + "=" * 80)
    print("步骤 1: 训练GQN模型")
    print("=" * 80)
    
    from main import train
    train()


def run_evaluation():
    """运行评估"""
    print("\n" + "=" * 80)
    print("步骤 2: 评估所有方法并生成对比图表")
    print("=" * 80)
    
    from experiments.evaluate import ExperimentRunner
    runner = ExperimentRunner()
    runner.run_all_experiments()


def run_testing():
    """运行测试"""
    print("\n" + "=" * 80)
    print("步骤 3: 测试最佳模型")
    print("=" * 80)
    
    from test import test_model
    model_path = os.path.join(Config.CHECKPOINT_DIR, 'best_model.pth')
    
    if os.path.exists(model_path):
        test_model(model_path, num_episodes=50)
    else:
        print(f"警告: 未找到模型文件 {model_path}")
        print("请先运行训练")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='LEO卫星网络星间路由 - 完整实验流程'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'eval', 'test', 'all'],
        default='all',
        help='运行模式: train(训练), eval(评估), test(测试), all(全部)'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='跳过训练（使用已有模型）'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LEO卫星网络星间路由 - 论文完整复现")
    print("GNN与DRL集成方法")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"运行模式: {args.mode}")
    print("=" * 80)
    
    try:
        if args.mode == 'train':
            run_training()
        
        elif args.mode == 'eval':
            run_evaluation()
        
        elif args.mode == 'test':
            run_testing()
        
        elif args.mode == 'all':
            if not args.skip_train:
                run_training()
            else:
                print("\n跳过训练，使用已有模型")
            
            run_evaluation()
            run_testing()
        
        print("\n" + "=" * 80)
        print("实验完成！")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结果保存在: {Config.RESULT_DIR}")
        print("=" * 80)
    
    except KeyboardInterrupt:
        print("\n\n实验被用户中断")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

