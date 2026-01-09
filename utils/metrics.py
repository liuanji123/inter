"""
性能指标计算
基于论文公式(16)-(21)
"""

import numpy as np
from typing import Dict, List


class MetricsCalculator:
    """性能指标计算器"""
    
    @staticmethod
    def calculate_all_metrics(env_metrics_list: List[Dict]) -> Dict:
        """
        计算所有性能指标的平均值
        
        Args:
            env_metrics_list: 环境指标列表
            
        Returns:
            averaged_metrics: 平均性能指标
        """
        if not env_metrics_list:
            return {}
        
        # 聚合所有指标
        metrics_dict = {
            'avg_e2e_delay': [],
            'packet_loss_rate': [],
            'link_utilization_variance': [],
            'avg_link_utilization': []
        }
        
        for metrics in env_metrics_list:
            for key in metrics_dict.keys():
                if key in metrics:
                    metrics_dict[key].append(metrics[key])
        
        # 计算平均值
        averaged_metrics = {}
        for key, values in metrics_dict.items():
            if values:
                averaged_metrics[key] = np.mean(values)
                averaged_metrics[f'{key}_std'] = np.std(values)
            else:
                averaged_metrics[key] = 0.0
                averaged_metrics[f'{key}_std'] = 0.0
        
        return averaged_metrics
    
    @staticmethod
    def compare_methods(results: Dict[str, Dict]) -> None:
        """
        比较不同方法的性能
        
        Args:
            results: {method_name: metrics_dict}
        """
        print("=" * 80)
        print("性能对比")
        print("=" * 80)
        
        metrics_to_compare = [
            'avg_e2e_delay',
            'packet_loss_rate', 
            'link_utilization_variance',
            'avg_link_utilization'
        ]
        
        print(f"{'方法':<15}", end="")
        for metric in metrics_to_compare:
            print(f"{metric:<25}", end="")
        print()
        print("-" * 80)
        
        for method_name, metrics in results.items():
            print(f"{method_name:<15}", end="")
            for metric in metrics_to_compare:
                value = metrics.get(metric, 0.0)
                std = metrics.get(f'{metric}_std', 0.0)
                print(f"{value:.6f}±{std:.6f}".ljust(25), end="")
            print()
        
        print("=" * 80)
    
    @staticmethod
    def format_metrics(metrics: Dict, method_name: str = "") -> str:
        """
        格式化输出指标
        
        Args:
            metrics: 指标字典
            method_name: 方法名称
            
        Returns:
            formatted: 格式化字符串
        """
        lines = []
        if method_name:
            lines.append(f"\n{'='*60}")
            lines.append(f"{method_name} - 性能指标")
            lines.append(f"{'='*60}")
        
        # 端到端延迟
        if 'avg_e2e_delay' in metrics:
            delay = metrics['avg_e2e_delay']
            delay_std = metrics.get('avg_e2e_delay_std', 0)
            lines.append(f"平均端到端延迟: {delay:.6f} ± {delay_std:.6f} 秒")
        
        # 丢包率
        if 'packet_loss_rate' in metrics:
            loss = metrics['packet_loss_rate']
            loss_std = metrics.get('packet_loss_rate_std', 0)
            lines.append(f"丢包率: {loss:.4f} ± {loss_std:.4f}")
        
        # 链路利用率
        if 'avg_link_utilization' in metrics:
            util = metrics['avg_link_utilization']
            util_std = metrics.get('avg_link_utilization_std', 0)
            lines.append(f"平均链路利用率: {util:.4f} ± {util_std:.4f}")
        
        # 链路利用率方差
        if 'link_utilization_variance' in metrics:
            var = metrics['link_utilization_variance']
            var_std = metrics.get('link_utilization_variance_std', 0)
            lines.append(f"链路利用率方差: {var:.6f} ± {var_std:.6f}")
        
        if method_name:
            lines.append(f"{'='*60}\n")
        
        return '\n'.join(lines)


if __name__ == "__main__":
    # 测试指标计算
    metrics_list = [
        {
            'avg_e2e_delay': 0.15,
            'packet_loss_rate': 0.02,
            'link_utilization_variance': 0.08,
            'avg_link_utilization': 0.6
        },
        {
            'avg_e2e_delay': 0.16,
            'packet_loss_rate': 0.03,
            'link_utilization_variance': 0.09,
            'avg_link_utilization': 0.65
        }
    ]
    
    calc = MetricsCalculator()
    avg_metrics = calc.calculate_all_metrics(metrics_list)
    
    print(calc.format_metrics(avg_metrics, "测试方法"))
    
    # 测试对比
    results = {
        'GQN': avg_metrics,
        'SP': {
            'avg_e2e_delay': 0.20,
            'packet_loss_rate': 0.05,
            'link_utilization_variance': 0.12,
            'avg_link_utilization': 0.5
        }
    }
    
    calc.compare_methods(results)

