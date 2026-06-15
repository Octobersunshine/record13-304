import numpy as np
from normality_test import NormalityTest


def example_basic_usage():
    print("=" * 60)
    print("示例 1: 基本使用")
    print("=" * 60)
    
    tester = NormalityTest(alpha=0.05)
    
    np.random.seed(42)
    
    normal_data = np.random.normal(loc=0, scale=1, size=30)
    print(f"正态分布数据（小样本，n=30）:")
    print(f"  前5个值: {normal_data[:5]}")
    print()
    
    result = tester.shapiro_wilk_test(normal_data)
    print("Shapiro-Wilk 检验结果:")
    print(f"  统计量: {result['statistic']:.4f}")
    print(f"  P值: {result['p_value']:.4f}")
    print(f"  结论: {result['conclusion']}")
    print()
    
    large_normal_data = np.random.normal(loc=5, scale=2, size=100)
    print(f"正态分布数据（大样本，n=100）:")
    result = tester.kolmogorov_smirnov_test(large_normal_data)
    print(f"  统计量: {result['statistic']:.4f}")
    print(f"  P值: {result['p_value']:.4f}")
    print(f"  估计均值: {result['estimated_mean']:.4f}")
    print(f"  估计标准差: {result['estimated_std']:.4f}")
    print(f"  结论: {result['conclusion']}")
    print()


def example_auto_selection():
    print("=" * 60)
    print("示例 2: 自动选择检验方法")
    print("=" * 60)
    
    tester = NormalityTest()
    
    np.random.seed(42)
    
    small_data = np.random.normal(0, 1, 20)
    result = tester.auto_test(small_data)
    print(f"样本量 {result['sample_size']}: {result['selection_reason']}")
    print(f"  检验方法: {result['test_name']}")
    print(f"  P值: {result['p_value']:.4f}")
    print(f"  结论: {result['conclusion']}")
    print()
    
    large_data = np.random.normal(0, 1, 100)
    result = tester.auto_test(large_data)
    print(f"样本量 {result['sample_size']}: {result['selection_reason']}")
    print(f"  检验方法: {result['test_name']}")
    print(f"  P值: {result['p_value']:.4f}")
    print(f"  结论: {result['conclusion']}")
    print()


def example_non_normal():
    print("=" * 60)
    print("示例 3: 非正态分布数据检验")
    print("=" * 60)
    
    tester = NormalityTest()
    
    np.random.seed(42)
    
    exp_data = np.random.exponential(scale=1.0, size=50)
    print(f"指数分布数据（n=50）:")
    
    result = tester.auto_test(exp_data)
    print(f"  检验方法: {result['test_name']}")
    print(f"  统计量: {result['statistic']:.4f}")
    print(f"  P值: {result['p_value']:.4f}")
    print(f"  结论: {result['conclusion']}")
    print()


def example_comprehensive():
    print("=" * 60)
    print("示例 4: 综合检验")
    print("=" * 60)
    
    tester = NormalityTest()
    
    np.random.seed(42)
    
    data = np.random.normal(0, 1, 40)
    result = tester.comprehensive_test(data)
    
    print(f"样本量: {result['overall']['sample_size']}")
    print()
    
    print("描述性统计:")
    stats = result['overall']['descriptive_stats']
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    print()
    
    for test_name, test_result in result['tests'].items():
        print(f"{test_result['test_name']} 检验:")
        print(f"  统计量: {test_result['statistic']:.4f}")
        print(f"  P值: {test_result['p_value']:.4f}")
        print(f"  结论: {test_result['conclusion']}")
        print()
    
    print(f"综合结论: {result['overall']['conclusion']}")
    print()


def example_custom_alpha():
    print("=" * 60)
    print("示例 5: 自定义显著性水平")
    print("=" * 60)
    
    tester_001 = NormalityTest(alpha=0.01)
    tester_010 = NormalityTest(alpha=0.10)
    
    np.random.seed(123)
    data = np.random.normal(0, 1, 30)
    
    print(f"数据: 正态分布（n=30）")
    print()
    
    for alpha, tester in [(0.01, tester_001), (0.10, tester_010)]:
        result = tester.shapiro_wilk_test(data)
        print(f"α = {alpha}:")
        print(f"  P值: {result['p_value']:.4f}")
        print(f"  结论: {result['conclusion']}")
        print()


if __name__ == "__main__":
    example_basic_usage()
    example_auto_selection()
    example_non_normal()
    example_comprehensive()
    example_custom_alpha()
    
    print("=" * 60)
    print("示例运行完成！")
    print("=" * 60)
