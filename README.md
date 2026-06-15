# 正态性检验服务

基于 Python + SciPy 实现的正态性检验服务，支持 Shapiro-Wilk 检验（小样本）和 Kolmogorov-Smirnov 检验（大样本）。

## 功能特性

- **Shapiro-Wilk 检验**: 适用于小样本数据（推荐 n < 50，最大支持 n ≤ 5000）
- **Kolmogorov-Smirnov 检验**: 适用于大样本数据（推荐 n ≥ 50）
- **自动选择**: 根据样本量自动选择最优检验方法
- **综合检验**: 同时执行两种检验并给出综合结论
- **RESTful API**: 基于 FastAPI 提供 HTTP 接口服务

## 项目结构

```
├── normality_test.py    # 正态性检验核心模块
├── main.py              # FastAPI 服务接口
├── test_normality.py    # 单元测试
├── example.py           # 使用示例
└── requirements.txt     # 项目依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 作为 Python 库使用

```python
import numpy as np
from normality_test import NormalityTest

tester = NormalityTest(alpha=0.05)

# 生成测试数据
data = np.random.normal(0, 1, 30)

# 自动选择检验方法
result = tester.auto_test(data)
print(result['conclusion'])

# 综合检验
result = tester.comprehensive_test(data)
print(result['overall']['conclusion'])
```

### 3. 启动 API 服务

```bash
python main.py
```

服务启动后访问:
- 首页: http://localhost:8000/
- 接口文档: http://localhost:8000/docs
- 服务说明: http://localhost:8000/info

### 4. API 调用示例

```bash
curl -X POST "http://localhost:8000/test" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0, 10.1],
    "alpha": 0.05,
    "test_method": "auto"
  }'
```

## 检验方法说明

### Shapiro-Wilk 检验
- **适用场景**: 小样本数据（n < 50）
- **原理**: 基于数据的顺序统计量与正态分布顺序统计量的相关性
- **优点**: 小样本下检验效能较高

### Kolmogorov-Smirnov 检验
- **适用场景**: 大样本数据（n ≥ 50）
- **原理**: 基于经验分布函数与理论正态分布函数的最大差异
- **优点**: 对分布形状敏感，不受参数估计影响

## 检验方法参数

| 参数 | 可选值 | 说明 |
|------|--------|------|
| test_method | auto | 根据样本量自动选择 |
| test_method | shapiro | Shapiro-Wilk 检验 |
| test_method | ks | Kolmogorov-Smirnov 检验 |
| test_method | comprehensive | 综合检验 |

## 运行测试

```bash
python -m unittest test_normality.py -v
```

## 运行示例

```bash
python example.py
```

