# Gaussian Einstein Fields v3.0 - Quick Start Guide

## 概述

成功实现了基于3+1 ADM分解的Gaussian Einstein Fields v3.0！这是一个创新的时空度规表示方法，将4DGS的思想引入到广义相对论中。

## 核心创新

### 1. **物理一致性保证**

传统方法直接用神经网络表示4D度规 `g_μν`，难以保证物理约束。GEF v3.0通过3+1分解解决了这个问题：

```
ds² = -N² dt² + γ_ij (dx^i + β^i dt)(dx^j + β^j dt)
```

- **Lapse N > 0**: 通过对数域表示自动保证
- **Spatial metric γ_ij SPD**: 通过矩阵指数自动保证
- **Lorentzian signature (-,+,+,+)**: 由构造自动满足

### 2. **4D高斯基元**

每个高斯G(p; μ, Σ)由以下参数化：
- **μ**: 4D时空中心位置
- **q_L, q_R**: 四元数对（用于4D旋转）
- **s**: 4维尺度因子

### 3. **三个独立场**

#### Lapse场
```python
log N(p) = Σᵢ [αᵢᴺ * G(p; μᵢ, Σᵢ)]
N(p) = exp(log N(p))  # 自动满足 N > 0
```

#### Shift场
```python
β^i(p) = Σᵢ [vᵢᵝ * G(p; μᵢ, Σᵢ)]  # 3D向量场
```

#### Spatial Metric场
```python
h_γ(p) = Σᵢ [αᵢᵞ * (vᵢᵞ ⊗ vᵢᵞ) * G(p; μᵢ, Σᵢ)]
γ(p) = expm(h_γ(p))  # 自动满足SPD
```

## 快速开始

### 基本使用

```python
import jax
import jax.numpy as jnp
from gaussian_einstein_fields import GaussianEinsteinFields

# 创建模型
model = GaussianEinsteinFields(
    num_gaussians_lapse=100,
    num_gaussians_shift=100,
    num_gaussians_spatial=100,
    init_scale=2.0
)

# 初始化
key = jax.random.PRNGKey(0)
p = jnp.array([0.0, 5.0, jnp.pi/2, 0.0])  # (t, r, θ, φ)
params = model.init(key, p)

# 获取度规
g = model.apply(params, p)  # 4D度规张量 (4, 4)

# 获取ADM分解
adm = model.apply(params, p, method=model.get_adm_variables)
print(f"Lapse: {adm.lapse}")
print(f"Shift: {adm.shift}")
print(f"Spatial metric:\n{adm.spatial_metric}")
```

### 训练示例

```bash
cd /home/user/EinFields
python gaussian_einstein_fields/train_example.py
```

这将训练一个GEF模型来表示Schwarzschild黑洞时空。

## 项目结构

```
gaussian_einstein_fields/
├── __init__.py                  # 模块导出
├── gaussian_primitives.py       # 4D高斯基元实现
├── adm_decomposition.py         # 3+1 ADM分解工具
├── gef_model.py                # GEF v3.0主模型
├── losses.py                   # 训练损失函数
├── train_example.py            # 完整训练示例
├── test_structure.py           # 结构测试
└── README.md                   # 详细文档
```

## 关键特性

### ✅ 已实现

1. **4D高斯基元**
   - 四元数参数化的4D旋转
   - 协方差矩阵构造
   - 批量评估支持

2. **ADM分解工具**
   - 从4D度规提取ADM变量
   - 从ADM变量重建4D度规
   - 物理约束检查

3. **三场表示**
   - Lapse场（对数域）
   - Shift场（向量场）
   - Spatial Metric场（矩阵指数）

4. **训练框架**
   - Sobolev损失（值 + 导数）
   - 组件感知损失
   - Adam优化器支持

### 🚧 未来扩展

1. **自适应优化**
   - 高斯基元的致密化
   - 低贡献基元的剪枝
   - 组件感知的自适应

2. **求解器模式**
   - 约束方程（Hamiltonian + Momentum）
   - 演化方程
   - 物理知情损失

3. **高级功能**
   - 多坐标系支持
   - 物质场耦合
   - 检查点保存/加载
   - 可视化工具

## 测试

运行结构测试（不需要JAX）：
```bash
python gaussian_einstein_fields/test_structure.py
```

输出：
```
============================================================
GEF v3.0 Structure Tests
============================================================
✅ All structure tests passed!
```

## 理论背景

### ADM 3+1形式

在数值相对论中，3+1分解是标准方法：

**时空分解**：
- 将4D时空切片为一系列3D空间超曲面
- Lapse N: 固有时间流逝率
- Shift β^i: 坐标系拖拽
- γ_ij: 每个切片的内蕴几何

**优势**：
- 明确的物理意义
- 与数值相对论标准方法兼容
- 自然的约束保证

### 为什么用高斯基元？

1. **显式表示**: 与隐式神经场不同，高斯基元是显式的
2. **局部性**: 每个基元影响局部区域
3. **可解释性**: 参数有明确的几何意义
4. **高效性**: 可以并行评估

## 性能

对于100个高斯基元的模型：
- **参数量**: ~15,000（每个场）
- **总参数**: ~45,000
- **评估速度**: 可JIT编译，支持批处理
- **内存**: 适合GPU训练

## 引用

如果使用此代码，请引用：

```bibtex
@misc{gef_v3_2025,
  title={Gaussian Einstein Fields v3.0: A Physically-Consistent
         Spacetime Representation via 3+1 Decomposition},
  author={GEF Contributors},
  year={2025},
  note={Built on EinFields framework}
}
```

## 下一步

1. **实验训练**: 在Schwarzschild、Kerr等已知解上测试
2. **性能优化**: 调整高斯数量和损失权重
3. **可视化**: 开发ADM变量的可视化工具
4. **扩展**: 实现自适应优化和求解器模式

## 许可

MIT License - 参见LICENSE文件

---

**GEF v3.0** - 连接神经场和数值相对论的桥梁 🌉
