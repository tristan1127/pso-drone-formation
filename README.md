# 🚁 PSO无人机集群编队队形优化

**基于粒子群优化算法的无人机集群编队调度系统**

> 工程智能基础 · 第1阶段实践任务 · 同济大学

---

[English](#-uav-swarm-formation-optimization-via-pso) | [中文](#项目简介)

---

## 项目简介

本项目以**无人机灯光表演编队调度**为真实工程背景，使用粒子群优化算法（PSO）求解20架无人机从随机散落位置集结成目标队形的最优速度分配方案，同时满足四个工程目标：

| 目标 | 数学表达 | 含义 |
|------|----------|------|
| 最短总路程 | `Σ \|\|start_i - target_i\|\|` | 匈牙利算法预先最优化 |
| 最短完成时间 | `max_i(d_i / v_i)` | 最慢的无人机决定整体集结时间 |
| 最少总能耗 | `Σ d_i × v_i²` | 空气阻力做功正比于速度平方×距离 |
| 全程无碰撞 | 飞行中任意时刻两机间距 > R_safe | 直线轨迹参数化检测 |

### 系统架构（两层解耦）

```
第一层  匈牙利算法（一次性，O(n³)）
        解决"谁飞哪"：构建20×20代价矩阵，求总路程最短的最优指派

第二层  PSO速度优化（迭代120代）
        解决"飞多快"：60个粒子在20维速度空间中搜索
        粒子 = 20架无人机的速度方案 [v1, v2, ..., v20]，单位 m/s
```

### 实验设计

对比两种目标队形，分析PSO在不同复杂度队形下的收敛特性：

- **箭形队形 ➤**：左侧竖线10架 + 右侧斜边10架，非对称，方向感强
- **五角星队形 ★**：外顶点5 + 内顶点5 + 边中点10，高度对称，路径天然分散

---

## 快速开始

### 环境要求

```bash
Python 3.8+
pip install numpy matplotlib scipy
```

### 运行

```bash
python pso_drone.py
```

运行完成后自动在当前目录生成全部输出文件（约2~3分钟）。

### 输出文件

| 文件 | 说明 |
|------|------|
| `drone_anim_arrow.gif` | 箭形编队飞行动画（三阶段） |
| `drone_anim_star.gif` | 五角星编队飞行动画（三阶段） |
| `drone_snap_arrow.png` | 箭形PSO演化快照（6宫格） |
| `drone_snap_star.png` | 五角星PSO演化快照（6宫格） |
| `drone_convergence.png` | 四目标收敛曲线对比（2×2） |
| `drone_final.png` | 最终队形结果对比（三图并排） |
| `drone_params.png` | 参数敏感性分析 |
| `drone_radar.png` | 四目标性能雷达图 |
| `drone_flowchart.png` | 算法流程图 |

---

## 算法说明

### PSO核心迭代公式

```
速度更新：
v = w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)
    ↑惯性    ↑个体认知          ↑社会学习

位置更新：
x = clip(x + v,  v_min,  v_max)
```

| 参数 | 值 | 含义 |
|------|----|------|
| `w` | 0.7 | 惯性权重，平衡全局/局部搜索 |
| `c1` | 1.5 | 个体认知因子，向自身历史最优学习 |
| `c2` | 1.5 | 社会学习因子，向全群最优学习 |
| `n_particles` | 60 | 粒子数量（候选方案数） |
| `n_iter` | 120 | 最大迭代代数 |
| `v_min / v_max` | 1.0 / 10.0 m/s | 速度搜索范围 |

### 碰撞检测（全向量化）

在飞行时间轴上均匀采样15个时刻，每个时刻用矩阵运算一次性计算所有配对距离：

```python
diff = P[:, None, :] - P[None, :, :]   # shape (n, n, 2)  广播
dmat = np.linalg.norm(diff, axis=2)    # shape (n, n)      所有配对距离
bad  = dmat < R_safe                   # 触发惩罚的配对
penalty += Σ (R_safe - dmat[bad])²
```

相比 Python 双重循环快约100倍。

### 适应度函数

```
F = w1·(f1/f1_ref) + w2·(f2/f2_ref) + w3·f3 + w4·(f4/f4_ref)

f1 = max(d_i / v_i)          完成时间，w1=1.0
f2 = Σ(d_i × v_i²)           总能耗，  w2=0.5
f3 = collision_penalty(...)   碰撞惩罚，w3=5.0
f4 = Σ d_i                   总路程，  w4=0.2
```

---

## 实验结果

### 主要指标对比

| 指标 | 箭形队形 | 五角星队形 |
|------|----------|------------|
| 总路程 (m) | 132.33 | 103.73 |
| 完成时间 (s) | 1.84 | 6.14 |
| 总能耗 | 4681 | 2412 |
| 碰撞惩罚 | 41.6 | 2.40 |

### 关键结论

1. **五角星总路程更短**，因为其目标点天然更均匀分散，匈牙利指派后各机路程差异小
2. **箭形完成时间更短**，PSO将速度压缩在较小范围内使各机同步到达
3. **五角星碰撞风险极低**（2.40 vs 41.6），对称队形使飞行路径天然错开
4. **多目标权衡体现PSO价值**：单目标优化无法同时兼顾时间、能耗、安全性，PSO通过加权适应度找到平衡点

---

## 项目结构

```
pso-drone-formation/
├── pso_drone.py          # 完整源代码（单文件，含所有模块）
├── README.md             # 项目说明（中英双语）
└── .gitignore            # 排除生成的图片和动画
```

> 图片和动画文件为运行生成物，不纳入版本控制。
> 运行 `python pso_drone.py` 即可本地复现全部结果。

---

## 依赖

```
numpy       矩阵运算、随机数生成
matplotlib  静态图表 + FuncAnimation动画
scipy       linear_sum_assignment（匈牙利算法）
```

---

---

# 🚁 UAV Swarm Formation Optimization via PSO

**UAV Swarm Formation Scheduling System Based on Particle Swarm Optimization**

> Fundamentals of Engineering Intelligence · Phase 1 Practice · Tongji University

---

## Overview

This project takes **UAV light show formation scheduling** as its real-world engineering background. A Particle Swarm Optimization (PSO) algorithm is used to find the optimal speed allocation for 20 UAVs flying from random initial positions to a target formation, simultaneously satisfying four engineering objectives:

| Objective | Mathematical Expression | Meaning |
|-----------|------------------------|---------|
| Minimum total distance | `Σ \|\|start_i - target_i\|\|` | Pre-optimized by Hungarian algorithm |
| Minimum completion time | `max_i(d_i / v_i)` | The slowest UAV determines when the formation is complete |
| Minimum total energy | `Σ d_i × v_i²` | Work against air drag is proportional to v²×distance |
| Collision-free flight | Any two UAVs maintain distance > R_safe at all times | Checked via parameterized linear trajectory sampling |

### System Architecture (Two-Layer Decoupling)

```
Layer 1  Hungarian Algorithm (one-time, O(n³))
         Solves "who flies where": builds a 20×20 cost matrix,
         finds the assignment that minimizes total path length.

Layer 2  PSO Speed Optimization (120 iterations)
         Solves "how fast each UAV flies": 60 particles search
         a 20-dimensional speed space.
         Particle = speed plan for all 20 UAVs [v1, v2, ..., v20] in m/s
```

### Experiment Design

Two target formations are compared to analyze PSO convergence under different geometric complexity:

- **Arrow Formation ➤**: 10 UAVs on the vertical bar + 10 on diagonal wings; asymmetric, directional
- **Star Formation ★**: 5 outer vertices + 5 inner vertices + 10 edge midpoints; highly symmetric, paths naturally dispersed

---

## Quick Start

### Requirements

```bash
Python 3.8+
pip install numpy matplotlib scipy
```

### Run

```bash
python pso_drone.py
```

All output files are generated automatically in the current directory (approximately 2–3 minutes).

### Output Files

| File | Description |
|------|-------------|
| `drone_anim_arrow.gif` | Arrow formation flight animation (3 stages) |
| `drone_anim_star.gif` | Star formation flight animation (3 stages) |
| `drone_snap_arrow.png` | Arrow formation PSO evolution snapshots (6-panel) |
| `drone_snap_star.png` | Star formation PSO evolution snapshots (6-panel) |
| `drone_convergence.png` | Four-objective convergence curves comparison (2×2) |
| `drone_final.png` | Final formation results comparison (3-panel) |
| `drone_params.png` | Parameter sensitivity analysis |
| `drone_radar.png` | Four-objective performance radar chart |
| `drone_flowchart.png` | Algorithm flowchart |

---

## Algorithm Details

### PSO Core Update Equations

```
Velocity update:
v = w·v + c1·r1·(pbest - x) + c2·r2·(gbest - x)
    ↑inertia  ↑cognitive          ↑social

Position update:
x = clip(x + v,  v_min,  v_max)
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `w` | 0.7 | Inertia weight; balances global vs. local search |
| `c1` | 1.5 | Cognitive factor; pulls particle toward its personal best |
| `c2` | 1.5 | Social factor; pulls particle toward the global best |
| `n_particles` | 60 | Number of particles (candidate solutions) |
| `n_iter` | 120 | Maximum number of iterations |
| `v_min / v_max` | 1.0 / 10.0 m/s | Speed search bounds |

### Collision Detection (Fully Vectorized)

15 time steps are sampled uniformly along the flight timeline. At each step, all pairwise distances are computed in a single matrix operation using NumPy broadcasting:

```python
diff = P[:, None, :] - P[None, :, :]   # shape (n, n, 2)  broadcasting
dmat = np.linalg.norm(diff, axis=2)    # shape (n, n)      all pairwise distances
bad  = dmat < R_safe                   # pairs that trigger penalty
penalty += Σ (R_safe - dmat[bad])²
```

This is approximately 100× faster than a naive Python double loop.

### Fitness Function

```
F = w1·(f1/f1_ref) + w2·(f2/f2_ref) + w3·f3 + w4·(f4/f4_ref)

f1 = max(d_i / v_i)          completion time,   w1 = 1.0
f2 = Σ(d_i × v_i²)           total energy,      w2 = 0.5
f3 = collision_penalty(...)   collision penalty, w3 = 5.0
f4 = Σ d_i                   total distance,    w4 = 0.2
```

---

## Results

### Key Metrics Comparison

| Metric | Arrow Formation | Star Formation |
|--------|----------------|----------------|
| Total distance (m) | 132.33 | 103.73 |
| Completion time (s) | 1.84 | 6.14 |
| Total energy | 4681 | 2412 |
| Collision penalty | 41.6 | 2.40 |

### Key Findings

1. **Star formation achieves shorter total distance** because its target points are naturally more evenly distributed, resulting in smaller variance in individual path lengths after Hungarian assignment.
2. **Arrow formation completes faster** because PSO compresses the speed range to a narrow band, enabling near-simultaneous arrival.
3. **Star formation has near-zero collision risk** (2.40 vs 41.6) because its symmetric geometry causes flight paths to naturally diverge.
4. **The multi-objective trade-off demonstrates PSO's value**: no single-objective optimizer can simultaneously balance time, energy, and safety. PSO finds the balanced optimum through a weighted fitness function.

---

## Repository Structure

```
pso-drone-formation/
├── pso_drone.py          # Complete source code (single file, all modules included)
├── README.md             # Project documentation (Chinese & English)
└── .gitignore            # Excludes generated images and animations
```

> Images and animation files are generated outputs and are not tracked by version control.
> Run `python pso_drone.py` to reproduce all results locally.

---

## Dependencies

```
numpy       Matrix operations, random number generation
matplotlib  Static charts + FuncAnimation
scipy       linear_sum_assignment (Hungarian algorithm)
```

---

## License

MIT License — free to use and modify with attribution.
