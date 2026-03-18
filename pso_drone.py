"""
================================================================================
基于粒子群优化算法（PSO）的无人机集群编队队形优化
UAV Swarm Formation Optimization via Particle Swarm Optimization

系统架构（两层解耦）
--------------------
第一层  匈牙利算法（一次性）
        解决"谁飞哪"：构建 N×N 代价矩阵，O(n³) 求最优指派，总路程最短

第二层  PSO速度优化（迭代120代）
        解决"飞多快"：每个粒子 = 20维速度向量 v_i ∈ [1,10] m/s
        四目标适应度：F = w1·完成时间 + w2·总能耗 + w3·碰撞惩罚 + w4·路程

碰撞检测（向量化）
        直线轨迹参数化，在 [0, T_max] 均匀采15个时刻
        全矩阵运算：diff(n,n,2) → dmat(n,n)，比双重Python循环快100倍

输出内容
--------
  静态图 ×7  + 飞行动画 ×2（箭形 / 五角星，保存为 gif）

运行环境
--------
Python 3.8+    pip install numpy matplotlib scipy
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment
import warnings
import time
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── 全局配色 ─────────────────────────────────────────────────
C_ORANGE = "#E87722"   # 箭形
C_BLUE   = "#4A90D9"   # 五角星
C_DARK   = "#1A1A2E"
C_GRAY   = "#F4F4F4"
C_GREEN  = "#27AE60"
C_RED    = "#E74C3C"


# ══════════════════════════════════════════════════════════════
# 一、队形坐标生成
# ══════════════════════════════════════════════════════════════

def make_arrow(n: int = 20, scale: float = 6.0) -> np.ndarray:
    """
    箭形队形 ➤
    左侧竖线 10 架（均匀分布）
    右侧上斜边 5 架 + 下斜边 5 架（构成箭头尖）
    """
    pts = []
    # 左侧竖线
    for i in range(10):
        pts.append([-scale * 0.5, scale * (i / 9.0 - 0.5)])
    # 右侧上斜边
    for i in range(5):
        t = (i + 1) / 5.0
        pts.append([-scale * 0.5 + scale * t * 0.5,  scale * 0.5 * t])
    # 右侧下斜边
    for i in range(5):
        t = (i + 1) / 5.0
        pts.append([-scale * 0.5 + scale * t * 0.5, -scale * 0.5 * t])
    return np.array(pts[:n], dtype=float)


def make_star(n: int = 20, scale: float = 5.5) -> np.ndarray:
    """
    五角星队形 ★
    外顶点 5 + 内顶点 5 + 每段边中点 2×5 = 10，共 20 点
    """
    pts = []
    Ro, Ri = scale, scale * 0.4
    for k in range(5):
        a = np.pi / 2 + k * 2 * np.pi / 5
        pts.append([Ro * np.cos(a), Ro * np.sin(a)])
    for k in range(5):
        a = np.pi / 2 + np.pi / 5 + k * 2 * np.pi / 5
        pts.append([Ri * np.cos(a), Ri * np.sin(a)])
    for k in range(5):
        ao  = np.pi / 2 + k       * 2 * np.pi / 5
        ai  = np.pi / 2 + np.pi/5 + k * 2 * np.pi / 5
        ao2 = np.pi / 2 + (k + 1) * 2 * np.pi / 5
        pts.append([(Ro*np.cos(ao)  + Ri*np.cos(ai))  / 2,
                    (Ro*np.sin(ao)  + Ri*np.sin(ai))  / 2])
        pts.append([(Ri*np.cos(ai)  + Ro*np.cos(ao2)) / 2,
                    (Ri*np.sin(ai)  + Ro*np.sin(ao2)) / 2])
    return np.array(pts[:n], dtype=float)


# ══════════════════════════════════════════════════════════════
# 二、匈牙利最优指派
# ══════════════════════════════════════════════════════════════

def hungarian_assign(starts: np.ndarray,
                     targets: np.ndarray) -> np.ndarray:
    """
    代价矩阵 cost[i,j] = dist(start_i, target_j)
    scipy.linear_sum_assignment → O(n³) 最优指派
    返回重排后的 targets，使 targets[i] 是无人机 i 的目标点
    """
    cost = np.linalg.norm(
        starts[:, None, :] - targets[None, :, :], axis=2)
    _, col = linear_sum_assignment(cost)
    return targets[col]


# ══════════════════════════════════════════════════════════════
# 三、碰撞检测（全向量化）
# ══════════════════════════════════════════════════════════════

def collision_penalty(starts: np.ndarray,
                      targets: np.ndarray,
                      speeds: np.ndarray,
                      R_safe: float = 1.5,
                      n_check: int = 15) -> float:
    """
    直线轨迹参数化碰撞检测。

    物理模型
        pos_i(t) = start_i + clip(t/T_i, 0, 1) × (target_i - start_i)
        T_i = d_i / v_i，到达后停在终点。

    向量化实现（避免 Python 双重循环）
        P  : (n, 2)    当前时刻所有无人机位置
        diff = P[:,None,:] - P[None,:,:]   → (n, n, 2)
        dmat = norm(diff, axis=2)           → (n, n)  所有配对距离
        bad  = dmat < R_safe               → 触发惩罚的配对
        penalty += Σ (R_safe - dmat[bad])² / 2

    返回
        惩罚值（越大说明碰撞越严重，0 表示全程无碰撞）
    """
    sp    = np.maximum(speeds, 0.1)
    d     = np.linalg.norm(targets - starts, axis=1)
    T     = d / sp
    T_max = float(np.max(T))
    if T_max < 1e-6:
        return 0.0

    penalty = 0.0
    for t in np.linspace(0, T_max, n_check):
        ratios = np.minimum(t / np.where(T > 1e-6, T, 1e9), 1.0)
        P      = starts + ratios[:, None] * (targets - starts)
        diff   = P[:, None, :] - P[None, :, :]
        dmat   = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dmat, R_safe + 1.0)
        bad    = dmat < R_safe
        if np.any(bad):
            penalty += np.sum((R_safe - dmat[bad]) ** 2) / 2
    return float(penalty)


# ══════════════════════════════════════════════════════════════
# 四、四目标适应度函数
# ══════════════════════════════════════════════════════════════

def fitness(speeds: np.ndarray,
            starts: np.ndarray,
            targets: np.ndarray,
            w1: float = 1.0,   # 完成时间权重
            w2: float = 0.5,   # 总能耗权重
            w3: float = 5.0,   # 碰撞惩罚权重
            w4: float = 0.2,   # 总路程权重（路程已由指派固定，用于分析）
            R_safe: float = 1.5) -> float:
    """
    四目标加权适应度（越小越好）

    f1  完成时间  = max_i(d_i / v_i)         最慢无人机决定整体
    f2  总能耗    = Σ d_i × v_i²             空气阻力功 ∝ v²×d
    f3  碰撞惩罚  = collision_penalty(...)   飞行全程15个时刻扫描
    f4  总路程    = Σ d_i                    匈牙利已最小化，此处记录

    归一化后加权（防止各目标量级差异过大导致某项压倒其他）
    """
    sp = np.maximum(speeds, 0.1)
    d  = np.linalg.norm(targets - starts, axis=1)

    f1 = float(np.max(d / sp))
    f2 = float(np.sum(d * sp ** 2))
    f3 = collision_penalty(starts, targets, sp, R_safe=R_safe)
    f4 = float(np.sum(d))

    # 参考值（以速度全取均值时的结果为基准）
    sp_ref = np.full_like(sp, (sp.min() + sp.max()) / 2)
    f1_ref = max(float(np.max(d / sp_ref)), 1e-6)
    f2_ref = max(float(np.sum(d * sp_ref ** 2)), 1e-6)
    f4_ref = max(f4, 1e-6)

    return (w1 * f1 / f1_ref
          + w2 * f2 / f2_ref
          + w3 * f3
          + w4 * f4 / f4_ref)


# ══════════════════════════════════════════════════════════════
# 五、PSO 核心类
# ══════════════════════════════════════════════════════════════

class PSO:
    """
    粒子群优化（专用于无人机编队速度分配）

    粒子编码
        x  ∈ R^N，x[i] = 无人机 i 的飞行速度 (m/s)
        搜索范围 [v_min, v_max] = [1.0, 10.0]

    核心迭代公式
        vel = w·vel + c1·r1·(pbest-x) + c2·r2·(gbest-x)
        x   = clip(x + vel, v_min, v_max)

    记录内容（用于绘图）
        hist_F   综合适应度  每代1个值
        hist_f1  完成时间
        hist_f2  总能耗
        hist_f3  碰撞惩罚
        _snap_speeds  每代 gbest 速度快照（用于动画回放）
    """

    def __init__(self,
                 starts:      np.ndarray,
                 targets:     np.ndarray,
                 n_particles: int   = 60,
                 n_iter:      int   = 120,
                 w:           float = 0.7,
                 c1:          float = 1.5,
                 c2:          float = 1.5,
                 v_min:       float = 1.0,
                 v_max:       float = 10.0,
                 w1:          float = 1.0,
                 w2:          float = 0.5,
                 w3:          float = 5.0,
                 w4:          float = 0.2,
                 R_safe:      float = 1.5,
                 seed:        int   = 42):

        self.starts  = starts.copy()
        self.targets = targets.copy()
        self.n       = len(starts)
        self.n_p     = n_particles
        self.n_iter  = n_iter
        self.w       = w
        self.c1      = c1
        self.c2      = c2
        self.vlo     = v_min
        self.vhi     = v_max
        self.fkw     = dict(w1=w1, w2=w2, w3=w3, w4=w4, R_safe=R_safe)

        np.random.seed(seed)

        # ── 初始化粒子 ────────────────────────────────────────
        self.x   = np.random.uniform(v_min, v_max, (n_particles, self.n))
        dv       = (v_max - v_min) * 0.15
        self.vel = np.random.uniform(-dv, dv, (n_particles, self.n))

        # ── 初始评估 ──────────────────────────────────────────
        self.scores    = np.array([self._F(p) for p in self.x])
        self.pbest_x   = self.x.copy()
        self.pbest_sc  = self.scores.copy()

        gb = np.argmin(self.pbest_sc)
        self.gbest_x   = self.pbest_x[gb].copy()
        self.gbest_sc  = float(self.pbest_sc[gb])

        # ── 历史记录 ──────────────────────────────────────────
        self.hist_F    = [self.gbest_sc]
        self.hist_f1   = []   # 完成时间
        self.hist_f2   = []   # 总能耗
        self.hist_f3   = []   # 碰撞惩罚
        self._snap_speeds = [self.gbest_x.copy()]  # 每代最优速度快照
        self._log(self.gbest_x)

    # ── 内部工具 ──────────────────────────────────────────────

    def _F(self, spd: np.ndarray) -> float:
        return fitness(spd, self.starts, self.targets, **self.fkw)

    def _log(self, spd: np.ndarray):
        sp = np.maximum(spd, 0.1)
        d  = np.linalg.norm(self.targets - self.starts, axis=1)
        self.hist_f1.append(float(np.max(d / sp)))
        self.hist_f2.append(float(np.sum(d * sp ** 2)))
        self.hist_f3.append(
            collision_penalty(self.starts, self.targets, sp,
                              R_safe=self.fkw["R_safe"]))

    # ── 单步迭代 ──────────────────────────────────────────────

    def step(self):
        r1 = np.random.rand(self.n_p, self.n)
        r2 = np.random.rand(self.n_p, self.n)

        # 速度更新（三项合力）
        self.vel = (self.w  * self.vel
                  + self.c1 * r1 * (self.pbest_x - self.x)
                  + self.c2 * r2 * (self.gbest_x - self.x))

        # 速度幅值截断
        dv_max   = (self.vhi - self.vlo) * 0.25
        self.vel = np.clip(self.vel, -dv_max, dv_max)

        # 位置更新 + 边界约束
        self.x   = np.clip(self.x + self.vel, self.vlo, self.vhi)

        # 评估新位置
        self.scores = np.array([self._F(p) for p in self.x])

        # 更新 pbest
        better               = self.scores < self.pbest_sc
        self.pbest_x[better] = self.x[better].copy()
        self.pbest_sc[better]= self.scores[better]

        # 更新 gbest
        gb = np.argmin(self.pbest_sc)
        if self.pbest_sc[gb] < self.gbest_sc:
            self.gbest_x  = self.pbest_x[gb].copy()
            self.gbest_sc = float(self.pbest_sc[gb])

        # 记录
        self.hist_F.append(self.gbest_sc)
        self._snap_speeds.append(self.gbest_x.copy())
        self._log(self.gbest_x)

    # ── 完整运行 ──────────────────────────────────────────────

    def run(self, verbose: bool = True) -> np.ndarray:
        for i in range(self.n_iter):
            self.step()
            if verbose and (i + 1) % 20 == 0:
                sp = np.maximum(self.gbest_x, 0.1)
                d  = np.linalg.norm(self.targets - self.starts, axis=1)
                print(f"  迭代 {i+1:3d}/{self.n_iter}"
                      f"  F={self.gbest_sc:.4f}"
                      f"  完成时间={np.max(d/sp):.2f}s"
                      f"  能耗={np.sum(d*sp**2):.1f}"
                      f"  碰撞惩罚={self.hist_f3[-1]:.3f}")
        return self.gbest_x.copy()

    # ── 辅助：计算某代某时刻的无人机位置 ────────────────────

    def positions_at(self, iter_idx: int, t_ratio: float) -> np.ndarray:
        """
        用第 iter_idx 代的最优速度方案，
        计算飞行进度 t_ratio（0~1）时所有无人机的位置。
        t_ratio=0 → 起点，t_ratio=1 → 终点（队形形成）
        """
        sp     = np.maximum(
            self._snap_speeds[min(iter_idx, len(self._snap_speeds)-1)], 0.1)
        d      = np.linalg.norm(self.targets - self.starts, axis=1)
        T      = d / sp
        T_max  = float(np.max(T))
        t      = t_ratio * T_max
        ratios = np.minimum(t / np.where(T > 1e-6, T, 1e9), 1.0)
        return self.starts + ratios[:, None] * (self.targets - self.starts)


# ══════════════════════════════════════════════════════════════
# 六、飞行动画（核心可视化）
# ══════════════════════════════════════════════════════════════

def make_animation(pso: PSO,
                   color: str,
                   name: str,
                   fps: int = 30,
                   save_path: str = None) -> animation.FuncAnimation:
    """
    三阶段飞行动画：

    阶段0  随机散落定格（1.5秒）
           20个点停在初始位置，让观众看清起点混乱状态
           右侧显示完整收敛曲线（告知PSO已跑完）

    阶段1  飞行过程（主体，约4秒）
           无人机从起点沿直线飞向目标，有尾迹拖尾
           标题实时显示飞行进度百分比

    阶段2  队形定格（2秒）
           所有无人机到位，标题变色高亮
           队形轮廓与无人机重合，视觉确认
    """
    # ── 帧数计算 ──────────────────────────────────────────────
    f_freeze_start  = int(fps * 1.5)   # 阶段0：45帧（1.5秒）
    f_fly           = int(fps * 4.0)   # 阶段1：120帧（4秒）
    f_freeze_end    = int(fps * 2.0)   # 阶段2：60帧（2秒）
    total_frames    = f_freeze_start + f_fly + f_freeze_end

    # 阶段边界帧号
    B1 = f_freeze_start               # 飞行开始帧
    B2 = f_freeze_start + f_fly       # 定格开始帧

    # ── 画布 ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6.5), facecolor="white")
    gs  = GridSpec(1, 2, width_ratios=[1.6, 1], figure=fig,
                   left=0.05, right=0.97, wspace=0.12)
    ax_fly  = fig.add_subplot(gs[0])
    ax_conv = fig.add_subplot(gs[1])

    # ── 飞行画面（左）────────────────────────────────────────
    ax_fly.set_facecolor("#0A0A18")
    ax_fly.set_xlim(-16, 16)
    ax_fly.set_ylim(-16, 16)
    ax_fly.set_aspect("equal")
    ax_fly.tick_params(colors="#444466", labelsize=7)
    ax_fly.grid(True, color="#14142A", lw=0.6)
    for sp in ax_fly.spines.values():
        sp.set_edgecolor("#1A1A33")

    # 目标队形轮廓（淡白圆圈，全程可见）
    ax_fly.scatter(pso.targets[:, 0], pso.targets[:, 1],
                   c="none", s=160, zorder=2, marker="o",
                   edgecolors="white", linewidths=0.9, alpha=0.20)

    # 飞行路径线（极淡，全程可见，暗示方向）
    for i in range(pso.n):
        ax_fly.plot([pso.starts[i, 0], pso.targets[i, 0]],
                    [pso.starts[i, 1], pso.targets[i, 1]],
                    color=color, alpha=0.06, lw=0.7, zorder=1)

    # 无人机散点（动态主体）
    scat = ax_fly.scatter([], [], c=color, s=90, zorder=6,
                          edgecolors="white", linewidths=0.9)

    # 尾迹线（每架一条，飞行阶段才有）
    TRAIL_LEN   = 30
    trail_buf   = [[] for _ in range(pso.n)]
    trail_lines = [ax_fly.plot([], [], color=color,
                               alpha=0.40, lw=1.0, zorder=4)[0]
                   for _ in range(pso.n)]

    # 到达标记（阶段2才显示的金色星星）
    arrived_scat = ax_fly.scatter([], [], c="#FFD700", s=160,
                                  zorder=7, marker="*",
                                  edgecolors="white", linewidths=0.5,
                                  alpha=0.0)   # 先透明

    ax_fly.set_xlabel("x (m)", fontsize=8, color="#666688")
    ax_fly.set_ylabel("y (m)", fontsize=8, color="#666688")
    title_txt = ax_fly.set_title("", fontsize=11,
                                  fontweight="bold", color="white", pad=10)

    # ── 收敛曲线（右）────────────────────────────────────────
    ax_conv.set_facecolor(C_GRAY)
    ax_conv.grid(True, color="#DDDDDD", lw=0.5)
    ax_conv.tick_params(colors="#888888", labelsize=8)
    for sp in ax_conv.spines.values():
        sp.set_edgecolor("#CCCCCC")
    ax_conv.set_title("PSO收敛曲线（120代）", fontsize=9,
                       fontweight="bold", color=C_DARK)
    ax_conv.set_xlabel("迭代代数", fontsize=8, color="#555555")
    ax_conv.set_ylabel("综合适应度 F", fontsize=8, color="#555555")

    # 完整收敛曲线（静态，一开始就画好）
    ax_conv.plot(pso.hist_F, color=color, lw=2.0, alpha=0.9, zorder=3)
    ax_conv.fill_between(range(len(pso.hist_F)),
                         pso.hist_F, alpha=0.12, color=color)
    f_min  = min(pso.hist_F)
    f_max  = max(pso.hist_F)
    margin = (f_max - f_min) * 0.12 + 0.01
    ax_conv.set_xlim(0, pso.n_iter)
    ax_conv.set_ylim(f_min - margin, f_max + margin)

    # 最优值标注线
    ax_conv.axhline(f_min, color=color, lw=1.0,
                    linestyle="--", alpha=0.5)
    ax_conv.text(pso.n_iter * 0.65, f_min + margin * 0.3,
                 f"最优 F={f_min:.2f}", fontsize=7.5,
                 color=color, alpha=0.85)

    # ── 阶段说明文字（右图底部）─────────────────────────────
    stage_txt = ax_conv.text(
        0.05, 0.06, "", transform=ax_conv.transAxes,
        fontsize=9, color=C_DARK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white", edgecolor=color, lw=1.2))

    # ── update 函数 ───────────────────────────────────────────
    def update(frame: int):

        # ── 阶段0：随机散落定格 ──────────────────────────────
        if frame < B1:
            pos = pso.starts.copy()

            # 清空尾迹
            for i in range(pso.n):
                trail_buf[i].clear()
                trail_lines[i].set_data([], [])

            arrived_scat.set_alpha(0.0)
            arrived_scat.set_offsets(np.zeros((0, 2)))

            scat.set_offsets(pos)
            scat.set_sizes([90] * pso.n)
            scat.set_facecolor(color)

            title_txt.set_text(f"{name}队形  ·  初始状态：20架无人机随机散落")
            title_txt.set_color("white")
            stage_txt.set_text("阶段0  初始散落")

        # ── 阶段1：飞行过程 ──────────────────────────────────
        elif frame < B2:
            ratio = (frame - B1) / max(f_fly - 1, 1)   # 0 → 1

            # 用 PSO 最优速度方案计算当前位置
            sp     = np.maximum(pso.gbest_x, 0.1)
            d      = np.linalg.norm(pso.targets - pso.starts, axis=1)
            T      = d / sp
            T_max  = float(np.max(T))
            t      = ratio * T_max
            ratios = np.minimum(
                t / np.where(T > 1e-6, T, 1e9), 1.0)
            pos = pso.starts + ratios[:, None] * (pso.targets - pso.starts)

            # 更新尾迹
            for i in range(pso.n):
                trail_buf[i].append(pos[i].copy())
                if len(trail_buf[i]) > TRAIL_LEN:
                    trail_buf[i].pop(0)
                if len(trail_buf[i]) > 1:
                    ta = np.array(trail_buf[i])
                    trail_lines[i].set_data(ta[:, 0], ta[:, 1])

            # 到达的无人机变金色星星
            arrived_mask = ratios >= 0.999
            if np.any(arrived_mask):
                arrived_scat.set_offsets(pso.targets[arrived_mask])
                arrived_scat.set_alpha(0.9)
            else:
                arrived_scat.set_offsets(np.zeros((0, 2)))
                arrived_scat.set_alpha(0.0)

            scat.set_offsets(pos)
            scat.set_facecolor(color)
            scat.set_sizes([90] * pso.n)

            pct = int(ratio * 100)
            n_arrived = int(np.sum(arrived_mask))
            title_txt.set_text(
                f"{name}队形  ·  飞行进度 {pct}%  "
                f"·  已到位 {n_arrived}/20 架")
            title_txt.set_color("white")
            stage_txt.set_text("阶段1  飞行中")

        # ── 阶段2：队形定格 ──────────────────────────────────
        else:
            # 所有无人机锁定在目标点
            pos = pso.targets.copy()

            for i in range(pso.n):
                trail_lines[i].set_data([], [])   # 清空尾迹，画面干净
            trail_buf_clear = [buf.clear() for buf in trail_buf]

            # 所有点变金色星星
            arrived_scat.set_offsets(pso.targets)
            arrived_scat.set_alpha(0.95)

            # 无人机点缩小变暗（队形标记为主）
            scat.set_offsets(pos)
            scat.set_facecolor(color)
            scat.set_sizes([55] * pso.n)

            title_txt.set_text(
                f"★  {name}队形集结完成！  "
                f"完成时间 {np.max(np.linalg.norm(pso.targets-pso.starts,axis=1)/np.maximum(pso.gbest_x,0.1)):.1f}s")
            title_txt.set_color("#FFD700")   # 金色
            stage_txt.set_text("阶段2  队形完成")

        return ([scat, arrived_scat, title_txt, stage_txt]
                + trail_lines)

    ani = animation.FuncAnimation(
        fig, update,
        frames=total_frames,
        interval=1000 // fps,
        blit=True
    )

    fig.suptitle(
        f"无人机集群编队优化  ——  {name}队形  "
        f"（PSO四目标优化：最短路程 + 最短时间 + 最少能耗 + 无碰撞）",
        fontsize=9, fontweight="bold", color=C_DARK, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        print(f"  保存动画：{save_path} ...")
        ani.save(save_path, writer="pillow", fps=fps,
                 savefig_kwargs={"facecolor": "white"})
        print(f"  ✓  {save_path}")
        plt.close()
    else:
        plt.show()

    return ani


# ══════════════════════════════════════════════════════════════
# 七、静态可视化
# ══════════════════════════════════════════════════════════════

# ── 7-A  演化快照（6宫格）────────────────────────────────────

def plot_snapshots(pso: PSO, name: str, color: str, filename: str):
    """
    6宫格快照：展示PSO第 0/10%/25%/50%/75%/100% 代
    每格显示：目标点轮廓 + 无人机在飞行50%进度处的位置
    """
    T      = pso.n_iter
    snaps  = [0, int(T*0.10), int(T*0.25),
              int(T*0.50), int(T*0.75), T]
    labels = ["初始方案（第0代）",
              f"第{snaps[1]}代", f"第{snaps[2]}代",
              f"第{snaps[3]}代", f"第{snaps[4]}代",
              f"第{T}代（最优）"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"PSO优化过程演化快照 — {name}队形\n"
                 f"（每格图示：飞行进度50%时无人机位置）",
                 fontsize=12, fontweight="bold", color=C_DARK, y=1.02)

    for ax, sidx, lbl in zip(axes.flat, snaps, labels):
        ax.set_facecolor(C_GRAY)
        ax.set_aspect("equal")
        ax.set_xlim(-16, 16); ax.set_ylim(-16, 16)

        # 目标点
        ax.scatter(pso.targets[:, 0], pso.targets[:, 1],
                   c="#CCCCCC", s=70, zorder=1, marker="o",
                   edgecolors="#AAAAAA", lw=0.8)

        # 飞行路径（淡箭头）
        for i in range(pso.n):
            ax.annotate("", xy=pso.targets[i], xytext=pso.starts[i],
                        arrowprops=dict(arrowstyle="->", color=color,
                                        alpha=0.10, lw=0.7))

        # 无人机中途位置
        mid = pso.positions_at(sidx, t_ratio=0.5)
        ax.scatter(mid[:, 0], mid[:, 1], c=color, s=60, zorder=5,
                   edgecolors="white", lw=0.8)

        sc  = pso.hist_F[min(sidx, len(pso.hist_F)-1)]
        t1  = pso.hist_f1[min(sidx, len(pso.hist_f1)-1)]
        ax.set_title(f"{lbl}\nF={sc:.3f}   完成时间={t1:.2f}s",
                     fontsize=9, fontweight="bold", color=C_DARK, pad=5)
        ax.tick_params(labelsize=7, colors="#999999")
        ax.grid(True, color="#DDDDDD", lw=0.4)
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  {filename}")


# ── 7-B  四目标收敛曲线（2×2）───────────────────────────────

def plot_convergence(pso_a: PSO, pso_s: PSO, filename: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle("PSO四目标收敛分析 — 箭形 vs 五角星",
                 fontsize=13, fontweight="bold", color=C_DARK)

    panels = [
        (axes[0,0], pso_a.hist_F,  pso_s.hist_F,
         "综合适应度 F（越小越好）",   "F 值"),
        (axes[0,1], pso_a.hist_f1, pso_s.hist_f1,
         "目标1：编队完成时间 (s)",    "时间 (s)"),
        (axes[1,0], pso_a.hist_f2, pso_s.hist_f2,
         "目标2：总能耗 (dist × v²)",  "能耗"),
        (axes[1,1], pso_a.hist_f3, pso_s.hist_f3,
         "目标3：碰撞惩罚值",          "惩罚值"),
    ]

    for ax, ha, hs, title, ylabel in panels:
        ax.set_facecolor(C_GRAY)
        ax.plot(ha, color=C_ORANGE, lw=2.2, label="箭形 ➤", zorder=3)
        ax.plot(hs, color=C_BLUE,   lw=2.2, label="五角星 ★", zorder=3)
        ax.fill_between(range(len(ha)), ha, alpha=0.10, color=C_ORANGE)
        ax.fill_between(range(len(hs)), hs, alpha=0.10, color=C_BLUE)
        ax.set_title(title, fontsize=10, fontweight="bold", color=C_DARK)
        ax.set_xlabel("迭代次数", fontsize=9, color="#555555")
        ax.set_ylabel(ylabel,     fontsize=9, color="#555555")
        ax.legend(fontsize=9)
        ax.grid(True, color="#DDDDDD", lw=0.5)
        ax.tick_params(colors="#888888")
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  {filename}")


# ── 7-C  最终结果对比（三图并排）────────────────────────────

def plot_final(pso_a: PSO, pso_s: PSO, filename: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("PSO无人机编队优化结果",
                 fontsize=13, fontweight="bold", color=C_DARK)

    def panel(ax, pos, tgt, starts, color, title):
        ax.set_facecolor(C_GRAY)
        ax.set_aspect("equal")
        ax.set_xlim(-15, 15); ax.set_ylim(-15, 15)
        if tgt is not None:
            ax.scatter(tgt[:,0], tgt[:,1], c="#CCCCCC",
                       s=90, marker="*", edgecolors="#AAAAAA", zorder=1)
        if starts is not None and tgt is not None:
            for i in range(len(pos)):
                ax.annotate("", xy=tgt[i], xytext=starts[i],
                            arrowprops=dict(arrowstyle="->", color=color,
                                            alpha=0.28, lw=0.9))
        ax.scatter(pos[:,0], pos[:,1], c=color, s=75, zorder=5,
                   edgecolors="white", lw=1.2)
        for i, (x, y) in enumerate(pos):
            ax.text(x+0.5, y+0.5, str(i+1),
                    fontsize=5.5, color="#444444", zorder=6)
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color=C_DARK, pad=8)
        ax.tick_params(labelsize=7, colors="#888888")
        ax.grid(True, color="#DDDDDD", lw=0.4)
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC")

    panel(axes[0], pso_a.starts, None, None,
          "#888888", "初始状态（随机散落）")
    panel(axes[1], pso_a.targets, pso_a.targets, pso_a.starts,
          C_ORANGE, "箭形队形 ➤（PSO最优调度）")
    panel(axes[2], pso_s.targets, pso_s.targets, pso_s.starts,
          C_BLUE, "五角星队形 ★（PSO最优调度）")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  {filename}")


# ── 7-D  参数敏感性分析 ──────────────────────────────────────

def plot_params(starts: np.ndarray, targets: np.ndarray, filename: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("white")
    fig.suptitle("PSO参数敏感性分析（箭形队形，控制变量法）",
                 fontsize=12, fontweight="bold", color=C_DARK)
    pal = [C_ORANGE, "#F5A623", C_GREEN, C_BLUE]

    for ax, vals, key, label in [
        (ax1, [0.4, 0.6, 0.7, 0.9], "w",           "惯性权重 w"),
        (ax2, [15,  30,  60,  100],  "n_particles", "粒子数量"),
    ]:
        ax.set_facecolor(C_GRAY)
        for v, col in zip(vals, pal):
            kw = dict(starts=starts, targets=targets,
                      n_iter=80, w=0.7, n_particles=60, seed=0)
            kw[key] = v
            p = PSO(**kw)
            p.run(verbose=False)
            ax.plot(p.hist_F, color=col, lw=1.8,
                    label=f"{label} = {v}", alpha=0.9)
        ax.set_title(f"{label}的影响", fontsize=10,
                     fontweight="bold", color=C_DARK)
        ax.set_xlabel("迭代次数", fontsize=9, color="#555555")
        ax.set_ylabel("综合适应度 F", fontsize=9, color="#555555")
        ax.legend(fontsize=9)
        ax.grid(True, color="#DDDDDD", lw=0.5)
        ax.tick_params(colors="#888888")
        for sp in ax.spines.values():
            sp.set_edgecolor("#CCCCCC")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  ✓  {filename}")
# ══════════════════════════════════════════════════════════════
# 八、主程序
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("  PSO 无人机集群编队队形优化")
    print("  20架无人机 | 四目标 | 箭形 vs 五角星 | 迭代120代")
    print("=" * 64)

    # ── 生成队形 & 起点 ──────────────────────────────────────
    arr_tgt  = make_arrow(n=20, scale=6.0)
    star_tgt = make_star(n=20, scale=5.5)

    # 时间戳种子：每次运行初始位置不同，同时打印种子方便复现
    seed = int(time.time())
    np.random.seed(seed)
    print(f"本次随机种子：{seed}（如需复现，将此数替换 seed 变量）")
    starts   = np.random.uniform(-11, 11, (20, 2))

    # ── 匈牙利指派（一次性）─────────────────────────────────
    print("\n【匈牙利算法最优指派】")
    arr_assigned  = hungarian_assign(starts, arr_tgt)
    star_assigned = hungarian_assign(starts, star_tgt)
    d_arr  = np.linalg.norm(arr_assigned  - starts, axis=1)
    d_star = np.linalg.norm(star_assigned - starts, axis=1)
    print(f"  箭形   总路程（指派后）= {np.sum(d_arr):.2f} m")
    print(f"  五角星 总路程（指派后）= {np.sum(d_star):.2f} m")

    # ── 箭形 PSO ─────────────────────────────────────────────
    print("\n【1/2  箭形队形 PSO 优化（120代）】")
    pso_a = PSO(
        starts=starts, targets=arr_assigned,
        n_particles=60, n_iter=120,
        w=0.7, c1=1.5, c2=1.5,
        v_min=1.0, v_max=10.0,
        w1=1.0, w2=0.5, w3=5.0, w4=0.2,
        R_safe=1.5, seed=42
    )
    pso_a.run(verbose=True)
    sp_a = np.maximum(pso_a.gbest_x, 0.1)
    print(f"\n  箭形最终指标：")
    print(f"    总路程    = {np.sum(d_arr):.2f} m")
    print(f"    完成时间  = {np.max(d_arr/sp_a):.2f} s")
    print(f"    总能耗    = {np.sum(d_arr*sp_a**2):.2f}")
    print(f"    碰撞惩罚  = {collision_penalty(starts,arr_assigned,sp_a):.4f}")
    print(f"    各机速度  = {np.round(sp_a,2)}")

    # ── 五角星 PSO ───────────────────────────────────────────
    print("\n【2/2  五角星队形 PSO 优化（120代）】")
    pso_s = PSO(
        starts=starts, targets=star_assigned,
        n_particles=60, n_iter=120,
        w=0.7, c1=1.5, c2=1.5,
        v_min=1.0, v_max=10.0,
        w1=1.0, w2=0.5, w3=5.0, w4=0.2,
        R_safe=1.5, seed=42
    )
    pso_s.run(verbose=True)
    sp_s = np.maximum(pso_s.gbest_x, 0.1)
    print(f"\n  五角星最终指标：")
    print(f"    总路程    = {np.sum(d_star):.2f} m")
    print(f"    完成时间  = {np.max(d_star/sp_s):.2f} s")
    print(f"    总能耗    = {np.sum(d_star*sp_s**2):.2f}")
    print(f"    碰撞惩罚  = {collision_penalty(starts,star_assigned,sp_s):.4f}")
    print(f"    各机速度  = {np.round(sp_s,2)}")

    # ── 生成静态图 ───────────────────────────────────────────
    print("\n【生成静态图表】")
    plot_snapshots(pso_a, "箭形",   C_ORANGE, "drone_snap_arrow.png")
    plot_snapshots(pso_s, "五角星", C_BLUE,   "drone_snap_star.png")
    plot_convergence(pso_a, pso_s,            "drone_convergence.png")
    plot_final(pso_a, pso_s,                  "drone_final.png")
    plot_params(starts, arr_assigned,         "drone_params.png")

    # ── 生成动画 ─────────────────────────────────────────────
    print("\n【生成飞行动画】")
    make_animation(pso_a, C_ORANGE, "箭形",
                   fps=30, save_path="drone_anim_arrow.gif")
    make_animation(pso_s, C_BLUE, "五角星",
                   fps=30, save_path="drone_anim_star.gif")

    # ── 输出汇总 ─────────────────────────────────────────────
    print("\n" + "="*64)
    print("完成 ✓  共 7 个输出文件：")
    print()
    print("  静态图（放PPT）：")
    print("    drone_snap_arrow.png    箭形演化快照（6宫格）")
    print("    drone_snap_star.png     五角星演化快照（6宫格）")
    print("    drone_convergence.png   四目标收敛曲线（2×2）")
    print("    drone_final.png         最终结果对比（三图并排）")
    print("    drone_params.png        参数敏感性分析")
    print()
    print("  动画（汇报时播放）：")
    print("    drone_anim_arrow.gif    箭形飞行动画")
    print("    drone_anim_star.gif     五角星飞行动画")
    print("="*64)
