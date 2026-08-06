# 热点建模：6 维特征向量（详细公式）

> 本文档从 README 的 `#### Section 2.1.1` 拆出来，专门讲 6 维特征向量的公式推导与代码映射。
> 代码实现见 `agents/hotspot_detector.py:_build_candidate()`（第 99-196 行）。

---

## 总公式

每维算 0-1 之间的得分，最后加权求和得综合评分：

$$\mathbf{H} = (I, P, V, N, R, D)$$

$$\text{Score}(H) = w^T \mathbf{H}, \quad w = (0.22, 0.11, 0.11, 0.17, 0.17, 0.22)$$

权重和 = 1.0（2026-08-06 移除 Lead Time 维度后重新归一化）。

---

## 6 维公式详解

### **$I$ Intensity（强度）**

$$I = \min\left(\max_{t \in \text{types}}\big(\text{score}_{type}(t)\big) + \min(N, 10) \times 0.02,\ 1.0\right)$$

| 异常类型 | 基础分 |
|---------|--------|
| `change_with_limitup` | 1.0 |
| `board_change_with_fundflow` | 0.9 |
| `board_resonance` | 0.7 |
| `risk_concentration` | 0.5 |
| 其他 | 0.4 |

- 取本组最强异常类型的得分
- 信号量（涨停股数 / 异动次数）每多 1 个加 0.02，最多加 0.20
- 上限 1.0

**代码**：`agents/hotspot_detector.py` 第 103-117 行

---

### **$P$ Persistence（持续性）**

$$P = \min(|G| \times 0.3,\ 1.0)$$

- $|G|$ = 本组 anomaly 数量
- 1 个 = 0.3，2 个 = 0.6，≥4 个 = 1.0
- 生产 TODO：应看时间序列稳定性，目前用数量近似

**代码**：`agents/hotspot_detector.py` 第 119-120 行

---

### **$V$ Virality（传播性）**

$$V = \min(\text{unique\_types} \times 0.4,\ 1.0)$$

- 跨异常类型数越多传播越广
- 1 种 = 0.4，2 种 = 0.8，≥3 种 = 1.0

**代码**：`agents/hotspot_detector.py` 第 122-123 行

---

### **$N$ Novelty（新颖度）**

$$N = \max(0,\ 1 - |\text{similar}| \times 0.2)$$

- 去 EpisodicMemory 找相似历史案例
- 0 相似 = 1.0（全新），1 相似 = 0.8，2 相似 = 0.6，≥5 相似 = 0.0
- TODO：升级向量检索（FAISS / Milvus）替代关键词匹配

**代码**：`agents/hotspot_detector.py` 第 125-127 行

---

### **$R$ Relevance（主体关联性）**

$$R = \begin{cases} 0.9 & \text{if 行业在 SM 知识图谱} \\ 0.5 & \text{else} \end{cases} + 0.1 \cdot \mathbb{1}[\text{政策敏感}]$$

- 在图谱里 = 0.9
- 不在图谱 = 0.5（陌生领域）
- 政策敏感行业（金融/医药/能源）= +0.1（上限 1.0）

**代码**：`agents/hotspot_detector.py` 第 129-136 行

---

### **$D$ Value Density（价值密度）**

$$D = \text{map}_{density}(\text{primary\_type})$$

| 异常类型 | 价值密度 |
|---------|---------|
| `change_with_limitup` | 0.9 |
| `board_change_with_fundflow` | 0.85 |
| `board_resonance` | 0.7 |
| `risk_concentration` | 0.6 |
| 其他 | 0.5 |

- 取 `anomaly_types[0]`（第一个异常类型）的固定密度值
- 跟 Intensity 的 max 不同：这里只看主类型

**代码**：`agents/hotspot_detector.py` 第 138-144 行

---

## 实际算例

以「消费电子」板块为例：假设 `board_resonance` + 7 个涨停 + 0 个相似历史 + 在 SM 知识图谱（政策敏感）：

$$
\begin{aligned}
I &= 0.7 + \min(7, 10) \times 0.02 = 0.84 \\
P &= \min(1, 3) \times 0.3 = 0.30 \\
V &= \min(1, 3) \times 0.4 = 0.40 \\
N &= \max(0, 1 - 0 \times 0.2) = 1.00 \\
R &= 0.9 + 0.1 = 1.00 \\
D &= 0.7 \\
\text{Score} &= 0.22 \times 0.84 + 0.11 \times 0.30 + 0.11 \times 0.40 + 0.17 \times 1.00 \\
&\quad + 0.17 \times 1.00 + 0.22 \times 0.70 \\
&\approx 0.770 \rightarrow \text{confidence: high}
\end{aligned}
$$

---

## 确信度划分

$$\text{confidence} = \begin{cases} \text{high} & \text{if Score} \geq 0.75 \\ \text{mid} & \text{if Score} \geq 0.55 \\ \text{low} & \text{otherwise} \end{cases}$$

| score | confidence | Orchestrator 处理 |
|-------|-----------|------------------|
| ≥ 0.75 | high | 直接 ReAct 跑 |
| 0.55-0.75 | mid | 默认 ReAct 跑 |
| 0.35-0.55 | low | 只观察，不写文章 |
| < 0.35 | — | 过滤掉 |

---

## 改进方向

| 维度 | 当前实现 | 改进 |
|------|---------|------|
| **$I$** | max + 信号量 | 按异常密度曲线（时间分布）|
| **$P$** | 数量近似 | 真时间序列 |
| **$V$** | 跨类型数 | 加"跨板块关联" |
| **$N$** | EM 关键词相似 | 升级向量检索（FAISS/Milvus）|
| **$R$** | 行业知识图谱命中 | 加"政策窗口期"（两会/季报/年报）|
| **$D$** | 异常类型映射 | 改成"writer 实际能写多少字 / 字数预估" |

> **2026-08-06 更新**: 原 7 维中的 $L$ Lead Time(提前量) 已移除。理由: Lead Time = "与竞品发现热点的时间差",本系统无外部竞品数据源,理论上是未知量; 更适合作为 evaluation 阶段的指标(对比竞品发布时间),而非 hotspot 建模维度。
