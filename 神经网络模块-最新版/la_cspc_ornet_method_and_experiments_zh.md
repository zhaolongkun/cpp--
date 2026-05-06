# LA-CSPC-ORNet：方法章节与实验设计草稿

## 1. 方法定位与任务定义

本文提出的方法不是目标检测算法，不是多目标跟踪算法，也不是控制命令生成算法，更不是 PID 控制算法。本文的主算法输入不是原始图像，也不是检测器本身，而是经过 detection 和 multi-object tracking 之后得到的时序偏差数据及其辅助特征。本文的主算法输出也不是电机控制命令，而是两类偏差状态：

1. 当前 clean offset state
   \[
   \mathbf{x}^{clean}_t = [\hat d_{x,t}, \hat d_{y,t}]
   \]
2. control-effective future predictive offset state
   \[
   \mathbf{x}^{pred}_{t+\Delta_t} = [\hat d^{pred}_{x,t+\Delta_t}, \hat d^{pred}_{y,t+\Delta_t}]
   \]

其中，\(\mathbf{x}^{clean}_t\) 表示当前时刻经过去噪和动态重建后的偏差状态，\(\mathbf{x}^{pred}_{t+\Delta_t}\) 表示考虑视觉处理、算法推理、控制线程以及执行器响应总时延之后，对控制真正生效时刻偏差状态的预测。本文的方法目标是在强噪声、强非线性、强时变和 zoom 扰动条件下，从 noisy offset sequence 中同时学习当前 clean offset 与 future predictive offset，从而为下游固定控制器提供更适合控制的偏差状态表示。

需要特别强调的是：PID 控制器仅在实验中作为 downstream validation 使用，用于验证本文重建与预测偏差状态的应用价值；PID 不属于本文主方法本体。

基于这一任务定义，本文将检测-跟踪后的偏差时序问题正式建模为一个时延感知的神经网络偏差重建与未来偏差预测问题，并提出：

- 中文名称：时延感知因果切换预测—校正偏差重建网络
- 英文名称：Latency-Aware Causal Switching Predict-Correct Offset Reconstruction Network
- 缩写：LA-CSPC-ORNet

## 2. 问题动机

传统偏差平滑或滤波方法通常默认只需恢复“当前时刻的较干净偏差”，但对于本文场景，这一定义并不充分。原因在于，从 detection/tracking 输出到控制真正生效之间存在明显的链路总时延，该时延至少包括：

1. 视觉处理延迟
2. 算法推理延迟
3. 控制线程调度延迟
4. 执行器响应延迟

如果模型只重建当前时刻的 clean offset，即使其噪声显著降低，后续控制器在实际生效时仍然会因为相位滞后而落后于真实目标动态。因此，本文的方法不再停留在“当前偏差去噪”层面，而是同时预测控制真正生效时刻的 future predictive offset，以实现时延补偿与超前控制。

## 3. 输入表示：post-detection/tracking offset sequence

记时刻 \(t\) 的输入向量为 \(\mathbf{o}_t\)。本文的输入不是原始图像，而是检测与多目标跟踪之后得到的偏差数据和辅助状态，定义为：

\[
\mathbf{o}_t =
[
d^{raw}_{x,t}, d^{raw}_{y,t},
\Delta d^{raw}_{x,t}, \Delta d^{raw}_{y,t},
\Delta^2 d^{raw}_{x,t}, \Delta^2 d^{raw}_{y,t},
\tau_{x,t}, \tau_{y,t},
\dot\tau_{x,t}, \dot\tau_{y,t},
c_t, a_t, m_t, u^{meas}_t, age_t,
z_t, \Delta z_t,
s^{sw}_t, s^{turn}_t,
\delta t_t,
\ell^{vis}_t, \ell^{alg}_t, \ell^{ctrl}_t, \ell^{act}_t,
h_t
]
\]

其中：

- \(d^{raw}_{x,t}, d^{raw}_{y,t}\)：检测-跟踪后的原始偏差
- \(\Delta d^{raw}_{x,t}, \Delta d^{raw}_{y,t}\)：一阶差分
- \(\Delta^2 d^{raw}_{x,t}, \Delta^2 d^{raw}_{y,t}\)：二阶差分
- \(\tau_{x,t}, \tau_{y,t}\)：因果趋势项，例如 causal EMA 或 causal median-EMA trend
- \(\dot\tau_{x,t}, \dot\tau_{y,t}\)：趋势速度
- \(c_t\)：检测置信度
- \(a_t\)：bbox area 或目标面积
- \(m_t\)：miss/lost 标志
- \(u^{meas}_t\)：当前时刻是否发生真实测量更新
- \(age_t\)：测量新鲜度，例如 meas age
- \(z_t\)：zoom 状态
- \(\Delta z_t\)：zoom 变化量
- \(s^{sw}_t\)：switch score，用于刻画局部方向切换强度
- \(s^{turn}_t\)：turn score，用于刻画局部转折强度
- \(\delta t_t\)：采样时间间隔
- \(\ell^{vis}_t\)：视觉处理延迟
- \(\ell^{alg}_t\)：算法推理延迟
- \(\ell^{ctrl}_t\)：控制线程延迟
- \(\ell^{act}_t\)：执行器响应延迟
- \(h_t\)：由总时延换算得到的预测 horizon

给定过去长度为 \(L\) 的因果窗口：

\[
\mathcal{O}_t = \{\mathbf{o}_{t-L+1}, \mathbf{o}_{t-L+2}, \ldots, \mathbf{o}_t\}
\]

本文的方法学习映射：

\[
f_\theta : \mathcal{O}_t \mapsto (\mathbf{x}^{clean}_t, \mathbf{x}^{pred}_{t+\Delta_t})
\]

即从 post-detection/tracking offset sequence 中同时重建当前 clean offset state 和未来 predictive offset state。

## 4. 模型总体结构

LA-CSPC-ORNet 包含以下五个部分：

1. 因果输入编码
2. 切换感知双状态时序建模
3. 方向受限的预测—校正前导机制
4. 当前 clean offset 输出头
5. 时延感知的 future predictive offset 输出头

整体可写为：

\[
\mathbf{e}_t = \text{Enc}(\mathcal{O}_t)
\]
\[
\mathbf{z}_t = \text{DualState}(\mathbf{e}_t, s^{sw}_t, s^{turn}_t, c_t, m_t, \Delta z_t)
\]
\[
\Delta \mathbf{d}^{lead}_t = \text{Lead}(\mathbf{z}_t, \mathbf{o}_t)
\]
\[
\mathbf{x}^{clean}_t = H_{clean}(\mathbf{z}_t, \Delta \mathbf{d}^{lead}_t)
\]
\[
\mathbf{x}^{pred}_{t+\Delta_t} = H_{pred}(\mathbf{z}_t, \mathbf{x}^{clean}_t, h_t)
\]

可选地，还可以输出辅助速度状态：

\[
\mathbf{v}_t = [\hat v_{x,t}, \hat v_{y,t}]
\]

但本文的核心输出仍然是两个偏差状态，而不是控制命令。

## 5. 因果输入编码

为满足在线部署要求，编码器严格满足因果性，即在时刻 \(t\) 只能使用 \(\mathcal{O}_t\) 中的当前及过去观测，不能访问未来时刻。本文推荐采用：

- Causal TCN + small GRU

作为主编码结构。其理由如下：

1. Causal TCN 对局部高频抖动、短时漏检、zoom 过渡扰动等局部模式具有较强表达能力；
2. 小型 GRU 适合对短窗口序列进行轻量递推建模；
3. 该结构比 LSTM 更轻，比轻量 Transformer 更适合低延迟在线部署。

因此，因果输入编码器可以写为：

\[
\mathbf{e}_t = \text{GRU}(\text{TCN}(\mathcal{O}_t))
\]

其中 TCN 负责局部时序模式提取，GRU 负责跨时间步的紧凑状态汇聚。

## 6. 切换感知双状态建模

这是本文的第一项核心创新。

### 6.1 设计动机

检测-跟踪后的偏差序列同时包含两类截然不同的动态：

1. 稳态段：表现为低频趋势变化，主要需要去噪和平滑
2. 事件段：表现为快速切换、方向翻转、转折峰值和短时恢复，主要需要快速响应和结构保持

如果只用单一隐状态统一建模，模型往往会在“平滑性”和“动态保持”之间退化成折中平滑器。因此，本文显式引入两个潜在状态：

- Slow state \(\mathbf{z}^{slow}_t\)：用于稳态去噪和连续性保持
- Fast state \(\mathbf{z}^{fast}_t\)：用于快速切换、转折保持和短时动态响应

### 6.2 状态递推

Slow state 和 fast state 分别进行因果更新：

\[
\tilde{\mathbf{z}}^{slow}_t = \text{GRU}_{slow}(\mathbf{e}_t, \mathbf{z}^{slow}_{t-1})
\]
\[
\tilde{\mathbf{z}}^{fast}_t = \text{GRU}_{fast}(\mathbf{e}_t, \mathbf{z}^{fast}_{t-1})
\]

定义 innovation：

\[
\mathbf{i}_t = [d^{raw}_{x,t}, d^{raw}_{y,t}] - [\tau_{x,t}, \tau_{y,t}]
\]

定义校正门：

\[
g_t^{corr} = \sigma(W_c[\mathbf{e}_t, s^{sw}_t, s^{turn}_t, \|\mathbf{i}_t\|_2] + b_c)
\]

则后验状态为：

\[
\mathbf{z}^{slow}_t = \tilde{\mathbf{z}}^{slow}_t + g_t^{corr}\odot\Phi_{slow}(\mathbf{e}_t, \tilde{\mathbf{z}}^{slow}_t)
\]
\[
\mathbf{z}^{fast}_t = \tilde{\mathbf{z}}^{fast}_t + g_t^{corr}\odot\Phi_{fast}(\mathbf{e}_t, \tilde{\mathbf{z}}^{fast}_t)
\]

### 6.3 切换感知融合

为了根据当前动态环境自适应地融合 slow 与 fast 两个状态，定义切换门：

\[
\alpha_t = \sigma(W_\alpha[\mathbf{e}_t, s^{sw}_t, s^{turn}_t, c_t, m_t, \Delta z_t] + b_\alpha)
\]

最终融合状态：

\[
\mathbf{z}_t = (1-\alpha_t)\mathbf{z}^{slow}_t + \alpha_t \mathbf{z}^{fast}_t
\]

解释如下：

- 在稳态段，\(\alpha_t\) 较小，模型更依赖 slow state
- 在高 switch / high turn / 高 zoom 扰动段，\(\alpha_t\) 增大，模型更依赖 fast state

因此，LA-CSPC-ORNet 不是普通单路径偏差平滑器，而是一个切换感知的双状态时序重建网络。

## 7. 方向受限的预测—校正前导机制

这是本文的第二项核心创新。

本文不允许网络自由输出任意方向的“补偿量”。原因在于，自由前导项容易被模型学习成任意平滑偏移，虽然可能降低短时噪声，却会在转折或高动态阶段引入错误方向的补偿。为此，本文将前导修正限制在一组物理上合理的方向基底之内。

### 7.1 方向基底构造

定义三个方向基底：

\[
\mathbf{d}^{innov}_t = \text{norm}(\mathbf{i}_t)
\]
\[
\mathbf{d}^{trend}_t = \text{norm}([\dot\tau_{x,t}, \dot\tau_{y,t}])
\]
\[
\mathbf{d}^{rawv}_t = \text{norm}\left(\left[\frac{\Delta d^{raw}_{x,t}}{\delta t_t}, \frac{\Delta d^{raw}_{y,t}}{\delta t_t}\right]\right)
\]

其中 `norm` 表示归一化方向向量。

### 7.2 方向与幅值解耦

根据融合状态预测方向混合权重：

\[
\mathbf{w}_t = \text{softmax}(W_d \mathbf{z}_t)
\]

由此得到方向：

\[
\mathbf{d}_t = \text{norm}(w_{t,1}\mathbf{d}^{innov}_t + w_{t,2}\mathbf{d}^{trend}_t + w_{t,3}\mathbf{d}^{rawv}_t)
\]

再预测前导幅值：

\[
m_t = \text{softplus}(W_m \mathbf{z}_t)
\]

定义前导激活门：

\[
\beta_t = \sigma(W_\beta[\mathbf{z}_t, s^{sw}_t, s^{turn}_t, \|\mathbf{i}_t\|_2] + b_\beta)
\]

得到前导修正：

\[
\Delta \mathbf{d}^{lead}_t = \beta_t \cdot m_t \cdot \mathbf{d}_t
\]

该前导项只有在高 switch / high turn / 高 innovation 条件下才会增强，在稳态段则保持较弱，从而实现：

1. 减小局部 lag
2. 保留转折结构
3. 避免无意义的全局超前

## 8. 双输出头：当前 clean offset 与 future predictive offset

这是本文最关键的任务扩展。

### 8.1 当前 clean offset 输出

首先由融合状态生成当前 clean offset state：

\[
\mathbf{x}^{clean}_t = [\hat d_{x,t}, \hat d_{y,t}] = H_{clean}(\mathbf{z}_t) + \Delta \mathbf{d}^{lead}_t
\]

如果启用辅助速度输出，则还可定义：

\[
\mathbf{v}_t = [\hat v_{x,t}, \hat v_{y,t}] = H_{vel}(\mathbf{z}_t)
\]

其中 \(\mathbf{x}^{clean}_t\) 用于表示当前时刻的近乎无噪声偏差状态。

### 8.2 未来 predictive offset 输出

仅重建当前 clean offset 仍无法解决链路总时延导致的控制落后问题。为此，本文进一步定义 future predictive head，用于预测控制真正生效时刻的偏差状态。

记总预测 horizon 为：

\[
\Delta_t = \ell^{vis}_t + \ell^{alg}_t + \ell^{ctrl}_t + \ell^{act}_t
\]

若换算成离散步长，则定义：

\[
h_t = \mathrm{round}\left(\frac{\Delta_t}{\delta t_t}\right)
\]

future predictive head 写为：

\[
\mathbf{x}^{pred}_{t+\Delta_t} = H_{pred}(\mathbf{z}_t, \mathbf{x}^{clean}_t, h_t)
\]

该输出表示控制真正生效时刻的偏差预测结果。本文建议在下游固定 PID 验证中，优先使用 \(\mathbf{x}^{pred}_{t+\Delta_t}\) 作为输入，以实现时延补偿和超前控制效果。

需要再次强调：本文仍然输出的是偏差状态，不是控制命令。

## 9. 无真值条件下的训练：多源伪标签 + 未来联合监督

由于不存在理想 clean offset 真值，本文采用多源伪标签与自监督未来联合训练。

### 9.1 当前 clean offset 的伪标签

当前 clean offset 的伪标签可来自：

1. 高质量片段的人工筛选轨迹
2. 延时对齐后的稳定偏差轨迹
3. 多平滑器一致性结果
4. 少量人工标注或人工筛选片段

记为：

\[
\tilde{\mathbf{y}}^{clean}_t = [\tilde d_{x,t}, \tilde d_{y,t}]
\]

若有辅助速度标签，也可定义：

\[
\tilde{\mathbf{v}}_t = [\tilde v_{x,t}, \tilde v_{y,t}]
\]

### 9.2 future predictive offset 的伪标签

future predictive offset 的伪标签来自未来时刻的原始偏差、趋势以及高质量弱监督结果。记为：

\[
\tilde{\mathbf{y}}^{pred}_{t+\Delta_t}
\]

它可由以下信息联合构造：

1. future raw offset
2. future trend
3. future 时刻的高质量弱监督轨迹
4. 延时对齐后的未来偏差状态

需要明确说明：

- 传统 KF / IMM / Robust IMM-KF 可以作为伪标签来源之一
- 但它们只是训练辅助，不是本文主方法

### 9.3 训练时使用未来信息，推理时严格因果

训练阶段允许访问 future raw / future trend，仅用于监督 future head：

\[
\mathbf{o}_{t+1}, \mathbf{o}_{t+2}, \ldots, \mathbf{o}_{t+h_t}
\]

推理阶段则严格只使用过去窗口 \(\mathcal{O}_t\)，不访问未来帧，因此方法满足实时因果部署要求。

## 10. 损失函数设计

总损失定义为：

\[
\mathcal{L} =
\lambda_{clean}\mathcal{L}_{clean}
+ \lambda_{vel}\mathcal{L}_{vel}
+ \lambda_{future}\mathcal{L}_{future}
+ \lambda_{smooth}\mathcal{L}_{smooth}
+ \lambda_{turn}\mathcal{L}_{turn}
+ \lambda_{lag}\mathcal{L}_{lag}
+ \lambda_{rob}\mathcal{L}_{rob}
\]

### 10.1 clean reconstruction loss

\[
\mathcal{L}_{clean} = \sum_t w_t \cdot \mathrm{Huber}(\mathbf{x}^{clean}_t, \tilde{\mathbf{y}}^{clean}_t)
\]

用于监督当前 clean offset state。

### 10.2 velocity consistency loss

若模型输出辅助速度：

\[
\mathcal{L}_{vel} = \sum_t w_t \cdot \mathrm{Huber}(\mathbf{v}_t, \tilde{\mathbf{v}}_t)
\]

### 10.3 future prediction loss

这是本文训练中的核心损失。其作用不是监督当前平滑结果，而是直接监督控制真正生效时刻的 future predictive offset：

\[
\mathcal{L}_{future} = \sum_t w_t^{future} \cdot \mathrm{Huber}(\mathbf{x}^{pred}_{t+\Delta_t}, \tilde{\mathbf{y}}^{pred}_{t+\Delta_t})
\]

该项迫使模型学会时延补偿，而不是仅仅学习当前帧去噪。

### 10.4 smoothness loss

平稳段要求平滑和连续，因此定义：

\[
\mathcal{L}_{smooth} = \sum_t (1-e_t)\|\Delta^2 \mathbf{x}^{clean}_t\|_1
\]

其中 \(e_t\) 是事件段权重；在事件段中，平滑项被减弱。

### 10.5 turn preservation loss

为了防止模型将转折峰值和方向翻转压扁，定义：

\[
\mathcal{L}_{turn} =
\sum_t e_t \Big(
\|\mathrm{Peak}(\mathbf{x}^{clean}_{t:t+k}) - \mathrm{Peak}(\tilde{\mathbf{y}}_{t:t+k})\|_1
+ \|\Delta \mathbf{x}^{clean}_t - \Delta \tilde{\mathbf{y}}_t\|_1
\Big)
\]

### 10.6 lag loss

lag loss 重点约束高 switch / high turn 片段中 future prediction 的相位滞后：

\[
\mathcal{L}_{lag} = \sum_t e_t \cdot \mathrm{LagProxy}(\mathbf{x}^{pred}_{t+\Delta_t}, \tilde{\mathbf{y}}^{pred}_{t+\Delta_t})
\]

这项损失直接服务于“时延补偿”目标。

### 10.7 robust loss

由于训练标签来自多源伪标签与弱监督结果，存在标签不一致和异常片段，因此引入鲁棒项：

\[
\mathcal{L}_{rob}
\]

可采用 uncertainty-weighted Huber 或 focal-Huber 等形式，以降低异常样本对训练的破坏。

## 11. 如何避免退化成普通低通滤波器

本文的方法并不是为了学习一个简单平滑器，而是为了学习一个动态偏差重建器和 future offset predictor。为避免退化成普通低通滤波器，本文通过以下机制进行约束：

1. dual-state 建模
   - slow state 学习稳态去噪
   - fast state 学习快速切换与转折响应
2. future supervision
   - future prediction loss 强迫隐状态编码未来动态，而不是只平滑当前值
3. switch-aware gating
   - 通过 switch score / turn score / confidence / miss / zoom delta 自适应提升 fast 分量
4. direction-constrained lead correction
   - 前导修正只能沿 innovation、trend velocity 与 raw local velocity 的合理方向发生
5. turn/lag 专项损失
   - 直接保护转折结构并抑制事件段相位滞后

因此，LA-CSPC-ORNet 学到的是一个时延感知的动态偏差重建与未来偏差预测器，而不是一个简单低通滤波器。

## 12. 在线推理与因果性

在线推理时，模型仅访问过去窗口：

\[
\mathcal{O}_t = \{\mathbf{o}_{t-L+1}, \ldots, \mathbf{o}_t\}
\]

未来 raw / future trend 仅在训练中用于监督 future head，不在推理阶段使用。因此，LA-CSPC-ORNet 满足严格因果的在线部署要求。

## 13. 创新点

本文的创新点集中在偏差重建本身，而不涉及检测器、PID 或控制命令生成。可归纳为以下四点：

1. 本文将 detection/tracking 后 noisy offset sequence 重新定义为一个时延感知的偏差重建与未来偏差预测问题，而不是继续沿用固定状态转移与高斯噪声假设下的滤波问题。
2. 本文提出 LA-CSPC-ORNet，通过切换感知双状态建模同时处理稳态去噪与快速动态响应，从而提升偏差重建的连续性和动态保持能力。
3. 本文提出方向受限的预测—校正前导机制，使局部前导修正仅沿合理动态方向发生，从而降低动态 lag 并保持转折结构。
4. 本文提出无理想真值条件下的多源伪标签与未来联合监督训练框架，实现当前 clean offset 和 future predictive offset 的联合学习。

## 14. 实验设计

### 14.1 主对比方法

主比较对象应包括：

1. Raw
2. EMA
3. Median
4. KF
5. EKF
6. IMM-KF
7. Robust IMM-KF
8. GRU
9. LSTM
10. TCN
11. Lightweight Transformer
12. CSPC-ORNet（仅当前重建）
13. LA-CSPC-ORNet（当前重建 + 未来预测）

### 14.2 消融实验

至少设计以下消融：

1. 无 dual-state
2. 无 lead correction
3. 无 future head
4. 无 delay-aware features
5. 无 turn loss
6. 无 lag loss
7. 无 switch-aware gate

### 14.3 专项评估片段

专项评估至少覆盖：

1. 高抖动段
2. zoom transition 段
3. lost/coasting 恢复段
4. 快速转折段
5. 高时延段

### 14.4 主指标

主指标必须以偏差重建与未来预测为中心，包括：

1. 当前偏差重建误差
2. 未来偏差预测误差
3. turn point retention
4. switch-segment lag
5. jitter energy
6. 恢复时间
7. zoom 过渡稳定性
8. 可选速度一致性误差

### 14.5 Downstream PID validation

在实验末尾可以增加一个独立小节：Downstream PID validation。该小节使用同一个固定 PID 控制器，分别接收：

1. raw offset
2. 传统滤波输出
3. 当前 clean offset
4. future predictive offset

比较：

1. 下游相位滞后
2. 超前控制效果
3. 中心保持误差

但必须明确写出：

> 该小节仅用于验证本文偏差重建与未来预测结果的应用价值，PID 控制器不是本文主方法本体。

## 15. 方法总结

本文最终提出的不是检测算法、控制命令生成算法或 PID 控制算法，而是一种面向 detection/tracking 后 noisy offset sequence 的时延感知神经网络偏差重建与未来偏差预测方法。LA-CSPC-ORNet 的目标是同时学习当前 clean offset state 和控制真正生效时刻的 future predictive offset state，从而在强噪声、强非线性、强时变与 zoom 扰动条件下，为下游固定控制器提供更适合控制的偏差状态表示。
