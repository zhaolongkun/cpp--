dscgnet_predictions.csv 说明

1. 先说明“第0行”

你说的第0行，是 csv 的表头行，也就是参数名 / 列名行，不是样本数据。

dscgnet_predictions.csv 的第0行是：

timestamp_ms,frame_id,track_id,det_dx,det_dy,track_dx,track_dy,first_filter_dx,first_filter_dy,fused_dx,fused_dy,cmd_x,cmd_y,error_px,model_ready,prediction_split,current_row_index,target_row_index,current_frame_id,target_frame_id,current_timestamp,target_timestamp,target_track_id,current_legacy_u_x,current_legacy_u_y,true_legacy_u_x,true_legacy_u_y,pred_legacy_u_x,pred_legacy_u_y,delta_pred_x,delta_pred_y,gate_x,gate_y,raw_delta_x,raw_delta_y

这一行的作用是定义后面每一列分别代表什么。


2. 这个文件整体在表达什么

这个文件是在下面这个任务定义下生成的：

- 模型任务：one-step-ahead prediction of legacy control signal
- 这里的 legacy control signal 就是原始数据里的 first_filter_dx 和 first_filter_dy
- 模型输入是它们的历史序列及其差分特征
- 模型输出是下一帧控制信号预测值

所以，这个文件本质上是在做一件事：

- 保留原始记录
- 在原始记录旁边附加模型对下一帧 legacy control signal 的预测结果


3. 第0行每个参数名都是什么意思

3.1 原始采集列

timestamp_ms
- 含义：该帧时间戳，单位毫秒。
- 来源：原始数据文件 track-fusion-move-baseline_new.csv。
- 作用：表示这一行对应的采样时刻，也用于计算相邻帧时间间隔 dt。

frame_id
- 含义：这一帧的编号。
- 来源：原始数据文件。
- 作用：表示时间顺序位置，也便于定位丢帧或对齐其他日志。

track_id
- 含义：目标轨迹编号。
- 来源：原始数据文件。
- 作用：表示这一行属于哪个跟踪目标。

det_dx, det_dy
- 含义：检测层给出的二维偏差量。
- 来源：原始数据文件。
- 作用：保留原始检测结果，方便与后续跟踪、滤波和控制链输出做对比。
- 说明：它们不是当前模型的主体输入状态。

track_dx, track_dy
- 含义：跟踪链路输出的二维偏差量。
- 来源：原始数据文件。
- 作用：保留跟踪结果，方便和检测结果、旧控制输出对比。

first_filter_dx, first_filter_dy
- 含义：旧控制链在真实系统中实际用于驱动电机的二维控制输出。
- 来源：原始数据文件。
- 作用：这是当前工程里定义的 legacy control signal / historical control output。
- 说明：这是本次模型训练和预测的核心序列。

fused_dx, fused_dy
- 含义：原始数据里已有的融合结果字段。
- 来源：原始数据文件。
- 作用：作为对照量保留。
- 说明：它们在本文件中不是当前监督目标的定义列，真正的目标列是 first_filter_dx, first_filter_dy 对应的下一帧。

cmd_x, cmd_y
- 含义：原系统当时发给执行侧的控制命令。
- 来源：原始数据文件。
- 作用：用于回看原系统的执行行为。
- 说明：它们不是本次监督学习的直接标签。

error_px
- 含义：该时刻的误差大小，单位像素。
- 来源：原始数据文件。
- 作用：用于辅助分析该时刻目标偏差程度。


3.2 导出控制流程列

model_ready
- 含义：这一行是否已经满足模型推理所需的历史窗口。
- 取值：
  - 0：还没有足够历史帧，当前行没有有效预测结果
  - 1：已经有足够历史帧，当前行存在有效预测结果

prediction_split
- 含义：这一行对应的样本所属阶段。
- 取值：
  - warmup：窗口尚未准备好，只是原始记录
  - train：有效样本，属于训练段
  - val：有效样本，属于验证段
  - test：有效样本，属于测试段

current_row_index
- 含义：有效样本中“当前时刻 t”在整张表中的行号。
- 说明：只有 model_ready=1 时才有值。

target_row_index
- 含义：有效样本中“目标时刻 t+1”在整张表中的行号。
- 说明：当前行本身就是目标时刻对应的那一行，所以有效样本里它对应当前行。

current_frame_id
- 含义：有效样本里当前时刻 t 的 frame_id。

target_frame_id
- 含义：有效样本里目标时刻 t+1 的 frame_id。

current_timestamp
- 含义：有效样本里当前时刻 t 的时间戳。

target_timestamp
- 含义：有效样本里目标时刻 t+1 的时间戳。

target_track_id
- 含义：有效样本里目标时刻 t+1 的 track_id。


3.3 当前状态、真实值、预测值列

current_legacy_u_x, current_legacy_u_y
- 含义：当前时刻 t 的 legacy control signal。
- 对应关系：
  - current_legacy_u_x = first_filter_dx(t)
  - current_legacy_u_y = first_filter_dy(t)
- 说明：只有 model_ready=1 时才有值。

true_legacy_u_x, true_legacy_u_y
- 含义：真实的下一帧 legacy control signal，也就是监督目标。
- 对应关系：
  - true_legacy_u_x = first_filter_dx(t+1)
  - true_legacy_u_y = first_filter_dy(t+1)
- 说明：这是拿来和模型预测值直接比较的真实值列。

pred_legacy_u_x, pred_legacy_u_y
- 含义：模型预测得到的下一帧 legacy control signal。
- 对应关系：
  - pred_legacy_u_x = u_pred_x(t+1)
  - pred_legacy_u_y = u_pred_y(t+1)
- 说明：这是模型最终输出的控制信号预测值。


3.4 中间输出列

delta_pred_x, delta_pred_y
- 含义：模型预测的单步状态转移增量。
- 对应关系：
  - delta_pred_x = pred_legacy_u_x - current_legacy_u_x
  - delta_pred_y = pred_legacy_u_y - current_legacy_u_y

gate_x, gate_y
- 含义：模型输出的门控量，范围在 0 到 1 之间。
- 作用：控制当前步允许放出的增量幅度。

raw_delta_x, raw_delta_y
- 含义：模型增量头输出的原始参数 z_t。
- 说明：它不是最终增量，要经过 tanh 和 gate 后才会变成 delta_pred。


4. 这些列是怎么来的

4.1 原始列的来源

下面这些列直接来自原始数据文件：

- timestamp_ms
- frame_id
- track_id
- det_dx
- det_dy
- track_dx
- track_dy
- first_filter_dx
- first_filter_dy
- fused_dx
- fused_dy
- cmd_x
- cmd_y
- error_px

这些列没有被重新计算，只是原样保留到了结果 csv 中。


4.2 模型输入是怎么构造的

模型并不是直接拿 det_dx, det_dy 做主体输入，而是以 legacy control signal 为核心状态：

legacy_u(t) = [first_filter_dx(t), first_filter_dy(t)]

基础输入特征是：

- f_x, f_y
- d_x, d_y
- dd_x, dd_y

其中：

f_x(t) = first_filter_dx(t)
f_y(t) = first_filter_dy(t)

d_x(t) = f_x(t) - f_x(t-1)
d_y(t) = f_y(t) - f_y(t-1)

dd_x(t) = d_x(t) - d_x(t-1)
dd_y(t) = d_y(t) - d_y(t-1)

如果质量分支开启，还会额外使用：

- conf
- log(area + 1)
- miss_flag
- dt

这些特征是模型内部使用的输入特征，不单独作为列写回这个 csv。


4.3 模型输出是怎么得到的

模型采用的是 state-transition parameterized output：

delta_pred = gate * (r_max * tanh(raw_delta))
pred_legacy_u = current_legacy_u + delta_pred

所以字段之间的联系是：

- raw_delta 决定原始转移趋势
- gate 决定当前步放行多少增量
- delta_pred 是最终单步预测增量
- pred_legacy_u 是最终预测控制信号


5. 这些列之间最重要的联系

如果只看最核心的关系，可以按下面理解：

5.1 时间关系

对于有效样本：

- current_* 表示时刻 t
- true_* 表示时刻 t+1 的真实值
- pred_* 表示模型对 t+1 的预测值

也就是：

- 输入窗口到 t 为止
- 输出目标是 t+1


5.2 控制信号关系

对每个有效样本，最重要的几组列是：

- 当前控制状态：
  - current_legacy_u_x
  - current_legacy_u_y

- 真实下一帧：
  - true_legacy_u_x
  - true_legacy_u_y

- 预测下一帧：
  - pred_legacy_u_x
  - pred_legacy_u_y

这三组列表示：

- 当前控制信号是什么
- 下一帧真实控制信号是什么
- 模型认为下一帧应该是什么


5.3 增量关系

有效样本中满足：

pred_legacy_u_x = current_legacy_u_x + delta_pred_x
pred_legacy_u_y = current_legacy_u_y + delta_pred_y

同时：

delta_pred_x = gate_x * (rmax_x * tanh(raw_delta_x))
delta_pred_y = gate_y * (rmax_y * tanh(raw_delta_y))

因此：

- raw_delta 是原始增量参数
- gate 是门控
- delta_pred 是最终放出的增量
- pred_legacy_u 是加到当前状态后的最终预测值


6. 为什么有些行预测列是空的

因为模型需要历史窗口。

当前配置里：

- seq_len = 16

所以：

- 前 16 行没有足够历史帧
- 这些行会被标成：
  - model_ready = 0
  - prediction_split = warmup
- 这些行的 pred_legacy_u、delta_pred、gate、raw_delta 会留空

这不是错误，而是正常现象。


7. 如何直接看“真实值 vs 预测值”

如果你想直接比较模型效果，优先看下面这些列：

- frame_id
- timestamp_ms
- model_ready
- current_legacy_u_x
- current_legacy_u_y
- true_legacy_u_x
- true_legacy_u_y
- pred_legacy_u_x
- pred_legacy_u_y
- delta_pred_x
- delta_pred_y
- gate_x
- gate_y

推荐先筛选：

- model_ready = 1

因为只有这些行才是真正完成了一帧前瞻预测的有效样本。


8. 一句话总结第0行

第0行不是数据，而是整张结果表的字段定义。

其中最关键的几列是：

- first_filter_dx, first_filter_dy：原始 legacy control signal
- current_legacy_u_x, current_legacy_u_y：当前时刻控制状态
- true_legacy_u_x, true_legacy_u_y：真实下一帧控制信号
- pred_legacy_u_x, pred_legacy_u_y：模型预测下一帧控制信号
- delta_pred_x, delta_pred_y：预测增量
- gate_x, gate_y：门控量
- raw_delta_x, raw_delta_y：原始增量参数

它们合起来描述的是：

- 模型如何根据历史控制状态，直接生成下一帧 legacy control signal。
