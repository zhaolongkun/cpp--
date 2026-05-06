论文与项目说明
==================================================

1. 文档用途
本文件用于说明 `paper/仪器与测量汇刊/new2` 这一版论文对应的方法实现过程、项目整体结构、主要创新点、关键实验结果以及代码映射关系，便于后续写论文、答辩汇报、代码复现和项目交接时统一口径。

本文件建议与以下内容配合阅读：
- 论文正文以 `paper/仪器与测量汇刊/new2/main.tex` 为准。
- 模型实现以 `模型-new2/models/dscgnet.py` 为主。
- 训练与消融实验实现以 `paper/仪器与测量汇刊/new2/picture/消融/run_ablation_experiments.py` 为主。
- 实时工程实现以仓库根目录下的 `src/`、`include/`、`config/` 为主。


2. 项目总览
本项目本质上是一个面向反无人机场景的二维视觉伺服智能控制系统。系统的输入是摄像头图像，经过目标检测、目标跟踪、目标选择、偏差估计、滤波与控制后，最终输出云台或电机的二维控制命令。

从工程视角看，这个仓库不是单独的检测项目，也不是单独的控制项目，而是一条完整的实时闭环链路，主要包含三层：

- 第一层：实时工程层
  使用 C++ 实现视觉采集、检测、跟踪、滤波、控制和执行器命令输出，入口主要在 `src/main.cpp` 和 `src/app/tracker_node.cpp`。

- 第二层：数据与学习层
  使用 Python/PyTorch 对控制日志和离线数据进行特征构造、模型训练、主实验对比和消融分析，核心目标是学习“下一帧 legacy control signal 的预测量”。

- 第三层：论文与出图层
  使用 LaTeX 汇总方法、实验和图表，核心论文文件为 `paper/仪器与测量汇刊/new2/main.tex`，图和结构示意图由同目录下的若干 Python 脚本生成。

论文提出的方法名称为 `DSCGNet`，全称可以概括为一种面向视觉控制时延补偿的 Direct Stability-Constrained Control Generation Network。它不是要推翻现有控制基线，而是在保留原有稳定控制链路的前提下，对视觉处理引入的一帧级滞后做可控的前瞻补偿。


3. 论文要解决的核心问题
当前系统已有一条可工作的传统控制链路。它的优点是结构清晰、响应稳定、工程可落地；但问题在于视觉处理链路天然存在时延。论文中给出的背景是摄像头运行在 30 FPS 左右，也就是单帧时间约 33 ms。考虑到图像采集、传输、检测与处理等环节，整体有效延迟大约在 50 ms 左右。

这带来的直接后果是：
- 控制器在时刻 t 拿到的视觉结果，实际上对应的是更早时刻的目标状态。
- 当目标机动较快时，电机始终在追“上一拍”的目标。
- 即使传统 PID 或滤波基线本身稳定，系统仍会表现出可感知的滞后。

因此，本文真正解决的问题不是“重新设计一个全新的控制器”，而是：

在不破坏原有稳定控制链路的前提下，预测下一帧的 legacy control signal，用一帧前瞻来补偿视觉时延。

如果一帧预测有效，就等价于把原来约 33 ms 的主要视觉滞后前移一拍，系统的有效剩余滞后就会明显下降。这也是本文的工程价值所在。


3.1 从检测算法到实际电机控制的全流程数据流向
这一节不再只讲论文抽象，而是按照当前 C++ 实时代码里的真实处理顺序，把信号从“摄像头图像”一直写到“电机收到命令”为止。

整条链路可以先概括成下面这一行：

摄像头图像
-> 检测器输出 `core::Detection`
-> 跟踪器输出 `core::TrackState`
-> 目标选择器确定 `controlled_id`
-> 自动变焦更新 `zoom_value`
-> 控制线程做状态外推与 `PnrImmKf` 滤波
-> 生成 `dx_raw/dy_raw`、`dx_hat/dy_hat`
-> 做轨迹预滤波和多级低通
-> 时序补偿器输出 `dx_ctrl/dy_ctrl`
-> X/Y 控制律生成 `cmd_base`
-> 满足门控条件后生成 `cmd_sent`
-> 电机驱动层缩放、限幅、俯仰保护
-> Python bridge
-> 实际 USB 电机


3.2 第 1 级：图像输入到检测输出
这一层主要发生在 `TrackerNode::vision_loop()` 和 `vision::YoloDetector` 里。

一、输入源
- `cam` 模式下，`vision::CameraCapture::read()` 从摄像头读一帧图像。
- `replay` 模式下，从回放 CSV 直接读当前帧检测结果。
- 如果启用了共享内存检测器，则从共享内存直接读取图像和检测框。

二、检测器输出
检测器的核心输出类型是 `std::vector<core::Detection>`，每个检测目标只保留三类核心信息：
- `bbox`
- `conf`
- `cls`

这里的 `bbox` 还是“检测框本身”，并不是最终会送去控制器的信号。到这一步为止，系统拿到的只是“这一帧看到哪些目标、每个目标大致在哪、置信度多少”。


3.3 第 2 级：检测框到轨迹状态
这一层主要发生在 `tracking::MultiObjectTracker::update()`。

跟踪器接收当前帧的 `detections`，再结合上一帧已有轨迹做预测、关联和更新，最终输出 `std::vector<core::TrackState>`。对控制最重要的字段包括：
- `track_id`
- `det_bbox`
- `bbox_hat`
- `cx, cy`
- `w, h`
- `vx, vy`
- `vw, vh`
- `miss_count`
- `matched_in_frame`

这里要特别区分两个框：
- `det_bbox`：这一帧真实检测到的框
- `bbox_hat`：跟踪器/Kalman 内部估计后的预测框

也就是说，从这一层开始，系统已经不再把“单帧检测框”当成唯一真值，而是开始构造时间连续的轨迹状态。


3.4 第 3 级：选出真正要控制的目标
这一层发生在 `control::TargetSelector::select()`。

规则很直接：
- 如果上一帧正在控制的 `preferred_id` 这一帧还存在，就继续控制它。
- 如果上一帧目标丢了，就退化为“选图像中心最近的轨迹”。

这一层的输出是：
- `controlled_id`

它的作用是把“多目标视觉问题”缩成“当前只控制哪一个目标”的单目标控制问题。后面的滤波、补偿和电机控制，全部只围绕这个 `controlled_id` 展开。


3.5 第 4 级：自动变焦会先影响控制链
这一层发生在 `TrackerNode::auto_zoom_step()`。

自动变焦不是附属功能，而是整条控制链的一部分，因为 zoom 的变化会直接影响：
- 目标框面积
- 目标中心速度
- 检测置信度
- 偏差量级

自动变焦使用的是当前受控目标的 `bbox_hat` 面积，而不是原始检测框边长。处理逻辑大致是：
- 根据 `bbox_hat.width() * bbox_hat.height()` 计算当前目标面积
- 与期望目标面积 `target_bbox_ratio` 对应的目标面积比较
- 经过中值滤波、EMA、滞回、持续帧数和动作间隔约束后
- 更新 `zoom_cmd_value_`

然后，视觉线程把当前 zoom 写入共享状态：
- `shared_.zoom_value = zoom_cmd_value_`

控制线程后面会用这个量做两件事：
- 作为 `PnrImmKf` 的质量信息输入之一
- 判断当前镜头是否已经稳定，决定能不能真正放行动作到电机


3.6 第 5 级：控制线程先把视觉状态外推到当前时刻
视觉线程和控制线程不是同一个时刻执行，所以控制线程刚拿到共享状态后，第一件事不是直接算控制，而是先算时间差：

- `pred_dt = now_ns - local.vision_time_ns`

含义是：
- 视觉结果是上一时刻采样出来的
- 控制计算发生在稍后的当前时刻
- 因此需要先把轨迹状态从视觉采样时刻外推到当前控制时刻

如果当前存在受控目标 `ct`，则先得到原始外推量：
- `cx_raw = ct.cx + ct.vx * pred_dt`
- `cy_raw = ct.cy + ct.vy * pred_dt`
- `w_raw = ct.w + ct.vw * pred_dt`
- `h_raw = ct.h + ct.vh * pred_dt`

这些量对应的是“跟踪器直接外推出来的目标状态”，还没有经过位置滤波器进一步稳化。


3.7 第 6 级：`PnrImmKf` 生成更稳定的中心和速度估计
这一层是控制前最关键的稳定化步骤之一，主要在 `filter::PnrImmKf` 中完成。

如果 `cfg_.filter.enable=true`，控制线程会执行：
- `pnr_filter_.predict(dt)`
- 在允许测量更新时执行 `pnr_filter_.update(ct.cx, ct.cy, q)`

这里的质量量 `q` 包括：
- `conf`
- `miss_count`
- `bbox_area`
- `zoom_value`
- `zoom_delta`
- `det_count`

然后滤波器输出更稳定的状态：
- `cx_hat = pnr_filter_.x()`
- `cy_hat = pnr_filter_.y()`
- `vx_hat = pnr_filter_.vx()`
- `vy_hat = -pnr_filter_.vy()`

因此，在当前实现里，同时存在两套中心量：
- `cx_raw, cy_raw`：轨迹直接外推得到的原始连续量
- `cx_hat, cy_hat`：经过 `PnrImmKf` 稳定后的连续量

这一步的意义是把“检测抖动、短时失配、zoom 变化、速度估计不稳”等问题先压下去，再把结果送给后面的控制链。


3.8 第 7 级：从目标中心坐标变成真正的控制偏差
控制器并不直接吃 `cx, cy`，而是把目标中心换算成相对图像中心的偏差。

当前代码中的定义是：
- `dx_raw = cx_raw - center_x`
- `dy_raw = center_y - cy_raw`
- `dx_hat = cx_hat - center_x`
- `dy_hat = center_y - cy_hat`

其中：
- `center_x = cfg_.camera.center_x`
- `center_y = cfg_.camera.center_y`

这说明：
- `dx_raw/dy_raw` 表示未经位置滤波稳化的原始连续偏差
- `dx_hat/dy_hat` 表示经过 `PnrImmKf` 后的稳化偏差

如果后面只看一条主线，那么真正进入后级信号清洗和控制补偿的，是 `dx_hat/dy_hat` 这一组更稳定的偏差。


3.9 第 8 级：轨迹预滤波、多级低通与 legacy-style 基线
`dx_hat/dy_hat` 还不会直接进电机。当前代码在这之后又做了一层“轨迹信号清洗”。

第一步：短窗口均值预滤波
- `track_prefilter_dx_window_`
- `track_prefilter_dy_window_`

代码会对最近 3 帧的 `dx_hat/dy_hat` 取均值，得到：
- `tracker_signal_dx`
- `tracker_signal_dy`

第二步：帧间跳变限幅
如果当前信号相对上一时刻跳得过大，代码会把单帧变化截断到阈值内：
- `kMaxStepPx = 60.0`

第三步：多级低通
当前实现里维护了三层递推低通：
- `offset_stage1_dx_, offset_stage1_dy_`
- `offset_stage2_dx_, offset_stage2_dy_`
- `offset_stage3_dx_, offset_stage3_dy_`

并映射到日志里的名字：
- `first_filter_dx, first_filter_dy`
- `second_filter_dx, second_filter_dy`
- `third_filter_dx, third_filter_dy`

同时还会继续做窗口均值，得到：
- `window8_second_filter_dx/dy`
- `window8_third_filter_dx/dy`
- `window8_10_third_filter_dx/dy`

从信号语义上看，这一层可以理解为：
- `track_dx/track_dy`：跟踪器稳化后的偏差
- `first_filter_dx/first_filter_dy`：第一层低通后的稳定基线
- `second/third filter`：更保守、更平滑的观测版本

这也是论文里 `legacy control signal` 最接近当前在线工程变量的地方。因为论文训练用的核心监督思路，本质上就是学习“稳定控制参考信号的下一步会怎么变化”。


3.10 第 9 级：时序补偿器把稳定基线变成前瞻控制量
这一层发生在：
- `TemporalCompensatorOnnx::step(raw_input_dx, raw_input_dy, track_signal_valid, false)`

进入时序补偿器的输入是：
- `raw_input_dx = tracker_signal_dx`
- `raw_input_dy = tracker_signal_dy`

也就是说，输入给时序模型的并不是原始检测框中心，而是已经经过“跟踪 + 位置滤波 + 短窗口平滑”的控制偏差信号。

当前补偿器内部的处理顺序是：

第一步：构造因果参考量 `e_ref`
如果 `input_is_reference=false`，补偿器会先对最近 `window_size` 个输入求均值，得到：
- `e_ref_x`
- `e_ref_y`

第二步：计算一阶差分
- `d_ref_x = e_ref_x - prev_ref_x`
- `d_ref_y = e_ref_y - prev_ref_y`

第三步：构造时序特征序列
每个时刻的特征是 4 维：
- `[e_ref_x, e_ref_y, d_ref_x, d_ref_y]`

第四步：当缓存长度达到 `seq_len` 后，送入 ONNX 模型
模型输出：
- `delta_pred_x`
- `delta_pred_y`

第五步：做增量限幅
- `delta_applied_x = clamp(delta_pred_x, -delta_max, delta_max)`
- `delta_applied_y = clamp(delta_pred_y, -delta_max, delta_max)`

第六步：得到最终补偿后的控制参考
- `u_x = e_ref_x + alpha * delta_applied_x`
- `u_y = e_ref_y + alpha * delta_applied_y`

控制线程把这两个量记成：
- `dx_ctrl = temporal_out.u_x`
- `dy_ctrl = temporal_out.u_y`

因此，这一层之后的信号语义非常清楚：
- `tracker_signal_dx/dy` 是稳化后的当前偏差
- `dx_ctrl/dy_ctrl` 是叠加了一步前瞻补偿后的控制偏差

如果 ONNX 模型还在 warmup、没有加载成功、或推理失败，代码会自动回退到因果基线 `e_ref`，也就是继续使用稳定基线，不会让控制链断掉。


3.11 第 10 级：`dx_ctrl/dy_ctrl` 如何变成电机命令
这一层是最终控制律。

一、X 轴控制
X 轴走的是比较标准的比例控制：
- `raw_x = pid_x_.step(-dx_ctrl, dt)`

然后还会叠加：
- deadband
- `cmd_limit` 限幅

最终得到：
- `cmd_base.cmd_x`

二、Y 轴控制
Y 轴更复杂，不是简单照抄 X 轴。当前实现会结合：
- `dy_ctrl` 的低通结果 `dy_f`
- 电机当前俯仰角 `motor_->get_pitch()`
- 电机俯仰角速度 `motor_->get_pitch_rate()`

主要逻辑包括：
- 对 `dy_ctrl` 先低通
- 用俯仰角速度做阻尼
- 做比例控制得到 `vy_raw`
- 做虚拟减速比缩放
- 做变化率限制
- 做最终 `vy_max` 限幅
- 在接近俯仰角上下边界时做硬保护

最终得到：
- `cmd_base.cmd_y`

因此，`cmd_base` 就是“控制线程在当前时刻认为应该发给电机的基础二维命令”。


3.12 第 11 级：不是有了 `cmd_base` 就立刻打电机
这一层是控制门控。

代码中真正放行动作的条件是：
- `target_detected && zoom_stable`

这里要区分两个概念：
- `target_valid`：受控 track 还存在
- `target_detected`：当前帧真的重新匹配到了检测框，也就是 `miss_count == 0`

这意味着：
- 如果只是 coasting，也就是轨迹还在但当前帧没真正检测到目标，系统可以继续估计和记日志
- 但不会继续把旧控制量直接发给电机

同样，如果镜头刚变焦，`zoom_stable=false`，系统也不会放行控制命令。

因此：
- 满足门控时：`cmd_sent = cmd_base`
- 不满足门控时：`cmd_sent = 0`

并且进入阻塞态时，系统会主动补发一次零命令，避免执行器保持上一帧速度继续转。


3.13 第 12 级：电机驱动层如何把 `cmd_sent` 变成真实硬件动作
这一层发生在 `actuator::MotorDriver`。

如果当前是：
- `MotorDriverMock`
  只记录命令，不发硬件
- `MotorDriverPyUsb`
  真正把控制命令送到硬件

在 `MotorDriverPyUsb` 中，`send(cmd)` 并不是立即直接写 USB，而是先把控制线程给出的命令缓存为：
- `command_state_.tracker_cmd = cmd`

随后后台 `command_loop()` 周期性读取该命令，再做最后一层硬件侧处理：

第一步：缩放
- `raw_x = round(cmd_x * scale_x)`
- `raw_y = cmd_y * scale_y`

第二步：俯仰保护
Y 轴会经过 `guard_y_command(raw_y)`，根据 IMU/俯仰角限制：
- 接近上限时禁止继续往上打
- 超过范围时强制反向
- 接近下限时禁止继续往下打

第三步：方向翻转与限速
在真正写出前，驱动层还会根据配置做：
- `invert_x`
- `invert_y`
- `speed_limit` 限幅

第四步：写给 Python bridge
最终通过：
- `write_command_pair(x, y)`

把命令以文本形式写给桥接脚本，再由桥接脚本走 USB 通讯发到实际电机。

因此，真正落到硬件上的命令，不是控制线程原始算出来的浮点值，而是：

`cmd_sent`
-> 驱动层缩放
-> 俯仰保护
-> 方向修正
-> 速度限幅
-> Python bridge
-> USB 电机


3.14 第 13 级：整条链路里每个变量分别代表什么
为了避免把不同层级的量混在一起，下面把最关键的变量再统一解释一次：

- `det_dx, det_dy`
  当前帧真实检测框中心相对图像中心的偏差。它最接近原始视觉测量。

- `dx_raw, dy_raw`
  跟踪器按 `pred_dt` 外推到当前控制时刻后的偏差，仍然偏“原始连续量”。

- `dx_hat, dy_hat`
  `PnrImmKf` 稳化后的偏差，是控制链真正使用的主输入。

- `track_dx, track_dy`
  `dx_hat/dy_hat` 经过短窗口预滤波后的轨迹偏差信号。

- `first_filter_dx, first_filter_dy`
  第一层低通后的稳定基线，最接近论文里 legacy-style control reference 的含义。

- `dx_ctrl, dy_ctrl`
  时序补偿器输出的最终控制偏差，也就是“稳定基线 + 前瞻补偿”后的结果。

- `cmd_base`
  控制线程算出的基础电机命令。

- `cmd_sent`
  满足门控条件后真正发给执行器驱动层的命令。


3.15 第 14 级：从日志角度如何验证这条信号链
当前代码已经把这条链的大部分关键节点打进了日志。

控制采集 CSV `track-fusion-move.csv` 里可以直接看到：
- `det_dx, det_dy`
- `track_dx, track_dy`
- `median_filter_dx, median_filter_dy`
- `first_filter_dx, first_filter_dy`
- `second_filter_dx, second_filter_dy`
- `third_filter_dx, third_filter_dy`
- `comp_dx, comp_dy`
- `fused_dx, fused_dy`
- `cmd_x, cmd_y`

其中可以这样理解：
- `det_*` 是检测侧信号
- `track_*` 和 `first_filter_*` 是稳定化后的传统链路信号
- `comp_*` 是时序补偿增量
- `fused_*` 是最终控制偏差
- `cmd_*` 是发往电机的控制命令

同时，`RuntimeSnapshot` 主日志里还会记录：
- `dx_raw, dy_raw`
- `dx_hat, dy_hat`
- `clean_dx, clean_dy`
- `cmd_base_x, cmd_base_y`
- `cmd_sent_x, cmd_sent_y`
- `infer_status`
- `infer_used_model`
- `zoom_value`
- `vision_latency_ms`

所以，无论是论文实验还是工程调试，都可以沿着“检测框 -> 轨迹偏差 -> 稳定基线 -> 时序补偿 -> 电机命令”这条链逐级回看。


3.16 一句话总结这条信号处理链
当前工程里，真正送到电机的并不是“检测框原始中心偏差”，而是：

检测框
-> 跟踪器连续化
-> `PnrImmKf` 稳化
-> 轨迹预滤波和多级低通
-> 时序补偿器做一帧前瞻修正
-> X/Y 控制律转成电机命令
-> zoom 门控与目标有效性门控
-> 驱动层俯仰保护、缩放和限幅
-> 最终发到真实电机


4. 现有工程基线是什么
论文不是凭空提出网络，而是建立在已有工程基线之上。按照论文与代码的对应关系，现有传统控制链路可以概括为：

目标检测框中心
-> Kalman/滤波估计
-> 相对图像中心的偏差计算
-> 分步限幅与平滑
-> 得到 legacy control signal
-> 交给控制器或执行器

在论文上下文中，legacy control signal 主要对应历史工程中已经验证过的一套稳定控制参考量，典型变量名是 `first_filter_dx`、`first_filter_dy` 这一类经过滤波和限制后的控制参考信号。

这条基线的价值有两个：
- 它已经在真实工程里证明过“能用、能稳、能跑”。
- 它为学习型方法提供了可靠监督目标，避免直接让神经网络从原始图像端到端输出电机命令。

因此，本文的方法定位非常明确：
不是取代基线，而是预测“基线的下一步会是什么”，再把这个预测结果作为时延补偿。


5. 论文实现过程

5.1 数据来源与问题重定义
本文不是把问题定义成“下一帧图像框中心预测”，也不是定义成“直接输出电机命令”，而是定义成：

给定过去一段时间内的 legacy control signal 及其相关质量信息，预测下一帧 legacy control signal。

这样做有三个好处：
- 监督目标直接对应已有工程可用控制量，目标定义稳定。
- 避开了图像域端到端学习的高复杂度和高不确定性。
- 预测结果可以直接嵌入现有控制链路，工程接入成本低。

论文中使用的数据文件是 `track-fusion-move-baseline_new.csv`。按正文描述：
- 原始记录总数：9451
- 有效样本数：9435
- 训练集：6604
- 验证集：1415
- 测试集：1416

样本构造方式是滑动窗口。历史窗口长度默认设置为 `T=16`，即使用过去 16 个时刻的时序信息来预测下一帧。


5.2 legacy control signal 的监督目标如何构造
核心监督目标不是直接拟合原始 bbox，而是拟合下一时刻的 legacy control signal：

u_legacy(t+1)

这意味着网络在训练时学的是：
- 当前传统控制参考在怎么变化；
- 这种变化趋势在下一帧会往哪里走；
- 如何在保持稳定的前提下给出一帧前瞻补偿。

这个问题设定非常贴合工程实际，因为控制器真正关心的是“下一拍应该打多大控制量”，而不是“目标框的理论几何位置”。


5.3 特征工程设计
论文最终采用 10 维输入特征，由 6 维运动特征和 4 维质量特征组成。

一、6 维运动特征：
- f_x(i) = u_x^L(i)
- f_y(i) = u_y^L(i)
- d_x(i) = f_x(i) - f_x(i-1)
- d_y(i) = f_y(i) - f_y(i-1)
- dd_x(i) = d_x(i) - d_x(i-1)
- dd_y(i) = d_y(i) - d_y(i-1)

其中：
- `f_x, f_y` 表示当前稳定控制参考本身；
- `d_x, d_y` 表示一阶变化量，反映最近变化速度；
- `dd_x, dd_y` 表示二阶变化量，反映变化趋势是否在加速或减速。

二、4 维质量特征：
- conf：检测置信度
- log(area + 1)：目标尺度信息
- miss_flag：是否存在丢帧或测量缺失
- dt：时间间隔

这部分设计对应代码中的特征构造逻辑，主要体现在 `run_ablation_experiments.py` 的 `build_feature_arrays()` 和 `build_samples()` 中。

这样设计的目的很明确：
- 运动特征负责描述“控制趋势”；
- 质量特征负责描述“当前这段趋势值是否可靠”。


5.4 DSCGNet 模型结构
模型核心代码在 `模型-new2/models/dscgnet.py`。默认结构可以概括为“双流运动编码 + 质量分支 + 融合输出头”。

一、X/Y 双流运动分支
模型没有把 6 维运动特征直接混在一起处理，而是按 x、y 两个通道分别建模：
- x 分支输入 `[f_x, d_x, dd_x]`
- y 分支输入 `[f_y, d_y, dd_y]`

每个运动分支大致包含：
- `CausalConv1d(3 -> 32, kernel=3, dilation=1)`
- `CausalConv1d(32 -> 32, kernel=3, dilation=2)`
- `GRU(32 -> 32)`
- 可选的因果注意力模块

这样做的考虑是：
- 卷积先提局部时序模式；
- 膨胀卷积扩大感受野；
- GRU 建模更长时间依赖；
- 因果注意力进一步突出与“下一拍预测”最相关的历史片段。

二、质量分支
质量分支处理 `[conf, log(area+1), miss_flag, dt]`，默认结构为：
- `Linear(4 -> 16)`
- `GRU(16 -> 16)`

这部分不是预测运动本身，而是学习“当前观测是否可信、是否该更保守或更积极地做补偿”。

三、融合层
两个运动分支各输出 32 维，质量分支输出 16 维，拼接后形成 80 维融合表示，再经过：
- `Linear(80 -> 64)`
- `ReLU`
- `Dropout(0.1)`

四、输出头
融合后分成两个头：
- delta head：输出下一步增量候选
- gate head：输出门控系数

这说明模型不是无约束地直接给出下一帧控制量，而是用“增量 + 门控”的方式控制修正幅度。


5.5 状态转移参数化输出
这是本文非常关键的设计之一。模型输出并不是直接回归

u_legacy(t+1)

而是先生成一个有界增量：

Delta_u_hat(t) = g_t * (r_max * tanh(z_t))

然后再与当前控制参考做状态转移组合：

u_legacy_hat(t+1) = u_legacy(t) + Delta_u_hat(t)

其中：
- `z_t` 是增量候选；
- `g_t` 是门控系数；
- `r_max = [12, 12]` 是每个维度的最大增量边界；
- `tanh` 和门控共同保证输出不会无限发散。

这套设计有很强的控制意义：
- 预测的是“下一步相对当前该怎么改”，而不是“完全重写当前控制量”；
- 增量被明确限幅，避免模型偶发异常时输出过激指令；
- 当前时刻的稳定基线始终保留在输出表达式中。


5.6 损失函数设计
论文采用的是控制导向的联合损失，而不是单一 MSE。总损失为：

L = lambda1 * L_ahead
  + lambda2 * L_inc
  + lambda3 * L_stab
  + lambda4 * L_dir
  + lambda5 * L_gate

默认权重为：
- lambda1 = 1.0
- lambda2 = 0.5
- lambda3 = 0.1
- lambda4 = 0.1
- lambda5 = 0.01
- tau = 12.0

各项含义如下：
- `L_ahead`：下一帧控制量拟合误差，正文采用 Huber 形式；
- `L_inc`：增量一致性，约束预测变化量与真实变化量一致；
- `L_stab`：稳定性项，抑制大幅抖动或不必要的突变；
- `L_dir`：方向一致性，减少方向反转风险；
- `L_gate`：门控稀疏约束，避免模型无必要地频繁开大门。

这套损失设计体现了本文不是单纯追求预测误差最小，而是明确把控制稳定性、方向正确性和门控保守性一起纳入训练目标。


5.7 训练配置
论文中的默认训练配置为：
- 框架：PyTorch
- 优化器：AdamW
- 学习率：1e-3
- 权重衰减：1e-4
- 批大小：64
- 历史窗口：T = 16
- 注意力头数：4
- dropout：0.1
- 训练轮数：500

训练过程中根据验证集的 Ahead MAE 保存最佳 checkpoint。

这一配置的意义在于：
- 学习率和权重衰减比较稳健，适合中小规模时序网络；
- `T=16` 对当前任务在实验中取得了最优综合表现；
- 整个模型总参数量只有 39284，说明设计目标从一开始就偏向轻量化和可部署性。


5.8 主实验对比
论文的主对比方法包括：
- Last Value
- Linear Extrapolation
- Kalman CV
- GRU Direct
- DSCGNet（本文方法）

主要评价指标包括：
- Ahead MAE：下一帧预测平均绝对误差
- Ahead RMSE：下一帧预测均方根误差
- Jitter：预测抖动程度
- Spike Rate：尖峰比例
- Sign Flip：符号翻转率

正文主表中的结果如下：

1. Last Value
   Ahead MAE = 1.7660
   Ahead RMSE = 4.4408
   Jitter = 0.0000
   Spike Rate = 0.0212
   Sign Flip = 1.0000

2. Linear Extrapolation
   Ahead MAE = 0.3135
   Ahead RMSE = 4.7372
   Jitter = 0.3137
   Spike Rate = 0.0247
   Sign Flip = 0.0275

3. Kalman CV
   Ahead MAE = 0.5643
   Ahead RMSE = 6.0284
   Jitter = 1.0566
   Spike Rate = 0.0318
   Sign Flip = 0.0307

4. GRU Direct
   Ahead MAE = 2.7794
   Ahead RMSE = 9.1456
   Jitter = 0.5465
   Spike Rate = 0.0170
   Sign Flip = 0.2066

5. DSCGNet
   Ahead MAE = 0.2157
   Ahead RMSE = 3.3362
   Jitter = 0.1903
   Spike Rate = 0.0254
   Sign Flip = 0.0258

从结果上看，DSCGNet 在 Ahead MAE 和 Ahead RMSE 上取得了最佳表现，同时保持较低的符号翻转率，说明它不仅更准，而且更适合控制链路。


5.9 消融实验结论
论文围绕四个因素做了系统消融：
- 历史窗口长度 T
- 质量分支是否开启
- 因果注意力是否开启
- 输出头是否采用状态转移参数化

一、历史长度消融
- T=8：MAE 0.8165，RMSE 6.2868，Jitter 2.8859
- T=12：MAE 0.8827，RMSE 6.2967，Jitter 2.8780
- T=16：MAE 0.7374，RMSE 6.1753，Jitter 2.9166
- T=20：MAE 1.0223，RMSE 6.2838，Jitter 2.9334

说明：当前任务中 T=16 的综合精度最好，过短看不够历史，过长会引入冗余信息。

二、质量分支消融
- 关闭质量分支：MAE 1.0664，RMSE 6.3291，Jitter 2.9568
- 开启质量分支：MAE 0.8374，RMSE 6.2753，Jitter 2.8361

说明：质量信息确实能提升预测可靠性。

三、因果注意力消融
- 关闭注意力：MAE 0.7808，RMSE 6.2778，Jitter 2.8614
- 开启注意力：MAE 0.6374，RMSE 6.0753，Jitter 2.8166

说明：注意力有助于模型更聚焦对当前预测最重要的历史时刻。

四、输出头形式消融
- 直接回归输出：MAE 6.0408，RMSE 11.9976，Jitter 3.6696
- 状态转移参数化输出：MAE 0.8022，RMSE 6.2583，Jitter 2.8491

说明：本文提出的有界状态转移输出头并不是“锦上添花”，而是性能提升的关键结构。


5.10 部署效率
论文正文给出的实验平台为：
- CPU：Intel Core i7-14700KF
- GPU：NVIDIA GeForce RTX 4070 Ti SUPER 16GB
- 内存：32GB

默认最终模型在单样本前向推理下的典型结果为：
- GPU 约 1.50 ms，对应约 664.68 Hz
- CPU 约 1.70 ms，对应约 588.56 Hz

这说明模型的计算开销相对整条视觉处理链路来说很小，具备较好的在线部署潜力。


6. 我的论文创新点

6.1 创新点一：重新定义问题，不做“图像预测”，而做“legacy control signal 的下一帧预测”
这篇论文最重要的创新之一，不是简单换一个网络，而是把任务定义得更贴合工程。

很多工作喜欢直接预测目标位置、目标轨迹，甚至直接回归控制命令；而本文选择预测“已经在工程里验证过的稳定控制参考量”的下一步。这样做带来的优势是：
- 监督信号稳定；
- 与现有控制链路兼容；
- 容易解释；
- 更容易部署。


6.2 创新点二：双流因果时序建模
模型把 x、y 两个方向分开建模，而不是粗暴拼接输入统一处理。每个方向都采用“因果卷积 + 膨胀卷积 + GRU + 可选因果注意力”的组合。

这种设计兼顾了三点：
- 因果性：只使用过去和当前信息，满足在线控制约束；
- 局部模式提取：卷积擅长提取短时趋势；
- 长时依赖建模：GRU 和注意力负责建模更长时间关系。

这比单纯的直接 GRU 回归更适合本任务。


6.3 创新点三：把检测质量信息显式引入补偿网络
很多时序预测方法只看运动量，不看观测是否可靠。本文把置信度、目标尺度、丢帧标记和时间间隔作为独立质量分支输入模型。

这使得网络不仅学“目标怎么动”，还学“当前观测值到底值不值得信”。这对视觉系统非常关键，因为真实工程里的误差很多时候不是来自目标运动本身，而是来自检测质量波动。


6.4 创新点四：状态转移参数化输出，而不是无约束直接回归
本文没有直接输出下一帧控制量，而是输出有界增量，再叠加到当前稳定参考上。这一点兼具算法和控制含义：
- 输出天然受限；
- 能防止大幅跃迁；
- 易于和现有控制链融合；
- 预测失败时不至于彻底破坏系统。

从消融实验看，这一设计是决定性能上限和稳定性的关键创新。


6.5 创新点五：面向控制的联合损失设计
本文的优化目标不是只最小化数值误差，而是把“预测准、变化稳、方向对、门控保守”一起优化。

这说明论文思路不是传统机器学习里那种“只要 MAE 低就算好”，而是明确服务于闭环控制场景，因此更符合工程应用要求。


6.6 创新点六：在实时工程语境下追求可落地部署
本文方法不是一个超大模型，也不是只在离线表格里好看。模型参数量只有 39284，推理时延毫秒级，同时仓库中已经存在实时控制系统和在线补偿接口。

这意味着本工作具有从离线建模走向在线接入的实际基础，而不是纯理论构想。


7. 项目与论文之间的关系
这个仓库的价值在于，论文不是孤立的，而是从真实工程中长出来的。可以把整个项目理解为下面三部分的闭环：

第一部分：实时 C++ 视觉控制系统
- 负责摄像头取流、目标检测、目标跟踪、目标选择、滤波、PID 和执行器控制。

第二部分：离线数据建模与实验
- 负责从控制日志中提取训练样本，训练 DSCGNet，跑主实验和消融实验，生成论文图表。

第三部分：论文撰写与结果整理
- 负责把方法动机、数学表达、实验表格、图像和项目背景系统化地整理出来。

因此，这不是“先写论文再补代码”，而是：

先有实时工程基线
-> 再从工程问题中抽取论文问题
-> 再做离线模型设计和验证
-> 最后回到工程部署语境解释其价值


8. 代码结构映射

8.1 实时工程主干
- `src/main.cpp`
  程序入口，负责解析参数并启动系统。

- `src/app/tracker_node.cpp`
  实时调度核心，串起视觉线程、控制线程、日志和执行器。

- `src/vision/`
  摄像头与目标检测。

- `src/tracking/`
  多目标跟踪与轨迹维护。

- `src/filter/`
  偏差滤波、状态估计和鲁棒处理。

- `src/control/`
  目标选择、PID、补偿分支、控制策略。

- `src/actuator/`
  电机或执行器命令输出。

- `src/core/`
  基础设施、配置和日志。


8.2 与时延补偿相关的在线接口
- `include/control/temporal_compensator_onnx.h`
- `src/control/temporal_compensator_onnx.cpp`

这部分说明仓库中已经存在“时序补偿模块”的在线接入接口。当前实时分支实现的是一个较轻量的 ONNX 时序补偿思路，而论文中的 DSCGNet 是更完整、更系统的一版方法论和实验方案。

也就是说：
- 工程里已经有补偿接口；
- 论文里给出了更成熟的建模方法与消融论证；
- 后续可以把论文方法继续向在线部署版本收敛。


8.3 论文与实验脚本
- `paper/仪器与测量汇刊/new2/main.tex`
  论文正文主文件。

- `模型-new2/models/dscgnet.py`
  DSCGNet 模型实现。

- `paper/仪器与测量汇刊/new2/picture/消融/run_ablation_experiments.py`
  主训练、对比和消融实验的关键脚本。

- `paper/仪器与测量汇刊/new2/generate_prediction_figures.py`
  预测效果图生成脚本。

- `paper/仪器与测量汇刊/new2/picture/fig_model_arch/draw_dscgnet_arch.py`
  网络结构图绘制脚本。

- `paper/仪器与测量汇刊/new2/picture/render_framework2.py`
  方法框架图绘制脚本。


9. 复现本文的大致流程
如果后续需要完整复现实验，可以按下面的顺序理解和执行：

第一步：理解实时工程基线
先看 `src/main.cpp`、`src/app/tracker_node.cpp`、`src/control/`、`src/filter/`，搞清楚传统链路怎么产生 legacy control signal。

第二步：确认训练数据来源与字段含义
结合训练脚本，确认 `track-fusion-move-baseline_new.csv` 中各字段与特征构造方法的对应关系。

第三步：运行主实验和消融实验
以 `run_ablation_experiments.py` 为主，复现实验配置、训练过程和评估指标。

第四步：检查模型结构
阅读 `模型-new2/models/dscgnet.py`，重点关注双流结构、质量分支、因果注意力和状态转移输出头。

第五步：生成论文插图
使用 `generate_prediction_figures.py`、`draw_dscgnet_arch.py`、`render_framework2.py` 等脚本重建论文图。

第六步：整理到 LaTeX
将实验结果、图表、公式和方法描述统一写入 `main.tex`。


10. 这个项目和论文的价值
从项目层面看，这个仓库已经具备较强的工程完整性：有实时链路、有日志、有控制基线、有在线接口、有离线训练代码，也有论文结果整理。

从论文层面看，本文的价值主要体现在：
- 问题定义贴近真实工程，而不是脱离工程的抽象预测问题；
- 方法结构轻量，适合实时部署；
- 结果不仅追求误差低，还关注控制稳定性；
- 与现有系统兼容，便于渐进式接入；
- 有消融实验支撑，不是只给最终结果。


11. 当前方法的边界与后续可扩展方向
虽然本文已经形成比较完整的方案，但它也有明确边界：

- 当前主要是对 legacy control signal 做一帧前瞻预测，不是完整端到端闭环控制替代。
- 数据规模目前仍偏中小，正文中提到的是 9451 条记录、6 条轨迹，后续还可以扩展场景和平台泛化验证。
- 在线 C++ 分支虽然已经有时序补偿接口，但论文版 DSCGNet 的完整落地、ONNX 化和硬件闭环验证还可以继续强化。
- 如果未来系统帧率、目标类型或执行器动态明显变化，特征范围、窗口长度和 `r_max` 等参数也需要重新校准。


12. 一句话总结
这篇论文的核心思想可以概括为：

在不破坏现有稳定视觉控制基线的前提下，把时延补偿问题重定义为“下一帧 legacy control signal 预测”，再通过双流因果时序建模、质量感知分支、有界状态转移输出和控制导向联合损失，得到一个既能提高前瞻性、又能控制风险、同时具备实时部署潜力的轻量化模型。


13. 建议的对外介绍口径
如果需要对老师、评审或项目成员快速介绍本工作，可以直接用下面这段话：

本项目是一个面向反无人机场景的实时视觉伺服控制系统。现有工程链路已经具备稳定跟踪与控制能力，但受制于视觉处理时延，电机响应存在明显的一帧级滞后。为了解决这个问题，本文没有重新推翻传统控制器，而是提出了 DSCGNet，把任务定义为对下一帧 legacy control signal 的预测。方法上，本文采用 x/y 双流因果卷积与 GRU 建模控制时序，再引入检测质量分支、状态转移参数化输出头和控制导向联合损失，在保证输出可控和稳定的前提下实现时延补偿。实验结果表明，本文方法在 Ahead MAE 和 Ahead RMSE 上优于多种基线，同时模型规模轻量、推理速度快，具备进一步在线部署的实际价值。
