# cpp智能控制

## 1. 项目定位

这个项目实现的是一条实时视觉跟踪与控制链路。输入是摄像头图像，输出是发给电机或云台的二维控制命令 `cmd_x, cmd_y`。工程不是单纯的目标检测程序，也不是单纯的 PID 控制程序，而是把以下几个环节串成了一条完整实时链路：

1. 相机取流
2. 目标检测
3. 多目标跟踪
4. 主控目标选择
5. 自动变焦
6. 像素偏差估计与滤波
7. 基础 PID 控制
8. 可选残差控制修正
9. 安全限幅与限速
10. 电机命令发送
11. 全流程日志记录

项目的主程序入口是：

- `src/main.cpp`
- `src/app/tracker_node.cpp`

主类是：

- `app::TrackerNode`

它负责把视觉线程和控制线程组织起来。

## 2. 代码结构

和主流程直接相关的目录如下：

- `src/main.cpp`
  - 解析命令行参数，构造 `TrackerNode`
- `src/app/tracker_node.cpp`
  - 整个系统的主流程调度
- `src/vision/`
  - 摄像头和检测器
- `src/tracking/`
  - 多目标跟踪
- `src/filter/`
  - 偏差滤波与状态估计
- `src/control/`
  - 目标选择、PID、残差控制策略
- `src/actuator/`
  - 电机输出
- `src/core/`
  - 配置和日志
- `config/tracker.yaml`
  - 相机模式的主要配置

## 3. 运行模式

项目支持两种模式：

1. `cam`
   - 实时摄像头模式
   - 从摄像头取流，在线检测、跟踪、变焦、控制
2. `replay`
   - 离线回放模式
   - 从 `csv` 读取检测框，复用后端控制逻辑

命令行参数由 `main.cpp` 解析：

- `--mode cam|replay`
- `--config config/tracker.yaml`
- `--replay_csv data/detections.csv`
- `--max_runtime_ms 5000`

## 4. 启动时做了什么

`TrackerNode::init()` 会完成以下初始化：

1. 生成运行 ID，用于日志标识
2. 加载残差 ONNX 模型、归一化参数和特征定义
3. 初始化偏差滤波器 `PnrImmKf`
4. 初始化电机驱动
   - `mock` 模式只记录命令，不发硬件
   - `pyusb` 模式通过 Python bridge 发 USB 命令
5. 打开日志文件
6. 根据运行模式：
   - `replay` 模式加载回放 CSV
   - `cam` 模式打开摄像头、设置自动对焦、尝试把变焦复位到 0、加载 ONNX 检测模型

需要注意的一点：

- 检测器配置里虽然可以给 `best.pt`
- 但 `vision::YoloDetector` 会优先寻找同目录下的 `best.onnx`
- 如果 `best.onnx` 存在，实际运行加载的是 `onnx`

也就是说，C++ 版本的实时检测链路实际走的是 OpenCV DNN + ONNX，不是 Python 里的 Ultralytics `.pt` 推理链。

## 5. 主线程和频率

系统启动后会创建两个线程：

1. `vision_thread_`
   - 运行 `vision_loop()`
   - 当前代码固定按 `30 Hz` 节拍跑
2. `control_thread_`
   - 运行 `control_loop()`
   - 按 `config.control.rate_hz` 运行

在当前 `tracker.yaml` 里，控制频率配置是：

- `rate_hz: 30`

所以当前默认配置下，视觉和控制都按约 `30 Hz` 跑。

## 6. 视觉链路：从相机到检测框

### 6.1 摄像头打开和读取

摄像头封装在：

- `src/vision/camera_capture.cpp`

类是：

- `vision::CameraCapture`

它负责：

1. 打开摄像头
2. 选择后端
   - Windows 下优先 `DSHOW`
   - 再回退 `CAP_ANY` 或 `MSMF`
3. 设置分辨率
4. 设置 MJPG
5. 设置缓冲区大小
6. 设置自动对焦
7. 读取帧
8. 读帧失败时做一次自动重连

当前相机配置来自 `tracker.yaml`：

- `width: 640`
- `height: 480`
- `backend: "dshow"`
- `use_mjpg: true`
- `auto_focus_enable: true`

### 6.2 检测器

检测器实现文件：

- `src/vision/yolo_detector.cpp`

类是：

- `vision::YoloDetector`

它完成的工作是：

1. 加载 ONNX 模型
2. 把图像做 blob 预处理
3. 调用 OpenCV DNN 前向推理
4. 解析输出张量
5. 做置信度筛选
6. 做 NMS
7. 返回 `core::Detection` 列表

`core::Detection` 只包含三类核心信息：

- `bbox`
- `conf`
- `cls`

### 6.3 检测器的切片策略

这个项目不是只做整图检测，还实现了局部切片推理：

- `tile_enable`
- `tile_size`
- `tile_overlap`
- `tile_max_per_frame`
- `tile_global_interval`
- `tile_priority_enable`
- `tile_priority_topk`
- `tile_priority_expand_ratio`
- `tile_priority_ttl`

逻辑是：

1. 周期性跑整图检测
2. 同时把图像切成多个 tile
3. 每帧只跑有限数量 tile
4. 如果上一帧已经有目标，就把上一帧目标附近设成 priority ROI
5. 下一帧优先跑该 ROI 附近的 tile
6. 把整图和 tile 检测结果合并，再做一次 NMS

这个设计的目的不是提高理论精度，而是在小目标场景里提高有效感受野利用率，避免整图下目标太小。

## 7. 跟踪链路：从检测框到稳定目标轨迹

### 7.1 多目标跟踪器

实现文件：

- `src/tracking/multi_object_tracker.cpp`

类是：

- `tracking::MultiObjectTracker`

当前支持两种代价风格：

- `ByteTrack`
- `BoTSORT` 风格代价

在 `tracker.yaml` 里默认是：

- `tracker_type: "bytetrack"`

### 7.2 更新逻辑

多目标跟踪器的更新流程是：

1. 先对现有 track 做预测
2. 按置信度把检测结果分成：
   - 高分检测
   - 低分检测
3. 第一阶段：
   - 用高分检测和已有 track 做匹配
4. 第二阶段：
   - 用低分检测和未匹配 track 再做一次补匹配
5. 对仍未匹配的 track 增加 `miss_count`
6. 对仍未匹配且高分的新检测创建新 track
7. 删除超过 `max_age` 的 track

输出的跟踪状态是 `core::TrackState`，其中包括：

- `track_id`
- `cx, cy`
- `w, h`
- `vx, vy`
- `vw, vh`
- `miss_count`
- `age`
- `matched_in_frame`

### 7.3 主控目标选择

文件：

- `src/control/target_selector.cpp`

类：

- `control::TargetSelector`

逻辑很直接：

1. 如果上一帧控制的 `track_id` 还存在，就继续用它
2. 否则从当前 tracks 里选离图像中心最近的目标

这意味着当前系统默认策略是“目标中心优先”，不是按类别优先，也不是按面积优先。

## 8. 自动变焦：从跟踪框面积到相机变焦命令

### 8.1 入口

自动变焦在：

- `TrackerNode::auto_zoom_step()`

它每个视觉周期执行一次，输入是：

- 当前跟踪结果 `tracks`
- 当前被控制目标 `controlled_id`

### 8.2 变焦状态机

代码里显式维护了三态：

- `Locked`
  - 当前已经有稳定目标，可以闭环调焦
- `Hold`
  - 刚丢目标，短时间保持当前变焦，不立刻乱搜
- `Search`
  - 长时间没目标，在变焦范围内往返搜索

### 8.3 变焦控制依据

变焦不是直接看 bbox 边长，而是看面积，并和期望目标面积比较。

当前做法是：

1. 取受控目标的 `bbox_hat`
2. 算当前面积 `current_area`
3. 算画面面积 `frame_area`
4. 由 `target_bbox_ratio` 反推出目标期望面积 `target_area`
5. 用 `target_area - filtered_area` 作为误差

其中 `filtered_area` 不是原始面积，而是：

1. 先对最近若干帧面积做中值滤波
2. 再做 EMA 平滑

这样做是为了避免目标框轻微抖动就触发频繁变焦。

### 8.4 防抖与安全机制

自动变焦里还有一整套稳定化机制：

1. 持续帧数判定 `zoom_persist_frames`
2. 动作最小时间间隔 `zoom_action_interval_ms`
3. 动态 deadband
4. 滞回 `zoom_hysteresis_ratio`
5. 重捕获确认帧数 `zoom_reacquire_confirm_frames`
6. 重捕获 freeze `zoom_reacquire_freeze_frames`
7. 丢目标后搜索性 zoom scan
8. 支持最大变焦后回退恢复

### 8.5 为什么控制线程要等 zoom 稳定

控制线程里有一个很重要的门控：

- 只有当 zoom 已经稳定了足够时间，才允许发电机控制命令

对应配置：

- `zoom_stable_required_ms`

原因很简单：

- 变焦变化会引起目标尺寸、像素速度、检测置信度同时变化
- 如果变焦刚动完就立刻继续控制，很容易把暂时性的视觉变化当成目标运动

所以代码里把“变焦稳定”作为控制使能条件之一。

## 9. 偏差数据是怎么从检测框变成 `dx, dy` 的

### 9.1 定义

控制环真正使用的不是 bbox 原始中心，而是预测后的目标中心偏差。

代码里偏差定义是：

- `dx_hat = cx_hat - center_x`
- `dy_hat = center_y - cy_hat`

含义是：

1. `dx_hat > 0`
   - 目标在画面中心右边
2. `dx_hat < 0`
   - 目标在画面中心左边
3. `dy_hat > 0`
   - 目标在画面中心上方
4. `dy_hat < 0`
   - 目标在画面中心下方

这里 `center_x, center_y` 来自相机配置：

- `center_x: 320`
- `center_y: 240`

### 9.2 为什么不是直接用检测框中心

如果直接用检测框中心，问题很明显：

1. 检测框每帧会抖
2. 跟踪框在短时丢帧时会断
3. 视觉到控制存在采样间隔
4. 目标在快速运动时会产生滞后

所以代码里默认走的是预测-滤波后的偏差，而不是原始检测框中心。

## 10. 偏差滤波：PNR IMM-KF 是怎么工作的

### 10.1 入口

滤波器类：

- `filter::PnrImmKf`

实现文件：

- `src/filter/pnr_imm_kf.cpp`

### 10.2 状态维度

这个滤波器内部维护 6 维状态：

- `x`
- `y`
- `vx`
- `vy`
- `ax_like`
- `ay_like`

虽然代码没有把最后两维直接命名成物理加速度，但在模型结构上它们承担的是加速度样的角色。

### 10.3 双模型 IMM

滤波器不是单模型，而是两个模型的 IMM：

1. `CV`
   - 匀速模型
2. `CA`
   - 匀加速度模型

每次循环里：

1. 按切换概率混合两个模型状态
2. 各自做预测
3. 更新各自似然
4. 再按模型概率融合成最终输出

这样做的目的很明确：

- 平稳段更偏向 CV
- 动态切换段允许 CA 占更大权重

### 10.4 鲁棒测量更新

测量更新时，代码没有直接把检测中心硬塞进滤波器，而是做了以下处理：

1. 根据置信度、丢帧数、bbox 面积、zoom 值构造质量量 `PnrFilterQuality`
2. 根据创新量和质量量，算动态的：
   - `alpha_q`
   - `alpha_r`
   - `outlier_prob`
   - `bias_x`
   - `bias_y`
3. 先对原始测量做偏置修正
4. 用 Huber 风格权重和门控概率抑制异常测量
5. 分别更新两个 IMM 子模型

即使 `filter.neural_enable=false`，这套滤波器也仍然是“鲁棒 IMM-KF”；只是其中小型神经自适应部分会关闭。

### 10.5 为什么说这是偏差处理核心

控制环最终用到的 `dx_hat, dy_hat, vx_hat, vy_hat`，在默认配置下都来自这个滤波器：

1. 先得到融合位置 `x(), y()`
2. 再得到融合速度 `vx(), vy()`
3. 再换算成图像中心偏差

所以从检测框到控制误差，中间最关键的稳定化环节就是这个滤波器。

## 11. 控制链路：从偏差到电机命令

### 11.1 控制线程输入

控制线程每次会取到：

1. 当前受控 track
2. 当前预测时间差 `pred_dt`
3. 当前偏差估计 `dx_hat, dy_hat`
4. 当前偏差速度 `vx_hat, vy_hat`
5. 当前测量是否有效
6. 当前 zoom 是否稳定

### 11.2 目标有效性判定

控制里区分三个状态：

1. `tracked`
   - 当前帧匹配到了测量
2. `coasting`
   - 当前目标存在但在短时失配中，依靠跟踪和滤波外推
3. `lost`
   - 当前目标不可用

真正允许发控制命令的是：

- `target_detected && zoom_stable`

如果目标存在但 zoom 不稳定，状态会记成：

- `zoom_guard`

### 11.3 基础 PID 控制器

基础控制器类：

- `control::Controller`

实现文件：

- `src/control/controller.cpp`

内部包含两个 PID：

- `pid_x_`
- `pid_y_`

以及两个 slew limiter：

- `slew_x_`
- `slew_y_`

控制步骤是：

1. 如果目标丢失且 `stop_when_lost=true`
   - reset PID
   - 输出零命令
2. 对 `dx_hat, dy_hat` 先做 deadband
   - 小于 `deadband_px` 的误差直接置零
3. 分别做 PID
4. 对 PID 输出做饱和限幅
   - `[-cmd_limit, cmd_limit]`
5. 再做每 tick 的 slew 限速

公式上就是：

`raw_x = kp_x * e_x + ki_x * ∫e_x dt + kd_x * de_x/dt`

`raw_y = kp_y * e_y + ki_y * ∫e_y dt + kd_y * de_y/dt`

然后再经过：

1. `cmd_limit`
2. `slew_per_tick`

### 11.4 当前 `tracker.yaml` 的 PID 参数

当前主配置里：

- `pid_x = {kp: 0.06, ki: 0.0, kd: 0.0}`
- `pid_y = {kp: 0.03, ki: 0.0, kd: 0.0}`
- `deadband_px = 10`
- `cmd_limit = 80`
- `slew_per_tick = 8`

所以当前默认并不是复杂 PID，而是偏保守的 P 控制 + deadband + 饱和 + 斜率限制。

## 12. 残差控制分支：处理后偏差数据怎么进一步修正

### 12.1 这条支路的定位

基础 PID 并不是系统最终唯一输出。代码里还留了一条残差控制支路：

1. 用当前视觉状态和基础控制命令构造特征
2. 送进 ONNX 模型
3. 让模型输出一个命令增量 `delta_cmd`
4. 再通过安全策略决定这份增量能用多少

对应类：

- `control::ResidualInferOnnx`
- `control::ResidualPolicy`

### 12.2 残差模型输入特征

当前 ONNX 分支输入 12 维特征：

1. `dx_hat`
2. `dy_hat`
3. `vx_hat`
4. `vy_hat`
5. `bbox_area`
6. `det_conf`
7. `lost_flag`
8. `is_meas_update`
9. `meas_age_ms`
10. `cmd_base_x`
11. `cmd_base_y`
12. `dt_ms`

也就是说，这个残差模型不是直接看图像，而是看“视觉估计状态 + 基础控制输出”。

### 12.3 残差策略为什么不是直接相加

残差模型输出并不会直接加到电机命令上。中间还有一层 `ResidualPolicy`：

1. 根据 `det_conf / lost_flag / is_meas_update / meas_age_ms` 算 `reliability_score`
2. 把这个分数映射成 `alpha_gate`
3. 只把一部分残差通过
4. 对残差幅值再做限幅
5. 对最终命令再做一次限速
6. 对最终命令再做一次饱和

这层策略的意义是：

- 目标不可靠时，模型残差不能乱改控制命令
- 测量陈旧时，残差必须变小
- 即使模型失真，也不能直接把电机打爆

### 12.4 当前项目里这条支路的实际状态

这条残差支路是“可选增强”，不是主链硬依赖。

如果 `model.onnx / scaler.json / feature_spec.json` 没加载成功：

- 系统会自动回退
- `delta=0`
- 只跑基础 PID + 安全限幅

所以项目能运行，不依赖这条残差 ONNX 一定存在。

## 13. 电机命令是怎么发出去的

### 13.1 执行器抽象

执行器接口：

- `actuator::MotorDriver`

有两个实现：

1. `MotorDriverMock`
   - 不发硬件
   - 只保留控制轨迹，方便调试
2. `MotorDriverPyUsb`
   - 通过 Python bridge 调 USB 电机

### 13.2 Mock 模式

如果 `actuator.mode = "mock"`：

- `send(cmd)` 什么都不发
- 但日志里仍然记录完整命令轨迹

这适合：

1. 先验证检测和控制链是否跑通
2. 离线分析 PID、残差和安全策略

### 13.3 PyUSB 模式

如果 `actuator.mode = "pyusb"`：

1. C++ 侧会启动一个 Python bridge 脚本
2. 通过标准输入把 `x y` 命令写给 bridge
3. bridge 再通过 USB 发给实际设备

这里当前不是直接在 C++ 里写 USB 协议，而是通过 Python bridge 复用已存在的硬件通讯逻辑。

### 13.4 发命令前还有哪些保护

真正发给电机前还有这些保护：

1. 目标必须有效
2. 变焦必须稳定
3. PID 自带 deadband
4. 基础控制输出有 `cmd_limit`
5. 基础控制输出有 `slew_per_tick`
6. 残差分支有可靠性门控
7. 残差分支有残差限幅
8. 残差分支有最终限速
9. 如果从允许控制切到阻塞控制，系统会主动发一次零命令

这意味着当前工程不是“检测到就直接打电机”，而是做了多层安全门控。

## 14. 日志是怎么记录的

日志类：

- `core::CsvLogger`

输出结构：

- `core::RuntimeSnapshot`

日志里会记录：

1. 图像尺寸
2. 目标框
3. `dx_hat, dy_hat`
4. `vx_hat, vy_hat`
5. 是否丢目标
6. 是否做了测量更新
7. 测量时延
8. `cmd_base`
9. `cmd_expert`
10. `cmd_sent`
11. 残差门控分数
12. 残差增量
13. clip / slew / saturation 标志
14. 当前是否真的用了残差模型
15. 当前目标数、track 数、主控 track id
16. 备注字段 `note`

当前电机反馈字段：

- `act_pos_x`
- `act_pos_y`
- `act_vel_x`
- `act_vel_y`

目前仍是占位值 `0`

也就是说，当前系统是“视觉闭环 + 控制输出记录”，但还没有把真实电机位置反馈接回主日志。

## 15. 从检测到电机的完整数据流

把整条链用一句话写清楚就是：

1. 摄像头输出一帧图像
2. 检测器得到若干 `bbox`
3. 多目标跟踪器把离散检测关联成连续 track
4. 目标选择器选出当前主控目标
5. 自动变焦根据主控目标面积调整 zoom
6. 控制线程从主控目标取出中心与速度
7. 可选地用 `PnrImmKf` 做预测与滤波
8. 得到平滑后的 `dx_hat, dy_hat, vx_hat, vy_hat`
9. 基础 PID 生成 `cmd_base`
10. 可选 ONNX 残差生成 `delta_cmd`
11. `ResidualPolicy` 做可靠性门控、残差限幅、最终限速
12. 得到 `cmd_final`
13. 如果目标有效且 zoom 稳定，则发送给电机
14. 否则复位控制器并发送零命令
15. 全部状态写入 `tracker_log.csv`

## 16. 当前工程里“偏差数据是怎么处理的”

如果只关心你最常问的核心问题：`dx,dy` 是怎么从检测结果处理成可控制数据的，可以简化成下面这个顺序：

1. 检测框中心 `cx, cy`
2. 跟踪器预测得到更连续的 `cx, cy, vx, vy`
3. `PnrImmKf` 融合当前位置、速度、测量置信度、zoom 信息
4. 生成 `cx_hat, cy_hat, vx_hat, vy_hat`
5. 转成相对图像中心的 `dx_hat, dy_hat`
6. PID 把 `dx_hat, dy_hat` 转成基础电机命令
7. 可选残差控制在可靠时做小幅校正
8. 安全策略做最终限幅与限速
9. 发给电机

所以真正送给电机的不是“检测框原始中心偏差”，而是“经过跟踪、预测、滤波、控制和安全策略后的控制命令”。

## 17. 当前 PID 控制链的局限

从代码实现角度看，当前 PID 控制链是可用的，但有这些明显边界：

1. PID 目前基本是 P 控制
   - `ki` 和 `kd` 默认是 0
2. 电机真实位置反馈还没接入主控制闭环
3. 视觉频率和控制频率当前都不高
4. zoom 稳定门控会主动牺牲一部分即时响应，换稳定性
5. 残差 ONNX 支路依赖额外模型文件，如果缺失就自动关闭

这意味着当前工程更适合：

1. 做实时视觉闭环验证
2. 做控制日志采集
3. 做控制工作点分析

它还不是一个带完整执行器反馈的最终工业闭环系统。

## 18. 当前主配置 `tracker.yaml` 的实际含义

当前配置体现的是一个“保守优先”的工作点：

1. 检测阈值高
2. 自动变焦开启
3. MOT 开启
4. 视觉滤波开启
5. 控制限速较强
6. 执行器默认 `mock`
7. 残差分支默认按配置结构存在，但能否启用取决于模型文件是否真正加载成功

因此，如果你发现：

- 画面已经检测到目标
- 但电机没动

要优先检查以下条件：

1. `actuator.mode` 是否还是 `mock`
2. `actuator.armed` 是否为 `false`
3. `zoom_stable_required_ms` 是否导致暂时阻塞控制
4. `target_detected` 是否为真
5. `deadband_px` 是否把当前偏差吃掉了

## 19. 直接运行命令

PowerShell 下最短可用命令是：

```powershell
Set-Location 'D:\kun-data\kun-code-data\反无\cpp智能控制'
$env:PATH='C:\Users\Administrator\miniconda3\envs\py310\Library\bin;' + $env:PATH
.\build\msvc-opencv-release\tracker.exe --mode cam --config .\config\tracker.yaml
```

如果还没编译，先执行：

```powershell
Set-Location 'D:\kun-data\kun-code-data\反无\cpp智能控制'
cmake --preset msvc-opencv-release
cmake --build --preset msvc-opencv-release -j 8
```

## 20. 一句话总结

这个项目的本质不是“跑一个 YOLO 模型”，而是：

一个以摄像头为输入、以目标中心偏差为中间状态、以 PID 和可选残差控制为核心、以安全限幅限速为保护、最终给电机发送二维控制命令的实时视觉控制系统。
