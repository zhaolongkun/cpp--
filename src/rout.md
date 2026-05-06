# 项目主流程说明

## 1. 系统整体链路

当前项目的主链路可以概括为：

```text
main.cpp
  -> 读取配置
  -> 构造 TrackerNode
  -> init()
      -> 初始化时序补偿模型
      -> 初始化输入源（相机或回放）
      -> 初始化 YOLO 检测器
      -> 初始化多目标跟踪器
      -> 初始化目标选择器
      -> 初始化滤波器
      -> 初始化控制器
      -> 初始化电机驱动
      -> 初始化日志
  -> run()
      -> 启动 vision_loop 视觉线程
      -> 启动 control_loop 控制线程
```

运行时的数据流为：

```text
图像/回放数据
  -> YOLO 检测
  -> 多目标跟踪
  -> 目标选择
  -> 写入 SharedState
  -> 控制线程读取目标状态
  -> 位置滤波与误差计算
  -> 时序补偿模型
  -> 速度控制器
  -> 电机命令发送
  -> 写日志
```

---

## 2. 程序入口

入口文件：

- `src/main.cpp`

`main.cpp` 的职责很单一：

1. 解析命令行参数
2. 读取 `config/tracker.yaml`
3. 构造 `TrackerNode`
4. 调用 `init()`
5. 调用 `run()`

也就是说，`main.cpp` 本身不做检测、跟踪和控制计算，它只是负责把整个系统启动起来。

---

## 3. TrackerNode 是整个系统的总调度器

核心文件：

- `include/app/tracker_node.h`
- `src/app/tracker_node.cpp`

`TrackerNode` 串起了这些模块：

- 相机采集：`vision::CameraCapture`
- 目标检测：`vision::YoloDetector`
- 多目标跟踪：`tracking::MultiObjectTracker`
- 目标选择：`control::TargetSelector`
- 控制器：`control::Controller`
- 时序补偿器：`control::TemporalCompensatorOnnx`
- 位置滤波器：`filter::PnrImmKf`
- 电机驱动：`actuator::MotorDriver`
- 日志器：`core::CsvLogger`

可以把 `TrackerNode` 理解为“把视觉链路、控制链路和执行链路接成闭环的总控制节点”。

---

## 4. 当前项目已经切换到的新算法

当前控制补偿链路已经不是旧的 `stage1_clean + residual` 两阶段方案。

现在使用的是单模型时序补偿方案：

```text
dx_raw, dy_raw
  -> 因果滑动均值
  -> 得到可靠基准 e_ref
  -> 构造特征 [e_ref, d_ref]
  -> CausalCNN+GRU ONNX 推理
  -> 预测 delta
  -> 有界融合
  -> 得到最终控制误差 u
  -> 送入速度控制器
```

其中：

- `e_ref`：由原始偏差经过因果滑动均值得到的可靠基准
- `d_ref`：相邻两次 `e_ref` 的差分
- `delta`：时序网络预测的一帧补偿增量
- `u = e_ref + alpha * clip(delta)`：最终送入控制器的融合误差

这套逻辑的目标是：

1. 保留工程基线的稳定性
2. 用预测增量补偿视觉链路的一帧滞后
3. 避免直接用神经网络完全替代基线控制

对应文件：

- `include/control/temporal_compensator_onnx.h`
- `src/control/temporal_compensator_onnx.cpp`

---

## 5. 初始化流程

初始化逻辑位于：

- `src/app/tracker_node.cpp` 的 `TrackerNode::init()`

### 5.1 初始化时序补偿模型

程序启动时会优先读取：

- `temporal_comp.model_path`
- `temporal_comp.stats_path`

也支持用环境变量覆盖：

- `TEMPORAL_MODEL_ONNX`
- `TEMPORAL_STATS_JSON`

如果模型和统计文件加载成功，会启用 ONNX 推理。

如果加载失败，程序不会崩溃，而是自动退回到“仅使用因果基线 `e_ref`”模式继续运行。

### 5.2 初始化输入源

程序支持两种运行模式：

1. `cam`
   直接打开相机
2. `replay`
   从 CSV 文件回放检测结果

### 5.3 初始化检测器

检测器在：

- `src/vision/yolo_detector.cpp`

负责加载 YOLO 模型并输出当前帧的检测框列表。

### 5.4 初始化电机驱动

根据配置项：

- `actuator.mode`

决定使用哪种电机驱动：

- `mock`：模拟模式，不向真实硬件发命令
- `pyusb`：通过 USB bridge 向真实电机发送命令

### 5.5 初始化日志

日志写入由：

- `core::CsvLogger`

负责，默认输出到：

- `logs/tracker_log.csv`

---

## 6. 视觉线程 vision_loop 的流程

视觉线程主要负责“感知前端”。

对应文件：

- `src/app/tracker_node.cpp`

### 6.1 获取输入

如果是 `cam` 模式：

- 从相机读取图像帧

如果是 `replay` 模式：

- 从回放 CSV 读取当前帧检测结果

### 6.2 YOLO 检测

检测器输出当前帧的多个 `Detection`。

对应文件：

- `src/vision/yolo_detector.cpp`

### 6.3 多目标跟踪

跟踪器主实现位于：

- `src/tracking/multi_object_tracker.cpp`

默认算法是 `ByteTrack` 风格的多目标跟踪。

配置位置：

- `config/tracker.yaml`

默认配置：

```yaml
tracker_type: "bytetrack"
```

如果改成：

```yaml
tracker_type: "botsort"
```

则会切到 `BoT-SORT` 风格的联合代价匹配。

### 6.4 跟踪器内部做了什么

`MultiObjectTracker::update()` 的核心步骤是：

1. 对现有轨迹做状态预测
2. 把检测框分成高分框和低分框
3. 高分框优先匹配已有轨迹
4. 低分框补充匹配未命中的轨迹
5. 未匹配的高分框生成新轨迹
6. 长时间丢失的轨迹删除

这就是当前跟踪部分的主体逻辑。

### 6.5 轨迹状态预测

每条轨迹由：

- `src/tracking/track.cpp`

管理，其内部状态预测和更新在：

- `src/tracking/kalman_bbox.cpp`

这里使用了简化的 `KalmanBBox` 状态模型，来维持轨迹在短时抖动或漏检时的连续性。

### 6.6 目标选择

目标选择器在：

- `src/control/target_selector.cpp`

它负责从多条轨迹中选出当前真正用于控制的目标。

### 6.7 自动变焦

视觉线程里还会运行：

- `auto_zoom_step()`

它根据目标框大小和跟踪状态控制相机变焦。

### 6.8 写入共享状态

视觉线程最终会把这些信息写入 `SharedState`：

- 当前帧号
- 检测结果
- 跟踪结果
- 当前控制目标 ID
- 当前时间戳
- 当前 zoom 值

这些共享数据供控制线程读取。

---

## 7. 控制线程 control_loop 的流程

控制线程主要负责“从目标状态生成电机命令”。

对应文件：

- `src/app/tracker_node.cpp`

### 7.1 读取共享状态

控制线程先从 `SharedState` 中取出：

- 当前轨迹列表
- 当前控制目标 ID
- 当前视觉时间戳
- 当前 zoom 状态

然后定位当前被控目标。

### 7.2 位置滤波

如果：

- `filter.enable = true`

则会调用：

- `src/filter/pnr_imm_kf.cpp`

中的 `PnrImmKf`

对目标位置做滤波和状态估计，输出更稳定的目标中心位置。

### 7.3 计算图像偏差

控制线程会把目标中心与图像中心比较，得到：

- `dx_raw`
- `dy_raw`

它们表示目标相对图像中心的水平和垂直偏差。

这是后续时序补偿和速度控制的基础输入。

### 7.4 时序补偿模块

当前新算法在这里生效。

对应实现：

- `src/control/temporal_compensator_onnx.cpp`

它的处理流程是：

1. 对 `dx_raw, dy_raw` 做因果滑动均值，得到 `e_ref`
2. 根据当前 `e_ref` 和上一时刻 `e_ref` 计算 `d_ref`
3. 构造时序特征 `[e_ref_x, e_ref_y, d_ref_x, d_ref_y]`
4. 等历史序列长度满足 `seq_len` 后，送入 ONNX 模型
5. 模型输出预测增量 `delta`
6. 对 `delta` 做 `clip`
7. 用 `u = e_ref + alpha * clip(delta)` 得到最终误差

如果模型未加载成功，或者还处于 warmup 阶段，则退回只使用 `e_ref`。

### 7.5 速度控制器

速度控制器位于：

- `src/control/controller.cpp`

当前控制器的逻辑不是“直接固定比例输出”，而是：

1. 先检查死区 `deadband_px`
2. 计算目标距离图像中心的距离 `distance_px`
3. 计算归一化方向
4. 根据距离比例计算速度比例
5. 根据最大速度得到目标速度
6. 根据方向分解出 `cmd_x` 和 `cmd_y`
7. 做最大速度限幅
8. 再经过 `SlewLimiter` 限制变化率

因此当前控制器的含义很清楚：

- 方向由偏差方向决定
- 速度大小由偏差距离决定
- 离中心越远，电机速度越大
- 最终速度不超过 `cmd_limit`

### 7.6 电机命令发送

控制线程最终得到：

- `cmd_sent`

然后调用：

- `motor_->send(cmd_sent)`

把命令发给电机驱动层。

### 7.7 安全保护

如果目标丢失，或者 zoom 状态不允许控制，则控制线程会停止正常控制输出，必要时发送零命令，避免电机乱转。

### 7.8 日志记录

控制线程最后会把运行状态写入 CSV，包括：

- 目标偏差
- 时序补偿输出
- 控制命令
- 是否用了模型
- 当前推理状态
- 跟踪状态

这些日志用于后续调试、回放和论文实验分析。

---

## 8. 电机命令是怎么发出去的

驱动接口定义在：

- `include/actuator/motor_driver.h`

真实发送逻辑在：

- `src/actuator/motor_driver_pyusb.cpp`

流程如下：

1. 接收 `cmd_x, cmd_y`
2. 根据 `scale_x, scale_y` 做缩放
3. 根据 `invert_x, invert_y` 判断是否反向
4. 用 `speed_limit` 再做一次最终限幅
5. 通过 USB bridge 把整数速度发给电机

如果当前模式是：

- `mock`

则不会向真实硬件发送命令，只做模拟输出。

---

## 9. 当前默认配置的含义

按照当前 `config/tracker.yaml`，默认关键配置是：

### 9.1 默认跟踪算法

```yaml
tracker_type: "bytetrack"
```

表示默认使用 `ByteTrack` 风格跟踪。

### 9.2 默认时序补偿配置

```yaml
temporal_comp:
  enable: true
  window_size: 2
  seq_len: 8
  alpha: 0.4
  delta_max: 10.0
```

含义是：

- 启用时序补偿
- 用长度为 2 的因果窗口构造基准
- 用长度为 8 的时序序列做推理
- 用 `alpha=0.4` 控制预测增量权重
- 用 `delta_max=10` 限制单次预测补偿幅度

### 9.3 默认控制器配置

控制器仍然使用速度型控制，关键参数包括：

- `deadband_px`
- `cmd_limit`
- `full_speed_px`
- `slew_per_tick`

### 9.4 默认电机模式

```yaml
actuator.mode: "mock"
```

表示默认不向真实电机发命令。

---

## 10. 现在最关键的源码文件

如果只看主干流程，最重要的是这些文件：

- `src/main.cpp`
  程序入口

- `src/app/tracker_node.cpp`
  总调度器，串起视觉线程和控制线程

- `src/vision/yolo_detector.cpp`
  YOLO 检测器

- `src/tracking/multi_object_tracker.cpp`
  多目标跟踪主逻辑

- `src/tracking/track.cpp`
  单条轨迹对象

- `src/tracking/kalman_bbox.cpp`
  轨迹状态预测与更新

- `src/control/target_selector.cpp`
  当前控制目标选择

- `src/filter/pnr_imm_kf.cpp`
  位置滤波与状态估计

- `src/control/temporal_compensator_onnx.cpp`
  新的时序补偿模型

- `src/control/controller.cpp`
  电机速度命令生成

- `src/actuator/motor_driver_pyusb.cpp`
  真实电机发送

---

## 11. 已经废弃的旧控制补偿链路

以下旧方案已经不再是当前主流程的一部分：

- `Stage1CleanInferOnnx`
- `ResidualInferOnnx`
- `ResidualPolicy`

旧的两阶段“清洗 + 残差”链路已经被新的单模型时序补偿方案替代。

因此，理解当前项目时，应以：

```text
可靠基准 e_ref + 时序网络预测 delta + 有界融合
```

作为主线，而不是旧的残差补偿框架。

---

## 12. 一句话总结整个项目

这个项目本质上是一个“YOLO 目标检测 + 多目标跟踪 + 目标选择 + 时序补偿 + 速度控制 + 电机执行”的闭环视觉伺服系统。

更具体地说，它先通过 YOLO 找到目标，再用 ByteTrack 风格的跟踪器保持轨迹连续性，然后选出当前需要控制的目标；控制线程根据目标相对图像中心的偏差，构造可靠基准，并用 CausalCNN+GRU 预测一帧补偿增量，再经过有界融合和速度控制器生成最终电机速度命令，最后把命令发给云台电机执行。
