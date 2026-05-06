================================================================================
给AI的指令文档（SCI一区目标·修订版）：YOLO12检测 + 多目标跟踪ID + 学习可靠性模块
           + 鲁棒IMM多速率估计（CRME++）+ 安全控制层 + 250Hz控制（无IMU）
           C++11实现 + 仅使用CMake编译（不使用Shell脚本）
           深度学习推理：仅使用 .pt（TorchScript/LibTorch），不使用ONNX/onnxruntime
================================================================================
【用途】
把这份TXT直接喂给AI，让AI按此方案为我的工程生成/修改 C++ 代码与实验脚本。
要求：工程可运行、可复现、强对比/消融实验完整；论文具备SCI一区档次的创新点（≥3点）。

================================================================================
0. 项目现状（必须理解）
================================================================================
- 相机：30Hz（真实帧率固定）
- 检测模型：YOLO12（Ultralytics系）
  权重路径（.pt，Windows记录）：D:\kun-data\kun-code-data\run\yolo12n\weights\best.pt
- 当前关键问题：
  P1 掉检：第一帧检测到，后面连续10帧检测不到，然后又检测到
  P2 跳变/翻转：数据可能上一时刻100，下一时刻-100（强外点/误检/目标切换/误关联）
  P3 低频测量导致控制卡顿：相机30Hz，但控制希望200~300Hz连续输入（不能生成高帧率图像）
  P4 无IMU：不能做角速度反馈内环，控制稳定性更依赖估计与安全策略
- 目标：在P1~P4存在时仍保证：目标持续被框住（带ID）、控制输入连续、抖动小、反向尖峰少、恢复快。

================================================================================
1. 论文的三大“创新模块”（必须落地到代码与实验）
================================================================================
注意：深度学习推理全部使用 LibTorch + TorchScript（.pt），不使用ONNX。

-------------------------------------------------------------------------------
创新模块①：ReliabilityNet（学习辅助测量可靠性模块，TorchScript .pt推理）
-------------------------------------------------------------------------------
【作用是什么】
给每一个“检测测量”输出可靠性：
- p_outlier ∈ [0,1]：外点/跳变/误检概率
- R_t：测量噪声（sigma_x, sigma_y → R_x=sigma_x^2, R_y=sigma_y^2）

【替换/升级了什么】
- 替换手工异常检测：3σ、固定阈值跳变判定（由学习模块预测外点概率）
- 替换手工R(conf,area)经验函数（由学习模块输出测量噪声）

【对系统问题的贡献】
- 针对P2（100→-100跳变）：外点被识别后skip update或强降权，减少估计被拉飞与反向控制尖峰
- 针对P1（掉检后误检重捕获）：重捕获初期常不稳，网络输出更大R/更高p_outlier避免误更新
- 针对P4（无IMU）：没有角速度反馈更怕误测量，ReliabilityNet降低错误测量进入控制回路的概率

【实现要求（C++推理）】
- 训练在Python完成，然后导出 TorchScript：
  reliability_net_ts.pt
- C++用 LibTorch 读取并推理：
  torch::jit::script::Module module = torch::jit::load("reliability_net_ts.pt");
- 输入张量 shape = (1, T, F) float32
- 输出：p_outlier（标量）、sigma_x、sigma_y（标量）

-------------------------------------------------------------------------------
创新模块②：CRME++（控制导向鲁棒多速率估计器：IMM + 鲁棒更新 + 250Hz predict-only）
-------------------------------------------------------------------------------
【作用是什么】
把30Hz、可能掉检/跳变的检测观测，变成250Hz连续的控制误差输入 dx_hat(t), dy_hat(t)。

组成：
- 多速率：30Hz测量更新 + 250Hz预测传播（predict-only）
- IMM多模型滤波（至少2模型）：
  M1：平稳模型（小Q）
  M2：机动模型（大Q或CA）
- 鲁棒更新：chi2 gating（硬拒绝）+ Huber/Student-t（软降权）
- 与ReliabilityNet联动：p_outlier高→skip update；R_t→自适应更新强度

【对系统问题的贡献】
- P3：250Hz predict-only 输出连续dx_hat/dy_hat，控制输入不再每帧突跳
- P1：掉检期间仍可predict维持状态（max_age内不断控）
- P2：IMM在平稳/机动间切换 + 鲁棒更新防外点拉飞
- P4：无IMU情况下用更强估计替代内环缺失的一部分“稳定性来源”

-------------------------------------------------------------------------------
创新模块③：Safety Shield（安全控制层）
-------------------------------------------------------------------------------
【作用是什么】
防止跳变/低可信/掉检导致的“反向猛拉、饱和、震荡”：
- 模式：NORMAL / SAFE_HOLD / SAFE_SLEW / RECOVER / LOST
- 根据：p_outlier、miss_count、jump_event、残差等控制输出保护
- 对PID输出做冻结、限速、渐进恢复、丢目标停机

【对系统问题的贡献】
- P2：减少反向尖峰（ReverseSpike）与饱和比例（SaturationRatio）
- P1：coasting/重捕获阶段更稳，减少恢复瞬间振荡
- P4：无IMU时安全层显著降低震荡概率

================================================================================
2. 总体系统流程（必须按此实现）
================================================================================
30Hz线程（视觉线程）：
  camera frame →
  YOLO12 detections（.pt，通过LibTorch或OpenCV DNN/TensorRT二选一，优先LibTorch） →
  多目标关联（IoU+Hungarian + gating，可选运动门控） →
  对每个匹配 measurement 构造序列特征 →
  ReliabilityNet（TorchScript .pt）推理得到 p_outlier, R_x, R_y →
  CRME++(IMM) 对 track 更新（或skip update） →
  维护 tracks（带track_id、miss_count、状态） →
  选择 controlled_id（中心最近且保持锁定优先） →
  可视化：bbox + track_id + controlled标记

250Hz线程（控制线程）：
  读取 controlled track 快照（线程安全） →
  CRME++ predict-only(dt=0.004) 得到 cx_hat, cy_hat →
  dx_hat = cx_hat - center_x; dy_hat = cy_hat - center_y →
  PID输出 cmd →
  Safety Shield 对 cmd 保护（hold/slew/recover/stop） →
  motor_driver.send(cmd)

================================================================================
3. C++工程架构（仅CMake，不使用Shell；C++11标准）
================================================================================
smart_gimbal_tracker_cpp/
├─ CMakeLists.txt
├─ config/
│  └─ tracker.yaml
├─ models/
│  ├─ yolo12_best.pt                 # YOLO12权重（推理用）
│  └─ reliability_net_ts.pt          # 可靠性网络TorchScript
├─ include/
│  ├─ core/ (types.h time_utils.h logger.h config.h)
│  ├─ vision/ (camera_capture.h yolo_detector.h bbox_utils.h)
│  ├─ tracking/
│  │   ├─ kalman_bbox.h
│  │   ├─ imm_filter.h
│  │   ├─ jump_detector.h
│  │   ├─ track.h
│  │   ├─ hungarian.h
│  │   └─ multi_object_tracker.h
│  ├─ learning/
│  │   ├─ reliability_net.h          # LibTorch推理封装（加载 reliability_net_ts.pt）
│  │   └─ feature_builder.h          # 构造(1,T,F)特征张量
│  ├─ control/
│  │   ├─ pid.h
│  │   ├─ slew_limiter.h
│  │   ├─ safety_shield.h
│  │   ├─ target_selector.h
│  │   └─ controller.h
│  ├─ actuator/
│  │   ├─ motor_driver.h
│  │   └─ motor_driver_mock.h
│  └─ app/
│     └─ tracker_node.h
├─ src/
│  ├─ ... 对应实现 .cpp
│  └─ main.cpp
└─ tools/
   ├─ train_reliability.py           # Python训练+导出 TorchScript
   ├─ export_torchscript.py          # 把PyTorch模型导出 reliability_net_ts.pt
   └─ eval_logs.py                   # 评测指标（不参与编译）

================================================================================
4. CMake编译要求（必须满足；C++11；不使用Shell）
================================================================================
- C++11（明确设置 CMAKE_CXX_STANDARD 11）
- 依赖：
  - OpenCV（相机与可视化）
  - Eigen3（滤波器矩阵）
  - yaml-cpp（读取tracker.yaml；若缺失给默认值）
  - LibTorch（必须，用于YOLO12 .pt推理 + ReliabilityNet .pt推理）
- 输出可执行：tracker

用户编译方式（写进README）：
  mkdir build
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j

运行方式：
  ./build/tracker --config config/tracker.yaml --mode cam
  ./build/tracker --config config/tracker.yaml --mode replay --dets data/detections.csv

注：replay模式用于论文对比，可不依赖YOLO推理后端。

================================================================================
5. 深度学习推理说明（仅.pt，不用ONNX）
================================================================================
5.1 YOLO12推理（C++）
- 优先实现两种模式（至少完成其中一种）：
  A) replay：从 detections.csv 回放（必须，用于论文复现）
  B) torchscript：用LibTorch加载 yolo12_best.pt（若该pt不是TorchScript，需要先在Python导出）
提示：
- Ultralytics训练出的best.pt通常不是TorchScript格式，C++直接load可能失败。
- 必须提供 Python 导出脚本：export_yolo_torchscript.py
  输出：models/yolo12_ts.pt
- C++端加载 yolo12_ts.pt 推理

5.2 ReliabilityNet推理（C++）
- Python训练后导出：models/reliability_net_ts.pt
- C++端使用 LibTorch 推理输出 p_outlier, sigma_x, sigma_y

================================================================================
6. 统一日志schema（用于创新点3：COTB++）
================================================================================
logs/tracker_log.csv 字段至少包含：
time_ns, control_tick, frame_seq,
det_count, track_count,
controlled_id, controlled_valid, miss_count,
dx_hat, dy_hat,
cmd_x, cmd_y,
p_outlier, R_x, R_y,
mode(NORMAL/SAFE_HOLD/SAFE_SLEW/RECOVER/LOST),
jump_event(0/1)

tools/eval_logs.py 必须输出：
- RMSE/MAE/P95
- JumpRate、FlipCount
- ReverseSpike、SaturationRatio
- CoastMax、T_recover、Peak_recover
- IDSW（controlled_id切换次数）

================================================================================
7. 对比/消融实验设计（必须）
================================================================================
Baseline链：
B0 YOLO-only + 30Hz PID
B1 YOLO + SORT(CV-KF) + 30Hz PID
B2 YOLO + ByteTrack(标准) + 30Hz PID
B3 B2 + 250Hz predict-only（无ReliabilityNet，无IMM，无Safety）

消融链（证明三创新模块贡献）：
O1 = B3 + ReliabilityNet（学p_outlier与R） + 单KF（无IMM）    → 证明创新模块①
O2 = O1 + IMM（CRME++）                                         → 证明创新模块②
O3 = O2 + Safety Shield（最终）                                 → 证明创新模块③

================================================================================
8. AI最终交付要求（必须）
================================================================================
AI必须输出：
- 完整可编译C++11工程（include/src + CMakeLists.txt）
- replay模式必须可运行（用于论文可复现实验）
- YOLO TorchScript导出脚本（若需要）：export_yolo_torchscript.py
- ReliabilityNet训练+TorchScript导出脚本：train_reliability.py + export_torchscript.py
- tools/eval_logs.py 输出COTB++全指标表
- README：CMake编译与运行命令（不使用Shell）

================================================================================
【结束】
该文档明确：C++11、仅CMake和g++编译、深度学习仅使用TorchScript/LibTorch的.pt模型。
并清晰说明三个创新模块分别做什么、替换什么、解决什么问题，以及如何通过消融实验证明贡献。
================================================================================



