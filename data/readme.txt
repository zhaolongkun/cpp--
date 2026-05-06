## 数据处理流程（不开启算法时）

检测框 → 融合数据的完整处理链：

1. YOLO检测输出：det_dx = cx - center_x，det_dy = center_y - cy
2. PnrImmKf卡尔曼滤波：对cx/cy做状态估计，输出dx_hat/dy_hat
3. track_prefilter：3帧均值滤波，输出tracker_signal_dx/dy
4. kMaxStepPx限幅：单帧跳变超过60px截断
5. offset_stage1 EMA平滑：
   - x轴：alpha=0.35，offset_stage1_dx_ = 0.35*tracker_signal_dx + 0.65*prev
   - y轴：alpha=0.15，offset_stage1_dy_ = 0.15*tracker_signal_dy + 0.85*prev
6. temporal_comp禁用时直接透传：dx_ctrl = baseline_dx，dy_ctrl = baseline_dy
7. 最终输出：fused_dx = dx_ctrl，fused_dy = dy_ctrl

## 不开启算法时，fused_dx/dy 等价于哪个数据

不开启算法（temporal_comp.enable: false）时：
- fused_dx = offset_stage1_dx_（EMA alpha=0.35 平滑后的卡尔曼估计x误差）
- fused_dy = offset_stage1_dy_（EMA alpha=0.15 平滑后的卡尔曼估计y误差）

即 fused_dx/dy 与 first_filter_dx/dy 列数据一致。

## 文件说明

- track-fusion-move.csv：完整采集数据，temporal_comp禁用，包含所有中间处理列
- track-fusion-move-baseline.csv：基线数据备份，直接用fused_dx/dy控制电机，无算法补偿
- track-fusion-move-baseline_new.csv：仅保留处理流程相关列的精简版基线数据
