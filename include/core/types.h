#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace core {

// 统一的边界框表示，采用左上角 + 右下角坐标。
struct BBox {
  double x1{0.0};
  double y1{0.0};
  double x2{0.0};
  double y2{0.0};

  // 便捷几何量，检测、跟踪和控制模块都会反复使用。
  double width() const { return x2 - x1; }
  double height() const { return y2 - y1; }
  double cx() const { return 0.5 * (x1 + x2); }
  double cy() const { return 0.5 * (y1 + y2); }
};

// 单帧检测结果。
struct Detection {
  BBox bbox;
  double conf{0.0};
  int cls{0};
};

// 跟踪器输出的统一轨迹状态。
struct TrackState {
  int track_id{-1};
  BBox det_bbox;
  BBox bbox_hat;
  double det_conf{0.0};
  double cx{0.0};
  double cy{0.0};
  double w{0.0};
  double h{0.0};
  double vx{0.0};
  double vy{0.0};
  double vw{0.0};
  double vh{0.0};
  int miss_count{0};
  int age{0};
  bool matched_in_frame{false};
};

// 发送给电机驱动层的二维速度命令。
struct MotorCmd {
  double cmd_x{0.0};
  double cmd_y{0.0};
};

// 运行时快照日志。
// 该结构把视觉、跟踪、滤波、残差补偿和执行器关键状态集中到一处，
// 便于做离线分析和论文实验统计。
struct RuntimeSnapshot {
  // 主日志字段。
  std::string run_id{"run_default"};
  uint64_t timestamp_ms{0};
  int frame_id{0};
  double dt_ms{0.0};
  int img_w{0};
  int img_h{0};
  BBox bbox;
  BBox bbox_raw;
  double bbox_area_px{0.0};
  double det_conf{0.0};
  double dx_raw{0.0};
  double dy_raw{0.0};
  double dx_hat{0.0};
  double dy_hat{0.0};
  double clean_dx{0.0};
  double clean_dy{0.0};
  double vx_hat{0.0};
  double vy_hat{0.0};
  int lost_flag{1};
  int is_meas_update{0};
  double meas_age_ms{0.0};
  double zoom_value{0.0};
  double zoom_delta{0.0};
  double vision_latency_ms{0.0};
  double cmd_base_x{0.0};
  double cmd_base_y{0.0};
  double cmd_expert_x{0.0};
  double cmd_expert_y{0.0};
  double cmd_sent_x{0.0};
  double cmd_sent_y{0.0};
  double reliability_score{0.0};
  double alpha_gate{0.0};
  double stage1_switch_gate{0.0};
  double delta_cmd_x{0.0};
  double delta_cmd_y{0.0};
  int residual_clip_flag{0};
  int slew_limit_flag{0};
  int final_sat_flag{0};
  int infer_used_model{0};
  int fallback_delta_zero{1};
  std::string infer_status;
  double act_pos_x{0.0};
  double act_pos_y{0.0};
  double act_vel_x{0.0};
  double act_vel_y{0.0};

  // 兼容/调试字段。
  int det_count{0};
  int track_count{0};
  int controlled_id{-1};
  int coast_count{0};
  std::string note;
};

}  // namespace core
