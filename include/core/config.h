#pragma once

#include <string>

namespace core {

struct PIDGains {
  double kp{0.0};
  double ki{0.0};
  double kd{0.0};
};

struct ModelConfig {
  std::string path{"D:/kun-data/kun-code-data/run/yolo12n/weights/best.pt"};
  double conf_detect{0.30};
  double conf_track{0.60};
  bool tile_enable{false};
  bool bbox_size_filter_enable{false};
  double bbox_size_filter_alpha{0.35};
  double bbox_size_filter_min_scale{0.85};
  double bbox_size_filter_max_scale{1.15};
  double bbox_size_filter_center_gate_px{80.0};
  double bbox_size_filter_center_deadband_px{2.0};
  double bbox_size_filter_size_deadband_px{2.0};
  double bbox_size_filter_max_center_step_px{8.0};
  double bbox_size_filter_max_size_step_px{6.0};
  int tile_size{960};
  double tile_overlap{0.25};
  int tile_max_per_frame{2};
  int tile_global_interval{6};
  bool tile_priority_enable{true};
  int tile_priority_topk{1};
  double tile_priority_expand_ratio{2.0};
  int tile_priority_ttl{8};
  bool use_shm_detector{false};
};

struct CameraConfig {
  int index{0};
  int width{640};
  int height{480};
  int center_x{320};
  int center_y{240};
  std::string backend{"any"};
  bool show_window{false};
  bool show_coasting{false};
  bool use_mjpg{true};
  bool auto_focus_enable{true};
  bool auto_zoom_enable{false};
  double target_bbox_ratio{0.12};
  double bbox_ratio_deadband{0.015};
  double bbox_ratio_alpha{0.25};
  int zoom_persist_frames{6};
  double zoom_step{1.0};
  double zoom_log_gain{0.20};
  double zoom_min_step{1.0};
  double zoom_max_step{12.0};
  double zoom_min{0.0};
  double zoom_max{260.0};
  bool show_zoom_debug{true};
  double zoom_fps_trigger{18.0};
  double zoom_trend_trigger{0.004};
  double zoom_area_deadband_px{100.0};
  double zoom_hysteresis_ratio{1.6};
  int zoom_action_interval_ms{250};
  int zoom_stable_required_ms{400};
  int zoom_median_window{5};
  double zoom_search_step{10.0};
  int zoom_search_every_n{3};
  bool zoom_max_lost_recover_enable{true};
  int zoom_max_lost_trigger_frames{8};
  double zoom_recover_step{15.0};
  bool zoom_out_enable{false};
  double zoom_out_trigger_mult{1.8};
  int zoom_hold_miss_frames{10};
  int zoom_reacquire_freeze_frames{6};
  int zoom_reacquire_confirm_frames{3};
};

struct MotConfig {
  int max_age{15};
  double iou_gate{0.25};
  std::string tracker_type{"bytetrack"};
  bool use_bytetrack{true};
  double track_high_thresh{0.6};
  double track_low_thresh{0.1};
  double new_track_thresh{0.7};
  double second_match_iou{0.15};
  double botsort_iou_weight{0.75};
  double botsort_center_weight{0.25};
  double botsort_max_center_dist_ratio{0.55};
};

struct ControlConfig {
  int rate_hz{250};
  double deadband_px{10.0};
  double cmd_limit{200.0};
  double full_speed_px{200.0};
  double slew_per_tick{30.0};
  PIDGains pid_x{0.06, 0.0, 0.0};
  PIDGains pid_y{0.03, 0.0, 0.0};
  bool stop_when_lost{true};
};

struct ActuatorConfig {
  std::string mode{"mock"};
  bool armed{false};
  std::string python_exe{"python"};
  std::string bridge_script{"tools/motor_usb_bridge.py"};
  int x_vendor_id{1155};
  int x_product_id{22288};
  int y_vendor_id{1156};
  int y_product_id{22289};
  bool invert_x{false};
  bool invert_y{false};
  double scale_x{1.0};
  double scale_y{1.0};
  int speed_limit{200};
  bool send_zero_on_close{true};
  bool debug{false};
  bool y_pitch_guard_enable{false};
  std::string y_pitch_port{"COM7"};
  int y_pitch_baud{115200};
  double y_pitch_min_deg{-60.0};
  double y_pitch_max_deg{0.0};
  double y_pitch_upper_stop_deg{-3.0};
  double y_pitch_lpf_alpha{0.20};
  double y_pitch_release_margin_deg{3.0};
  int y_pitch_timeout_ms{250};
  bool y_pitch_debug{false};
  bool y_pitch_positive_increases_angle{true};
  double y_pitch_soft_zone_deg{2.0};
  bool y_pitch_takeover_enable{true};
  double y_pitch_takeover_target_deg{-20.0};
  double y_pitch_takeover_speed{1.0};
  int y_pitch_takeover_interval_ms{20};
  int y_pitch_session_idle_ms{150};
  bool y_pitch_jump_guard_enable{true};
  double y_pitch_max_step_deg{4.0};
  double y_pitch_jump_speed_scale{0.50};
  double y_pitch_jump_speed_cap{20.0};
};

struct FilterConfig {
  bool enable{true};
  bool neural_enable{true};
  bool meas_only_when_matched{true};
  bool offset_lpf_enable{true};
  double offset_lpf_alpha{0.35};
  double offset_lpf_alpha_y{0.15};
  double switch_prob{0.04};
  double gate_chi2{9.21};
  double huber_c{2.5};
  double q_pos{180.0};
  double q_vel{80.0};
  double q_acc{25.0};
  double r_pos{144.0};
  double alpha_q_min{0.35};
  double alpha_q_max{3.00};
  double alpha_r_min{0.35};
  double alpha_r_max{4.50};
  double bias_limit_px{4.0};
  double outlier_prob_min{0.05};
  double outlier_prob_max{0.95};
};

struct ResidualConfig {
  bool enable{true};
  double w_conf{0.45};
  double w_lost{0.25};
  double w_meas{0.15};
  double w_age{0.15};
  double age_tau_ms{200.0};
  double meas_age_hard_ms{300.0};
  double gate_r_low{0.35};
  double gate_r_high{0.75};
  double alpha_smooth{0.70};
  double residual_limit_ratio{0.35};
  double residual_min_scale{0.30};
  double final_slew_per_tick{30.0};
};

struct TemporalCompConfig {
  bool enable{true};
  int window_size{2};
  int seq_len{8};
  double alpha{0.4};
  double delta_max{10.0};
  std::string model_path{"妯″瀷-new2/checkpoints/causal_cnn_gru.onnx"};
  std::string stats_path{"妯″瀷-new2/checkpoints/stats.json"};
};

struct LogConfig {
  bool enable{true};
  std::string path{"logs/tracker_log.csv"};
  bool dedup_by_frame_id{true};
  std::string profile{"full"};
};

struct AppConfig {
  ModelConfig model;
  CameraConfig camera;
  MotConfig mot;
  ControlConfig control;
  ActuatorConfig actuator;
  FilterConfig filter;
  ResidualConfig residual;
  TemporalCompConfig temporal_comp;
  LogConfig log;
};

class ConfigLoader {
 public:
  static AppConfig load_or_default(const std::string& path);
};

}  // namespace core
