// TrackerNode 是系统的总调度节点：
// 它把相机/回放、检测、跟踪、选目标、滤波、控制、电机输出和日志串成一个完整闭环。
#include "app/tracker_node.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include "core/time_utils.h"
#include "vision/bbox_utils.h"

namespace {

// 启动阶段的可选追踪日志，主要用于排查 init 阶段卡死或失败。
void startup_trace(const std::string& msg) {
  const char* trace_path = std::getenv("TRACKER_STARTUP_TRACE");
  if (trace_path == nullptr || trace_path[0] == '\0') {
    return;
  }
  std::ofstream ofs(trace_path, std::ios::app);
  if (!ofs.is_open()) {
    return;
  }
  ofs << msg << '\n';
}

// 回放 CSV 的最小分割函数，满足当前日志/检测文件格式即可。
std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::string token;
  std::stringstream ss(line);
  while (std::getline(ss, token, ',')) {
    out.push_back(token);
  }
  return out;
}

// 以下两个工具函数用于健壮地解析回放 CSV。
bool parse_int(const std::string& s, int& out) {
  try {
    out = std::stoi(s);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_double(const std::string& s, double& out) {
  try {
    out = std::stod(s);
    return true;
  } catch (...) {
    return false;
  }
}

double signed_y_speed_for_pitch_error(double pitch_error_deg, double speed_mag, bool positive_increases_angle) {
  const double mag = std::abs(speed_mag);
  if (mag == 0.0 || pitch_error_deg == 0.0) {
    return 0.0;
  }
  const bool need_increase_angle = pitch_error_deg > 0.0;
  return (need_increase_angle == positive_increases_angle) ? mag : -mag;
}

double clamp_y_away_from_upper_limit(double cmd_y, bool positive_increases_angle) {
  return positive_increases_angle ? std::min(cmd_y, 0.0) : std::max(cmd_y, 0.0);
}

double clamp_y_away_from_lower_limit(double cmd_y, bool positive_increases_angle) {
  return positive_increases_angle ? std::max(cmd_y, 0.0) : std::min(cmd_y, 0.0);
}

double median_from_window(const std::deque<double>& window) {
  if (window.empty()) {
    return 0.0;
  }
  std::vector<double> values(window.begin(), window.end());
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

double mean_from_window(const std::deque<double>& window) {
  if (window.empty()) {
    return 0.0;
  }
  double sum = 0.0;
  for (double value : window) {
    sum += value;
  }
  return sum / static_cast<double>(window.size());
}

double bbox_center_distance_px(const core::BBox& a, const core::BBox& b) {
  const double dx = a.cx() - b.cx();
  const double dy = a.cy() - b.cy();
  return std::sqrt(dx * dx + dy * dy);
}

void smooth_detection_bbox_size(std::vector<core::Detection>& detections, const core::TrackState* hint,
                                const core::ModelConfig& cfg, bool& filter_ready, int& filter_track_id,
                                std::deque<double>& cx_window, std::deque<double>& cy_window,
                                std::deque<double>& w_window, std::deque<double>& h_window, double& filter_cx,
                                double& filter_cy, double& filter_w, double& filter_h) {
  constexpr size_t kBboxPreFilterWindow = 8;
  auto reset_filter_state = [&]() {
    filter_ready = false;
    filter_track_id = -1;
    cx_window.clear();
    cy_window.clear();
    w_window.clear();
    h_window.clear();
    filter_cx = 0.0;
    filter_cy = 0.0;
    filter_w = 0.0;
    filter_h = 0.0;
  };
  if (!cfg.bbox_size_filter_enable || hint == nullptr || hint->miss_count > 0 || detections.empty()) {
    reset_filter_state();
    return;
  }

  const core::BBox& ref_bbox = hint->bbox_hat;
  const double ref_w = std::max(1.0, ref_bbox.width());
  const double ref_h = std::max(1.0, ref_bbox.height());
  const double gate_px = cfg.bbox_size_filter_center_gate_px;

  int best_idx = -1;
  double best_cost = std::numeric_limits<double>::max();
  for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
    const auto& det_bbox = detections[i].bbox;
    const double center_dist = bbox_center_distance_px(det_bbox, ref_bbox);
    if (center_dist > gate_px) {
      continue;
    }
    const double cost = center_dist - 80.0 * vision::iou(det_bbox, ref_bbox);
    if (cost < best_cost) {
      best_cost = cost;
      best_idx = i;
    }
  }

  if (best_idx < 0) {
    reset_filter_state();
    return;
  }

  auto& det_bbox = detections[best_idx].bbox;
  const double cx = det_bbox.cx();
  const double cy = det_bbox.cy();
  const double cur_w = std::max(1.0, det_bbox.width());
  const double cur_h = std::max(1.0, det_bbox.height());
  const double alpha = cfg.bbox_size_filter_alpha;
  const bool track_switched = (!filter_ready) || (filter_track_id != hint->track_id);
  if (track_switched) {
    cx_window.clear();
    cy_window.clear();
    w_window.clear();
    h_window.clear();
    cx_window.push_back(cx);
    cy_window.push_back(cy);
    w_window.push_back(cur_w);
    h_window.push_back(cur_h);
    const double mean_cx = mean_from_window(cx_window);
    const double mean_cy = mean_from_window(cy_window);
    const double mean_w = mean_from_window(w_window);
    const double mean_h = mean_from_window(h_window);
    filter_cx = mean_cx;
    filter_cy = mean_cy;
    filter_w = mean_w;
    filter_h = mean_h;
    filter_ready = true;
    filter_track_id = hint->track_id;
  } else {
    cx_window.push_back(cx);
    cy_window.push_back(cy);
    w_window.push_back(cur_w);
    h_window.push_back(cur_h);
    while (cx_window.size() > kBboxPreFilterWindow) {
      cx_window.pop_front();
    }
    while (cy_window.size() > kBboxPreFilterWindow) {
      cy_window.pop_front();
    }
    while (w_window.size() > kBboxPreFilterWindow) {
      w_window.pop_front();
    }
    while (h_window.size() > kBboxPreFilterWindow) {
      h_window.pop_front();
    }
    const double mean_cx = mean_from_window(cx_window);
    const double mean_cy = mean_from_window(cy_window);
    const double mean_w = mean_from_window(w_window);
    const double mean_h = mean_from_window(h_window);
    filter_cx = alpha * mean_cx + (1.0 - alpha) * filter_cx;
    filter_cy = alpha * mean_cy + (1.0 - alpha) * filter_cy;
    filter_w = alpha * mean_w + (1.0 - alpha) * filter_w;
    filter_h = alpha * mean_h + (1.0 - alpha) * filter_h;
  }

  const double ref_cx = ref_bbox.cx();
  const double ref_cy = ref_bbox.cy();
  const double smooth_cx = std::clamp(filter_cx, ref_cx - gate_px, ref_cx + gate_px);
  const double smooth_cy = std::clamp(filter_cy, ref_cy - gate_px, ref_cy + gate_px);
  const double smooth_w =
      std::clamp(filter_w, ref_w * cfg.bbox_size_filter_min_scale, ref_w * cfg.bbox_size_filter_max_scale);
  const double smooth_h =
      std::clamp(filter_h, ref_h * cfg.bbox_size_filter_min_scale, ref_h * cfg.bbox_size_filter_max_scale);
  filter_cx = smooth_cx;
  filter_cy = smooth_cy;
  filter_w = smooth_w;
  filter_h = smooth_h;

  det_bbox = vision::cxcywh_to_bbox(smooth_cx, smooth_cy, smooth_w, smooth_h);
}

// 把字符串统一转成小写，便于处理 mode / tracker_type / actuator.mode 这类配置项。
std::string to_lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// 运行日志中的 run_id，按时间戳生成，便于区分不同批次实验。
std::string make_run_id() {
  using sys_clock = std::chrono::system_clock;
  const auto now = sys_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
  std::ostringstream oss;
  oss << "run_" << ms;
  return oss.str();
}

}  // namespace

namespace app {

#ifdef _WIN32
// 在 Windows + OpenCV GUI 模式下，尽量自动补齐 Qt 插件路径，避免窗口初始化失败。
void configure_qt_plugin_path_if_needed() {
  if (std::getenv("QT_QPA_PLATFORM_PLUGIN_PATH") != nullptr) {
    return;
  }
  const std::vector<std::string> candidates = {
      "C:/Users/Administrator/miniconda3/envs/py310/Library/lib/qt6/plugins/platforms",
      "C:/Users/Administrator/miniconda3/envs/py310/Lib/site-packages/PyQt5/Qt5/plugins/platforms",
      "C:/Qt/6.5.3/msvc2019_64/plugins/platforms",
      "C:/Qt/6.9.0/msvc2022_64/plugins/platforms"};

  for (const auto& p : candidates) {
    if (std::filesystem::exists(p)) {
      _putenv_s("QT_QPA_PLATFORM_PLUGIN_PATH", p.c_str());
      return;
    }
  }
}
#endif

TrackerNode::TrackerNode(core::AppConfig cfg, Mode mode, std::string replay_csv, uint64_t max_runtime_ms)
    : cfg_(std::move(cfg)),
      mode_(mode),
      replay_csv_path_(std::move(replay_csv)),
      mot_(cfg_.mot),
      controller_(cfg_.control),
      pid_x_(cfg_.control.pid_x.kp, cfg_.control.pid_x.ki, cfg_.control.pid_x.kd),
      pid_y_(cfg_.control.pid_y.kp, cfg_.control.pid_y.ki, cfg_.control.pid_y.kd),
      temporal_compensator_(cfg_.temporal_comp),
      pnr_filter_(cfg_.filter),
      max_runtime_ms_(max_runtime_ms) {}

TrackerNode::~TrackerNode() { stop(); }

bool TrackerNode::init() {
  // ---- 初始化阶段 1：重置运行期状态 ----
  startup_trace("init:begin");
  run_id_ = make_run_id();
  control_dump_path_ = "data/track-fusion-move.csv";
  last_log_timestamp_ms_ = 0;
  temporal_comp_warned_ = false;
  temporal_compensator_.configure(cfg_.temporal_comp);
  startup_trace("init:temporal_comp_configured");

  // ---- 初始化阶段 2：加载新版单模型时序补偿器 ----
  const char* env_temporal_model = std::getenv("TEMPORAL_MODEL_ONNX");
  const char* env_temporal_stats = std::getenv("TEMPORAL_STATS_JSON");
  const std::string temporal_model =
      (env_temporal_model != nullptr) ? std::string(env_temporal_model) : cfg_.temporal_comp.model_path;
  const std::string temporal_stats =
      (env_temporal_stats != nullptr) ? std::string(env_temporal_stats) : cfg_.temporal_comp.stats_path;
  startup_trace("init:temporal_comp_load_begin");
  if (cfg_.temporal_comp.enable && temporal_compensator_.load(temporal_model, temporal_stats)) {
    startup_trace("init:temporal_comp_loaded");
    std::cerr << "[temporal_comp] enabled with ONNX model.\n";
  } else {
    if (cfg_.temporal_comp.enable) {
      startup_trace("init:temporal_comp_load_failed");
      std::cerr << "[temporal_comp] disabled (load failed), fallback to causal baseline only.\n";
    } else {
      startup_trace("init:temporal_comp_disabled_by_config");
      std::cerr << "[temporal_comp] disabled by config.\n";
    }
  }

  // ---- 初始化阶段 4：重置位置滤波与 zoom 相关状态 ----
  pnr_filter_.reset();
  pnr_last_meas_frame_ = -1;
  pnr_last_meas_time_ns_ = 0;
  pnr_last_zoom_value_ = 0.0;
  control_zoom_initialized_ = false;
  control_last_zoom_value_ = 0.0;
  control_zoom_stable_since_ns_ = 0;
  control_zero_sent_while_blocked_ = false;
  median_filter_dx_window_.clear();
  median_filter_dy_window_.clear();
  track_prefilter_dx_window_.clear();
  track_prefilter_dy_window_.clear();
  offset_filter_ready_ = false;
  offset_filter_track_id_ = -1;
  offset_stage1_dx_ = 0.0;
  offset_stage1_dy_ = 0.0;
  offset_stage2_dx_ = 0.0;
  offset_stage2_dy_ = 0.0;
  offset_stage3_dx_ = 0.0;
  offset_stage3_dy_ = 0.0;
  offset_stage2_dx_window_.clear();
  offset_stage2_dy_window_.clear();
  offset_stage3_dx_window_.clear();
  offset_stage3_dy_window_.clear();
  offset_window8_stage3_dx_window_.clear();
  offset_window8_stage3_dy_window_.clear();
  bbox_size_filter_ready_ = false;
  bbox_size_filter_track_id_ = -1;
  bbox_size_filter_cx_window_.clear();
  bbox_size_filter_cy_window_.clear();
  bbox_size_filter_w_window_.clear();
  bbox_size_filter_h_window_.clear();
  bbox_size_filter_cx_ = 0.0;
  bbox_size_filter_cy_ = 0.0;
  bbox_size_filter_w_ = 0.0;
  bbox_size_filter_h_ = 0.0;
  bbox_size_second_filter_ready_ = false;
  bbox_size_second_filter_track_id_ = -1;
  bbox_size_second_filter_w_ = 0.0;
  bbox_size_second_filter_h_ = 0.0;

  // ---- 初始化阶段 5：选择电机驱动实现 ----
  // pyusb 对应真实硬件；mock 用于安全调试和无硬件环境。
  {
    const std::string amode = to_lower_copy(cfg_.actuator.mode);
    if (amode == "pyusb") {
      motor_ = std::make_unique<actuator::MotorDriverPyUsb>(cfg_.actuator);
      std::cerr << "[actuator] mode=pyusb armed=" << (cfg_.actuator.armed ? "true" : "false") << '\n';
    } else {
      motor_ = std::make_unique<actuator::MotorDriverMock>();
      std::cerr << "[actuator] mode=mock\n";
    }
  }

  // ---- 初始化阶段 6：打开日志文件 ----
  if (cfg_.log.enable && !logger_.open(cfg_.log.path, cfg_.log.dedup_by_frame_id, cfg_.log.profile)) {
    startup_trace("init:logger_open_failed path=" + cfg_.log.path);
    std::cerr << "[tracker] Failed to open log file: " << cfg_.log.path << '\n';
    return false;
  }
  startup_trace("init:logger_open_ok path=" + cfg_.log.path);

  try {
    const auto dump_path = std::filesystem::u8path(control_dump_path_);
    if (dump_path.has_parent_path()) {
      std::filesystem::create_directories(dump_path.parent_path());
    }
    control_dump_ofs_.open(dump_path, std::ios::out | std::ios::trunc);
    if (control_dump_ofs_.is_open()) {
      control_dump_ofs_ << "timestamp_ms,frame_id,track_id,controlled_id,det_conf,miss_count,matched_in_frame,"
                           "det_dx,det_dy,track_dx,track_dy,median_filter_dx,median_filter_dy,first_filter_dx,first_filter_dy,"
                           "no.2_first_filter_x,no.2_first_filter_y,no.3_first_filter_x,no.3_first_filter_y,"
                           "window8_no.2_first_filter_x,window8_no.2_first_filter_y,"
                           "window8_no.3_first_filter_x,window8_no.3_first_filter_y,"
                           "window8_10_no.3_first_filter_x,window8_10_no.3_first_filter_y,"
                           "comp_dx,comp_dy,fused_dx,fused_dy,"
                           "det_w,det_h,track_w,track_h,first_filter_w,first_filter_h,no.2_first_filter_w,no.2_first_filter_h,"
                           "infer_used_model,infer_status,cmd_x,cmd_y,error_px\n";
      control_dump_ofs_.flush();
    } else {
      std::cerr << "[tracker] warning: failed to open control dump file: " << control_dump_path_ << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "[tracker] warning: failed to prepare control dump file: " << e.what() << '\n';
  }

#ifndef HAVE_OPENCV
  // 没有 OpenCV 的构建只能跑 replay，不能直接开相机。
  if (mode_ == Mode::Cam) {
    std::cerr << "[tracker] cam mode requires OpenCV-enabled build. Use preset: msvc-opencv-release\n";
    return false;
  }
#endif

  // ---- 初始化阶段 7：准备输入源 ----
  if (mode_ == Mode::Replay) {
    if (!load_replay_csv(replay_csv_path_)) {
      startup_trace("init:replay_load_failed path=" + replay_csv_path_);
      std::cerr << "[tracker] Failed to load replay csv: " << replay_csv_path_ << '\n';
      return false;
    }
    startup_trace("init:replay_load_ok path=" + replay_csv_path_);
  } else {
#ifdef _WIN32
    if (cfg_.camera.show_window) {
      configure_qt_plugin_path_if_needed();
    }
#endif
    // 相机模式下还需要初始化 zoom、检测器和可选 GUI 窗口。
    if (cfg_.model.use_shm_detector) {
      if (!shm_reader_.open()) {
        std::cerr << "[tracker] ShmDetectionReader open failed\n";
        return false;
      }
      startup_trace("init:shm_reader_ok");
    } else {
      camera_.set_backend(cfg_.camera.backend);
      camera_.set_use_mjpg(cfg_.camera.use_mjpg);
      if (!camera_.open(cfg_.camera.index, cfg_.camera.width, cfg_.camera.height)) {
        startup_trace("init:camera_open_failed");
        std::cerr << "[tracker] Failed to open camera index " << cfg_.camera.index << '\n';
        return false;
      }
      startup_trace("init:camera_open_ok");
      (void)camera_.set_auto_focus(cfg_.camera.auto_focus_enable);
      bool zoom_reset_ok = camera_.set_zoom(0.0, 0.0, cfg_.camera.zoom_max);
      if (!zoom_reset_ok && cfg_.camera.auto_focus_enable) {
        (void)camera_.set_auto_focus(false);
        zoom_reset_ok = camera_.set_zoom(0.0, 0.0, cfg_.camera.zoom_max);
        (void)camera_.set_auto_focus(cfg_.camera.auto_focus_enable);
      }
      if (!zoom_reset_ok) {
        std::cerr << "[camera] warning: failed to reset zoom to 0 at startup.\n";
      }
      zoom_cmd_ready_ = true;
      zoom_cmd_value_ = 0.0;
      zoom_mode_ = ZoomMode::Search;
      zoom_hold_miss_count_ = 0;
      zoom_reacquire_freeze_left_ = 0;
      zoom_reacquire_confirm_count_ = 0;
      zoom_search_direction_ = 1;
      zoom_strategy_armed_ = false;
      zoom_area_window_.clear();
      zoom_area_filtered_ready_ = false;
      zoom_prev_area_ready_ = false;
      zoom_prev_track_id_ = -1;
      zoom_hysteresis_active_ = false;
      zoom_lost_at_max_frames_ = 0;
      zoom_max_recover_active_ = false;
      zoom_last_action_ns_ = 0;
      zoom_set_fail_streak_ = 0;
      detector_.set_conf_threshold(cfg_.model.conf_detect);
      vision::YoloDetector::TilingConfig tcfg;
      tcfg.enable = cfg_.model.tile_enable;
      tcfg.tile_size = cfg_.model.tile_size;
      tcfg.tile_overlap = cfg_.model.tile_overlap;
      tcfg.tile_max_per_frame = cfg_.model.tile_max_per_frame;
      tcfg.tile_global_interval = cfg_.model.tile_global_interval;
      tcfg.priority_enable = cfg_.model.tile_priority_enable;
      tcfg.priority_topk = cfg_.model.tile_priority_topk;
      tcfg.priority_expand_ratio = cfg_.model.tile_priority_expand_ratio;
      tcfg.priority_ttl = cfg_.model.tile_priority_ttl;
      detector_.set_tiling_config(tcfg);
      if (!detector_.load(cfg_.model.path)) {
        startup_trace("init:detector_load_failed path=" + cfg_.model.path);
        std::cerr << "[tracker] Detector load failed for: " << cfg_.model.path << '\n';
        return false;
      }
      startup_trace("init:detector_load_ok path=" + cfg_.model.path);
    }
  }
  startup_trace("init:return_true");
  return true;
}

void TrackerNode::run() {
  if (running_.exchange(true)) {
    return;
  }
  run_start_ns_ = core::time::now_ns();

  vision_thread_ = std::thread(&TrackerNode::vision_loop, this);
  control_thread_ = std::thread(&TrackerNode::control_loop, this);

  if (vision_thread_.joinable()) {
    vision_thread_.join();
  }
  running_.store(false);
  frame_cv_.notify_all();
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
}

void TrackerNode::wait_actuator_ready() {
  if (!motor_) return;
  std::cerr << "[startup] waiting for takeover (max 30s)...\n";
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!motor_->takeover_done()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      std::cerr << "[startup] takeover timeout, proceeding anyway.\n";
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  std::cerr << "[startup] takeover done.\n";
}

void TrackerNode::init_pitch_angle() {
  if (!motor_) return;
  constexpr double pitch_init_deg = -25.0;
  constexpr double pitch_init_tol =   1.0;
  constexpr double vy_init        =   5.0;
  constexpr int    timeout_ms     = 8000;
  const bool positive_increases_angle = cfg_.actuator.y_pitch_positive_increases_angle;

  std::cerr << "[startup] initializing pitch to " << pitch_init_deg << " deg...\n";
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

  while (std::chrono::steady_clock::now() < deadline) {
    const double pitch = motor_->get_pitch();
    const double err   = pitch_init_deg - pitch;
    if (std::fabs(err) < pitch_init_tol) {
      motor_->send(core::MotorCmd{0.0, 0.0});
      std::cerr << "[startup] pitch initialized at " << pitch << " deg.\n";
      return;
    }
    // invert_y=true: cmd_y>0 使俯仰角减小，负值使角度增大
    // err>0 表示需要增大角度 → 发负值；err<0 需要减小角度 → 发正值
    const double cmd_y = (err > 0) ? -vy_init : vy_init;
    const double cmd_y_runtime = signed_y_speed_for_pitch_error(err, vy_init, positive_increases_angle);
    motor_->send(core::MotorCmd{0.0, cmd_y_runtime});
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }

  motor_->send(core::MotorCmd{0.0, 0.0});
  std::cerr << "[startup] pitch init timeout, current pitch=" << motor_->get_pitch() << " deg.\n";
}

void TrackerNode::stop() {
  running_.store(false);
  frame_cv_.notify_all();
  if (vision_thread_.joinable()) {
    vision_thread_.join();
  }
  if (control_thread_.joinable()) {
    control_thread_.join();
  }
  zoom_strategy_armed_ = false;
  zoom_lost_at_max_frames_ = 0;
  zoom_max_recover_active_ = false;
  zoom_mode_ = ZoomMode::Search;
  zoom_hold_miss_count_ = 0;
  zoom_reacquire_freeze_left_ = 0;
  zoom_reacquire_confirm_count_ = 0;
  zoom_search_direction_ = 1;
  pnr_filter_.reset();
  pnr_last_meas_frame_ = -1;
  pnr_last_meas_time_ns_ = 0;
  pnr_last_zoom_value_ = 0.0;
  control_zoom_initialized_ = false;
  control_last_zoom_value_ = 0.0;
  control_zoom_stable_since_ns_ = 0;
  control_zero_sent_while_blocked_ = false;
  temporal_compensator_.reset();
  median_filter_dx_window_.clear();
  median_filter_dy_window_.clear();
  offset_filter_ready_ = false;
  offset_filter_track_id_ = -1;
  track_prefilter_dx_window_.clear();
  track_prefilter_dy_window_.clear();
  offset_stage1_dx_ = 0.0;
  offset_stage1_dy_ = 0.0;
  offset_stage2_dx_ = 0.0;
  offset_stage2_dy_ = 0.0;
  offset_stage3_dx_ = 0.0;
  offset_stage3_dy_ = 0.0;
  offset_stage2_dx_window_.clear();
  offset_stage2_dy_window_.clear();
  offset_stage3_dx_window_.clear();
  offset_stage3_dy_window_.clear();
  offset_window8_stage3_dx_window_.clear();
  offset_window8_stage3_dy_window_.clear();
  bbox_size_filter_ready_ = false;
  bbox_size_filter_track_id_ = -1;
  bbox_size_filter_cx_window_.clear();
  bbox_size_filter_cy_window_.clear();
  bbox_size_filter_w_window_.clear();
  bbox_size_filter_h_window_.clear();
  bbox_size_filter_cx_ = 0.0;
  bbox_size_filter_cy_ = 0.0;
  bbox_size_filter_w_ = 0.0;
  bbox_size_filter_h_ = 0.0;
  bbox_size_second_filter_ready_ = false;
  bbox_size_second_filter_track_id_ = -1;
  bbox_size_second_filter_w_ = 0.0;
  bbox_size_second_filter_h_ = 0.0;
  if (motor_) {
    motor_->close();
  }
  logger_.close();
  if (control_dump_ofs_.is_open()) {
    control_dump_ofs_.flush();
    control_dump_ofs_.close();
  }
  camera_.close();
}

bool TrackerNode::runtime_limit_reached(uint64_t now_ns) const {
  if (max_runtime_ms_ == 0 || run_start_ns_ == 0 || now_ns < run_start_ns_) {
    return false;
  }
  const uint64_t limit_ns = max_runtime_ms_ * 1000000ULL;
  return (now_ns - run_start_ns_) >= limit_ns;
}

bool TrackerNode::load_replay_csv(const std::string& csv_path) {
  // 回放模式下，把 frame_seq -> detections 建成索引，便于按帧读取。
  std::ifstream ifs(csv_path);
  if (!ifs.is_open()) {
    return false;
  }

  replay_by_frame_.clear();
  replay_max_frame_ = -1;

  std::string line;
  bool first_line = true;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }

    if (first_line) {
      first_line = false;
      if (line.find("frame_seq") != std::string::npos) {
        continue;
      }
    }

    const auto cols = split_csv(line);
    if (cols.size() < 7) {
      continue;
    }

    int frame_seq = 0;
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
    double conf = 0.0;
    int cls = 0;

    if (!parse_int(cols[0], frame_seq) || !parse_double(cols[1], x1) || !parse_double(cols[2], y1) ||
        !parse_double(cols[3], x2) || !parse_double(cols[4], y2) || !parse_double(cols[5], conf) ||
        !parse_int(cols[6], cls)) {
      continue;
    }

    core::Detection det;
    det.bbox = {x1, y1, x2, y2};
    det.conf = conf;
    det.cls = cls;
    replay_by_frame_[frame_seq].push_back(det);
    replay_max_frame_ = std::max(replay_max_frame_, frame_seq);
  }

  return replay_max_frame_ >= 0;
}

std::vector<core::Detection> TrackerNode::replay_detections_for_frame(int frame_seq) const {
  const auto it = replay_by_frame_.find(frame_seq);
  if (it == replay_by_frame_.end()) {
    return {};
  }
  return it->second;
}

void TrackerNode::auto_zoom_step(const std::vector<core::TrackState>& tracks, int controlled_id) {
#ifndef HAVE_OPENCV
  (void)tracks;
  (void)controlled_id;
  return;
#else
  // 自动变焦只在相机模式下生效；回放模式不需要这条逻辑。
  if (mode_ != Mode::Cam || !cfg_.camera.auto_zoom_enable) {
    return;
  }

  const double zmin = cfg_.camera.zoom_min;
  const double zmax = cfg_.camera.zoom_max;
  if (!zoom_cmd_ready_) {
    zoom_cmd_value_ = std::clamp(camera_.get_zoom(zmin), zmin, zmax);
    zoom_cmd_ready_ = true;
  }

  // 先尝试拿到当前被控制的目标；如果选中的目标丢了，再找一个最稳定的近邻轨迹兜底。
  const int max_zoom_coast = std::max(1, std::min(cfg_.mot.max_age, 8));
  const core::TrackState* ct = nullptr;
  for (const auto& t : tracks) {
    if (t.track_id == controlled_id && t.miss_count <= max_zoom_coast) {
      ct = &t;
      break;
    }
  }

  // 兜底策略：如果当前控制目标丢失，则选一个距离中心近且面积较合理的轨迹继续维持变焦。
  if (ct == nullptr) {
    double best_score = std::numeric_limits<double>::max();
    for (const auto& t : tracks) {
      if (t.miss_count > max_zoom_coast) {
        continue;
      }
      const double dx = t.cx - static_cast<double>(cfg_.camera.center_x);
      const double dy = t.cy - static_cast<double>(cfg_.camera.center_y);
      const double d2 = dx * dx + dy * dy;
      const double area = std::max(1.0, t.bbox_hat.width() * t.bbox_hat.height());
      const double score = static_cast<double>(t.miss_count) * 1e6 + d2 - 0.01 * area;
      if (score < best_score) {
        best_score = score;
        ct = &t;
      }
    }
  }

  // 如果完全没有可用目标，则进入 hold/search 逻辑，避免镜头一直停在错误焦段。
  if (ct == nullptr) {
    ++zoom_no_target_frames_;
    zoom_pending_sign_ = 0;
    zoom_pending_frames_ = 0;
    zoom_hysteresis_active_ = false;
    zoom_debug_valid_ = false;
    zoom_reacquire_confirm_count_ = 0;
    zoom_lost_at_max_frames_ = 0;
    zoom_max_recover_active_ = false;

    // 目标丢失时，先从锁定态切到保持态，再视情况切到搜索态。
    if (zoom_mode_ == ZoomMode::Locked) {
      zoom_mode_ = ZoomMode::Hold;
      zoom_hold_miss_count_ = 0;
    }

    if (zoom_mode_ == ZoomMode::Hold) {
      ++zoom_hold_miss_count_;
      const int hold_frames = std::max(1, cfg_.camera.zoom_hold_miss_frames);
      if (zoom_hold_miss_count_ < hold_frames) {
        zoom_debug_value_ = zoom_cmd_value_;
        return;
      }
      zoom_mode_ = ZoomMode::Search;
      zoom_hold_miss_count_ = 0;
    }

    // 搜索态下按固定步长来回扫焦段，提高重新捕获目标的概率。
    const int search_every_n = std::max(1, cfg_.camera.zoom_search_every_n);
    const bool frame_ready = (zoom_no_target_frames_ >= search_every_n && (zoom_no_target_frames_ % search_every_n) == 0);
    const uint64_t now_ns = core::time::now_ns();
    const uint64_t interval_ns = static_cast<uint64_t>(std::max(0, cfg_.camera.zoom_action_interval_ms)) * 1000000ULL;
    const bool interval_ready =
        (zoom_last_action_ns_ == 0 || now_ns <= zoom_last_action_ns_ || now_ns - zoom_last_action_ns_ >= interval_ns);

    if (zoom_mode_ == ZoomMode::Search && frame_ready && interval_ready) {
      const double search_step = std::max(0.5, cfg_.camera.zoom_search_step);
      double next_zoom = std::clamp(zoom_cmd_value_ + static_cast<double>(zoom_search_direction_) * search_step, zmin, zmax);

      // 打到边界后翻转方向，避免一直卡在最大或最小焦段。
      if (std::abs(next_zoom - zoom_cmd_value_) < 1e-9) {
        zoom_search_direction_ = (zoom_search_direction_ >= 0) ? -1 : +1;
        next_zoom = std::clamp(zoom_cmd_value_ + static_cast<double>(zoom_search_direction_) * search_step, zmin, zmax);
      }

      if (std::abs(next_zoom - zoom_cmd_value_) >= 1e-9) {
        bool ok = camera_.set_zoom(next_zoom, zmin, zmax);
        if (!ok && cfg_.camera.auto_focus_enable) {
          (void)camera_.set_auto_focus(false);
          ok = camera_.set_zoom(next_zoom, zmin, zmax);
        }
        if (ok) {
          zoom_cmd_value_ = next_zoom;
          zoom_last_action_ns_ = now_ns;
          zoom_set_fail_streak_ = 0;
          if (zoom_cmd_value_ >= zmax - 1e-6) {
            zoom_search_direction_ = -1;
          } else if (zoom_cmd_value_ <= zmin + 1e-6) {
            zoom_search_direction_ = +1;
          }
        } else {
          ++zoom_set_fail_streak_;
        }
      }
    }
    if (zoom_no_target_frames_ > 30) {
      zoom_ema_ready_ = false;
      zoom_prev_ratio_ready_ = false;
      zoom_prev_area_ready_ = false;
      zoom_prev_track_id_ = -1;
      zoom_area_window_.clear();
      zoom_area_filtered_ready_ = false;
      zoom_hysteresis_active_ = false;
      zoom_lost_at_max_frames_ = 0;
      zoom_max_recover_active_ = false;
      zoom_mode_ = ZoomMode::Search;
      zoom_hold_miss_count_ = 0;
      zoom_reacquire_confirm_count_ = 0;
    }
    zoom_debug_value_ = zoom_cmd_value_;
    return;
  }
  zoom_no_target_frames_ = 0;
  zoom_hold_miss_count_ = 0;
  zoom_lost_at_max_frames_ = 0;
  zoom_max_recover_active_ = false;
  // 重新看到目标后，先经过若干帧确认，避免偶发误检立刻改变变焦策略。
  if (zoom_mode_ == ZoomMode::Search) {
    ++zoom_reacquire_confirm_count_;
    const int need_confirm = std::max(1, cfg_.camera.zoom_reacquire_confirm_frames);
    if (zoom_reacquire_confirm_count_ < need_confirm) {
      zoom_debug_valid_ = true;
      zoom_debug_value_ = zoom_cmd_value_;
      return;
    }
    zoom_mode_ = ZoomMode::Locked;
    zoom_reacquire_confirm_count_ = 0;
    zoom_reacquire_freeze_left_ = std::max(0, cfg_.camera.zoom_reacquire_freeze_frames);
    zoom_area_window_.clear();
    zoom_area_filtered_ready_ = false;
    zoom_prev_area_ready_ = false;
    zoom_prev_track_id_ = -1;
    zoom_hysteresis_active_ = false;
    zoom_pending_sign_ = 0;
    zoom_pending_frames_ = 0;
    zoom_debug_valid_ = true;
    zoom_debug_value_ = zoom_cmd_value_;
    return;
  }
  zoom_reacquire_confirm_count_ = 0;
  if (zoom_mode_ == ZoomMode::Hold) {
    zoom_mode_ = ZoomMode::Locked;
  }
  if (!zoom_strategy_armed_) {
    zoom_strategy_armed_ = true;
    zoom_area_window_.clear();
    zoom_area_filtered_ready_ = false;
    zoom_prev_area_ready_ = false;
    zoom_prev_track_id_ = -1;
    zoom_hysteresis_active_ = false;
    zoom_pending_sign_ = 0;
    zoom_pending_frames_ = 0;
    zoom_lost_at_max_frames_ = 0;
    zoom_max_recover_active_ = false;
  }

  const double bw = std::max(1.0, ct->bbox_hat.width());
  const double bh = std::max(1.0, ct->bbox_hat.height());
  const double current_area = bw * bh;
  const double frame_area = static_cast<double>(cfg_.camera.width) * static_cast<double>(cfg_.camera.height);
  zoom_debug_valid_ = true;
  zoom_debug_ratio_ = (frame_area > 1.0) ? std::sqrt(current_area / frame_area) : 0.0;

  // 切换目标或刚开始跟踪时，先重新初始化 zoom 内部滤波状态，这一帧不调焦。
  if (!zoom_prev_area_ready_ || zoom_prev_track_id_ != ct->track_id || !zoom_area_filtered_ready_) {
    zoom_prev_area_ready_ = true;
    zoom_prev_track_id_ = ct->track_id;
    zoom_prev_area_ = current_area;
    zoom_area_window_.clear();
    zoom_area_window_.push_back(current_area);
    zoom_area_filtered_ = current_area;
    zoom_area_filtered_ready_ = true;
    zoom_hysteresis_active_ = false;
    zoom_debug_error_ = 0.0;
    zoom_debug_area_err_ = 0.0;
    zoom_debug_value_ = zoom_cmd_value_;
    zoom_pending_sign_ = 0;
    zoom_pending_frames_ = 0;
    return;
  }

  if (zoom_reacquire_freeze_left_ > 0) {
    --zoom_reacquire_freeze_left_;
    zoom_pending_sign_ = 0;
    zoom_pending_frames_ = 0;
    zoom_hysteresis_active_ = false;
    zoom_debug_value_ = zoom_cmd_value_;
    return;
  }

  zoom_prev_area_ = current_area;

  // 用中值 + 指数平滑估计目标面积，减小检测框面积抖动对 zoom 的影响。
  const int win_n = std::clamp(cfg_.camera.zoom_median_window, 3, 9);
  zoom_area_window_.push_back(current_area);
  if (static_cast<int>(zoom_area_window_.size()) > win_n) {
    zoom_area_window_.erase(zoom_area_window_.begin());
  }

  std::vector<double> tmp = zoom_area_window_;
  const auto mid_it = tmp.begin() + (tmp.size() / 2);
  std::nth_element(tmp.begin(), mid_it, tmp.end());
  const double area_med = *mid_it;

  double area_mean = 0.0;
  for (double v : zoom_area_window_) {
    area_mean += v;
  }
  area_mean /= static_cast<double>(zoom_area_window_.size());
  double area_var = 0.0;
  for (double v : zoom_area_window_) {
    const double dv = v - area_mean;
    area_var += dv * dv;
  }
  area_var /= static_cast<double>(zoom_area_window_.size());
  const double area_sigma = std::sqrt(std::max(0.0, area_var));

  const double alpha = std::clamp(cfg_.camera.bbox_ratio_alpha, 0.05, 1.0);
  zoom_area_filtered_ = alpha * area_med + (1.0 - alpha) * zoom_area_filtered_;

  // 根据目标面积和目标面积期望值的差，决定是继续 zoom in 还是 zoom out。
  const double target_ratio = std::max(0.01, cfg_.camera.target_bbox_ratio);
  const double target_area = std::clamp(target_ratio * target_ratio * frame_area, 1.0, frame_area);
  const double area_err = target_area - zoom_area_filtered_;
  zoom_debug_error_ = area_err;
  zoom_debug_area_err_ = area_err;

  const double db_base = std::max(1.0, cfg_.camera.zoom_area_deadband_px);
  const double db_dynamic = std::max(db_base, 2.0 * area_sigma);
  const double hyst_ratio = std::max(1.05, cfg_.camera.zoom_hysteresis_ratio);
  const double db_in = db_dynamic * hyst_ratio;
  const double db_out = db_dynamic;

  const double abs_err = std::abs(area_err);
  if (!zoom_hysteresis_active_) {
    zoom_hysteresis_active_ = (abs_err > db_in);
  } else if (abs_err < db_out) {
    zoom_hysteresis_active_ = false;
  }

  if (!zoom_hysteresis_active_) {
    zoom_pending_sign_ = 0;
    zoom_pending_frames_ = 0;
    zoom_debug_value_ = zoom_cmd_value_;
    return;
  }

  const int direction = (area_err >= 0.0) ? +1 : -1;
  if (direction < 0) {
    const double current_ratio = (frame_area > 1.0) ? std::sqrt(std::max(1.0, zoom_area_filtered_) / frame_area) : 0.0;
    const double trigger_ratio = target_ratio * std::max(1.0, cfg_.camera.zoom_out_trigger_mult);
    const bool allow_zoom_out = cfg_.camera.zoom_out_enable && current_ratio >= trigger_ratio;
    if (!allow_zoom_out) {
      zoom_pending_sign_ = 0;
      zoom_pending_frames_ = 0;
      zoom_debug_value_ = zoom_cmd_value_;
      return;
    }
  }
  if (direction != zoom_pending_sign_) {
    zoom_pending_sign_ = direction;
    zoom_pending_frames_ = 1;
  } else {
    ++zoom_pending_frames_;
  }

  // 只有误差持续若干帧且到达动作间隔后，才真正发出 zoom 指令，避免镜头来回抽动。
  const int need_frames = std::max(1, cfg_.camera.zoom_persist_frames);
  if (zoom_pending_frames_ < need_frames) {
    zoom_debug_value_ = zoom_cmd_value_;
    return;
  }

  const uint64_t now_ns = core::time::now_ns();
  const uint64_t interval_ns = static_cast<uint64_t>(std::max(0, cfg_.camera.zoom_action_interval_ms)) * 1000000ULL;
  if (zoom_last_action_ns_ > 0 && now_ns > zoom_last_action_ns_ && now_ns - zoom_last_action_ns_ < interval_ns) {
    zoom_debug_value_ = zoom_cmd_value_;
    return;
  }

  const double step = std::max(0.1, cfg_.camera.zoom_step);
  const double next_zoom = std::clamp(zoom_cmd_value_ + direction * step, zmin, zmax);

  if (std::abs(next_zoom - zoom_cmd_value_) < 1e-6) {
    zoom_debug_value_ = zoom_cmd_value_;
    return;
  }

  bool ok = camera_.set_zoom(next_zoom, zmin, zmax);
  if (!ok && cfg_.camera.auto_focus_enable) {
    (void)camera_.set_auto_focus(false);
    ok = camera_.set_zoom(next_zoom, zmin, zmax);
  }

  if (ok) {
    zoom_cmd_value_ = next_zoom;
    zoom_last_action_ns_ = now_ns;
    zoom_set_fail_streak_ = 0;
  } else {
    ++zoom_set_fail_streak_;
  }

  if (!ok && !zoom_not_supported_reported_) {
    std::cerr << "[camera] zoom control not supported by this camera/backend, auto zoom inactive.\n";
    zoom_not_supported_reported_ = true;
  }
  zoom_debug_value_ = zoom_cmd_value_;
  zoom_pending_frames_ = 0;
#endif
}

void TrackerNode::vision_loop() {
  // 视觉线程负责“采集/回放 -> 检测 -> 跟踪 -> 选目标 -> 更新共享状态”。
  using clock = std::chrono::steady_clock;
  constexpr double kVisionDt = 1.0 / 30.0;
  constexpr double kShmPollDt = 1.0 / 240.0;
  const bool use_fast_shm_poll = (mode_ == Mode::Cam && cfg_.model.use_shm_detector);
  const auto period = std::chrono::duration<double>(use_fast_shm_poll ? kShmPollDt : kVisionDt);
  auto next_tick = clock::now();

  double mot_dt = kVisionDt;
  uint64_t last_packet_ns = 0;
  int frame_seq = 0;
  bool window_initialized = false;
#ifdef HAVE_OPENCV
  if (mode_ == Mode::Cam && cfg_.camera.show_window) {
    try {
      cv::namedWindow("tracker", cv::WINDOW_NORMAL);
      window_initialized = true;
    } catch (...) {
      window_initialized = false;
    }
  }
#endif
  while (running_.load()) {
    const uint64_t vision_now_ns = core::time::now_ns();
    if (runtime_limit_reached(vision_now_ns)) {
      std::cerr << "[tracker] max runtime reached, stopping.\n";
      running_.store(false);
      break;
    }
    if (zoom_last_vision_ns_ > 0 && vision_now_ns > zoom_last_vision_ns_) {
      const double dt = static_cast<double>(vision_now_ns - zoom_last_vision_ns_) * 1e-9;
      if (dt > 1e-6) {
        zoom_debug_fps_ = 1.0 / dt;
      }
    }
    zoom_last_vision_ns_ = vision_now_ns;

    // cam 模式直接从相机读图；replay 模式则从 CSV 中取当前帧检测结果。
    std::vector<core::Detection> detections;
    cv::Mat frame;

    SharedState size_filter_hint;
    {
      std::scoped_lock lock(mu_);
      size_filter_hint = shared_;
    }
    const core::TrackState* size_filter_track = nullptr;
    for (const auto& t : size_filter_hint.tracks) {
      if (t.track_id == size_filter_hint.controlled_id && t.miss_count <= std::max(1, cfg_.mot.max_age)) {
        size_filter_track = &t;
        break;
      }
    }

    if (mode_ == Mode::Replay) {
      mot_dt = kVisionDt;
      detections = replay_detections_for_frame(frame_seq);
      smooth_detection_bbox_size(detections, size_filter_track, cfg_.model, bbox_size_filter_ready_,
                                 bbox_size_filter_track_id_, bbox_size_filter_cx_window_,
                                 bbox_size_filter_cy_window_, bbox_size_filter_w_window_,
                                 bbox_size_filter_h_window_, bbox_size_filter_cx_, bbox_size_filter_cy_,
                                 bbox_size_filter_w_, bbox_size_filter_h_);
    } else {
      if (cfg_.model.use_shm_detector) {
#ifdef HAVE_OPENCV
        int shm_seq = -1;
        if (!shm_reader_.read_packet(frame, detections, static_cast<float>(cfg_.model.conf_detect), &shm_seq)) {
          next_tick += std::chrono::duration_cast<clock::duration>(period);
          core::time::sleep_until(next_tick);
          continue;
        }
#else
        next_tick += std::chrono::duration_cast<clock::duration>(period);
        core::time::sleep_until(next_tick);
        continue;
#endif
        mot_dt = kVisionDt;
        if (last_packet_ns > 0 && vision_now_ns > last_packet_ns) {
          mot_dt = std::clamp(static_cast<double>(vision_now_ns - last_packet_ns) * 1e-9, kShmPollDt, 0.2);
        }
        last_packet_ns = vision_now_ns;
        smooth_detection_bbox_size(detections, size_filter_track, cfg_.model, bbox_size_filter_ready_,
                                   bbox_size_filter_track_id_, bbox_size_filter_cx_window_,
                                   bbox_size_filter_cy_window_, bbox_size_filter_w_window_,
                                   bbox_size_filter_h_window_, bbox_size_filter_cx_, bbox_size_filter_cy_,
                                   bbox_size_filter_w_, bbox_size_filter_h_);
      } else if (!camera_.read(frame)) {
        mot_dt = kVisionDt;
        detections.clear();
        smooth_detection_bbox_size(detections, nullptr, cfg_.model, bbox_size_filter_ready_,
                                   bbox_size_filter_track_id_, bbox_size_filter_cx_window_,
                                   bbox_size_filter_cy_window_, bbox_size_filter_w_window_,
                                   bbox_size_filter_h_window_, bbox_size_filter_cx_, bbox_size_filter_cy_,
                                   bbox_size_filter_w_, bbox_size_filter_h_);
      } else {
        mot_dt = kVisionDt;
        // 如果上一帧已有控制目标，则把它的预测框作为优先 ROI，提高局部检测效率。
        SharedState hint;
        {
          std::scoped_lock lock(mu_);
          hint = shared_;
        }
        const core::TrackState* ht = nullptr;
        for (const auto& t : hint.tracks) {
          if (t.track_id == hint.controlled_id && t.miss_count <= std::max(1, cfg_.mot.max_age)) {
            ht = &t;
            break;
          }
        }
        if (ht != nullptr) {
#ifdef HAVE_OPENCV
          detector_.set_priority_roi(ht->bbox_hat, frame.cols, frame.rows);
#else
          detector_.set_priority_roi(ht->bbox_hat, cfg_.camera.width, cfg_.camera.height);
#endif
        }
        detections = detector_.infer(frame);
        smooth_detection_bbox_size(detections, ht, cfg_.model, bbox_size_filter_ready_, bbox_size_filter_track_id_,
                                   bbox_size_filter_cx_window_, bbox_size_filter_cy_window_,
                                   bbox_size_filter_w_window_, bbox_size_filter_h_window_, bbox_size_filter_cx_,
                                   bbox_size_filter_cy_, bbox_size_filter_w_, bbox_size_filter_h_);
      }
    }

    // 多目标跟踪器负责把当前帧检测关联成稳定轨迹。
    mot_.update(detections, mot_dt);
    auto tracks = mot_.get_tracks();

    int prev_id = -1;
    {
      std::scoped_lock lock(mu_);
      prev_id = shared_.controlled_id;
    }

    // 目标选择器从当前轨迹里选出要控制的那一个 track_id。
    const int controlled_id = selector_.select(tracks, prev_id, static_cast<double>(cfg_.camera.center_x),
                                               static_cast<double>(cfg_.camera.center_y));
    // 自动变焦依赖“当前控制目标”的状态，因此在选目标之后执行。
    auto_zoom_step(tracks, controlled_id);
    const uint64_t vision_sample_ns = core::time::now_ns();

    {
      // 共享状态只保存控制线程真正需要的信息，避免跨线程传整帧图像。
      std::scoped_lock lock(mu_);
      shared_.frame_seq = frame_seq;
      shared_.det_count = static_cast<int>(detections.size());
      shared_.tracks = tracks;
      shared_.controlled_id = controlled_id;
      shared_.vision_time_ns = vision_sample_ns;
      shared_.zoom_value = zoom_cmd_value_;
      ++shared_generation_;
    }
    frame_cv_.notify_one();

    // 相机模式下可选显示 GUI，方便调试跟踪框、控制目标和变焦状态。
    if (mode_ == Mode::Cam && cfg_.camera.show_window && window_initialized && !frame.empty()) {
      for (const auto& t : tracks) {
        const bool is_coasting = (t.miss_count > 0);
        if (is_coasting && !cfg_.camera.show_coasting) {
          continue;
        }
        const auto b = t.bbox_hat;
        cv::Scalar color = (t.track_id == controlled_id) ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 0);
        if (is_coasting) {
          color = cv::Scalar(0, 165, 255);
        }
        cv::rectangle(frame, cv::Point(static_cast<int>(b.x1), static_cast<int>(b.y1)),
                      cv::Point(static_cast<int>(b.x2), static_cast<int>(b.y2)), color, 2);
        std::string tag = "ID=" + std::to_string(t.track_id);
        if (is_coasting) {
          tag += " COAST(" + std::to_string(t.miss_count) + ")";
        }
        cv::putText(frame, tag, cv::Point(static_cast<int>(b.x1), static_cast<int>(b.y1) - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
      }
      cv::circle(frame, cv::Point(cfg_.camera.center_x, cfg_.camera.center_y), 4, cv::Scalar(0, 0, 255), -1);
      if (cfg_.camera.show_zoom_debug) {
        std::ostringstream oss;
        const char zmode =
            (zoom_mode_ == ZoomMode::Locked) ? 'L' : ((zoom_mode_ == ZoomMode::Hold) ? 'H' : 'S');
        if (zoom_debug_valid_) {
          oss << "zoom=" << static_cast<int>(std::round(zoom_debug_value_)) << " ratio=" << std::fixed
              << std::setprecision(3) << zoom_debug_ratio_ << " err=" << std::showpos << std::setprecision(3)
              << zoom_debug_error_ << std::noshowpos << " hold=" << zoom_pending_frames_
              << " area_err=" << static_cast<int>(std::round(zoom_debug_area_err_)) << " zfail=" << zoom_set_fail_streak_
              << " rec=" << (zoom_max_recover_active_ ? 1 : 0) << " zm=" << zmode
              << " hm=" << zoom_hold_miss_count_ << " rf=" << zoom_reacquire_freeze_left_
              << " fps=" << std::setprecision(1) << zoom_debug_fps_;
        } else {
          oss << "zoom=n/a ratio=n/a err=n/a"
              << " zm=" << zmode << " hm=" << zoom_hold_miss_count_ << " rf=" << zoom_reacquire_freeze_left_;
        }
        cv::putText(frame, oss.str(), cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
      }
      try {
        cv::imshow("tracker", frame);
      } catch (...) {
        window_initialized = false;
      }
      int key = -1;
      try {
        key = cv::waitKey(1);
      } catch (...) {
        key = -1;
      }
      if (key == '1') {
        const bool ok = camera_.set_zoom(cfg_.camera.zoom_min, cfg_.camera.zoom_min, cfg_.camera.zoom_max);
        if (ok) {
          zoom_cmd_ready_ = true;
          zoom_cmd_value_ = cfg_.camera.zoom_min;
          zoom_set_fail_streak_ = 0;
        }
        std::cerr << "[camera] manual zoom= " << cfg_.camera.zoom_min << (ok ? " ok\n" : " failed\n");
      } else if (key == '2') {
        const bool ok = camera_.set_zoom(cfg_.camera.zoom_max, cfg_.camera.zoom_min, cfg_.camera.zoom_max);
        if (ok) {
          zoom_cmd_ready_ = true;
          zoom_cmd_value_ = cfg_.camera.zoom_max;
          zoom_set_fail_streak_ = 0;
        }
        std::cerr << "[camera] manual zoom= " << cfg_.camera.zoom_max << (ok ? " ok\n" : " failed\n");
      }
      if (key == 27 || key == 'q' || key == 'Q') {
        running_.store(false);
        break;
      }
#ifdef HAVE_OPENCV
      // 允许直接点击窗口关闭按钮结束程序。
      try {
        if (cv::getWindowProperty("tracker", cv::WND_PROP_VISIBLE) < 1) {
          running_.store(false);
          break;
        }
      } catch (...) {
        window_initialized = false;
      }
#endif
    }

    ++frame_seq;
    if (mode_ == Mode::Replay && frame_seq > replay_max_frame_ + cfg_.mot.max_age + 5) {
      running_.store(false);
      break;
    }

    next_tick += std::chrono::duration_cast<clock::duration>(period);
    core::time::sleep_until(next_tick);
  }

  frame_cv_.notify_all();
  if (mode_ == Mode::Cam && cfg_.camera.show_window) {
    cv::destroyAllWindows();
  }
}

void TrackerNode::control_loop() {
  // 控制线程负责”读取共享状态 -> 滤波/预测 -> 计算偏差 -> 生成命令 -> 发给电机 -> 记录日志”。
  const double dt = 1.0 / static_cast<double>(cfg_.control.rate_hz);

  uint64_t last_gen = 0;
  while (running_.load()) {
    SharedState local;
    {
      std::unique_lock<std::mutex> lock(mu_);
      frame_cv_.wait(lock, [&] { return shared_generation_ != last_gen || !running_.load(); });
      if (!running_.load()) break;
      local = shared_;
      last_gen = shared_generation_;
    }

    // zoom 稳定性保护：镜头刚变化时，暂时不让控制命令直接驱动电机。
    const uint64_t now_ns = core::time::now_ns();
    double zoom_delta_for_log = 0.0;
    if (!control_zoom_initialized_ || !std::isfinite(control_last_zoom_value_)) {
      control_zoom_initialized_ = true;
      control_last_zoom_value_ = local.zoom_value;
      control_zoom_stable_since_ns_ = now_ns;
    } else {
      constexpr double kZoomStableEps = 1e-3;
      zoom_delta_for_log = local.zoom_value - control_last_zoom_value_;
      if (std::abs(local.zoom_value - control_last_zoom_value_) > kZoomStableEps) {
        control_last_zoom_value_ = local.zoom_value;
        control_zoom_stable_since_ns_ = now_ns;
        control_zero_sent_while_blocked_ = false;
      }
    }
    const uint64_t zoom_stable_required_ns =
        static_cast<uint64_t>(std::max(0, cfg_.camera.zoom_stable_required_ms)) * 1000000ULL;
    bool zoom_stable = true;
    if (zoom_stable_required_ns > 0) {
      zoom_stable = (now_ns >= control_zoom_stable_since_ns_) &&
                    ((now_ns - control_zoom_stable_since_ns_) >= zoom_stable_required_ns);
    }

    // 视觉线程和控制线程不同步，pred_dt 用于把目标状态外推到当前控制时刻。
    const double pred_dt = local.vision_time_ns > 0 ? static_cast<double>(now_ns - local.vision_time_ns) * 1e-9 : 0.0;

    // 从共享轨迹里找到当前被选中的控制目标。
    bool target_valid = false;
    core::TrackState ct;
    for (const auto& t : local.tracks) {
      if (t.track_id == local.controlled_id) {
        ct = t;
        target_valid = true;
        break;
      }
    }

    // 这批变量会同时服务于控制和日志记录。
    double dx_hat = 0.0;
    double dy_hat = 0.0;
    double vx_hat = 0.0;
    double vy_hat = 0.0;
    core::BBox controlled_bbox;
    core::BBox raw_bbox;
    double bbox_area_px = 0.0;
    double dx_raw = 0.0;
    double dy_raw = 0.0;
    double det_dx = 0.0;
    double det_dy = 0.0;
    double track_dx = 0.0;
    double track_dy = 0.0;
    double first_filter_dx = 0.0;
    double first_filter_dy = 0.0;
    double second_filter_dx = 0.0;
    double second_filter_dy = 0.0;
    double third_filter_dx = 0.0;
    double third_filter_dy = 0.0;
    double window8_second_filter_dx = 0.0;
    double window8_second_filter_dy = 0.0;
    double window8_third_filter_dx = 0.0;
    double window8_third_filter_dy = 0.0;
    double window8_10_third_filter_dx = 0.0;
    double window8_10_third_filter_dy = 0.0;
    double det_w = 0.0;
    double det_h = 0.0;
    double track_w = 0.0;
    double track_h = 0.0;
    double first_filter_w = 0.0;
    double first_filter_h = 0.0;
    double second_filter_w = 0.0;
    double second_filter_h = 0.0;
    double det_conf = 0.0;
    int coast_count = 0;
    std::string note = "lost";
    bool is_meas_update = false;
    double meas_age_ms = 0.0;

    // 先走位置滤波链，得到更稳定的中心和速度估计；否则直接使用轨迹预测值。
    if (cfg_.filter.enable) {
      if (target_valid) {
        if (!pnr_filter_.initialized()) {
          pnr_filter_.init(ct.cx, ct.cy);
          pnr_last_zoom_value_ = local.zoom_value;
        }

        pnr_filter_.predict(dt);

        filter::PnrFilterQuality q;
        q.conf = (ct.miss_count == 0) ? 1.0 : 0.35;
        q.miss_count = ct.miss_count;
        q.bbox_area = std::max(1.0, ct.bbox_hat.width() * ct.bbox_hat.height());
        q.zoom_value = local.zoom_value;
        q.zoom_delta = local.zoom_value - pnr_last_zoom_value_;
        q.det_count = local.det_count;
        pnr_last_zoom_value_ = local.zoom_value;

        // meas_only_when_matched 打开时，只在“本帧真实匹配到检测框”时更新滤波器。
        bool allow_update = (local.frame_seq != pnr_last_meas_frame_);
        if (cfg_.filter.meas_only_when_matched) {
          allow_update = allow_update && (ct.miss_count == 0);
        }
        if (allow_update) {
          pnr_filter_.update(ct.cx, ct.cy, q);
          is_meas_update = true;
          pnr_last_meas_frame_ = local.frame_seq;
          pnr_last_meas_time_ns_ = now_ns;
        }

        // raw 表示直接由轨迹外推得到的中心；hat 表示经过滤波器后的更稳定估计。
        const double cx_raw = ct.cx + ct.vx * pred_dt;
        const double cy_raw = ct.cy + ct.vy * pred_dt;
        const double w_raw = std::max(1.0, ct.w + ct.vw * pred_dt);
        const double h_raw = std::max(1.0, ct.h + ct.vh * pred_dt);
        const double cx_hat = pnr_filter_.x();
        const double cy_hat = pnr_filter_.y();
        const double w_hat = std::max(1.0, ct.w + ct.vw * pred_dt);
        const double h_hat = std::max(1.0, ct.h + ct.vh * pred_dt);

        dx_raw = cx_raw - static_cast<double>(cfg_.camera.center_x);
        dy_raw = static_cast<double>(cfg_.camera.center_y) - cy_raw;
        dx_hat = cx_hat - static_cast<double>(cfg_.camera.center_x);
        dy_hat = static_cast<double>(cfg_.camera.center_y) - cy_hat;
        vx_hat = pnr_filter_.vx();
        vy_hat = -pnr_filter_.vy();
        raw_bbox = vision::cxcywh_to_bbox(cx_raw, cy_raw, w_raw, h_raw);
        controlled_bbox = vision::cxcywh_to_bbox(cx_hat, cy_hat, w_hat, h_hat);
        bbox_area_px = std::max(1.0, raw_bbox.width() * raw_bbox.height());
        det_conf = ct.det_conf;
        coast_count = ct.miss_count;
        note = (ct.miss_count > 0) ? "coasting" : "tracked";
      } else {
        pnr_filter_.reset();
        pnr_last_meas_frame_ = -1;
        pnr_last_meas_time_ns_ = 0;
        pnr_last_zoom_value_ = local.zoom_value;
      }
    } else if (target_valid) {
      const double cx_raw = ct.cx + ct.vx * pred_dt;
      const double cy_raw = ct.cy + ct.vy * pred_dt;
      const double cx_hat = ct.cx + ct.vx * pred_dt;
      const double cy_hat = ct.cy + ct.vy * pred_dt;
      const double w_raw = std::max(1.0, ct.w + ct.vw * pred_dt);
      const double h_raw = std::max(1.0, ct.h + ct.vh * pred_dt);
      const double w_hat = std::max(1.0, ct.w + ct.vw * pred_dt);
      const double h_hat = std::max(1.0, ct.h + ct.vh * pred_dt);

      dx_raw = cx_raw - static_cast<double>(cfg_.camera.center_x);
      dy_raw = static_cast<double>(cfg_.camera.center_y) - cy_raw;
      dx_hat = cx_hat - static_cast<double>(cfg_.camera.center_x);
      dy_hat = static_cast<double>(cfg_.camera.center_y) - cy_hat;
      vx_hat = ct.vx;
      vy_hat = -ct.vy;
      raw_bbox = vision::cxcywh_to_bbox(cx_raw, cy_raw, w_raw, h_raw);
      controlled_bbox = vision::cxcywh_to_bbox(cx_hat, cy_hat, w_hat, h_hat);
      bbox_area_px = std::max(1.0, raw_bbox.width() * raw_bbox.height());
      det_conf = ct.det_conf;
      coast_count = ct.miss_count;
      note = (ct.miss_count > 0) ? "coasting" : "tracked";
    }

    if (pnr_last_meas_time_ns_ > 0 && now_ns >= pnr_last_meas_time_ns_) {
      meas_age_ms = static_cast<double>(now_ns - pnr_last_meas_time_ns_) * 1e-6;
    }

    const bool target_detected = target_valid && (ct.miss_count == 0);
    if (target_detected) {
      det_dx = ct.det_bbox.cx() - static_cast<double>(cfg_.camera.center_x);
      det_dy = static_cast<double>(cfg_.camera.center_y) - ct.det_bbox.cy();
      det_w = std::max(1.0, ct.det_bbox.width());
      det_h = std::max(1.0, ct.det_bbox.height());
    }

    const bool track_signal_valid = target_valid;
    const bool track_switched = track_signal_valid && ((!offset_filter_ready_) || (offset_filter_track_id_ != ct.track_id));
    double tracker_signal_dx = dx_hat;
    double tracker_signal_dy = dy_hat;
    if (track_signal_valid) {
      constexpr size_t kTrackPrefilterWindow = 3;
      if (track_switched) {
        track_prefilter_dx_window_.clear();
        track_prefilter_dy_window_.clear();
      }
      track_prefilter_dx_window_.push_back(dx_hat);
      track_prefilter_dy_window_.push_back(dy_hat);
      while (track_prefilter_dx_window_.size() > kTrackPrefilterWindow) {
        track_prefilter_dx_window_.pop_front();
      }
      while (track_prefilter_dy_window_.size() > kTrackPrefilterWindow) {
        track_prefilter_dy_window_.pop_front();
      }
      tracker_signal_dx = mean_from_window(track_prefilter_dx_window_);
      tracker_signal_dy = mean_from_window(track_prefilter_dy_window_);
      // 帧间差分限幅：单帧跳变超过阈值时截断，防止误检导致电机过冲
      constexpr double kMaxStepPx = 60.0;
      if (offset_filter_ready_) {
        const double step_x = tracker_signal_dx - offset_stage1_dx_;
        const double step_y = tracker_signal_dy - offset_stage1_dy_;
        if (std::abs(step_x) > kMaxStepPx) {
          tracker_signal_dx = offset_stage1_dx_ + std::copysign(kMaxStepPx, step_x);
        }
        if (std::abs(step_y) > kMaxStepPx) {
          tracker_signal_dy = offset_stage1_dy_ + std::copysign(kMaxStepPx, step_y);
        }
      }
    } else {
      track_prefilter_dx_window_.clear();
      track_prefilter_dy_window_.clear();
    }
    if (track_signal_valid) {
      if (!cfg_.filter.offset_lpf_enable || track_switched) {
        offset_stage1_dx_ = tracker_signal_dx;
        offset_stage1_dy_ = tracker_signal_dy;
        offset_stage2_dx_ = tracker_signal_dx;
        offset_stage2_dy_ = tracker_signal_dy;
        offset_stage3_dx_ = tracker_signal_dx;
        offset_stage3_dy_ = tracker_signal_dy;
        offset_stage2_dx_window_.clear();
        offset_stage2_dy_window_.clear();
        offset_stage3_dx_window_.clear();
        offset_stage3_dy_window_.clear();
        offset_window8_stage3_dx_window_.clear();
        offset_window8_stage3_dy_window_.clear();
        offset_filter_ready_ = true;
        offset_filter_track_id_ = ct.track_id;
      } else {
        const double alpha = cfg_.filter.offset_lpf_alpha;
        const double alpha_y = cfg_.filter.offset_lpf_alpha_y;
        offset_stage1_dx_ = alpha * tracker_signal_dx + (1.0 - alpha) * offset_stage1_dx_;
        offset_stage1_dy_ = alpha_y * tracker_signal_dy + (1.0 - alpha_y) * offset_stage1_dy_;
        offset_stage2_dx_ = alpha * offset_stage1_dx_ + (1.0 - alpha) * offset_stage2_dx_;
        offset_stage2_dy_ = alpha * offset_stage1_dy_ + (1.0 - alpha) * offset_stage2_dy_;
        offset_stage3_dx_ = alpha * offset_stage2_dx_ + (1.0 - alpha) * offset_stage3_dx_;
        offset_stage3_dy_ = alpha * offset_stage2_dy_ + (1.0 - alpha) * offset_stage3_dy_;
      }
      first_filter_dx = offset_stage1_dx_;
      first_filter_dy = offset_stage1_dy_;
      second_filter_dx = offset_stage2_dx_;
      second_filter_dy = offset_stage2_dy_;
      third_filter_dx = offset_stage3_dx_;
      third_filter_dy = offset_stage3_dy_;
      offset_stage2_dx_window_.push_back(second_filter_dx);
      offset_stage2_dy_window_.push_back(second_filter_dy);
      while (offset_stage2_dx_window_.size() > 8) {
        offset_stage2_dx_window_.pop_front();
      }
      while (offset_stage2_dy_window_.size() > 8) {
        offset_stage2_dy_window_.pop_front();
      }
      window8_second_filter_dx = mean_from_window(offset_stage2_dx_window_);
      window8_second_filter_dy = mean_from_window(offset_stage2_dy_window_);
      offset_stage3_dx_window_.push_back(third_filter_dx);
      offset_stage3_dy_window_.push_back(third_filter_dy);
      while (offset_stage3_dx_window_.size() > 8) {
        offset_stage3_dx_window_.pop_front();
      }
      while (offset_stage3_dy_window_.size() > 8) {
        offset_stage3_dy_window_.pop_front();
      }
      window8_third_filter_dx = mean_from_window(offset_stage3_dx_window_);
      window8_third_filter_dy = mean_from_window(offset_stage3_dy_window_);
      offset_window8_stage3_dx_window_.push_back(window8_third_filter_dx);
      offset_window8_stage3_dy_window_.push_back(window8_third_filter_dy);
      while (offset_window8_stage3_dx_window_.size() > 10) {
        offset_window8_stage3_dx_window_.pop_front();
      }
      while (offset_window8_stage3_dy_window_.size() > 10) {
        offset_window8_stage3_dy_window_.pop_front();
      }
      window8_10_third_filter_dx = mean_from_window(offset_window8_stage3_dx_window_);
      window8_10_third_filter_dy = mean_from_window(offset_window8_stage3_dy_window_);
    } else {
      offset_filter_ready_ = false;
      offset_filter_track_id_ = -1;
      offset_stage1_dx_ = 0.0;
      offset_stage1_dy_ = 0.0;
      offset_stage2_dx_ = 0.0;
      offset_stage2_dy_ = 0.0;
      offset_stage3_dx_ = 0.0;
      offset_stage3_dy_ = 0.0;
      offset_stage2_dx_window_.clear();
      offset_stage2_dy_window_.clear();
      offset_stage3_dx_window_.clear();
      offset_stage3_dy_window_.clear();
      offset_window8_stage3_dx_window_.clear();
      offset_window8_stage3_dy_window_.clear();
      bbox_size_second_filter_ready_ = false;
      bbox_size_second_filter_track_id_ = -1;
      bbox_size_second_filter_w_ = 0.0;
      bbox_size_second_filter_h_ = 0.0;
    }
    if (target_valid) {
      track_w = std::max(1.0, raw_bbox.width());
      track_h = std::max(1.0, raw_bbox.height());
      first_filter_w = std::max(1.0, controlled_bbox.width());
      first_filter_h = std::max(1.0, controlled_bbox.height());
      if (target_detected) {
        const bool size_track_switched =
            (!bbox_size_second_filter_ready_) || (bbox_size_second_filter_track_id_ != ct.track_id);
        if (size_track_switched) {
          bbox_size_second_filter_w_ = first_filter_w;
          bbox_size_second_filter_h_ = first_filter_h;
          bbox_size_second_filter_ready_ = true;
          bbox_size_second_filter_track_id_ = ct.track_id;
        } else {
          const double alpha = cfg_.model.bbox_size_filter_alpha;
          bbox_size_second_filter_w_ = alpha * first_filter_w + (1.0 - alpha) * bbox_size_second_filter_w_;
          bbox_size_second_filter_h_ = alpha * first_filter_h + (1.0 - alpha) * bbox_size_second_filter_h_;
        }
        second_filter_w = bbox_size_second_filter_w_;
        second_filter_h = bbox_size_second_filter_h_;
      }
    }
    double baseline_dx = track_signal_valid ? tracker_signal_dx : 0.0;
    double baseline_dy = track_signal_valid ? tracker_signal_dy : 0.0;
    double dx_ctrl = baseline_dx;
    double dy_ctrl = baseline_dy;
    double median_filter_dx = 0.0;
    double median_filter_dy = 0.0;
    double alpha_gate = 0.0;
    double delta_applied_x = 0.0;
    double delta_applied_y = 0.0;
    std::string infer_status = "disabled";
    int infer_used_model = 0;

    const double raw_input_dx = track_signal_valid ? tracker_signal_dx : 0.0;
    const double raw_input_dy = track_signal_valid ? tracker_signal_dy : 0.0;
    const auto temporal_out = temporal_compensator_.step(raw_input_dx, raw_input_dy, track_signal_valid, false);
    infer_status = temporal_out.status;
    if (track_signal_valid) {
      track_dx = tracker_signal_dx;
      track_dy = tracker_signal_dy;
      median_filter_dx_window_.push_back(track_dx);
      median_filter_dy_window_.push_back(track_dy);
      while (median_filter_dx_window_.size() > 3) {
        median_filter_dx_window_.pop_front();
      }
      while (median_filter_dy_window_.size() > 3) {
        median_filter_dy_window_.pop_front();
      }
      median_filter_dx = median_from_window(median_filter_dx_window_);
      median_filter_dy = median_from_window(median_filter_dy_window_);
      dx_ctrl = temporal_out.u_x;
      dy_ctrl = temporal_out.u_y;
      alpha_gate = temporal_out.alpha;
      delta_applied_x = temporal_out.delta_applied_x;
      delta_applied_y = temporal_out.delta_applied_y;
      infer_used_model = temporal_out.used_model ? 1 : 0;
    } else {
      median_filter_dx_window_.clear();
      median_filter_dy_window_.clear();
    }
    if (temporal_out.buffer_ready && !temporal_out.used_model && !temporal_comp_warned_ &&
        temporal_out.status != "warmup" && temporal_out.status != "disabled" && temporal_out.status != "target_invalid") {
      std::cerr << "[temporal_comp] infer fallback: " << temporal_out.status << ", use causal baseline only.\n";
      temporal_comp_warned_ = true;
    }

    // 电机接受位置误差、板上自带速度PID；本侧只做死区过滤，直接把 fused_dx/dy 发出去。
    core::MotorCmd cmd_base;
    core::MotorCmd cmd_sent;
    static bool prev_target_valid_y = false;
    static bool y_recover_mode = false;
    static double dy_f = 0.0;
    static double vy_last = 0.0;
    static int hold_count_y = 0;
    static bool hold_active_y = false;
    static double gyro_pitch_rate_f = 0.0;

    // ── 每帧超界检测（不依赖目标是否存在）────────────────────────────────────
    {
      constexpr double pitch_min_deg   = -60.0;
      constexpr double pitch_max_deg   =   3.0;
      constexpr double pitch_guard_deg =   5.0;
      if (motor_ && motor_->get_pitch_valid()) {
        const double p = motor_->get_pitch();
        if (p > pitch_max_deg - pitch_guard_deg || p < pitch_min_deg + pitch_guard_deg) {
          y_recover_mode = true;
          hold_count_y  = 0;
          hold_active_y = false;
        }
      }
    }

    if (target_detected) {
      constexpr double pitch_min_deg   = -60.0;
      constexpr double pitch_max_deg   =   3.0;
      constexpr double pitch_guard_deg =   5.0;
      constexpr double alpha_y         =   0.2;
      constexpr double alpha_g         =   0.2;
      constexpr double kp_y            =   0.2;
      constexpr double kg_y            =   1.0;
      constexpr double conf_low_y      =   0;
      constexpr double conf_stop_y     =   0;
      constexpr double vy_max          =  20.0;
      constexpr double hold_err_y      =   5.0;
      constexpr double hold_gyro_y     =   1.0;
      constexpr double release_err_y   =   8.0;
      constexpr double release_gyro_y  =   1.5;
      constexpr int    hold_count_threshold_y = 3;
      const bool positive_increases_angle = cfg_.actuator.y_pitch_positive_increases_angle;

      const bool   pitch_valid     = motor_ ? motor_->get_pitch_valid() : false;
      const double pitch_deg       = motor_ ? motor_->get_pitch()       : 0.0;
      const double gyro_pitch_rate = motor_ ? motor_->get_pitch_rate()  : 0.0;

      // X轴
      const double raw_x = pid_x_.step(-dx_ctrl, dt);
      cmd_base.cmd_x = (std::abs(dx_ctrl) > cfg_.control.deadband_px)
                           ? std::clamp(raw_x, -cfg_.control.cmd_limit, cfg_.control.cmd_limit)
                           : 0.0;

      // 目标重新获得时重置状态
      if (target_valid && !prev_target_valid_y) {
        dy_f = dy_ctrl;
        vy_last = 0.0;
        hold_count_y = 0;
        hold_active_y = false;
      }

      // 低通滤波
      dy_f = alpha_y * dy_ctrl + (1.0 - alpha_y) * dy_f;
      dy_f = std::clamp(dy_f, -40.0, 40.0);
      gyro_pitch_rate_f = alpha_g * gyro_pitch_rate + (1.0 - alpha_g) * gyro_pitch_rate_f;

      // 无有效俯仰角数据或死区内，禁止Y轴
      if (!pitch_valid || std::fabs(dy_f) < 3.0) {
        cmd_base.cmd_y = 0.0;
        vy_last = 0.0;
        prev_target_valid_y = target_valid;
      }
      else
      {
        // 分段增益 × 置信度缩放
        const double kp_scale   = (std::fabs(dy_f) < 20.0) ? 0.8 :
                                   (std::fabs(dy_f) < 35.0) ? 0.9 : 1.0;
        const double conf_scale = (det_conf < conf_low_y) ? 0.5 : 1.0;
        const double kp_eff     = kp_scale * conf_scale * kp_y;
        // vy_raw = 比例项 - 陀螺仪阻尼项
        double vy_raw = kp_eff * dy_f - kg_y * gyro_pitch_rate_f;
        vy_raw = std::clamp(vy_raw, -cfg_.control.cmd_limit, cfg_.control.cmd_limit);
        if (det_conf < conf_stop_y) vy_raw = 0.0;

        // Y轴虚拟减速（等效齿轮减速比，可调）
        constexpr double y_virtual_ratio = 2;
        vy_raw /= y_virtual_ratio;

        // 变化率限制 + 限幅
        double vy_cmd = vy_last + std::clamp(vy_raw - vy_last, -0.5, 0.5);
        vy_cmd = std::clamp(vy_cmd, -vy_max, vy_max);
        const double vy_cmd_pre_pitch_limit = vy_cmd;

        // 静止保持（仅正常跟踪模式）- 已注释
        // if (!y_recover_mode) { ... }

        // ── 最终输出层 pitch 硬限制 ──────────────────────────────────────────
        // 此处在 vy_cmd 已经过变化率限制、最终限幅、hold 门控之后执行，
        // 是真正的最后一道防线，确保任何情况下都不会越界。
        // invert_y=true 的含义：cmd_y > 0 → 电机使俯仰角减小；cmd_y < 0 → 角度增大。

        // 俯仰角接近上限（pitch_max_deg=3°）时进入保护区：
        // 此时禁止发 cmd_y < 0（负速度会继续增大角度，推向上限外）
        // std::max(vy_cmd, 0.0) 将负值截为 0，只允许正值（减小角度方向）或停止
        if (pitch_deg >= pitch_max_deg - pitch_guard_deg)
          vy_cmd = std::max(vy_cmd, 0.0);  // 上限保护：截断负速度，防止角度继续增大

        // 俯仰角接近下限（pitch_min_deg=-60°）时进入保护区：
        // 此时禁止发 cmd_y > 0（正速度会继续减小角度，推向下限外）
        // std::min(vy_cmd, 0.0) 将正值截为 0，只允许负值（增大角度方向）或停止
        if (pitch_deg <= pitch_min_deg + pitch_guard_deg)
          vy_cmd = std::min(vy_cmd, 0.0);  // 下限保护：截断正速度，防止角度继续减小

        if (pitch_deg >= pitch_max_deg - pitch_guard_deg) {
          vy_cmd = clamp_y_away_from_upper_limit(vy_cmd_pre_pitch_limit, positive_increases_angle);
        }
        if (pitch_deg <= pitch_min_deg + pitch_guard_deg) {
          vy_cmd = clamp_y_away_from_lower_limit(vy_cmd_pre_pitch_limit, positive_increases_angle);
        }
        vy_cmd = std::clamp(vy_cmd, -2.5, 2.5);
        vy_last = vy_cmd;
        cmd_base.cmd_y = vy_cmd;
        prev_target_valid_y = target_valid;
      }
    }
    else
    {
      pid_x_.reset();
      pid_y_.reset();
      cmd_base = core::MotorCmd{};
      prev_target_valid_y = false;
      dy_f = 0.0;
      vy_last = 0.0;
      hold_count_y = 0;
      hold_active_y = false;
      gyro_pitch_rate_f = 0.0;

      // 目标丢失时，若处于恢复模式，仍需继续驱动电机回到 home 位置
      if (y_recover_mode && motor_) {
        constexpr double pitch_home_deg = -30.0;
        constexpr double pitch_home_tol =   1.0;
        constexpr double vy_recover     =   2.0;
        const bool positive_increases_angle = cfg_.actuator.y_pitch_positive_increases_angle;
        const double pitch_deg = motor_->get_pitch();
        const bool   pitch_valid = motor_->get_pitch_valid();
        if (pitch_valid) {
          const double err = pitch_home_deg - pitch_deg;
          if (std::fabs(err) <= pitch_home_tol) {
            y_recover_mode = false;  // 已到达 home，退出恢复模式
            motor_->send(core::MotorCmd{0.0, 0.0});
          } else {
            // invert_y=true：cmd_y>0 使角度减小；err>0 需增大角度 → 发负值
            const double vy = (err > 0) ? -vy_recover : vy_recover;
            const double vy_runtime = signed_y_speed_for_pitch_error(err, vy_recover, positive_increases_angle);
            motor_->send(core::MotorCmd{0.0, vy_runtime});
          }
        } else {
          y_recover_mode = false;  // 无 pitch 数据，放弃恢复
        }
      } else {
        y_recover_mode = false;
      }
    }

    // 训练采集阶段暂时没有人工 expert 控制量，因此先用基础命令占位。
    core::MotorCmd cmd_expert = cmd_base;

    // 只有在”目标有效 + zoom 已稳定”时才允许真正发电机命令。
    const bool allow_motor_control = target_detected && zoom_stable;
    if (allow_motor_control) {
      cmd_sent = cmd_base;
      if (!std::isfinite(cmd_sent.cmd_x) || !std::isfinite(cmd_sent.cmd_y)) {
        pid_x_.reset();
        pid_y_.reset();
        cmd_sent = core::MotorCmd{};
      }
      if (motor_) {
        motor_->send(cmd_sent);
      }
      control_zero_sent_while_blocked_ = false;
    } else {
      pid_x_.reset();
      pid_y_.reset();
      cmd_sent = core::MotorCmd{};
      if (target_detected && !zoom_stable) {
        note = "zoom_guard";
      } else if (!target_detected && target_valid) {
        note = "coasting";
      } else if (!target_detected) {
        note = "lost";
      }
      // 进入阻塞态时补发一次零命令，避免执行器保持上一帧速度继续转。
      if (motor_ && !control_zero_sent_while_blocked_) {
        motor_->send(cmd_sent);
        control_zero_sent_while_blocked_ = true;
      }
    }

    // 最后一段把全过程状态打到 CSV，用于离线分析、训练和论文实验。
    if (control_dump_ofs_.is_open() && track_signal_valid) {
      using sys_clock = std::chrono::system_clock;
      const uint64_t dump_now_ms = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(sys_clock::now().time_since_epoch()).count());
      control_dump_ofs_ << dump_now_ms << ',' << local.frame_seq << ',' << ct.track_id << ',' << local.controlled_id
                        << ',' << det_conf << ',' << ct.miss_count << ',' << (ct.matched_in_frame ? 1 : 0) << ','
                        << det_dx << ',' << det_dy << ',' << track_dx << ',' << track_dy << ',' << median_filter_dx
                        << ',' << median_filter_dy << ',' << first_filter_dx << ',' << first_filter_dy << ','
                        << second_filter_dx << ',' << second_filter_dy << ',' << third_filter_dx << ','
                        << third_filter_dy << ',' << window8_second_filter_dx << ',' << window8_second_filter_dy << ','
                        << window8_third_filter_dx << ',' << window8_third_filter_dy << ','
                        << window8_10_third_filter_dx << ',' << window8_10_third_filter_dy << ','
                        << delta_applied_x << ',' << delta_applied_y << ',' << dx_ctrl << ',' << dy_ctrl << ','
                        << det_w << ',' << det_h << ',' << track_w << ',' << track_h << ',' << first_filter_w << ','
                        << first_filter_h << ',' << second_filter_w << ',' << second_filter_h << ','
                        << infer_used_model << ',' << infer_status << ','
                        << cmd_sent.cmd_x << ',' << cmd_sent.cmd_y << ','
                        << std::sqrt(dx_ctrl * dx_ctrl + dy_ctrl * dy_ctrl) << '\n';
      control_dump_ofs_.flush();
    }

    if (cfg_.log.enable) {
      using sys_clock = std::chrono::system_clock;
      const uint64_t now_ms = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::milliseconds>(sys_clock::now().time_since_epoch()).count());
      double dt_ms = static_cast<double>(dt * 1000.0);
      if (last_log_timestamp_ms_ > 0 && now_ms >= last_log_timestamp_ms_) {
        dt_ms = static_cast<double>(now_ms - last_log_timestamp_ms_);
      }
      last_log_timestamp_ms_ = now_ms;

      core::RuntimeSnapshot snap;
      snap.run_id = run_id_;
      snap.timestamp_ms = now_ms;
      snap.frame_id = local.frame_seq;
      snap.dt_ms = dt_ms;
      snap.img_w = cfg_.camera.width;
      snap.img_h = cfg_.camera.height;
      snap.bbox = controlled_bbox;
      snap.bbox_raw = raw_bbox;
      snap.bbox_area_px = bbox_area_px;
      snap.det_conf = det_conf;
      snap.dx_raw = dx_raw;
      snap.dy_raw = dy_raw;
      snap.dx_hat = dx_hat;
      snap.dy_hat = dy_hat;
      snap.clean_dx = dx_ctrl;
      snap.clean_dy = dy_ctrl;
      snap.vx_hat = vx_hat;
      snap.vy_hat = vy_hat;
      snap.lost_flag = target_detected ? 0 : 1;
      snap.is_meas_update = is_meas_update ? 1 : 0;
      snap.meas_age_ms = meas_age_ms;
      snap.zoom_value = local.zoom_value;
      snap.zoom_delta = zoom_delta_for_log;
      snap.vision_latency_ms = pred_dt * 1000.0;
      snap.cmd_base_x = cmd_base.cmd_x;
      snap.cmd_base_y = cmd_base.cmd_y;
      snap.cmd_expert_x = cmd_expert.cmd_x;
      snap.cmd_expert_y = cmd_expert.cmd_y;
      snap.cmd_sent_x = cmd_sent.cmd_x;
      snap.cmd_sent_y = cmd_sent.cmd_y;
      snap.reliability_score = 0.0;
      snap.alpha_gate = alpha_gate;
      snap.stage1_switch_gate = 0.0;
      snap.delta_cmd_x = delta_applied_x;
      snap.delta_cmd_y = delta_applied_y;
      snap.residual_clip_flag = 0;
      snap.slew_limit_flag = 0;
      snap.final_sat_flag = 0;
      snap.infer_used_model = infer_used_model;
      snap.fallback_delta_zero = temporal_out.used_model ? 0 : 1;
      snap.infer_status = infer_status;
      // 当前电机接口还没有回读编码器，所以执行器位置/速度先写占位值。
      snap.act_pos_x = 0.0;
      snap.act_pos_y = 0.0;
      snap.act_vel_x = 0.0;
      snap.act_vel_y = 0.0;
      snap.det_count = local.det_count;
      snap.track_count = static_cast<int>(local.tracks.size());
      snap.controlled_id = local.controlled_id;
      snap.coast_count = coast_count;
      snap.note = note;
      logger_.write(snap);
    }

  }
}

}  // namespace app
