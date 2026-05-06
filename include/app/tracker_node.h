#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "actuator/motor_driver_mock.h"
#include "actuator/motor_driver_pyusb.h"
#include "control/controller.h"
#include "control/pid.h"
#include "control/target_selector.h"
#include "control/temporal_compensator_onnx.h"
#include "core/config.h"
#include "core/logger.h"
#include "filter/pnr_imm_kf.h"
#include "tracking/multi_object_tracker.h"
#include "vision/shm_detection_reader.h"
#include "vision/camera_capture.h"
#include "vision/yolo_detector.h"

namespace app {

class TrackerNode {
 public:
  // Cam: 实时相机输入
  // Replay: 从离线 CSV 回放检测序列
  enum class Mode { Cam, Replay };
  // 自动变焦状态机。
  enum class ZoomMode { Locked, Hold, Search };

  TrackerNode(core::AppConfig cfg, Mode mode, std::string replay_csv, uint64_t max_runtime_ms = 0);
  ~TrackerNode();

  // 初始化相机、模型、执行器、日志和线程依赖资源。
  bool init();
  // 启动视觉线程和控制线程。
  void run();
  void stop();
  // 阻塞直到 actuator takeover 完成（或超时）。
  void wait_actuator_ready();
  void init_pitch_angle();

 private:
  // 视觉线程与控制线程之间共享的最小状态。
  struct SharedState {
    int frame_seq{0};
    int det_count{0};
    std::vector<core::TrackState> tracks;
    int controlled_id{-1};
    uint64_t vision_time_ns{0};
    double zoom_value{0.0};
  };

  bool load_replay_csv(const std::string& csv_path);
  std::vector<core::Detection> replay_detections_for_frame(int frame_seq) const;
  bool runtime_limit_reached(uint64_t now_ns) const;

  // 视觉线程：采图/回放 -> 检测 -> 跟踪 -> 共享状态更新
  void vision_loop();
  // 控制线程：读共享状态 -> 滤波/补偿 -> 电机命令输出
  void control_loop();
  // 自动变焦策略入口。
  void auto_zoom_step(const std::vector<core::TrackState>& tracks, int controlled_id);

  core::AppConfig cfg_;
  Mode mode_{Mode::Replay};
  std::string replay_csv_path_;

  vision::CameraCapture camera_;
  vision::YoloDetector detector_;
  vision::ShmDetectionReader shm_reader_;
  tracking::MultiObjectTracker mot_;
  control::TargetSelector selector_;
  control::Controller controller_;
  control::PID pid_x_;
  control::PID pid_y_;
  control::TemporalCompensatorOnnx temporal_compensator_;
  filter::PnrImmKf pnr_filter_;
  std::unique_ptr<actuator::MotorDriver> motor_;
  core::CsvLogger logger_;
  std::ofstream control_dump_ofs_;
  std::string control_dump_path_{"D:/kun-data/kun-code-data/反无/cpp智能控制/data/test.csv"};

  mutable std::mutex mu_;
  std::condition_variable frame_cv_;
  SharedState shared_;
  uint64_t shared_generation_{0};

  std::unordered_map<int, std::vector<core::Detection>> replay_by_frame_;
  int replay_max_frame_{-1};
  bool zoom_ema_ready_{false};
  double zoom_ratio_ema_{0.0};
  int zoom_no_target_frames_{0};
  int zoom_pending_sign_{0};
  int zoom_pending_frames_{0};
  bool zoom_cmd_ready_{false};
  double zoom_cmd_value_{0.0};
  ZoomMode zoom_mode_{ZoomMode::Search};
  int zoom_hold_miss_count_{0};
  int zoom_reacquire_freeze_left_{0};
  int zoom_reacquire_confirm_count_{0};
  int zoom_search_direction_{1};  // +1 放大, -1 缩小
  int zoom_set_fail_streak_{0};
  bool zoom_prev_ratio_ready_{false};
  double zoom_prev_ratio_{0.0};
  bool zoom_prev_area_ready_{false};
  double zoom_prev_area_{0.0};
  int zoom_prev_track_id_{-1};
  std::vector<double> zoom_area_window_;
  bool zoom_area_filtered_ready_{false};
  double zoom_area_filtered_{0.0};
  bool zoom_hysteresis_active_{false};
  int zoom_lost_at_max_frames_{0};
  bool zoom_max_recover_active_{false};
  uint64_t zoom_last_action_ns_{0};
  double zoom_debug_area_err_{0.0};
  double zoom_debug_fps_{0.0};
  uint64_t zoom_last_vision_ns_{0};
  bool zoom_strategy_armed_{false};
  bool zoom_not_supported_reported_{false};
  bool zoom_debug_valid_{false};
  double zoom_debug_ratio_{0.0};
  double zoom_debug_error_{0.0};
  double zoom_debug_value_{0.0};
  int pnr_last_meas_frame_{-1};
  uint64_t pnr_last_meas_time_ns_{0};
  double pnr_last_zoom_value_{0.0};
  bool control_zoom_initialized_{false};
  double control_last_zoom_value_{0.0};
  uint64_t control_zoom_stable_since_ns_{0};
  bool control_zero_sent_while_blocked_{false};
  bool temporal_comp_warned_{false};
  std::deque<double> median_filter_dx_window_;
  std::deque<double> median_filter_dy_window_;
  std::deque<double> track_prefilter_dx_window_;
  std::deque<double> track_prefilter_dy_window_;
  bool offset_filter_ready_{false};
  int offset_filter_track_id_{-1};
  double offset_stage1_dx_{0.0};
  double offset_stage1_dy_{0.0};
  double offset_stage2_dx_{0.0};
  double offset_stage2_dy_{0.0};
  double offset_stage3_dx_{0.0};
  double offset_stage3_dy_{0.0};
  std::deque<double> offset_stage2_dx_window_;
  std::deque<double> offset_stage2_dy_window_;
  std::deque<double> offset_stage3_dx_window_;
  std::deque<double> offset_stage3_dy_window_;
  std::deque<double> offset_window8_stage3_dx_window_;
  std::deque<double> offset_window8_stage3_dy_window_;
  bool bbox_size_filter_ready_{false};
  int bbox_size_filter_track_id_{-1};
  std::deque<double> bbox_size_filter_cx_window_;
  std::deque<double> bbox_size_filter_cy_window_;
  std::deque<double> bbox_size_filter_w_window_;
  std::deque<double> bbox_size_filter_h_window_;
  double bbox_size_filter_cx_{0.0};
  double bbox_size_filter_cy_{0.0};
  double bbox_size_filter_w_{0.0};
  double bbox_size_filter_h_{0.0};
  bool bbox_size_second_filter_ready_{false};
  int bbox_size_second_filter_track_id_{-1};
  double bbox_size_second_filter_w_{0.0};
  double bbox_size_second_filter_h_{0.0};
  std::string run_id_{"run_default"};
  uint64_t last_log_timestamp_ms_{0};
  uint64_t max_runtime_ms_{0};
  uint64_t run_start_ns_{0};

  std::atomic<bool> running_{false};
  std::thread vision_thread_;
  std::thread control_thread_;
};

}  // namespace app
