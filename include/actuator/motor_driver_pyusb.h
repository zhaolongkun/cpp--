#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include "actuator/motor_driver.h"
#include "core/config.h"

namespace actuator {

class MotorDriverPyUsb final : public MotorDriver {
 public:
  explicit MotorDriverPyUsb(core::ActuatorConfig cfg);
  ~MotorDriverPyUsb() override;

  void send(const core::MotorCmd& cmd) override;
  void close() override;

  bool takeover_done() const override { return takeover_done_.load(); }
  double get_pitch_rate() const override;
  double get_pitch() const override;
  bool get_pitch_valid() const override;

 private:
  struct CommandState {
    core::MotorCmd tracker_cmd{};
    bool has_tracker_update{false};
    bool session_active{false};
    bool takeover_active{false};
    bool recovery_active{false};
    int last_sent_x{0};
    double last_sent_y{0.0};
    std::chrono::steady_clock::time_point last_tracker_update{};
  };

  struct PitchState {
    bool has_sample{false};
    bool step_limited{false};
    double raw_pitch_deg{0.0};
    double filtered_pitch_deg{0.0};
    double raw_step_deg{0.0};
    double pitch_rate_dps{0.0};
    std::chrono::steady_clock::time_point last_update{};
  };

  bool ensure_open();
  void start_pitch_guard();
  void stop_pitch_guard();
  void start_command_loop();
  void stop_command_loop();
  void command_loop();
  void write_command_pair(int x, double y);
  void pitch_reader_loop();
  double guard_y_command(double cmd_y);
  void update_pitch(double pitch_deg, double pitch_rate_dps);
  static bool parse_pitch_from_line(const std::string& line, double& pitch_deg, double& pitch_rate_dps);
  static std::string quote_arg(const std::string& s);

  core::ActuatorConfig cfg_;
  bool warned_{false};
  bool pitch_warned_{false};
  std::atomic<bool> stop_pitch_thread_{false};
  std::atomic<bool> stop_command_thread_{false};
  std::atomic<bool> takeover_done_{false};
  std::thread pitch_thread_;
  std::thread command_thread_;
  std::mutex cmd_mutex_;
  mutable std::mutex pitch_mutex_;
  CommandState command_state_;
  PitchState pitch_state_;

#ifdef _WIN32
  HANDLE pipe_write_{INVALID_HANDLE_VALUE};
  HANDLE child_process_{INVALID_HANDLE_VALUE};
#else
  FILE* pipe_{nullptr};
#endif
};

}  // namespace actuator
