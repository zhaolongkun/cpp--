#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#endif

#include "actuator/motor_driver_pyusb.h"

#include <algorithm>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace actuator {

// ── helpers ──────────────────────────────────────────────────────────────────

std::string MotorDriverPyUsb::quote_arg(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\\\"";
    else          out += c;
  }
  out += '"';
  return out;
}

bool MotorDriverPyUsb::parse_pitch_from_line(const std::string& line, double& pitch_deg, double& pitch_rate_dps) {
  // 格式：IMU,<timestamp>,<roll>,<pitch>,<yaw>,<roll_rate>,<pitch_rate>,<yaw_rate>
  // 下标：0       1          2       3      4       5            6           7
  if (line.empty()) return false;
  std::vector<std::string> fields;
  std::istringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) fields.push_back(tok);
  if (fields.size() < 7) return false;
  try {
    pitch_deg      = std::stod(fields[3]);
    pitch_rate_dps = std::stod(fields[6]);
    return true;
  } catch (...) {
    return false;
  }
}

// ── ctor / dtor ───────────────────────────────────────────────────────────────

MotorDriverPyUsb::MotorDriverPyUsb(core::ActuatorConfig cfg)
    : cfg_(std::move(cfg)) {
  if (!cfg_.y_pitch_takeover_enable) takeover_done_.store(true);
  if (cfg_.y_pitch_guard_enable) {
    start_pitch_guard();
  }
  start_command_loop();
}

MotorDriverPyUsb::~MotorDriverPyUsb() {
  stop_command_loop();
  stop_pitch_guard();
#ifdef _WIN32
  if (pipe_write_ != INVALID_HANDLE_VALUE) {
    if (cfg_.send_zero_on_close) write_command_pair(0, 0);
    CloseHandle(pipe_write_);
    pipe_write_ = INVALID_HANDLE_VALUE;
  }
  if (child_process_ != INVALID_HANDLE_VALUE) {
    TerminateProcess(child_process_, 0);
    CloseHandle(child_process_);
    child_process_ = INVALID_HANDLE_VALUE;
  }
#else
  if (pipe_) {
    if (cfg_.send_zero_on_close) write_command_pair(0, 0);
    pclose(pipe_);
    pipe_ = nullptr;
  }
#endif
}

// ── pipe ─────────────────────────────────────────────────────────────────────

bool MotorDriverPyUsb::ensure_open() {
#ifdef _WIN32
  if (pipe_write_ != INVALID_HANDLE_VALUE) return true;
  if (!cfg_.armed) return false;

  // 构造命令行：python.exe script.py --x-vid ... --y-vid ...
  std::ostringstream cmd;
  cmd << quote_arg(cfg_.python_exe) << " " << quote_arg(cfg_.bridge_script)
      << " --x_vid " << cfg_.x_vendor_id  << " --x_pid " << cfg_.x_product_id
      << " --y_vid " << cfg_.y_vendor_id  << " --y_pid " << cfg_.y_product_id;
  std::string cmdstr = cmd.str();
  std::cerr << "[motor_pyusb] cmd: " << cmdstr << '\n';

  // 创建匿名管道，子进程 stdin 读端，父进程写端
  HANDLE hReadPipe = INVALID_HANDLE_VALUE;
  HANDLE hWritePipe = INVALID_HANDLE_VALUE;
  SECURITY_ATTRIBUTES sa{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0)) {
    if (!warned_) {
      std::cerr << "[motor_pyusb] CreatePipe failed\n";
      warned_ = true;
    }
    return false;
  }
  // 父进程写端不需要被子进程继承
  SetHandleInformation(hWritePipe, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOA si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput  = hReadPipe;
  si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
  si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);

  PROCESS_INFORMATION pi{};
  if (!CreateProcessA(nullptr, cmdstr.data(), nullptr, nullptr,
                      TRUE, 0, nullptr, nullptr, &si, &pi)) {
    if (!warned_) {
      std::cerr << "[motor_pyusb] CreateProcess failed cmd: " << cmdstr << '\n';
      warned_ = true;
    }
    CloseHandle(hReadPipe);
    CloseHandle(hWritePipe);
    return false;
  }
  CloseHandle(hReadPipe);          // 父进程不需要读端
  CloseHandle(pi.hThread);
  pipe_write_    = hWritePipe;
  child_process_ = pi.hProcess;
  return true;
#else
  if (pipe_) return true;
  if (!cfg_.armed) return false;
  std::ostringstream cmd;
  cmd << quote_arg(cfg_.python_exe) << " " << quote_arg(cfg_.bridge_script)
      << " --x_vid " << cfg_.x_vendor_id  << " --x_pid " << cfg_.x_product_id
      << " --y_vid " << cfg_.y_vendor_id  << " --y_pid " << cfg_.y_product_id;
  pipe_ = popen(cmd.str().c_str(), "w");
  if (!pipe_ && !warned_) {
    std::cerr << "[motor_pyusb] popen failed\n";
    warned_ = true;
  }
  return pipe_ != nullptr;
#endif
}

void MotorDriverPyUsb::write_command_pair(int x, double y) {
  if (!ensure_open()) return;
  if (cfg_.invert_x) x = -x;
  if (cfg_.invert_y) y = -y;
  x = std::clamp(x, -cfg_.speed_limit, cfg_.speed_limit);
  y = std::clamp(y, static_cast<double>(-cfg_.speed_limit), static_cast<double>(cfg_.speed_limit));
#ifdef _WIN32
  std::ostringstream oss;
  oss << x << " " << std::fixed << std::setprecision(2) << y << "\n";
  std::string line = oss.str();
  DWORD written = 0;
  WriteFile(pipe_write_, line.c_str(), static_cast<DWORD>(line.size()), &written, nullptr);
#else
  std::fprintf(pipe_, "%d %.2f\n", x, y);
  std::fflush(pipe_);
#endif
  if (cfg_.debug)
    std::cerr << "[motor_pyusb] send x=" << x << " y=" << y << '\n';
}

// ── public send ───────────────────────────────────────────────────────────────

void MotorDriverPyUsb::send(const core::MotorCmd& cmd) {
  std::scoped_lock lock(cmd_mutex_);
  command_state_.tracker_cmd = cmd;
  command_state_.has_tracker_update = true;
  command_state_.last_tracker_update = std::chrono::steady_clock::now();
  command_state_.session_active = true;
}

void MotorDriverPyUsb::close() {
  stop_command_loop();
  stop_pitch_guard();
#ifdef _WIN32
  if (pipe_write_ != INVALID_HANDLE_VALUE) {
    if (cfg_.send_zero_on_close) write_command_pair(0, 0);
    CloseHandle(pipe_write_);
    pipe_write_ = INVALID_HANDLE_VALUE;
  }
  if (child_process_ != INVALID_HANDLE_VALUE) {
    TerminateProcess(child_process_, 0);
    CloseHandle(child_process_);
    child_process_ = INVALID_HANDLE_VALUE;
  }
#else
  if (pipe_) {
    if (cfg_.send_zero_on_close) write_command_pair(0, 0);
    pclose(pipe_);
    pipe_ = nullptr;
  }
#endif
}

// ── pitch guard thread ────────────────────────────────────────────────────────

void MotorDriverPyUsb::start_pitch_guard() {
  stop_pitch_thread_.store(false);
  pitch_thread_ = std::thread(&MotorDriverPyUsb::pitch_reader_loop, this);
}

void MotorDriverPyUsb::stop_pitch_guard() {
  stop_pitch_thread_.store(true);
  if (pitch_thread_.joinable()) pitch_thread_.join();
}

void MotorDriverPyUsb::pitch_reader_loop() {
#ifdef _WIN32
  const std::string port = cfg_.y_pitch_port;
  const int baud = cfg_.y_pitch_baud;

  std::string port_path = "\\\\.\\" + port;
  HANDLE h = CreateFileA(port_path.c_str(), GENERIC_READ, 0, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (h == INVALID_HANDLE_VALUE) {
    if (!pitch_warned_) {
      std::cerr << "[motor_pyusb] pitch serial open failed: " << port << '\n';
      pitch_warned_ = true;
    }
    return;
  }

  DCB dcb{};
  dcb.DCBlength = sizeof(dcb);
  GetCommState(h, &dcb);
  dcb.BaudRate = static_cast<DWORD>(baud);
  dcb.ByteSize = 8;
  dcb.Parity   = NOPARITY;
  dcb.StopBits = ONESTOPBIT;
  SetCommState(h, &dcb);

  COMMTIMEOUTS to{};
  to.ReadIntervalTimeout         = 50;
  to.ReadTotalTimeoutConstant    = 50;
  to.ReadTotalTimeoutMultiplier  = 10;
  SetCommTimeouts(h, &to);

  std::string buf;
  char ch = 0;
  DWORD nread = 0;

  while (!stop_pitch_thread_.load()) {
    if (ReadFile(h, &ch, 1, &nread, nullptr) && nread == 1) {
      if (ch == '\n') {
        double pitch = 0.0;
        double pitch_rate = 0.0;
        if (parse_pitch_from_line(buf, pitch, pitch_rate)) {
          update_pitch(pitch, pitch_rate);
        }
        buf.clear();
      } else if (ch != '\r') {
        buf += ch;
      }
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
  CloseHandle(h);
#else
  // Linux: 使用标准 termios
  (void)cfg_;
  if (!pitch_warned_) {
    std::cerr << "[motor_pyusb] pitch serial not implemented on this platform\n";
    pitch_warned_ = true;
  }
#endif
}

void MotorDriverPyUsb::update_pitch(double pitch_deg, double pitch_rate_dps) {
  std::scoped_lock lock(pitch_mutex_);
  const double alpha = std::clamp(cfg_.y_pitch_lpf_alpha, 0.01, 1.0);
  if (!pitch_state_.has_sample) {
    pitch_state_.filtered_pitch_deg = pitch_deg;
  } else {
    const double step = pitch_deg - pitch_state_.filtered_pitch_deg;
    pitch_state_.step_limited = (std::abs(step) > cfg_.y_pitch_max_step_deg);
    pitch_state_.raw_step_deg = step;
    pitch_state_.filtered_pitch_deg = alpha * pitch_deg + (1.0 - alpha) * pitch_state_.filtered_pitch_deg;
  }
  pitch_state_.raw_pitch_deg = pitch_deg;
  pitch_state_.pitch_rate_dps = pitch_rate_dps;
  pitch_state_.has_sample = true;
  pitch_state_.last_update = std::chrono::steady_clock::now();
}

// ── get_pitch_rate ────────────────────────────────────────────────────────────

double MotorDriverPyUsb::get_pitch_rate() const {
  std::scoped_lock lock(pitch_mutex_);
  return pitch_state_.pitch_rate_dps;
}

double MotorDriverPyUsb::get_pitch() const {
  std::scoped_lock lock(pitch_mutex_);
  return pitch_state_.filtered_pitch_deg;
}

bool MotorDriverPyUsb::get_pitch_valid() const {
  std::scoped_lock lock(pitch_mutex_);
  return pitch_state_.has_sample;
}

// ── guard_y_command ───────────────────────────────────────────────────────────
// 返回经过俯仰角保护后的 Y 命令值。
// 同时处理 takeover / recovery 状态。
double MotorDriverPyUsb::guard_y_command(double cmd_y) {
  if (!cfg_.y_pitch_guard_enable) return cmd_y;

  PitchState ps;
  {
    std::scoped_lock lock(pitch_mutex_);
    ps = pitch_state_;
  }

  // 串口超时：无有效数据时放行
  if (!ps.has_sample) return cmd_y;
  const auto now = std::chrono::steady_clock::now();
  const auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - ps.last_update).count();
  if (age_ms > cfg_.y_pitch_timeout_ms) return cmd_y;

  const double pitch = ps.filtered_pitch_deg;
  const double upper = cfg_.y_pitch_upper_stop_deg;   // -3°
  const double pmin  = cfg_.y_pitch_min_deg;          // -60°
  const double pmax  = cfg_.y_pitch_max_deg;          //  0°

  // positive_increases_angle: cmd_y>0 使角度增大（向上限方向）
  const bool pos_up = cfg_.y_pitch_positive_increases_angle;

  // 接近上限：不允许继续向上限方向运动
  if (pitch >= upper) {
    const bool moving_toward_upper = pos_up ? (cmd_y > 0) : (cmd_y < 0);
    if (moving_toward_upper) {
      if (cfg_.y_pitch_debug)
        std::cerr << "[pitch_guard] near upper limit pitch=" << pitch << ", blocking cmd_y=" << cmd_y << '\n';
      return 0.0;
    }
  }

  // 超出最大范围：强制反向
  if (pitch > pmax) {
    const double reverse = pos_up ? -std::abs(cmd_y) : std::abs(cmd_y);
    if (cfg_.y_pitch_debug)
      std::cerr << "[pitch_guard] above max pitch=" << pitch << ", reversing to " << reverse << '\n';
    return reverse;
  }

  // 超出最小范围：强制反向
  if (pitch < pmin) {
    const double reverse = pos_up ? std::abs(cmd_y) : -std::abs(cmd_y);
    if (cfg_.y_pitch_debug)
      std::cerr << "[pitch_guard] below min pitch=" << pitch << ", reversing to " << reverse << '\n';
    return reverse;
  }

  return cmd_y;
}

// ── command loop ──────────────────────────────────────────────────────────────

void MotorDriverPyUsb::start_command_loop() {
  stop_command_thread_.store(false);
  command_thread_ = std::thread(&MotorDriverPyUsb::command_loop, this);
}

void MotorDriverPyUsb::stop_command_loop() {
  stop_command_thread_.store(true);
  if (command_thread_.joinable()) command_thread_.join();
}

void MotorDriverPyUsb::command_loop() {
  using clock = std::chrono::steady_clock;
  using ms    = std::chrono::milliseconds;

  // ── takeover 初始化阶段 ──────────────────────────────────────────────────
  // 上电后立即发 cmd_y，使俯仰角从任意起始值减小到 takeover_target（-20°）。
  // 只要 pitch_valid 且 pitch <= target 才停止，其余情况一律发命令。
  bool takeover_done = !cfg_.y_pitch_takeover_enable;
  auto last_takeover_tick = clock::now();

  const int    takeover_interval_ms = cfg_.y_pitch_takeover_interval_ms;
  const double takeover_target      = cfg_.y_pitch_takeover_target_deg;
  const double takeover_speed       = cfg_.y_pitch_takeover_speed;
  // positive_increases_angle=true → cmd_y>0 使角度增大，要减小角度需发负值
  const int    takeover_cmd_y = cfg_.y_pitch_positive_increases_angle
                                ? -static_cast<int>(takeover_speed)
                                :  static_cast<int>(takeover_speed);

  while (!stop_command_thread_.load()) {
    std::this_thread::sleep_for(ms(1));

    const auto now = clock::now();

    // ── 读取当前俯仰角 ────────────────────────────────────────────────────
    double pitch = 0.0;
    bool   pitch_valid = false;
    {
      std::scoped_lock lock(pitch_mutex_);
      if (pitch_state_.has_sample) {
        const auto age = std::chrono::duration_cast<ms>(now - pitch_state_.last_update).count();
        if (age <= cfg_.y_pitch_timeout_ms) {
          pitch       = pitch_state_.filtered_pitch_deg;
          pitch_valid = true;
        }
      }
    }

    // ── takeover 阶段 ─────────────────────────────────────────────────────
    if (!takeover_done) {
      const auto elapsed = std::chrono::duration_cast<ms>(now - last_takeover_tick).count();
      if (elapsed < takeover_interval_ms) continue;
      last_takeover_tick = now;

      // 有角度数据且已到目标 → 交接
      if (pitch_valid && pitch <= takeover_target) {
        takeover_done = true;
        takeover_done_.store(true);
        write_command_pair(0, 0);
        if (cfg_.y_pitch_debug)
          std::cerr << "[takeover] done, pitch=" << pitch << '\n';
        continue;
      }

      // 否则一律发命令
      write_command_pair(0, takeover_cmd_y);
      if (cfg_.y_pitch_debug)
        std::cerr << "[takeover] cmd_y=" << takeover_cmd_y << " pitch=" << pitch << '\n';
      continue;
    }

    // ── 正常跟踪阶段：由检测框命令接手 ───────────────────────────────────
    CommandState cs;
    {
      std::scoped_lock lock(cmd_mutex_);
      cs = command_state_;
    }

    // session 超时检测
    bool session_timed_out = false;
    if (cs.session_active && cs.has_tracker_update) {
      const auto idle = std::chrono::duration_cast<ms>(now - cs.last_tracker_update).count();
      if (idle > cfg_.y_pitch_session_idle_ms) {
        session_timed_out = true;
      }
    }

    if (!cs.session_active || !cs.has_tracker_update || session_timed_out) {
      if (cs.last_sent_x != 0 || cs.last_sent_y != 0) {
        write_command_pair(0, 0);
        std::scoped_lock lock(cmd_mutex_);
        command_state_.last_sent_x = 0;
        command_state_.last_sent_y = 0;
      }
      continue;
    }

    const int    raw_x   = static_cast<int>(std::round(cs.tracker_cmd.cmd_x * cfg_.scale_x));
    const double raw_y   = cs.tracker_cmd.cmd_y * cfg_.scale_y;

    const double guarded_y = guard_y_command(raw_y);
    const double final_y   = guarded_y;

    if (raw_x != cs.last_sent_x || final_y != cs.last_sent_y) {
      write_command_pair(raw_x, final_y);
      std::scoped_lock lock(cmd_mutex_);
      command_state_.last_sent_x = raw_x;
      command_state_.last_sent_y = final_y;
    }
  }
}

}  // namespace actuator
