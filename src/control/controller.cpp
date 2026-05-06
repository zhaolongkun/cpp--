// 基础控制器：
// 根据目标偏差方向和距离大小，生成限幅后的二维电机速度命令。
#include "control/controller.h"

#include <algorithm>
#include <cmath>

namespace control {

Controller::Controller(const core::ControlConfig& cfg)
    : cfg_(cfg),
      slew_x_(cfg.slew_per_tick),
      slew_y_(cfg.slew_per_tick) {}

core::MotorCmd Controller::step(double dx_hat, double dy_hat, double dt_sec, bool target_valid) {
  // 当前版本控制律不依赖 dt_sec，保留参数是为了兼容旧接口。
  (void)dt_sec;
  core::MotorCmd cmd;
  if (!target_valid && cfg_.stop_when_lost) {
    reset();
    return cmd;
  }

  // 死区内的微小偏差不驱动电机，防止在中心附近反复抖动。
  if (std::abs(dx_hat) <= cfg_.deadband_px) {
    dx_hat = 0.0;
  }
  if (std::abs(dy_hat) <= cfg_.deadband_px) {
    dy_hat = 0.0;
  }

  // 用图像平面上的二维偏差计算“离中心有多远”。
  const double distance_px = std::hypot(dx_hat, dy_hat);
  if (distance_px <= 1e-9) {
    // 目标已经在中心附近时，让输出平滑回零。
    cmd.cmd_x = slew_x_.limit(0.0);
    cmd.cmd_y = slew_y_.limit(0.0);
    return cmd;
  }

  // 先归一化得到运动方向，再单独计算速度大小。
  const double dir_x = dx_hat / distance_px;
  const double dir_y = dy_hat / distance_px;

  // 距离越远，速度比例越大；达到 full_speed_px 后按满速输出。
  const double speed_ratio =
      std::clamp((distance_px - cfg_.deadband_px) / (cfg_.full_speed_px - cfg_.deadband_px), 0.0, 1.0);
  const double target_speed = cfg_.cmd_limit * speed_ratio;

  // 把“方向”和“速度标量”组合成二维电机命令。
  double cmd_x = dir_x * target_speed;
  double cmd_y = dir_y * target_speed;

  // 额外做一次二维范数限幅，保证合速度不超过最大速度。
  const double cmd_norm = std::hypot(cmd_x, cmd_y);
  if (cmd_norm > cfg_.cmd_limit && cmd_norm > 1e-9) {
    const double scale = cfg_.cmd_limit / cmd_norm;
    cmd_x *= scale;
    cmd_y *= scale;
  }

  // 单轴再做一次夹紧，避免数值误差导致越界。
  cmd_x = std::clamp(cmd_x, -cfg_.cmd_limit, cfg_.cmd_limit);
  cmd_y = std::clamp(cmd_y, -cfg_.cmd_limit, cfg_.cmd_limit);

  // 最后通过 slew limiter 限制单个控制周期内的变化量。
  cmd.cmd_x = slew_x_.limit(cmd_x);
  cmd.cmd_y = slew_y_.limit(cmd_y);
  return cmd;
}

void Controller::reset() {
  // 目标丢失或系统停机时，把输出状态重置到零。
  slew_x_.reset(0.0);
  slew_y_.reset(0.0);
}

}  // namespace control
