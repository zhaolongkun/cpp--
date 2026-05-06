// 传统 PID 控制器实现。
// 目前主控制链已经不再直接依赖 PID，但保留该实现用于兼容与实验对比。
#include "control/pid.h"
#include <algorithm>

namespace control {

PID::PID(double kp, double ki, double kd, double integral_limit)
    : kp_(kp), ki_(ki), kd_(kd), integral_limit_(integral_limit) {}

void PID::set_gains(double kp, double ki, double kd) {
  // 支持运行时切换参数，便于调参与实验。
  kp_ = kp;
  ki_ = ki;
  kd_ = kd;
}

void PID::reset() {
  // 清空积分项和上一时刻误差，避免状态跨目标残留。
  integral_ = 0.0;
  prev_error_ = 0.0;
  has_prev_ = false;
}

double PID::step(double error, double dt_sec) {
  // 时间步异常时直接输出零，防止导数项被放大。
  if (dt_sec <= 1e-9) {
    return 0.0;
  }
  // 积分项累计历史误差，用于消除稳态偏差。
  integral_ += error * dt_sec;
  integral_ = std::clamp(integral_, -integral_limit_, integral_limit_);

  double derivative = 0.0;
  if (has_prev_) {
    // 导数项描述误差变化速度。
    derivative = (error - prev_error_) / dt_sec;
  }

  prev_error_ = error;
  has_prev_ = true;

  // 返回标准 PID 三项叠加结果。
  return kp_ * error + ki_ * integral_ + kd_ * derivative;
}

}  // namespace control
