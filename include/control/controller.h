#pragma once

#include "control/slew_limiter.h"
#include "core/config.h"
#include "core/types.h"

namespace control {

class Controller {
 public:
  explicit Controller(const core::ControlConfig& cfg);

  // 根据目标偏差计算电机速度命令。
  // 当前实现采用“归一化方向 + 距离比例速度 + slew limit”的工程控制律。
  core::MotorCmd step(double dx_hat, double dy_hat, double dt_sec, bool target_valid);
  void reset();

 private:
  core::ControlConfig cfg_;
  SlewLimiter slew_x_;
  SlewLimiter slew_y_;
};

}  // namespace control
