#include "control/slew_limiter.h"

#include <algorithm>

namespace control {

double SlewLimiter::limit(double target) {
  // 每个控制周期只允许输出变化一定幅度，避免控制命令突变过大，
  // 这一步主要用于保护执行机构，同时降低电机速度指令的抖动感。
  const double lo = value_ - per_tick_;
  const double hi = value_ + per_tick_;
  value_ = std::clamp(target, lo, hi);
  return value_;
}

}  // namespace control
