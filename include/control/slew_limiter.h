#pragma once

namespace control {

class SlewLimiter {
 public:
  explicit SlewLimiter(double per_tick = 0.0) : per_tick_(per_tick) {}

  // 每个控制 tick 允许的最大变化量。
  void set_per_tick(double per_tick) { per_tick_ = per_tick; }
  void reset(double value = 0.0) { value_ = value; }
  // 对目标值做斜率限制，避免命令突变。
  double limit(double target);

 private:
  double per_tick_{0.0};
  double value_{0.0};
};

}  // namespace control
