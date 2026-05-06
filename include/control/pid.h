#pragma once

namespace control {

class PID {
 public:
  PID() = default;
  // integral_limit: 积分项上限（对称），防止 ki 非零时积分飞散。
  PID(double kp, double ki, double kd, double integral_limit = 500.0);

  // 在线修改增益参数。
  void set_gains(double kp, double ki, double kd);
  void reset();
  // 单步 PID 计算，error 为当前误差，dt_sec 为控制周期。
  double step(double error, double dt_sec);

 private:
  double kp_{0.0};
  double ki_{0.0};
  double kd_{0.0};
  double integral_limit_{500.0};
  double integral_{0.0};
  double prev_error_{0.0};
  bool has_prev_{false};
};

}  // namespace control
