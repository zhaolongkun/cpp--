#pragma once

#include <array>

#include "core/config.h"

namespace filter {

// 当前观测质量信息，用于自适应调节滤波强度。
struct PnrFilterQuality {
  double conf{1.0};
  int miss_count{0};
  double bbox_area{0.0};
  double zoom_value{0.0};
  double zoom_delta{0.0};
  int det_count{0};
};

// 供外部记录和调试的滤波内部状态。
struct PnrFilterDebug {
  double alpha_q{1.0};
  double alpha_r{1.0};
  double outlier_prob{0.0};
  double gate_d2{0.0};
  double model_prob_cv{0.5};
  double model_prob_ca{0.5};
  double bias_x{0.0};
  double bias_y{0.0};
  bool used_measurement{false};
};

// PNR-IMM-KF：双模型 IMM + 自适应噪声/偏置修正的工程滤波器。
class PnrImmKf {
 public:
  explicit PnrImmKf(const core::FilterConfig& cfg);

  void reset();
  bool initialized() const { return initialized_; }

  void init(double x_px, double y_px);
  void predict(double dt_sec);
  void update(double z_x_px, double z_y_px, const PnrFilterQuality& q);

  double x() const { return x_fused_[0]; }
  double y() const { return x_fused_[1]; }
  double vx() const { return x_fused_[2]; }
  double vy() const { return x_fused_[3]; }
  const PnrFilterDebug& debug() const { return dbg_; }

 private:
  // 内部启发式“神经适配器”的输出。
  struct NeuralOutput {
    double bias_x{0.0};
    double bias_y{0.0};
    double alpha_q{1.0};
    double alpha_r{1.0};
    double outlier_prob{0.0};
    double delta_switch{0.0};
  };

  NeuralOutput neural_step(double residual_x, double residual_y, const PnrFilterQuality& q);
  void mix_states(double switch_prob);
  void predict_models(double dt_sec, double alpha_q);
  double update_model(int model_idx, double z_x, double z_y, double alpha_r, double outlier_prob);
  void fuse_models();

  static constexpr int kN = 6;
  static constexpr int kM = 2;

  using Vec = std::array<double, kN>;
  using Mat = std::array<std::array<double, kN>, kN>;

  core::FilterConfig cfg_;
  bool initialized_{false};

  std::array<Vec, kM> x_models_{};
  std::array<Mat, kM> p_models_{};
  std::array<double, kM> model_prob_{0.5, 0.5};

  Vec x_fused_{};
  Mat p_fused_{};

  std::array<double, 8> h_{};  // 递归记忆状态，用于自适应调节
  bool neural_ready_{false};
  double switch_delta_{0.0};

  PnrFilterDebug dbg_{};
};

}  // namespace filter
