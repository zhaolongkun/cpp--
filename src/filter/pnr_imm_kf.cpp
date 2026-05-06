#include "filter/pnr_imm_kf.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

using Vec6 = std::array<double, 6>;
using Mat6 = std::array<std::array<double, 6>, 6>;

constexpr double kPi = 3.14159265358979323846;

double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

double sigmoid(double x) {
  if (x >= 0.0) {
    const double e = std::exp(-x);
    return 1.0 / (1.0 + e);
  }
  const double e = std::exp(x);
  return e / (1.0 + e);
}

Vec6 vec_zero() {
  // 下面这一组 helper 都是为了避免引入大型线性代数库，
  // 直接用定长数组把 6 维 IMM/KF 所需运算写清楚。
  Vec6 out{};
  out.fill(0.0);
  return out;
}

Mat6 mat_zero() {
  Mat6 out{};
  for (auto& r : out) {
    r.fill(0.0);
  }
  return out;
}

Mat6 mat_eye() {
  Mat6 out = mat_zero();
  for (int i = 0; i < 6; ++i) {
    out[i][i] = 1.0;
  }
  return out;
}

Vec6 mat_vec_mul(const Mat6& a, const Vec6& x) {
  Vec6 out = vec_zero();
  for (int r = 0; r < 6; ++r) {
    double s = 0.0;
    for (int c = 0; c < 6; ++c) {
      s += a[r][c] * x[c];
    }
    out[r] = s;
  }
  return out;
}

Mat6 mat_mul(const Mat6& a, const Mat6& b) {
  Mat6 out = mat_zero();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      double s = 0.0;
      for (int k = 0; k < 6; ++k) {
        s += a[r][k] * b[k][c];
      }
      out[r][c] = s;
    }
  }
  return out;
}

Mat6 mat_add(const Mat6& a, const Mat6& b) {
  Mat6 out = mat_zero();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out[r][c] = a[r][c] + b[r][c];
    }
  }
  return out;
}

Mat6 mat_sub(const Mat6& a, const Mat6& b) {
  Mat6 out = mat_zero();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out[r][c] = a[r][c] - b[r][c];
    }
  }
  return out;
}

Mat6 mat_transpose(const Mat6& a) {
  Mat6 out = mat_zero();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out[r][c] = a[c][r];
    }
  }
  return out;
}

void enforce_symmetric(Mat6& p) {
  for (int r = 0; r < 6; ++r) {
    for (int c = r + 1; c < 6; ++c) {
      const double v = 0.5 * (p[r][c] + p[c][r]);
      p[r][c] = v;
      p[c][r] = v;
    }
    p[r][r] = std::max(1e-9, p[r][r]);
  }
}

Mat6 outer(const Vec6& a, const Vec6& b) {
  Mat6 out = mat_zero();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out[r][c] = a[r] * b[c];
    }
  }
  return out;
}

Vec6 vec_add(const Vec6& a, const Vec6& b) {
  Vec6 out = vec_zero();
  for (int i = 0; i < 6; ++i) {
    out[i] = a[i] + b[i];
  }
  return out;
}

Vec6 vec_sub(const Vec6& a, const Vec6& b) {
  Vec6 out = vec_zero();
  for (int i = 0; i < 6; ++i) {
    out[i] = a[i] - b[i];
  }
  return out;
}

Vec6 vec_scale(const Vec6& a, double s) {
  Vec6 out = vec_zero();
  for (int i = 0; i < 6; ++i) {
    out[i] = a[i] * s;
  }
  return out;
}

Mat6 mat_scale(const Mat6& a, double s) {
  Mat6 out = mat_zero();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out[r][c] = a[r][c] * s;
    }
  }
  return out;
}

Mat6 build_f_cv(double dt) {
  // CV: constant velocity，状态只对位置和速度做传播。
  Mat6 f = mat_eye();
  f[0][2] = dt;
  f[1][3] = dt;
  f[4][4] = 0.60;
  f[5][5] = 0.60;
  return f;
}

Mat6 build_f_ca(double dt) {
  // CA: constant acceleration，额外把加速度项纳入传播。
  Mat6 f = mat_eye();
  const double dt2 = dt * dt;
  f[0][2] = dt;
  f[1][3] = dt;
  f[0][4] = 0.5 * dt2;
  f[1][5] = 0.5 * dt2;
  f[2][4] = dt;
  f[3][5] = dt;
  return f;
}

Mat6 build_q(double dt, double q_pos, double q_vel, double q_acc, bool use_ca_model) {
  // 过程噪声矩阵采用对角近似，工程上重点是可调和稳定。
  Mat6 q = mat_zero();
  const double dt2 = dt * dt;
  q[0][0] = q_pos * dt2;
  q[1][1] = q_pos * dt2;
  q[2][2] = q_vel * dt;
  q[3][3] = q_vel * dt;
  q[4][4] = q_acc * (use_ca_model ? dt : 1.0);
  q[5][5] = q_acc * (use_ca_model ? dt : 1.0);
  return q;
}

bool invert_2x2(double a00, double a01, double a10, double a11, double& i00, double& i01, double& i10, double& i11,
                double& det_out) {
  const double det = a00 * a11 - a01 * a10;
  det_out = det;
  if (std::abs(det) < 1e-12) {
    return false;
  }
  const double inv_det = 1.0 / det;
  i00 = a11 * inv_det;
  i01 = -a01 * inv_det;
  i10 = -a10 * inv_det;
  i11 = a00 * inv_det;
  return true;
}

}  // namespace

namespace filter {

PnrImmKf::PnrImmKf(const core::FilterConfig& cfg) : cfg_(cfg) { reset(); }

void PnrImmKf::reset() {
  // 清空滤波器内部状态、模型概率和“类神经”自适应记忆。
  initialized_ = false;
  x_fused_ = vec_zero();
  p_fused_ = mat_zero();
  for (auto& x : x_models_) {
    x = vec_zero();
  }
  for (auto& p : p_models_) {
    p = mat_zero();
  }
  model_prob_ = {0.5, 0.5};
  h_.fill(0.0);
  neural_ready_ = false;
  switch_delta_ = 0.0;
  dbg_ = PnrFilterDebug{};
}

void PnrImmKf::init(double x_px, double y_px) {
  // 两个子模型都从同一个初始位置出发，只是后续传播模型不同。
  initialized_ = true;
  model_prob_ = {0.5, 0.5};
  for (int j = 0; j < kM; ++j) {
    auto& x = x_models_[j];
    auto& p = p_models_[j];
    x = vec_zero();
    x[0] = x_px;
    x[1] = y_px;
    p = mat_zero();
    p[0][0] = cfg_.r_pos;
    p[1][1] = cfg_.r_pos;
    p[2][2] = 400.0;
    p[3][3] = 400.0;
    p[4][4] = 400.0;
    p[5][5] = 400.0;
  }
  x_fused_ = x_models_[0];
  p_fused_ = p_models_[0];
  dbg_ = PnrFilterDebug{};
}

void PnrImmKf::mix_states(double switch_prob) {
  // IMM 的核心步骤之一：
  // 在预测前先按模型转移概率把两个模型的状态做混合，
  // 表达“当前帧可能从另一种运动模式切过来”。
  std::array<std::array<double, 2>, 2> pi{};
  pi[0][0] = 1.0 - switch_prob;
  pi[0][1] = switch_prob;
  pi[1][0] = switch_prob;
  pi[1][1] = 1.0 - switch_prob;

  std::array<double, 2> c{};
  c[0] = model_prob_[0] * pi[0][0] + model_prob_[1] * pi[1][0];
  c[1] = model_prob_[0] * pi[0][1] + model_prob_[1] * pi[1][1];
  c[0] = std::max(1e-9, c[0]);
  c[1] = std::max(1e-9, c[1]);

  std::array<Vec, kM> x_mix{};
  std::array<Mat, kM> p_mix{};

  for (int j = 0; j < kM; ++j) {
    Vec xj = vec_zero();
    for (int i = 0; i < kM; ++i) {
      const double w = model_prob_[i] * pi[i][j] / c[j];
      xj = vec_add(xj, vec_scale(x_models_[i], w));
    }
    x_mix[j] = xj;

    Mat pj = mat_zero();
    for (int i = 0; i < kM; ++i) {
      const double w = model_prob_[i] * pi[i][j] / c[j];
      const Vec dx = vec_sub(x_models_[i], xj);
      const Mat spread = mat_add(p_models_[i], outer(dx, dx));
      pj = mat_add(pj, mat_scale(spread, w));
    }
    enforce_symmetric(pj);
    p_mix[j] = pj;
  }

  x_models_ = x_mix;
  p_models_ = p_mix;
  model_prob_[0] = c[0];
  model_prob_[1] = c[1];
  const double sum = model_prob_[0] + model_prob_[1];
  model_prob_[0] /= std::max(1e-9, sum);
  model_prob_[1] /= std::max(1e-9, sum);
}

void PnrImmKf::predict_models(double dt_sec, double alpha_q) {
  const double dt = std::clamp(dt_sec, 1e-4, 0.2);
  const double qpos = cfg_.q_pos * alpha_q;
  const double qvel = cfg_.q_vel * alpha_q;
  const double qacc = cfg_.q_acc * alpha_q;

  const Mat f_cv = build_f_cv(dt);
  const Mat f_ca = build_f_ca(dt);
  const Mat q_cv = build_q(dt, qpos, qvel, qacc, false);
  const Mat q_ca = build_q(dt, qpos, qvel, qacc, true);

  for (int j = 0; j < kM; ++j) {
    // 模型 0 使用 CV，模型 1 使用 CA。
    const Mat& f = (j == 0) ? f_cv : f_ca;
    const Mat& q = (j == 0) ? q_cv : q_ca;
    x_models_[j] = mat_vec_mul(f, x_models_[j]);
    Mat p = mat_mul(mat_mul(f, p_models_[j]), mat_transpose(f));
    p_models_[j] = mat_add(p, q);
    enforce_symmetric(p_models_[j]);
  }
}

void PnrImmKf::predict(double dt_sec) {
  if (!initialized_) {
    return;
  }
  // 过程噪声和模型切换概率都不是固定常数，
  // 而是根据上一帧神经启发模块的输出动态调节。
  const double alpha_q = std::clamp(dbg_.alpha_q, cfg_.alpha_q_min, cfg_.alpha_q_max);
  const double switch_prob = std::clamp(cfg_.switch_prob + 0.08 * switch_delta_, 0.001, 0.45);
  mix_states(switch_prob);
  predict_models(dt_sec, alpha_q);
  fuse_models();
  dbg_.used_measurement = false;
}

PnrImmKf::NeuralOutput PnrImmKf::neural_step(double residual_x, double residual_y, const PnrFilterQuality& q) {
  NeuralOutput out;
  if (!cfg_.neural_enable) {
    return out;
  }

  // 这一段不是训练得到的真实神经网络，而是手工设计的“小型递归启发器”：
  // 输入残差、检测置信度、丢失次数、变焦等质量信息，
  // 输出对 Q/R、自适应偏置、离群概率、模型切换倾向的估计。
  const double conf = clamp01(q.conf);
  const double miss = std::clamp(static_cast<double>(q.miss_count) / 10.0, 0.0, 1.5);
  const double area = std::clamp(std::log1p(std::max(0.0, q.bbox_area)) / 10.0, 0.0, 2.0);
  const double zoom = std::clamp(q.zoom_value / 260.0, 0.0, 1.5);
  const double zoom_rate = std::clamp(q.zoom_delta / 30.0, -1.0, 1.0);
  const double rx = std::clamp(residual_x / 120.0, -2.5, 2.5);
  const double ry = std::clamp(residual_y / 120.0, -2.5, 2.5);

  std::array<double, 8> nh{};
  nh[0] = std::tanh(0.72 * h_[0] + 0.35 * rx - 0.18 * (1.0 - conf) + 0.12 * miss + 0.05 * zoom_rate);
  nh[1] = std::tanh(0.70 * h_[1] + 0.35 * ry - 0.18 * (1.0 - conf) + 0.12 * miss - 0.05 * zoom_rate);
  nh[2] = std::tanh(0.66 * h_[2] + 0.30 * std::abs(rx) + 0.30 * std::abs(ry) + 0.15 * miss + 0.10 * area);
  nh[3] = std::tanh(0.68 * h_[3] + 0.40 * (1.0 - conf) + 0.18 * miss + 0.08 * std::abs(zoom_rate));
  nh[4] = std::tanh(0.64 * h_[4] + 0.30 * zoom + 0.20 * zoom_rate);
  nh[5] = std::tanh(0.65 * h_[5] + 0.20 * area - 0.12 * miss);
  nh[6] = std::tanh(0.62 * h_[6] + 0.30 * (std::abs(rx) + std::abs(ry)) - 0.15 * conf);
  nh[7] = std::tanh(0.62 * h_[7] + 0.20 * static_cast<double>(q.det_count > 0) - 0.10 * miss);
  h_ = nh;
  neural_ready_ = true;

  const double bias_lat_x = 0.8 * h_[0] + 0.2 * h_[2] - 0.15 * h_[3];
  const double bias_lat_y = 0.8 * h_[1] + 0.2 * h_[2] - 0.15 * h_[3];
  out.bias_x = cfg_.bias_limit_px * 0.5 * std::tanh(bias_lat_x);
  out.bias_y = cfg_.bias_limit_px * 0.5 * std::tanh(bias_lat_y);

  const double q_log = std::clamp(0.50 * h_[2] + 0.20 * h_[6] + 0.15 * miss - 0.10 * conf, -1.2, 1.2);
  out.alpha_q = std::clamp(std::exp(q_log), cfg_.alpha_q_min, cfg_.alpha_q_max);

  const double r_log = std::clamp(0.75 * h_[3] + 0.30 * h_[6] + 0.15 * miss + 0.10 * (1.0 - conf), -1.4, 1.6);
  out.alpha_r = std::clamp(std::exp(r_log), cfg_.alpha_r_min, cfg_.alpha_r_max);

  const double outlier_logits =
      1.2 * (std::abs(rx) + std::abs(ry)) + 0.9 * miss + 0.5 * (1.0 - conf) + 0.15 * std::abs(zoom_rate) - 1.7;
  out.outlier_prob =
      std::clamp(sigmoid(outlier_logits), cfg_.outlier_prob_min, cfg_.outlier_prob_max);

  out.delta_switch = std::tanh(0.55 * (h_[0] * h_[0] + h_[1] * h_[1]) + 0.20 * h_[6] - 0.10 * conf - 0.10);
  return out;
}

double PnrImmKf::update_model(int model_idx, double z_x, double z_y, double alpha_r, double outlier_prob) {
  auto& x = x_models_[model_idx];
  auto& p = p_models_[model_idx];

  // 测量噪声按当前场景自适应缩放，检测质量越差，等价测量噪声越大。
  double r = cfg_.r_pos * alpha_r;
  r = std::max(1e-6, r);

  const double nu_x = z_x - x[0];
  const double nu_y = z_y - x[1];

  double s00 = p[0][0] + r;
  double s01 = p[0][1];
  double s10 = p[1][0];
  double s11 = p[1][1] + r;

  double i00 = 0.0;
  double i01 = 0.0;
  double i10 = 0.0;
  double i11 = 0.0;
  double det = 0.0;
  if (!invert_2x2(s00, s01, s10, s11, i00, i01, i10, i11, det)) {
    return 1e-12;
  }

  const double d2 = nu_x * (i00 * nu_x + i01 * nu_y) + nu_y * (i10 * nu_x + i11 * nu_y);
  dbg_.gate_d2 = std::max(dbg_.gate_d2, d2);

  // Huber 权重 + 离群概率联合抑制异常观测，避免单次坏框把状态拉飞。
  const double res_norm = std::sqrt(std::max(0.0, d2));
  const double huber_w = (res_norm <= cfg_.huber_c) ? 1.0 : (cfg_.huber_c / std::max(1e-6, res_norm));
  const double gate_factor = (d2 > cfg_.gate_chi2 * (1.0 + 2.0 * outlier_prob)) ? 0.25 : 1.0;
  const double robust_w = std::clamp(huber_w * (1.0 - 0.75 * outlier_prob) * gate_factor, 0.05, 1.0);

  const double r_eff = r / robust_w;
  s00 = p[0][0] + r_eff;
  s01 = p[0][1];
  s10 = p[1][0];
  s11 = p[1][1] + r_eff;
  if (!invert_2x2(s00, s01, s10, s11, i00, i01, i10, i11, det)) {
    return 1e-12;
  }

  std::array<std::array<double, 2>, 6> k{};
  for (int row = 0; row < 6; ++row) {
    const double p0 = p[row][0];
    const double p1 = p[row][1];
    k[row][0] = p0 * i00 + p1 * i10;
    k[row][1] = p0 * i01 + p1 * i11;
  }

  for (int row = 0; row < 6; ++row) {
    x[row] += k[row][0] * nu_x + k[row][1] * nu_y;
  }

  // 协方差更新采用 Joseph 形式，数值稳定性比简单 P=(I-KH)P 更好。
  Mat i_kh = mat_eye();
  for (int row = 0; row < 6; ++row) {
    i_kh[row][0] -= k[row][0];
    i_kh[row][1] -= k[row][1];
  }

  const Mat left = mat_mul(i_kh, p);
  const Mat joseph = mat_mul(left, mat_transpose(i_kh));
  Mat krkt = mat_zero();
  for (int r0 = 0; r0 < 6; ++r0) {
    for (int c0 = 0; c0 < 6; ++c0) {
      krkt[r0][c0] = r_eff * (k[r0][0] * k[c0][0] + k[r0][1] * k[c0][1]);
    }
  }
  p = mat_add(joseph, krkt);
  enforce_symmetric(p);

  const double det_safe = std::max(1e-12, det);
  const double d2_eff = nu_x * (i00 * nu_x + i01 * nu_y) + nu_y * (i10 * nu_x + i11 * nu_y);
  const double norm = 1.0 / (2.0 * kPi * std::sqrt(det_safe));
  const double like = norm * std::exp(-0.5 * std::max(0.0, d2_eff));
  if (!std::isfinite(like) || like <= 0.0) {
    return 1e-12;
  }
  return std::max(1e-12, like);
}

void PnrImmKf::update(double z_x_px, double z_y_px, const PnrFilterQuality& q) {
  if (!initialized_) {
    init(z_x_px, z_y_px);
  }

  // 先根据当前测量与融合预测的残差，估计“这一帧观测靠不靠谱”。
  const double residual_x = z_x_px - x_fused_[0];
  const double residual_y = z_y_px - x_fused_[1];
  const NeuralOutput neural = neural_step(residual_x, residual_y, q);
  switch_delta_ = neural.delta_switch;

  dbg_.alpha_q = neural.alpha_q;
  dbg_.alpha_r = neural.alpha_r;
  dbg_.outlier_prob = neural.outlier_prob;
  dbg_.bias_x = neural.bias_x;
  dbg_.bias_y = neural.bias_y;
  dbg_.gate_d2 = 0.0;

  const double z_x = z_x_px - neural.bias_x;
  const double z_y = z_y_px - neural.bias_y;

  // 两个模型分别更新，再按似然重新分配模型概率。
  std::array<double, 2> like{};
  for (int j = 0; j < kM; ++j) {
    like[j] = update_model(j, z_x, z_y, neural.alpha_r, neural.outlier_prob);
  }

  model_prob_[0] = std::max(1e-12, model_prob_[0] * like[0]);
  model_prob_[1] = std::max(1e-12, model_prob_[1] * like[1]);
  const double sum = model_prob_[0] + model_prob_[1];
  model_prob_[0] /= std::max(1e-12, sum);
  model_prob_[1] /= std::max(1e-12, sum);

  fuse_models();
  dbg_.model_prob_cv = model_prob_[0];
  dbg_.model_prob_ca = model_prob_[1];
  dbg_.used_measurement = true;
}

void PnrImmKf::fuse_models() {
  // 最终对外只暴露一个融合状态，
  // 它是两个子模型按当前模型概率加权后的结果。
  x_fused_ = vec_zero();
  for (int j = 0; j < kM; ++j) {
    x_fused_ = vec_add(x_fused_, vec_scale(x_models_[j], model_prob_[j]));
  }

  p_fused_ = mat_zero();
  for (int j = 0; j < kM; ++j) {
    const Vec dx = vec_sub(x_models_[j], x_fused_);
    const Mat spread = mat_add(p_models_[j], outer(dx, dx));
    p_fused_ = mat_add(p_fused_, mat_scale(spread, model_prob_[j]));
  }
  enforce_symmetric(p_fused_);
}

}  // namespace filter
