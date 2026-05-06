#pragma once

#include <array>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace control {

// 新算法输出：
// e_ref 是因果均值基线；
// delta_pred 是模型预测的下一帧基线增量；
// u 是经过有界融合后的最终控制误差。
struct TemporalCompResult {
  double e_ref_x{0.0};
  double e_ref_y{0.0};
  double delta_pred_x{0.0};
  double delta_pred_y{0.0};
  double delta_applied_x{0.0};
  double delta_applied_y{0.0};
  double u_x{0.0};
  double u_y{0.0};
  double alpha{0.0};
  bool used_model{false};
  bool buffer_ready{false};
  std::string status{"disabled"};
};

// 单模型时序补偿器：
// 1. 对 dx_raw/dy_raw 做因果滑动均值，得到 e_ref
// 2. 用 [e_ref, d_ref] 的历史序列做 ONNX 推理，预测下一帧增量 delta
// 3. 按 u = e_ref + alpha * clip(delta) 进行有界融合
class TemporalCompensatorOnnx {
 public:
  explicit TemporalCompensatorOnnx(const core::TemporalCompConfig& cfg = {});

  bool load(const std::string& model_path, const std::string& stats_json_path);
  void configure(const core::TemporalCompConfig& cfg);
  void reset();
  bool ready() const { return ready_; }

  TemporalCompResult step(double signal_x, double signal_y, bool target_detected, bool input_is_reference = false);

 private:
  bool load_stats_json(const std::string& path);
  static bool read_text_file(const std::string& path, std::string& out_text);
  static bool parse_number_array(const std::string& text, const std::string& key, std::vector<double>& out);
  static bool parse_int_value(const std::string& text, const std::string& key, int& out);

 private:
  core::TemporalCompConfig cfg_{};
  bool ready_{false};
  bool has_prev_ref_{false};
  std::array<double, 2> prev_ref_{0.0, 0.0};
  std::deque<std::array<double, 2>> raw_ring_;
  std::deque<std::array<float, 4>> feature_ring_;
  std::vector<float> mean_{0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> std_{1.0f, 1.0f, 1.0f, 1.0f};
  std::string model_path_;
  std::string stats_path_;

#ifdef HAVE_ONNXRUNTIME
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "temporal_comp"};
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_name_strs_;
  std::vector<std::string> output_name_strs_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
#endif
};

}  // namespace control
