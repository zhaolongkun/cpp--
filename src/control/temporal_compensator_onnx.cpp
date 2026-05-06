#include "control/temporal_compensator_onnx.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

namespace {

double clamp(double v, double lo, double hi) { return std::max(lo, std::min(v, hi)); }

}  // namespace

namespace control {

TemporalCompensatorOnnx::TemporalCompensatorOnnx(const core::TemporalCompConfig& cfg) : cfg_(cfg) {}

bool TemporalCompensatorOnnx::read_text_file(const std::string& path, std::string& out_text) {
  try {
    std::ifstream ifs(std::filesystem::u8path(path), std::ios::binary);
    if (!ifs.is_open()) {
      return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    out_text = oss.str();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[temporal_comp] read_text_file exception: " << e.what() << '\n';
    return false;
  }
}

bool TemporalCompensatorOnnx::parse_number_array(const std::string& text, const std::string& key,
                                                 std::vector<double>& out) {
  const std::string k = "\"" + key + "\"";
  const size_t kp = text.find(k);
  if (kp == std::string::npos) {
    return false;
  }
  const size_t lb = text.find('[', kp);
  const size_t rb = text.find(']', lb);
  if (lb == std::string::npos || rb == std::string::npos || rb <= lb) {
    return false;
  }
  const std::string body = text.substr(lb + 1, rb - lb - 1);
  std::regex re(R"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)");
  std::sregex_iterator it(body.begin(), body.end(), re);
  std::sregex_iterator end;
  out.clear();
  for (; it != end; ++it) {
    out.push_back(std::stod((*it)[0].str()));
  }
  return !out.empty();
}

bool TemporalCompensatorOnnx::parse_int_value(const std::string& text, const std::string& key, int& out) {
  const std::regex re("\"" + key + R"(\"\s*:\s*([-+]?\d+))");
  std::smatch m;
  if (!std::regex_search(text, m, re)) {
    return false;
  }
  out = std::stoi(m[1].str());
  return true;
}

void TemporalCompensatorOnnx::configure(const core::TemporalCompConfig& cfg) {
  cfg_ = cfg;
  cfg_.window_size = std::max(1, cfg_.window_size);
  cfg_.seq_len = std::max(1, cfg_.seq_len);
  cfg_.alpha = clamp(cfg_.alpha, 0.0, 1.0);
  cfg_.delta_max = std::max(0.0, cfg_.delta_max);
  reset();
}

void TemporalCompensatorOnnx::reset() {
  has_prev_ref_ = false;
  prev_ref_ = {0.0, 0.0};
  raw_ring_.clear();
  feature_ring_.clear();
}

bool TemporalCompensatorOnnx::load_stats_json(const std::string& path) {
  std::string text;
  if (!read_text_file(path, text)) {
    std::cerr << "[temporal_comp] failed to read stats json: " << path << '\n';
    return false;
  }

  std::vector<double> mean_d;
  std::vector<double> std_d;
  if (!parse_number_array(text, "mean", mean_d) || !parse_number_array(text, "std", std_d)) {
    std::cerr << "[temporal_comp] stats json missing mean/std\n";
    return false;
  }
  if (mean_d.size() != 4 || std_d.size() != 4) {
    std::cerr << "[temporal_comp] stats json dimension mismatch\n";
    return false;
  }

  int seq_len = cfg_.seq_len;
  int input_dim = 4;
  (void)parse_int_value(text, "seq_len", seq_len);
  (void)parse_int_value(text, "input_dim", input_dim);
  if (input_dim != 4) {
    std::cerr << "[temporal_comp] unsupported input_dim: " << input_dim << '\n';
    return false;
  }
  cfg_.seq_len = std::max(1, seq_len);

  mean_.resize(4);
  std_.resize(4);
  for (size_t i = 0; i < 4; ++i) {
    mean_[i] = static_cast<float>(mean_d[i]);
    std_[i] = static_cast<float>(std::abs(std_d[i]) < 1e-8 ? 1.0 : std_d[i]);
  }
  return true;
}

bool TemporalCompensatorOnnx::load(const std::string& model_path, const std::string& stats_json_path) {
  ready_ = false;
  model_path_ = model_path;
  stats_path_ = stats_json_path;
  reset();

  if (!load_stats_json(stats_path_)) {
    return false;
  }

#ifdef HAVE_ONNXRUNTIME
  try {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    const auto model_fs_path = std::filesystem::u8path(model_path);
    session_ = std::make_unique<Ort::Session>(env_, model_fs_path.c_str(), session_options_);
    Ort::AllocatorWithDefaultOptions allocator;
    input_name_strs_.clear();
    output_name_strs_.clear();
    input_names_.clear();
    output_names_.clear();
    const size_t in_count = session_->GetInputCount();
    const size_t out_count = session_->GetOutputCount();
    if (in_count < 1 || out_count < 1) {
      std::cerr << "[temporal_comp] invalid onnx io count\n";
      return false;
    }
    for (size_t i = 0; i < in_count; ++i) {
      auto n = session_->GetInputNameAllocated(i, allocator);
      input_name_strs_.emplace_back(n.get());
    }
    for (size_t i = 0; i < out_count; ++i) {
      auto n = session_->GetOutputNameAllocated(i, allocator);
      output_name_strs_.emplace_back(n.get());
    }
    for (const auto& s : input_name_strs_) {
      input_names_.push_back(s.c_str());
    }
    for (const auto& s : output_name_strs_) {
      output_names_.push_back(s.c_str());
    }
  } catch (const std::exception& e) {
    std::cerr << "[temporal_comp] failed to create onnx session: " << e.what() << '\n';
    return false;
  }
  ready_ = true;
  std::cerr << "[temporal_comp] onnx loaded: " << model_path_ << '\n';
  return true;
#else
  std::cerr << "[temporal_comp] ONNX Runtime not enabled in this build.\n";
  return false;
#endif
}

TemporalCompResult TemporalCompensatorOnnx::step(double signal_x, double signal_y, bool target_detected,
                                                 bool input_is_reference) {
  TemporalCompResult out;
  out.alpha = cfg_.alpha;

  if (!cfg_.enable) {
    out.e_ref_x = signal_x;
    out.e_ref_y = signal_y;
    out.u_x = signal_x;
    out.u_y = signal_y;
    out.status = "disabled";
    return out;
  }

  if (!target_detected) {
    reset();
    out.status = "target_invalid";
    return out;
  }

  std::array<double, 2> e_ref{signal_x, signal_y};
  if (!input_is_reference) {
    raw_ring_.push_back({signal_x, signal_y});
    while (static_cast<int>(raw_ring_.size()) > cfg_.window_size) {
      raw_ring_.pop_front();
    }

    e_ref = {0.0, 0.0};
    for (const auto& v : raw_ring_) {
      e_ref[0] += v[0];
      e_ref[1] += v[1];
    }
    const size_t denom = std::max<size_t>(1, raw_ring_.size());
    e_ref[0] /= static_cast<double>(denom);
    e_ref[1] /= static_cast<double>(denom);
  } else {
    raw_ring_.clear();
  }

  std::array<double, 2> d_ref{0.0, 0.0};
  if (has_prev_ref_) {
    d_ref[0] = e_ref[0] - prev_ref_[0];
    d_ref[1] = e_ref[1] - prev_ref_[1];
  }
  prev_ref_ = e_ref;
  has_prev_ref_ = true;

  out.e_ref_x = e_ref[0];
  out.e_ref_y = e_ref[1];
  out.u_x = e_ref[0];
  out.u_y = e_ref[1];

  feature_ring_.push_back(
      {static_cast<float>(e_ref[0]), static_cast<float>(e_ref[1]), static_cast<float>(d_ref[0]), static_cast<float>(d_ref[1])});
  while (static_cast<int>(feature_ring_.size()) > cfg_.seq_len) {
    feature_ring_.pop_front();
  }

  out.buffer_ready = static_cast<int>(feature_ring_.size()) >= cfg_.seq_len;
  if (!out.buffer_ready) {
    out.status = "warmup";
    return out;
  }
  if (!ready_) {
    out.status = "backend_not_ready";
    return out;
  }

#ifdef HAVE_ONNXRUNTIME
  try {
    std::vector<float> input_data;
    input_data.reserve(static_cast<size_t>(cfg_.seq_len) * 4U);
    for (const auto& row : feature_ring_) {
      for (size_t i = 0; i < 4; ++i) {
        const float normed = static_cast<float>((row[i] - mean_[i]) / std_[i]);
        input_data.push_back(std::clamp(normed, -8.0f, 8.0f));
      }
    }

    const std::array<int64_t, 3> input_shape = {1, static_cast<int64_t>(cfg_.seq_len), 4};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), input_data.size(),
                                                              input_shape.data(), input_shape.size());
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1, output_names_.data(), 1);
    if (outputs.empty() || !outputs[0].IsTensor()) {
      out.status = "invalid_output";
      return out;
    }
    const float* pred = outputs[0].GetTensorData<float>();
    out.delta_pred_x = pred[0];
    out.delta_pred_y = pred[1];
    out.delta_applied_x = clamp(out.delta_pred_x, -cfg_.delta_max, cfg_.delta_max);
    out.delta_applied_y = clamp(out.delta_pred_y, -cfg_.delta_max, cfg_.delta_max);
    out.u_x = out.e_ref_x + cfg_.alpha * out.delta_applied_x;
    out.u_y = out.e_ref_y + cfg_.alpha * out.delta_applied_y;
    out.used_model = true;
    out.status = "ok";
    return out;
  } catch (const std::exception& e) {
    out.status = std::string("infer_exception: ") + e.what();
    return out;
  }
#else
  out.status = "onnxruntime_disabled";
  return out;
#endif
}

}  // namespace control
