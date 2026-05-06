// 配置加载器：
// 负责从 YAML/简易文本配置中读取参数，并在末尾统一做安全范围校验。
#include "core/config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

#ifdef HAVE_YAML_CPP
#include <yaml-cpp/yaml.h>
#endif

namespace {

#ifdef HAVE_YAML_CPP

// YAML 模式下的通用标量读取函数：只有字段存在时才覆盖默认值。
template <typename T>
void load_scalar(const YAML::Node& node, const char* key, T& dst) {
  if (node && node[key]) {
    dst = node[key].as<T>();
  }
}

// 读取内联 PID 配置块。
void load_pid(const YAML::Node& root, const char* key, core::PIDGains& gains) {
  if (!root[key]) {
    return;
  }
  const auto n = root[key];
  load_scalar(n, "kp", gains.kp);
  load_scalar(n, "ki", gains.ki);
  load_scalar(n, "kd", gains.kd);
}

#else

// 无 yaml-cpp 时，退回到简单文本解析，这些工具函数服务于轻量解析器。
std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::string unquote(std::string s) {
  s = trim(std::move(s));
  if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

bool parse_bool(const std::string& s, bool& out) {
  std::string t = trim(s);
  std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (t == "true" || t == "1") {
    out = true;
    return true;
  }
  if (t == "false" || t == "0") {
    out = false;
    return true;
  }
  return false;
}

// 兼容 `pid_x: {kp: ..., ki: ..., kd: ...}` 这种内联写法。
void parse_pid_inline(const std::string& text, core::PIDGains& gains) {
  std::smatch m;
  const std::regex kp_re(R"(kp\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))");
  const std::regex ki_re(R"(ki\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))");
  const std::regex kd_re(R"(kd\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))");

  if (std::regex_search(text, m, kp_re)) {
    gains.kp = std::stod(m[1].str());
  }
  if (std::regex_search(text, m, ki_re)) {
    gains.ki = std::stod(m[1].str());
  }
  if (std::regex_search(text, m, kd_re)) {
    gains.kd = std::stod(m[1].str());
  }
}

#endif

}  // namespace

namespace core {

AppConfig ConfigLoader::load_or_default(const std::string& path) {
  // 先用代码内置默认值初始化，配置文件只覆盖用户显式指定的部分。
  AppConfig cfg;
  try {
#ifdef HAVE_YAML_CPP
    const YAML::Node root = YAML::LoadFile(path);

    // 按模块分别读取，便于配置结构和 AppConfig 对齐。
    if (root["model"]) {
      const auto n = root["model"];
      load_scalar(n, "path", cfg.model.path);
      load_scalar(n, "conf_detect", cfg.model.conf_detect);
      load_scalar(n, "conf_track", cfg.model.conf_track);
      load_scalar(n, "tile_enable", cfg.model.tile_enable);
      load_scalar(n, "bbox_size_filter_enable", cfg.model.bbox_size_filter_enable);
      load_scalar(n, "bbox_size_filter_alpha", cfg.model.bbox_size_filter_alpha);
      load_scalar(n, "bbox_size_filter_min_scale", cfg.model.bbox_size_filter_min_scale);
      load_scalar(n, "bbox_size_filter_max_scale", cfg.model.bbox_size_filter_max_scale);
      load_scalar(n, "bbox_size_filter_center_gate_px", cfg.model.bbox_size_filter_center_gate_px);
      load_scalar(n, "bbox_size_filter_center_deadband_px", cfg.model.bbox_size_filter_center_deadband_px);
      load_scalar(n, "bbox_size_filter_size_deadband_px", cfg.model.bbox_size_filter_size_deadband_px);
      load_scalar(n, "bbox_size_filter_max_center_step_px", cfg.model.bbox_size_filter_max_center_step_px);
      load_scalar(n, "bbox_size_filter_max_size_step_px", cfg.model.bbox_size_filter_max_size_step_px);
      load_scalar(n, "tile_size", cfg.model.tile_size);
      load_scalar(n, "tile_overlap", cfg.model.tile_overlap);
      load_scalar(n, "tile_max_per_frame", cfg.model.tile_max_per_frame);
      load_scalar(n, "tile_global_interval", cfg.model.tile_global_interval);
      load_scalar(n, "tile_priority_enable", cfg.model.tile_priority_enable);
      load_scalar(n, "tile_priority_topk", cfg.model.tile_priority_topk);
      load_scalar(n, "tile_priority_expand_ratio", cfg.model.tile_priority_expand_ratio);
      load_scalar(n, "tile_priority_ttl", cfg.model.tile_priority_ttl);
      load_scalar(n, "use_shm_detector", cfg.model.use_shm_detector);
    }

    if (root["camera"]) {
      const auto n = root["camera"];
      load_scalar(n, "index", cfg.camera.index);
      load_scalar(n, "width", cfg.camera.width);
      load_scalar(n, "height", cfg.camera.height);
      load_scalar(n, "center_x", cfg.camera.center_x);
      load_scalar(n, "center_y", cfg.camera.center_y);
      load_scalar(n, "backend", cfg.camera.backend);
      load_scalar(n, "show_window", cfg.camera.show_window);
      load_scalar(n, "show_coasting", cfg.camera.show_coasting);
      load_scalar(n, "use_mjpg", cfg.camera.use_mjpg);
      load_scalar(n, "auto_focus_enable", cfg.camera.auto_focus_enable);
      load_scalar(n, "auto_zoom_enable", cfg.camera.auto_zoom_enable);
      load_scalar(n, "target_bbox_ratio", cfg.camera.target_bbox_ratio);
      load_scalar(n, "bbox_ratio_deadband", cfg.camera.bbox_ratio_deadband);
      load_scalar(n, "bbox_ratio_alpha", cfg.camera.bbox_ratio_alpha);
      load_scalar(n, "zoom_persist_frames", cfg.camera.zoom_persist_frames);
      load_scalar(n, "zoom_step", cfg.camera.zoom_step);
      load_scalar(n, "zoom_log_gain", cfg.camera.zoom_log_gain);
      load_scalar(n, "zoom_min_step", cfg.camera.zoom_min_step);
      load_scalar(n, "zoom_max_step", cfg.camera.zoom_max_step);
      load_scalar(n, "zoom_min", cfg.camera.zoom_min);
      load_scalar(n, "zoom_max", cfg.camera.zoom_max);
      load_scalar(n, "show_zoom_debug", cfg.camera.show_zoom_debug);
      load_scalar(n, "zoom_fps_trigger", cfg.camera.zoom_fps_trigger);
      load_scalar(n, "zoom_trend_trigger", cfg.camera.zoom_trend_trigger);
      load_scalar(n, "zoom_area_deadband_px", cfg.camera.zoom_area_deadband_px);
      load_scalar(n, "zoom_hysteresis_ratio", cfg.camera.zoom_hysteresis_ratio);
      load_scalar(n, "zoom_action_interval_ms", cfg.camera.zoom_action_interval_ms);
      load_scalar(n, "zoom_stable_required_ms", cfg.camera.zoom_stable_required_ms);
      load_scalar(n, "zoom_median_window", cfg.camera.zoom_median_window);
      load_scalar(n, "zoom_search_step", cfg.camera.zoom_search_step);
      load_scalar(n, "zoom_search_every_n", cfg.camera.zoom_search_every_n);
      load_scalar(n, "zoom_max_lost_recover_enable", cfg.camera.zoom_max_lost_recover_enable);
      load_scalar(n, "zoom_max_lost_trigger_frames", cfg.camera.zoom_max_lost_trigger_frames);
      load_scalar(n, "zoom_recover_step", cfg.camera.zoom_recover_step);
      load_scalar(n, "zoom_out_enable", cfg.camera.zoom_out_enable);
      load_scalar(n, "zoom_out_trigger_mult", cfg.camera.zoom_out_trigger_mult);
      load_scalar(n, "zoom_hold_miss_frames", cfg.camera.zoom_hold_miss_frames);
      load_scalar(n, "zoom_reacquire_freeze_frames", cfg.camera.zoom_reacquire_freeze_frames);
      load_scalar(n, "zoom_reacquire_confirm_frames", cfg.camera.zoom_reacquire_confirm_frames);
    }

    if (root["mot"]) {
      const auto n = root["mot"];
      load_scalar(n, "max_age", cfg.mot.max_age);
      load_scalar(n, "iou_gate", cfg.mot.iou_gate);
      load_scalar(n, "tracker_type", cfg.mot.tracker_type);
      load_scalar(n, "use_bytetrack", cfg.mot.use_bytetrack);
      load_scalar(n, "track_high_thresh", cfg.mot.track_high_thresh);
      load_scalar(n, "track_low_thresh", cfg.mot.track_low_thresh);
      load_scalar(n, "new_track_thresh", cfg.mot.new_track_thresh);
      load_scalar(n, "second_match_iou", cfg.mot.second_match_iou);
      load_scalar(n, "botsort_iou_weight", cfg.mot.botsort_iou_weight);
      load_scalar(n, "botsort_center_weight", cfg.mot.botsort_center_weight);
      load_scalar(n, "botsort_max_center_dist_ratio", cfg.mot.botsort_max_center_dist_ratio);
    }

    if (root["control"]) {
      const auto n = root["control"];
      load_scalar(n, "rate_hz", cfg.control.rate_hz);
      load_scalar(n, "deadband_px", cfg.control.deadband_px);
      load_scalar(n, "cmd_limit", cfg.control.cmd_limit);
      load_scalar(n, "full_speed_px", cfg.control.full_speed_px);
      load_scalar(n, "slew_per_tick", cfg.control.slew_per_tick);
      load_scalar(n, "stop_when_lost", cfg.control.stop_when_lost);
      load_pid(n, "pid_x", cfg.control.pid_x);
      load_pid(n, "pid_y", cfg.control.pid_y);
    }

    if (root["actuator"]) {
      const auto n = root["actuator"];
      load_scalar(n, "mode", cfg.actuator.mode);
      load_scalar(n, "armed", cfg.actuator.armed);
      load_scalar(n, "python_exe", cfg.actuator.python_exe);
      load_scalar(n, "bridge_script", cfg.actuator.bridge_script);
      load_scalar(n, "x_vendor_id", cfg.actuator.x_vendor_id);
      load_scalar(n, "x_product_id", cfg.actuator.x_product_id);
      load_scalar(n, "y_vendor_id", cfg.actuator.y_vendor_id);
      load_scalar(n, "y_product_id", cfg.actuator.y_product_id);
      load_scalar(n, "invert_x", cfg.actuator.invert_x);
      load_scalar(n, "invert_y", cfg.actuator.invert_y);
      load_scalar(n, "scale_x", cfg.actuator.scale_x);
      load_scalar(n, "scale_y", cfg.actuator.scale_y);
      load_scalar(n, "speed_limit", cfg.actuator.speed_limit);
      load_scalar(n, "send_zero_on_close", cfg.actuator.send_zero_on_close);
      load_scalar(n, "debug", cfg.actuator.debug);
      load_scalar(n, "y_pitch_guard_enable", cfg.actuator.y_pitch_guard_enable);
      load_scalar(n, "y_pitch_port", cfg.actuator.y_pitch_port);
      load_scalar(n, "y_pitch_baud", cfg.actuator.y_pitch_baud);
      load_scalar(n, "y_pitch_min_deg", cfg.actuator.y_pitch_min_deg);
      load_scalar(n, "y_pitch_max_deg", cfg.actuator.y_pitch_max_deg);
      load_scalar(n, "y_pitch_upper_stop_deg", cfg.actuator.y_pitch_upper_stop_deg);
      load_scalar(n, "y_pitch_lpf_alpha", cfg.actuator.y_pitch_lpf_alpha);
      load_scalar(n, "y_pitch_release_margin_deg", cfg.actuator.y_pitch_release_margin_deg);
      load_scalar(n, "y_pitch_timeout_ms", cfg.actuator.y_pitch_timeout_ms);
      load_scalar(n, "y_pitch_debug", cfg.actuator.y_pitch_debug);
      load_scalar(n, "y_pitch_positive_increases_angle", cfg.actuator.y_pitch_positive_increases_angle);
      load_scalar(n, "y_pitch_soft_zone_deg", cfg.actuator.y_pitch_soft_zone_deg);
      load_scalar(n, "y_pitch_takeover_enable", cfg.actuator.y_pitch_takeover_enable);
      load_scalar(n, "y_pitch_takeover_target_deg", cfg.actuator.y_pitch_takeover_target_deg);
      load_scalar(n, "y_pitch_takeover_speed", cfg.actuator.y_pitch_takeover_speed);
      load_scalar(n, "y_pitch_takeover_interval_ms", cfg.actuator.y_pitch_takeover_interval_ms);
      load_scalar(n, "y_pitch_session_idle_ms", cfg.actuator.y_pitch_session_idle_ms);
      load_scalar(n, "y_pitch_jump_guard_enable", cfg.actuator.y_pitch_jump_guard_enable);
      load_scalar(n, "y_pitch_max_step_deg", cfg.actuator.y_pitch_max_step_deg);
      load_scalar(n, "y_pitch_jump_speed_scale", cfg.actuator.y_pitch_jump_speed_scale);
      load_scalar(n, "y_pitch_jump_speed_cap", cfg.actuator.y_pitch_jump_speed_cap);
    }

    if (root["filter"]) {
      const auto n = root["filter"];
      load_scalar(n, "enable", cfg.filter.enable);
      load_scalar(n, "neural_enable", cfg.filter.neural_enable);
      load_scalar(n, "meas_only_when_matched", cfg.filter.meas_only_when_matched);
      load_scalar(n, "offset_lpf_enable", cfg.filter.offset_lpf_enable);
      load_scalar(n, "offset_lpf_alpha", cfg.filter.offset_lpf_alpha);
      load_scalar(n, "offset_lpf_alpha_y", cfg.filter.offset_lpf_alpha_y);
      load_scalar(n, "switch_prob", cfg.filter.switch_prob);
      load_scalar(n, "gate_chi2", cfg.filter.gate_chi2);
      load_scalar(n, "huber_c", cfg.filter.huber_c);
      load_scalar(n, "q_pos", cfg.filter.q_pos);
      load_scalar(n, "q_vel", cfg.filter.q_vel);
      load_scalar(n, "q_acc", cfg.filter.q_acc);
      load_scalar(n, "r_pos", cfg.filter.r_pos);
      load_scalar(n, "alpha_q_min", cfg.filter.alpha_q_min);
      load_scalar(n, "alpha_q_max", cfg.filter.alpha_q_max);
      load_scalar(n, "alpha_r_min", cfg.filter.alpha_r_min);
      load_scalar(n, "alpha_r_max", cfg.filter.alpha_r_max);
      load_scalar(n, "bias_limit_px", cfg.filter.bias_limit_px);
      load_scalar(n, "outlier_prob_min", cfg.filter.outlier_prob_min);
      load_scalar(n, "outlier_prob_max", cfg.filter.outlier_prob_max);
    }

    if (root["residual"]) {
      const auto n = root["residual"];
      load_scalar(n, "enable", cfg.residual.enable);
      load_scalar(n, "w_conf", cfg.residual.w_conf);
      load_scalar(n, "w_lost", cfg.residual.w_lost);
      load_scalar(n, "w_meas", cfg.residual.w_meas);
      load_scalar(n, "w_age", cfg.residual.w_age);
      load_scalar(n, "age_tau_ms", cfg.residual.age_tau_ms);
      load_scalar(n, "meas_age_hard_ms", cfg.residual.meas_age_hard_ms);
      load_scalar(n, "gate_r_low", cfg.residual.gate_r_low);
      load_scalar(n, "gate_r_high", cfg.residual.gate_r_high);
      load_scalar(n, "alpha_smooth", cfg.residual.alpha_smooth);
      load_scalar(n, "residual_limit_ratio", cfg.residual.residual_limit_ratio);
      load_scalar(n, "residual_min_scale", cfg.residual.residual_min_scale);
      load_scalar(n, "final_slew_per_tick", cfg.residual.final_slew_per_tick);
    }

    if (root["temporal_comp"]) {
      const auto n = root["temporal_comp"];
      load_scalar(n, "enable", cfg.temporal_comp.enable);
      load_scalar(n, "window_size", cfg.temporal_comp.window_size);
      load_scalar(n, "seq_len", cfg.temporal_comp.seq_len);
      load_scalar(n, "alpha", cfg.temporal_comp.alpha);
      load_scalar(n, "delta_max", cfg.temporal_comp.delta_max);
      load_scalar(n, "model_path", cfg.temporal_comp.model_path);
      load_scalar(n, "stats_path", cfg.temporal_comp.stats_path);
    }

    if (root["log"]) {
      const auto n = root["log"];
      load_scalar(n, "enable", cfg.log.enable);
      load_scalar(n, "path", cfg.log.path);
      load_scalar(n, "dedup_by_frame_id", cfg.log.dedup_by_frame_id);
      load_scalar(n, "profile", cfg.log.profile);
    }
#else
    // 轻量文本解析器按 section + key:value 形式读取配置。
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
      throw std::runtime_error("cannot open config file");
    }

    std::string section;
    std::string line;
    while (std::getline(ifs, line)) {
      auto hash_pos = line.find('#');
      if (hash_pos != std::string::npos) {
        line = line.substr(0, hash_pos);
      }
      line = trim(line);
      if (line.empty()) {
        continue;
      }

      if (line.back() == ':' && line.find(':') == line.size() - 1) {
        section = trim(line.substr(0, line.size() - 1));
        continue;
      }

      const auto pos = line.find(':');
      if (pos == std::string::npos || section.empty()) {
        continue;
      }

      const std::string key = trim(line.substr(0, pos));
      const std::string val = trim(line.substr(pos + 1));

      if (section == "model") {
        if (key == "path") cfg.model.path = unquote(val);
        if (key == "conf_detect") cfg.model.conf_detect = std::stod(val);
        if (key == "conf_track") cfg.model.conf_track = std::stod(val);
        if (key == "tile_enable") parse_bool(val, cfg.model.tile_enable);
        if (key == "bbox_size_filter_enable") parse_bool(val, cfg.model.bbox_size_filter_enable);
        if (key == "bbox_size_filter_alpha") cfg.model.bbox_size_filter_alpha = std::stod(val);
        if (key == "bbox_size_filter_min_scale") cfg.model.bbox_size_filter_min_scale = std::stod(val);
        if (key == "bbox_size_filter_max_scale") cfg.model.bbox_size_filter_max_scale = std::stod(val);
        if (key == "bbox_size_filter_center_gate_px") cfg.model.bbox_size_filter_center_gate_px = std::stod(val);
        if (key == "bbox_size_filter_center_deadband_px") cfg.model.bbox_size_filter_center_deadband_px = std::stod(val);
        if (key == "bbox_size_filter_size_deadband_px") cfg.model.bbox_size_filter_size_deadband_px = std::stod(val);
        if (key == "bbox_size_filter_max_center_step_px") cfg.model.bbox_size_filter_max_center_step_px = std::stod(val);
        if (key == "bbox_size_filter_max_size_step_px") cfg.model.bbox_size_filter_max_size_step_px = std::stod(val);
        if (key == "tile_size") cfg.model.tile_size = std::stoi(val);
        if (key == "tile_overlap") cfg.model.tile_overlap = std::stod(val);
        if (key == "tile_max_per_frame") cfg.model.tile_max_per_frame = std::stoi(val);
        if (key == "tile_global_interval") cfg.model.tile_global_interval = std::stoi(val);
        if (key == "tile_priority_enable") parse_bool(val, cfg.model.tile_priority_enable);
        if (key == "tile_priority_topk") cfg.model.tile_priority_topk = std::stoi(val);
        if (key == "tile_priority_expand_ratio") cfg.model.tile_priority_expand_ratio = std::stod(val);
        if (key == "tile_priority_ttl") cfg.model.tile_priority_ttl = std::stoi(val);
        if (key == "use_shm_detector") cfg.model.use_shm_detector = (val == "true" || val == "1");
      } else if (section == "camera") {
        if (key == "index") cfg.camera.index = std::stoi(val);
        if (key == "width") cfg.camera.width = std::stoi(val);
        if (key == "height") cfg.camera.height = std::stoi(val);
        if (key == "center_x") cfg.camera.center_x = std::stoi(val);
        if (key == "center_y") cfg.camera.center_y = std::stoi(val);
        if (key == "backend") cfg.camera.backend = unquote(val);
        if (key == "show_window") parse_bool(val, cfg.camera.show_window);
        if (key == "show_coasting") parse_bool(val, cfg.camera.show_coasting);
        if (key == "use_mjpg") parse_bool(val, cfg.camera.use_mjpg);
        if (key == "auto_focus_enable") parse_bool(val, cfg.camera.auto_focus_enable);
        if (key == "auto_zoom_enable") parse_bool(val, cfg.camera.auto_zoom_enable);
        if (key == "target_bbox_ratio") cfg.camera.target_bbox_ratio = std::stod(val);
        if (key == "bbox_ratio_deadband") cfg.camera.bbox_ratio_deadband = std::stod(val);
        if (key == "bbox_ratio_alpha") cfg.camera.bbox_ratio_alpha = std::stod(val);
        if (key == "zoom_persist_frames") cfg.camera.zoom_persist_frames = std::stoi(val);
        if (key == "zoom_step") cfg.camera.zoom_step = std::stod(val);
        if (key == "zoom_log_gain") cfg.camera.zoom_log_gain = std::stod(val);
        if (key == "zoom_min_step") cfg.camera.zoom_min_step = std::stod(val);
        if (key == "zoom_max_step") cfg.camera.zoom_max_step = std::stod(val);
        if (key == "zoom_min") cfg.camera.zoom_min = std::stod(val);
        if (key == "zoom_max") cfg.camera.zoom_max = std::stod(val);
        if (key == "show_zoom_debug") parse_bool(val, cfg.camera.show_zoom_debug);
        if (key == "zoom_fps_trigger") cfg.camera.zoom_fps_trigger = std::stod(val);
        if (key == "zoom_trend_trigger") cfg.camera.zoom_trend_trigger = std::stod(val);
        if (key == "zoom_area_deadband_px") cfg.camera.zoom_area_deadband_px = std::stod(val);
        if (key == "zoom_hysteresis_ratio") cfg.camera.zoom_hysteresis_ratio = std::stod(val);
        if (key == "zoom_action_interval_ms") cfg.camera.zoom_action_interval_ms = std::stoi(val);
        if (key == "zoom_stable_required_ms") cfg.camera.zoom_stable_required_ms = std::stoi(val);
        if (key == "zoom_median_window") cfg.camera.zoom_median_window = std::stoi(val);
        if (key == "zoom_search_step") cfg.camera.zoom_search_step = std::stod(val);
        if (key == "zoom_search_every_n") cfg.camera.zoom_search_every_n = std::stoi(val);
        if (key == "zoom_max_lost_recover_enable") parse_bool(val, cfg.camera.zoom_max_lost_recover_enable);
        if (key == "zoom_max_lost_trigger_frames") cfg.camera.zoom_max_lost_trigger_frames = std::stoi(val);
        if (key == "zoom_recover_step") cfg.camera.zoom_recover_step = std::stod(val);
        if (key == "zoom_out_enable") parse_bool(val, cfg.camera.zoom_out_enable);
        if (key == "zoom_out_trigger_mult") cfg.camera.zoom_out_trigger_mult = std::stod(val);
        if (key == "zoom_hold_miss_frames") cfg.camera.zoom_hold_miss_frames = std::stoi(val);
        if (key == "zoom_reacquire_freeze_frames") cfg.camera.zoom_reacquire_freeze_frames = std::stoi(val);
        if (key == "zoom_reacquire_confirm_frames") cfg.camera.zoom_reacquire_confirm_frames = std::stoi(val);
      } else if (section == "mot") {
        if (key == "max_age") cfg.mot.max_age = std::stoi(val);
        if (key == "iou_gate") cfg.mot.iou_gate = std::stod(val);
        if (key == "tracker_type") cfg.mot.tracker_type = unquote(val);
        if (key == "use_bytetrack") parse_bool(val, cfg.mot.use_bytetrack);
        if (key == "track_high_thresh") cfg.mot.track_high_thresh = std::stod(val);
        if (key == "track_low_thresh") cfg.mot.track_low_thresh = std::stod(val);
        if (key == "new_track_thresh") cfg.mot.new_track_thresh = std::stod(val);
        if (key == "second_match_iou") cfg.mot.second_match_iou = std::stod(val);
        if (key == "botsort_iou_weight") cfg.mot.botsort_iou_weight = std::stod(val);
        if (key == "botsort_center_weight") cfg.mot.botsort_center_weight = std::stod(val);
        if (key == "botsort_max_center_dist_ratio") cfg.mot.botsort_max_center_dist_ratio = std::stod(val);
      } else if (section == "control") {
        if (key == "rate_hz") cfg.control.rate_hz = std::stoi(val);
        if (key == "deadband_px") cfg.control.deadband_px = std::stod(val);
        if (key == "cmd_limit") cfg.control.cmd_limit = std::stod(val);
        if (key == "full_speed_px") cfg.control.full_speed_px = std::stod(val);
        if (key == "slew_per_tick") cfg.control.slew_per_tick = std::stod(val);
        if (key == "stop_when_lost") parse_bool(val, cfg.control.stop_when_lost);
        if (key == "pid_x") parse_pid_inline(val, cfg.control.pid_x);
        if (key == "pid_y") parse_pid_inline(val, cfg.control.pid_y);
      } else if (section == "actuator") {
        if (key == "mode") cfg.actuator.mode = unquote(val);
        if (key == "armed") parse_bool(val, cfg.actuator.armed);
        if (key == "python_exe") cfg.actuator.python_exe = unquote(val);
        if (key == "bridge_script") cfg.actuator.bridge_script = unquote(val);
        if (key == "x_vendor_id") cfg.actuator.x_vendor_id = std::stoi(val);
        if (key == "x_product_id") cfg.actuator.x_product_id = std::stoi(val);
        if (key == "y_vendor_id") cfg.actuator.y_vendor_id = std::stoi(val);
        if (key == "y_product_id") cfg.actuator.y_product_id = std::stoi(val);
        if (key == "invert_x") parse_bool(val, cfg.actuator.invert_x);
        if (key == "invert_y") parse_bool(val, cfg.actuator.invert_y);
        if (key == "scale_x") cfg.actuator.scale_x = std::stod(val);
        if (key == "scale_y") cfg.actuator.scale_y = std::stod(val);
        if (key == "speed_limit") cfg.actuator.speed_limit = std::stoi(val);
        if (key == "send_zero_on_close") parse_bool(val, cfg.actuator.send_zero_on_close);
        if (key == "debug") parse_bool(val, cfg.actuator.debug);
        if (key == "y_pitch_guard_enable") parse_bool(val, cfg.actuator.y_pitch_guard_enable);
        if (key == "y_pitch_port") cfg.actuator.y_pitch_port = unquote(val);
        if (key == "y_pitch_baud") cfg.actuator.y_pitch_baud = std::stoi(val);
        if (key == "y_pitch_min_deg") cfg.actuator.y_pitch_min_deg = std::stod(val);
        if (key == "y_pitch_max_deg") cfg.actuator.y_pitch_max_deg = std::stod(val);
        if (key == "y_pitch_upper_stop_deg") cfg.actuator.y_pitch_upper_stop_deg = std::stod(val);
        if (key == "y_pitch_lpf_alpha") cfg.actuator.y_pitch_lpf_alpha = std::stod(val);
        if (key == "y_pitch_release_margin_deg") cfg.actuator.y_pitch_release_margin_deg = std::stod(val);
        if (key == "y_pitch_timeout_ms") cfg.actuator.y_pitch_timeout_ms = std::stoi(val);
        if (key == "y_pitch_debug") parse_bool(val, cfg.actuator.y_pitch_debug);
        if (key == "y_pitch_positive_increases_angle") parse_bool(val, cfg.actuator.y_pitch_positive_increases_angle);
        if (key == "y_pitch_soft_zone_deg") cfg.actuator.y_pitch_soft_zone_deg = std::stod(val);
        if (key == "y_pitch_takeover_enable") parse_bool(val, cfg.actuator.y_pitch_takeover_enable);
        if (key == "y_pitch_takeover_target_deg") cfg.actuator.y_pitch_takeover_target_deg = std::stod(val);
        if (key == "y_pitch_takeover_speed") cfg.actuator.y_pitch_takeover_speed = std::stod(val);
        if (key == "y_pitch_takeover_interval_ms") cfg.actuator.y_pitch_takeover_interval_ms = std::stoi(val);
        if (key == "y_pitch_session_idle_ms") cfg.actuator.y_pitch_session_idle_ms = std::stoi(val);
        if (key == "y_pitch_jump_guard_enable") parse_bool(val, cfg.actuator.y_pitch_jump_guard_enable);
        if (key == "y_pitch_max_step_deg") cfg.actuator.y_pitch_max_step_deg = std::stod(val);
        if (key == "y_pitch_jump_speed_scale") cfg.actuator.y_pitch_jump_speed_scale = std::stod(val);
        if (key == "y_pitch_jump_speed_cap") cfg.actuator.y_pitch_jump_speed_cap = std::stod(val);
      } else if (section == "filter") {
        if (key == "enable") parse_bool(val, cfg.filter.enable);
        if (key == "neural_enable") parse_bool(val, cfg.filter.neural_enable);
        if (key == "meas_only_when_matched") parse_bool(val, cfg.filter.meas_only_when_matched);
        if (key == "offset_lpf_enable") parse_bool(val, cfg.filter.offset_lpf_enable);
        if (key == "offset_lpf_alpha") cfg.filter.offset_lpf_alpha = std::stod(val);
        if (key == "offset_lpf_alpha_y") cfg.filter.offset_lpf_alpha_y = std::stod(val);
        if (key == "switch_prob") cfg.filter.switch_prob = std::stod(val);
        if (key == "gate_chi2") cfg.filter.gate_chi2 = std::stod(val);
        if (key == "huber_c") cfg.filter.huber_c = std::stod(val);
        if (key == "q_pos") cfg.filter.q_pos = std::stod(val);
        if (key == "q_vel") cfg.filter.q_vel = std::stod(val);
        if (key == "q_acc") cfg.filter.q_acc = std::stod(val);
        if (key == "r_pos") cfg.filter.r_pos = std::stod(val);
        if (key == "alpha_q_min") cfg.filter.alpha_q_min = std::stod(val);
        if (key == "alpha_q_max") cfg.filter.alpha_q_max = std::stod(val);
        if (key == "alpha_r_min") cfg.filter.alpha_r_min = std::stod(val);
        if (key == "alpha_r_max") cfg.filter.alpha_r_max = std::stod(val);
        if (key == "bias_limit_px") cfg.filter.bias_limit_px = std::stod(val);
        if (key == "outlier_prob_min") cfg.filter.outlier_prob_min = std::stod(val);
        if (key == "outlier_prob_max") cfg.filter.outlier_prob_max = std::stod(val);
      } else if (section == "residual") {
        if (key == "enable") parse_bool(val, cfg.residual.enable);
        if (key == "w_conf") cfg.residual.w_conf = std::stod(val);
        if (key == "w_lost") cfg.residual.w_lost = std::stod(val);
        if (key == "w_meas") cfg.residual.w_meas = std::stod(val);
        if (key == "w_age") cfg.residual.w_age = std::stod(val);
        if (key == "age_tau_ms") cfg.residual.age_tau_ms = std::stod(val);
        if (key == "meas_age_hard_ms") cfg.residual.meas_age_hard_ms = std::stod(val);
        if (key == "gate_r_low") cfg.residual.gate_r_low = std::stod(val);
        if (key == "gate_r_high") cfg.residual.gate_r_high = std::stod(val);
        if (key == "alpha_smooth") cfg.residual.alpha_smooth = std::stod(val);
        if (key == "residual_limit_ratio") cfg.residual.residual_limit_ratio = std::stod(val);
        if (key == "residual_min_scale") cfg.residual.residual_min_scale = std::stod(val);
        if (key == "final_slew_per_tick") cfg.residual.final_slew_per_tick = std::stod(val);
      } else if (section == "temporal_comp") {
        if (key == "enable") parse_bool(val, cfg.temporal_comp.enable);
        if (key == "window_size") cfg.temporal_comp.window_size = std::stoi(val);
        if (key == "seq_len") cfg.temporal_comp.seq_len = std::stoi(val);
        if (key == "alpha") cfg.temporal_comp.alpha = std::stod(val);
        if (key == "delta_max") cfg.temporal_comp.delta_max = std::stod(val);
        if (key == "model_path") cfg.temporal_comp.model_path = unquote(val);
        if (key == "stats_path") cfg.temporal_comp.stats_path = unquote(val);
      } else if (section == "log") {
        if (key == "enable") parse_bool(val, cfg.log.enable);
        if (key == "path") cfg.log.path = unquote(val);
        if (key == "dedup_by_frame_id") parse_bool(val, cfg.log.dedup_by_frame_id);
        if (key == "profile") cfg.log.profile = unquote(val);
      }
    }
#endif
  } catch (const std::exception& e) {
    // 配置解析失败时不让程序直接崩溃，而是保留默认值继续运行。
    std::cerr << "[config] Failed to read " << path << ", fallback to defaults: " << e.what() << '\n';
  }

  // 向后兼容旧版配置：如果只设置了 use_bytetrack，则自动推断 tracker_type。
  std::string tt = cfg.mot.tracker_type;
  std::transform(tt.begin(), tt.end(), tt.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (tt.empty()) {
    cfg.mot.tracker_type = cfg.mot.use_bytetrack ? "bytetrack" : "botsort";
  }

  // 统一整理跟踪参数，保证阈值关系正确且数值处于安全范围。
  if (cfg.mot.track_low_thresh > cfg.mot.track_high_thresh) {
    std::swap(cfg.mot.track_low_thresh, cfg.mot.track_high_thresh);
  }
  if (cfg.mot.new_track_thresh < cfg.mot.track_high_thresh) {
    cfg.mot.new_track_thresh = cfg.mot.track_high_thresh;
  }
  cfg.mot.track_low_thresh = std::clamp(cfg.mot.track_low_thresh, 0.0, 1.0);
  cfg.mot.track_high_thresh = std::clamp(cfg.mot.track_high_thresh, 0.0, 1.0);
  cfg.mot.new_track_thresh = std::clamp(cfg.mot.new_track_thresh, 0.0, 1.0);
  cfg.mot.iou_gate = std::clamp(cfg.mot.iou_gate, 0.0, 1.0);
  cfg.mot.second_match_iou = std::clamp(cfg.mot.second_match_iou, 0.0, 1.0);
  cfg.mot.botsort_iou_weight = std::clamp(cfg.mot.botsort_iou_weight, 0.0, 1.0);
  cfg.mot.botsort_center_weight = std::clamp(cfg.mot.botsort_center_weight, 0.0, 1.0);
  const double wsum = cfg.mot.botsort_iou_weight + cfg.mot.botsort_center_weight;
  if (wsum <= 1e-9) {
    cfg.mot.botsort_iou_weight = 0.75;
    cfg.mot.botsort_center_weight = 0.25;
  } else {
    cfg.mot.botsort_iou_weight /= wsum;
    cfg.mot.botsort_center_weight /= wsum;
  }
  cfg.mot.botsort_max_center_dist_ratio = std::clamp(cfg.mot.botsort_max_center_dist_ratio, 0.05, 1.5);
  cfg.mot.max_age = std::max(1, cfg.mot.max_age);
  cfg.camera.zoom_out_trigger_mult = std::clamp(cfg.camera.zoom_out_trigger_mult, 1.0, 5.0);
  cfg.camera.zoom_stable_required_ms = std::clamp(cfg.camera.zoom_stable_required_ms, 0, 10000);
  cfg.camera.zoom_hold_miss_frames = std::max(1, cfg.camera.zoom_hold_miss_frames);
  cfg.camera.zoom_reacquire_freeze_frames = std::max(0, cfg.camera.zoom_reacquire_freeze_frames);
  cfg.camera.zoom_reacquire_confirm_frames = std::max(1, cfg.camera.zoom_reacquire_confirm_frames);

  cfg.model.tile_size = std::max(160, cfg.model.tile_size);
  cfg.model.tile_overlap = std::clamp(cfg.model.tile_overlap, 0.0, 0.8);
  cfg.model.bbox_size_filter_alpha = std::clamp(cfg.model.bbox_size_filter_alpha, 0.0, 1.0);
  cfg.model.bbox_size_filter_min_scale = std::clamp(cfg.model.bbox_size_filter_min_scale, 0.10, 1.0);
  cfg.model.bbox_size_filter_max_scale =
      std::clamp(cfg.model.bbox_size_filter_max_scale, cfg.model.bbox_size_filter_min_scale, 5.0);
  cfg.model.bbox_size_filter_center_gate_px = std::clamp(cfg.model.bbox_size_filter_center_gate_px, 1.0, 10000.0);
  cfg.model.bbox_size_filter_center_deadband_px =
      std::clamp(cfg.model.bbox_size_filter_center_deadband_px, 0.0, 1000.0);
  cfg.model.bbox_size_filter_size_deadband_px =
      std::clamp(cfg.model.bbox_size_filter_size_deadband_px, 0.0, 1000.0);
  cfg.model.bbox_size_filter_max_center_step_px =
      std::clamp(cfg.model.bbox_size_filter_max_center_step_px, 0.0, 1000.0);
  cfg.model.bbox_size_filter_max_size_step_px =
      std::clamp(cfg.model.bbox_size_filter_max_size_step_px, 0.0, 1000.0);
  cfg.model.tile_max_per_frame = std::max(1, cfg.model.tile_max_per_frame);
  cfg.model.tile_global_interval = std::max(1, cfg.model.tile_global_interval);
  cfg.model.tile_priority_topk = std::max(1, cfg.model.tile_priority_topk);
  cfg.model.tile_priority_expand_ratio = std::clamp(cfg.model.tile_priority_expand_ratio, 1.0, 6.0);
  cfg.model.tile_priority_ttl = std::max(1, cfg.model.tile_priority_ttl);

  // 控制器和执行器参数最后统一做安全夹紧，避免异常配置损坏硬件。
  cfg.control.rate_hz = std::clamp(cfg.control.rate_hz, 1, 120);
  cfg.control.deadband_px = std::clamp(cfg.control.deadband_px, 0.0, 5000.0);
  cfg.control.cmd_limit = std::clamp(cfg.control.cmd_limit, 0.0, 200.0);
  cfg.control.full_speed_px = std::clamp(cfg.control.full_speed_px, cfg.control.deadband_px + 1.0, 5000.0);
  cfg.control.slew_per_tick = std::clamp(cfg.control.slew_per_tick, 0.0, 30.0);

  cfg.actuator.scale_x = std::clamp(cfg.actuator.scale_x, 0.01, 1000.0);
  cfg.actuator.scale_y = std::clamp(cfg.actuator.scale_y, 0.01, 1000.0);
  cfg.actuator.speed_limit = std::clamp(cfg.actuator.speed_limit, 1, 200);
  cfg.actuator.y_pitch_baud = std::clamp(cfg.actuator.y_pitch_baud, 1200, 3000000);
  if (cfg.actuator.y_pitch_min_deg > cfg.actuator.y_pitch_max_deg) {
    std::swap(cfg.actuator.y_pitch_min_deg, cfg.actuator.y_pitch_max_deg);
  }
  cfg.actuator.y_pitch_upper_stop_deg =
      std::clamp(cfg.actuator.y_pitch_upper_stop_deg, cfg.actuator.y_pitch_min_deg, cfg.actuator.y_pitch_max_deg);
  cfg.actuator.y_pitch_lpf_alpha = std::clamp(cfg.actuator.y_pitch_lpf_alpha, 0.0, 1.0);
  cfg.actuator.y_pitch_release_margin_deg = std::clamp(cfg.actuator.y_pitch_release_margin_deg, 0.0, 30.0);
  cfg.actuator.y_pitch_timeout_ms = std::clamp(cfg.actuator.y_pitch_timeout_ms, 10, 5000);
  cfg.actuator.y_pitch_soft_zone_deg = std::clamp(cfg.actuator.y_pitch_soft_zone_deg, 0.0, 30.0);
  cfg.actuator.y_pitch_takeover_speed = std::clamp(cfg.actuator.y_pitch_takeover_speed, 0.0, cfg.control.cmd_limit);
  cfg.actuator.y_pitch_takeover_interval_ms = std::clamp(cfg.actuator.y_pitch_takeover_interval_ms, 5, 1000);
  cfg.actuator.y_pitch_session_idle_ms = std::clamp(cfg.actuator.y_pitch_session_idle_ms, 20, 5000);
  cfg.actuator.y_pitch_max_step_deg = std::clamp(cfg.actuator.y_pitch_max_step_deg, 0.1, 45.0);
  cfg.actuator.y_pitch_jump_speed_scale = std::clamp(cfg.actuator.y_pitch_jump_speed_scale, 0.0, 1.0);
  cfg.actuator.y_pitch_jump_speed_cap = std::clamp(cfg.actuator.y_pitch_jump_speed_cap, 0.0, cfg.control.cmd_limit);
  if (cfg.actuator.mode.empty()) {
    cfg.actuator.mode = "mock";
  }
  if (cfg.actuator.y_pitch_port.empty()) {
    cfg.actuator.y_pitch_port = "COM7";
  }

  // 滤波参数主要限制在数值稳定和经验可用范围内。
  cfg.filter.switch_prob = std::clamp(cfg.filter.switch_prob, 0.001, 0.45);
  cfg.filter.offset_lpf_alpha = std::clamp(cfg.filter.offset_lpf_alpha, 0.0, 1.0);
  cfg.filter.offset_lpf_alpha_y = std::clamp(cfg.filter.offset_lpf_alpha_y, 0.0, 1.0);
  cfg.filter.gate_chi2 = std::clamp(cfg.filter.gate_chi2, 1.0, 100.0);
  cfg.filter.huber_c = std::clamp(cfg.filter.huber_c, 0.1, 10.0);
  cfg.filter.q_pos = std::max(1e-6, cfg.filter.q_pos);
  cfg.filter.q_vel = std::max(1e-6, cfg.filter.q_vel);
  cfg.filter.q_acc = std::max(1e-6, cfg.filter.q_acc);
  cfg.filter.r_pos = std::max(1e-6, cfg.filter.r_pos);
  cfg.filter.alpha_q_min = std::clamp(cfg.filter.alpha_q_min, 0.01, 5.0);
  cfg.filter.alpha_q_max = std::clamp(cfg.filter.alpha_q_max, cfg.filter.alpha_q_min, 20.0);
  cfg.filter.alpha_r_min = std::clamp(cfg.filter.alpha_r_min, 0.01, 5.0);
  cfg.filter.alpha_r_max = std::clamp(cfg.filter.alpha_r_max, cfg.filter.alpha_r_min, 40.0);
  cfg.filter.bias_limit_px = std::clamp(cfg.filter.bias_limit_px, 0.0, 100.0);
  cfg.filter.outlier_prob_min = std::clamp(cfg.filter.outlier_prob_min, 0.0, 0.99);
  cfg.filter.outlier_prob_max = std::clamp(cfg.filter.outlier_prob_max, cfg.filter.outlier_prob_min, 1.0);

  // 残差策略参数同样在这里统一归一化和限幅。
  cfg.residual.w_conf = std::clamp(cfg.residual.w_conf, 0.0, 1.0);
  cfg.residual.w_lost = std::clamp(cfg.residual.w_lost, 0.0, 1.0);
  cfg.residual.w_meas = std::clamp(cfg.residual.w_meas, 0.0, 1.0);
  cfg.residual.w_age = std::clamp(cfg.residual.w_age, 0.0, 1.0);
  const double residual_wsum = cfg.residual.w_conf + cfg.residual.w_lost + cfg.residual.w_meas + cfg.residual.w_age;
  if (residual_wsum > 1e-9) {
    cfg.residual.w_conf /= residual_wsum;
    cfg.residual.w_lost /= residual_wsum;
    cfg.residual.w_meas /= residual_wsum;
    cfg.residual.w_age /= residual_wsum;
  } else {
    cfg.residual.w_conf = 0.45;
    cfg.residual.w_lost = 0.25;
    cfg.residual.w_meas = 0.15;
    cfg.residual.w_age = 0.15;
  }
  cfg.residual.age_tau_ms = std::clamp(cfg.residual.age_tau_ms, 1.0, 5000.0);
  cfg.residual.meas_age_hard_ms = std::clamp(cfg.residual.meas_age_hard_ms, 1.0, 10000.0);
  cfg.residual.gate_r_low = std::clamp(cfg.residual.gate_r_low, 0.0, 1.0);
  cfg.residual.gate_r_high = std::clamp(cfg.residual.gate_r_high, cfg.residual.gate_r_low + 1e-6, 1.0);
  cfg.residual.alpha_smooth = std::clamp(cfg.residual.alpha_smooth, 0.0, 0.999);
  cfg.residual.residual_limit_ratio = std::clamp(cfg.residual.residual_limit_ratio, 0.0, 1.0);
  cfg.residual.residual_min_scale = std::clamp(cfg.residual.residual_min_scale, 0.0, 1.0);
  cfg.residual.final_slew_per_tick = std::clamp(cfg.residual.final_slew_per_tick, 0.0, 200.0);

  cfg.temporal_comp.window_size = std::max(1, cfg.temporal_comp.window_size);
  cfg.temporal_comp.seq_len = std::max(1, cfg.temporal_comp.seq_len);
  cfg.temporal_comp.alpha = std::clamp(cfg.temporal_comp.alpha, 0.0, 1.0);
  cfg.temporal_comp.delta_max = std::max(0.0, cfg.temporal_comp.delta_max);

  return cfg;
}

}  // namespace core
