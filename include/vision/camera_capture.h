#pragma once

#include <string>

#include "vision/cv_compat.h"

namespace vision {

// 相机采集封装，提供打开、读帧、对焦和变焦控制。
class CameraCapture {
 public:
  CameraCapture() = default;
  ~CameraCapture();

  void set_backend(const std::string& backend);
  void set_use_mjpg(bool enable);
  bool open(int camera_index, int width, int height);
  bool read(cv::Mat& frame);
  bool set_auto_focus(bool enable);
  bool set_focus(double value);
  bool set_zoom(double value, double min_value, double max_value);
  double get_zoom(double fallback = 0.0);
  bool adjust_zoom(double delta, double min_value, double max_value);
  bool is_open() const;
  void close();

 private:
  bool apply_auto_focus_setting();
  bool open_internal();

  cv::VideoCapture cap_;
  int camera_index_{0};
  int width_{640};
  int height_{480};
  std::string backend_{"any"};
  bool use_mjpg_{true};
  bool auto_focus_enable_{true};
  bool auto_focus_requested_{false};
  bool zoom_initialized_{false};
  double cached_zoom_{0.0};
};

}  // namespace vision
