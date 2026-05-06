#include "vision/camera_capture.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

namespace vision {

CameraCapture::~CameraCapture() { close(); }

void CameraCapture::set_backend(const std::string& backend) { backend_ = backend; }

void CameraCapture::set_use_mjpg(bool enable) { use_mjpg_ = enable; }

bool CameraCapture::open(int camera_index, int width, int height) {
  camera_index_ = camera_index;
  width_ = width;
  height_ = height;
  return open_internal();
}

bool CameraCapture::open_internal() {
  close();
#ifdef HAVE_OPENCV
#ifdef _WIN32
  std::string b = backend_;
  std::transform(b.begin(), b.end(), b.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  std::vector<int> api_order;
  if (b == "dshow") {
    api_order = {cv::CAP_DSHOW, cv::CAP_ANY, cv::CAP_MSMF};
  } else if (b == "msmf") {
    api_order = {cv::CAP_MSMF, cv::CAP_ANY, cv::CAP_DSHOW};
  } else {
    // "any": prioritize DSHOW for stability, then fallback.
    api_order = {cv::CAP_DSHOW, cv::CAP_ANY, cv::CAP_MSMF};
  }

  bool opened = false;
  for (int api : api_order) {
    // Windows 下不同采集后端稳定性差异很大，这里按优先级逐个尝试。
    if (api == cv::CAP_ANY) {
      opened = cap_.open(camera_index_);
    } else {
      opened = cap_.open(camera_index_, api);
    }
    if (opened) {
      break;
    }
  }
#else
  const bool opened = cap_.open(camera_index_);
#endif
#else
  const bool opened = cap_.open(camera_index_);
#endif

  if (!opened) {
    return false;
  }

#ifdef HAVE_OPENCV
  // 采集参数保持和已经验证过的 Python 版本一致，
  // 这样便于和旧系统做性能/时延对齐。
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
  if (use_mjpg_) {
    cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  }
  cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
  (void)apply_auto_focus_setting();
  const double z = cap_.get(cv::CAP_PROP_ZOOM);
  if (std::isfinite(z) && z >= 0.0) {
    cached_zoom_ = z;
    zoom_initialized_ = true;
  } else {
    zoom_initialized_ = false;
  }
#else
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
#endif
  return true;
}

bool CameraCapture::read(cv::Mat& frame) {
  if (!cap_.isOpened()) {
    return false;
  }
  if (cap_.read(frame) && !frame.empty()) {
    return true;
  }

  // 某些 USB 摄像头会偶发 read 失败，这里做一次重连兜底。
  if (!open_internal()) {
    return false;
  }
  return cap_.read(frame) && !frame.empty();
}

bool CameraCapture::set_auto_focus(bool enable) {
  auto_focus_enable_ = enable;
  auto_focus_requested_ = true;
  return apply_auto_focus_setting();
}

bool CameraCapture::apply_auto_focus_setting() {
#ifdef HAVE_OPENCV
  if (!auto_focus_requested_ || !cap_.isOpened()) {
    return true;
  }
  // 自动对焦是易变参数，只有用户明确设置后才真正写入相机。
  return cap_.set(cv::CAP_PROP_AUTOFOCUS, auto_focus_enable_ ? 1.0 : 0.0);
#else
  return false;
#endif
}

bool CameraCapture::set_focus(double value) {
#ifdef HAVE_OPENCV
  // 手动设置焦距前先强制关闭自动对焦，避免相机固件把手动值又改回去。
  auto_focus_enable_ = false;
  auto_focus_requested_ = true;
  (void)apply_auto_focus_setting();
  if (!cap_.isOpened()) {
    return false;
  }
  return cap_.set(cv::CAP_PROP_FOCUS, value);
#else
  (void)value;
  return false;
#endif
}

bool CameraCapture::set_zoom(double value, double min_value, double max_value) {
#ifdef HAVE_OPENCV
  if (!cap_.isOpened() || !std::isfinite(value)) {
    return false;
  }
  if (max_value < min_value) {
    std::swap(min_value, max_value);
  }
  const double target = std::clamp(std::round(value), min_value, max_value);
  // 某些 UVC 相机只接受整数 zoom 档位，所以这里先 round 再写入。
  const bool ok = cap_.set(cv::CAP_PROP_ZOOM, target);
  if (ok) {
    cached_zoom_ = target;
    zoom_initialized_ = true;
  }
  return ok;
#else
  (void)value;
  (void)min_value;
  (void)max_value;
  return false;
#endif
}

double CameraCapture::get_zoom(double fallback) {
#ifdef HAVE_OPENCV
  if (!cap_.isOpened()) {
    return fallback;
  }
  const double queried = cap_.get(cv::CAP_PROP_ZOOM);
  if (std::isfinite(queried) && queried >= 0.0) {
    cached_zoom_ = queried;
    zoom_initialized_ = true;
    return queried;
  }
  if (zoom_initialized_) {
    return cached_zoom_;
  }
  return fallback;
#else
  return fallback;
#endif
}

bool CameraCapture::adjust_zoom(double delta, double min_value, double max_value) {
#ifdef HAVE_OPENCV
  if (!cap_.isOpened() || !std::isfinite(delta) || std::abs(delta) < 1e-9) {
    return false;
  }

  if (max_value < min_value) {
    std::swap(min_value, max_value);
  }

  double current = cached_zoom_;
  const double queried = cap_.get(cv::CAP_PROP_ZOOM);
  if (std::isfinite(queried) && queried >= 0.0) {
    current = queried;
    cached_zoom_ = queried;
    zoom_initialized_ = true;
  } else if (!zoom_initialized_) {
    current = min_value;
    cached_zoom_ = current;
    zoom_initialized_ = true;
  }

  const double next = std::clamp(current + delta, min_value, max_value);
  if (std::abs(next - current) < 1e-9) {
    return true;
  }

  // adjust_zoom 只是计算下一档目标值，真正的写相机仍复用 set_zoom。
  return set_zoom(next, min_value, max_value);
#else
  (void)delta;
  (void)min_value;
  (void)max_value;
  return false;
#endif
}

bool CameraCapture::is_open() const { return cap_.isOpened(); }

void CameraCapture::close() {
  if (cap_.isOpened()) {
    cap_.release();
  }
}

}  // namespace vision
