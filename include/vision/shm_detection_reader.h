#pragma once
#include <string>
#include <vector>
#include "core/types.h"

#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#endif

namespace vision {

class ShmDetectionReader {
 public:
  static constexpr int kMaxDets = 64;
  static constexpr int kFrameW = 640;
  static constexpr int kFrameH = 480;
  static constexpr int kFrameBytes = kFrameW * kFrameH * 3;
  static constexpr int kDetBytes = 4 + 4 + kMaxDets * 5 * 4;
  static constexpr int kShmSize = kDetBytes + kFrameBytes;

  explicit ShmDetectionReader(const std::string& name = "yolo_det_shm");
  ~ShmDetectionReader();

  bool open();
  std::vector<core::Detection> read(float conf_threshold = 0.0f);

#ifdef HAVE_OPENCV
  // Reads a coherent shm snapshot and returns true only when a new packet is available.
  bool read_packet(cv::Mat& frame, std::vector<core::Detection>& detections, float conf_threshold = 0.0f,
                   int* seq_out = nullptr);
  // Returns latest frame from shm (BGR). Empty if not available.
  cv::Mat read_frame();
#endif

 private:
  std::string name_;
  void* handle_{nullptr};
  void* view_{nullptr};
  int last_seq_{-1};
};

}  // namespace vision
