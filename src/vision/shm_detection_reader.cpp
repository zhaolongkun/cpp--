#include "vision/shm_detection_reader.h"
#include <cstring>
#include <iostream>
#define NOMINMAX
#include <windows.h>

#ifdef HAVE_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace vision {

ShmDetectionReader::ShmDetectionReader(const std::string& name) : name_(name) {}

ShmDetectionReader::~ShmDetectionReader() {
  if (view_) UnmapViewOfFile(view_);
  if (handle_) CloseHandle(handle_);
}

bool ShmDetectionReader::open() {
  handle_ = OpenFileMappingA(FILE_MAP_READ, FALSE, name_.c_str());
  if (!handle_) {
    std::cerr << "[shm_reader] OpenFileMapping failed: " << name_ << '\n';
    return false;
  }
  view_ = MapViewOfFile(handle_, FILE_MAP_READ, 0, 0, kShmSize);
  if (!view_) {
    std::cerr << "[shm_reader] MapViewOfFile failed\n";
    return false;
  }
  return true;
}

std::vector<core::Detection> ShmDetectionReader::read(float conf_threshold) {
  if (!view_) return {};
  uint8_t buf[kShmSize];
  memcpy(buf, view_, kShmSize);

  int32_t seq, count;
  memcpy(&seq, buf, 4);
  memcpy(&count, buf + 4, 4);

  if (seq == last_seq_ || count <= 0 || count > kMaxDets) return {};
  last_seq_ = seq;

  std::vector<core::Detection> dets;
  dets.reserve(count);
  const uint8_t* p = buf + 8;
  for (int i = 0; i < count; ++i, p += 20) {
    float x1, y1, x2, y2, conf;
    memcpy(&x1, p,      4);
    memcpy(&y1, p + 4,  4);
    memcpy(&x2, p + 8,  4);
    memcpy(&y2, p + 12, 4);
    memcpy(&conf, p + 16, 4);
    if (conf < conf_threshold) continue;
    core::Detection d;
    d.bbox = core::BBox{x1, y1, x2, y2};
    d.conf = conf;
    d.cls = 0;
    dets.push_back(d);
  }
  return dets;
}

#ifdef HAVE_OPENCV
bool ShmDetectionReader::read_packet(cv::Mat& frame, std::vector<core::Detection>& detections, float conf_threshold,
                                     int* seq_out) {
  frame.release();
  detections.clear();
  if (!view_) {
    return false;
  }

  uint8_t buf[kShmSize];
  memcpy(buf, view_, kShmSize);

  int32_t seq = 0;
  int32_t count = 0;
  memcpy(&seq, buf, 4);
  memcpy(&count, buf + 4, 4);

  if (seq == last_seq_ || count < 0 || count > kMaxDets) {
    return false;
  }
  last_seq_ = seq;
  if (seq_out != nullptr) {
    *seq_out = seq;
  }

  detections.reserve(static_cast<size_t>(count));
  const uint8_t* p = buf + 8;
  for (int i = 0; i < count; ++i, p += 20) {
    float x1, y1, x2, y2, conf;
    memcpy(&x1, p, 4);
    memcpy(&y1, p + 4, 4);
    memcpy(&x2, p + 8, 4);
    memcpy(&y2, p + 12, 4);
    memcpy(&conf, p + 16, 4);
    if (conf < conf_threshold) {
      continue;
    }
    core::Detection d;
    d.bbox = core::BBox{x1, y1, x2, y2};
    d.conf = conf;
    d.cls = 0;
    detections.push_back(d);
  }

  frame = cv::Mat(kFrameH, kFrameW, CV_8UC3);
  memcpy(frame.data, buf + kDetBytes, kFrameBytes);
  return true;
}

cv::Mat ShmDetectionReader::read_frame() {
  if (!view_) return {};
  const uint8_t* src = static_cast<const uint8_t*>(view_) + kDetBytes;
  cv::Mat frame(kFrameH, kFrameW, CV_8UC3);
  memcpy(frame.data, src, kFrameBytes);
  return frame;
}
#endif

}  // namespace vision
