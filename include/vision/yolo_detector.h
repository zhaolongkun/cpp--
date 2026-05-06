#pragma once

#include <string>
#include <vector>

#include "core/types.h"
#include "vision/cv_compat.h"

namespace vision {

// YOLO 检测器封装，支持整图推理和分块推理。
class YoloDetector {
 public:
  // 分块推理配置。
  struct TilingConfig {
    bool enable{false};
    int tile_size{960};
    double tile_overlap{0.25};
    int tile_max_per_frame{2};
    int tile_global_interval{6};
    bool priority_enable{true};
    int priority_topk{1};
    double priority_expand_ratio{2.0};
    int priority_ttl{8};
  };

  // 单个分块区域。
  struct TileRegion {
    int x{0};
    int y{0};
    int w{0};
    int h{0};
  };

  bool load(const std::string& path);
  void set_conf_threshold(double conf) { conf_threshold_ = conf; }
  void set_nms_threshold(double nms) { nms_threshold_ = nms; }
  void set_tiling_config(const TilingConfig& cfg);
  // 根据当前关注目标设置 priority ROI，后续优先检测其附近区域。
  void set_priority_roi(const core::BBox& bbox, int frame_width, int frame_height);
  void clear_priority_roi();
  std::vector<core::Detection> infer(const cv::Mat& frame) const;

 private:
  std::vector<core::Detection> infer_single(const cv::Mat& frame, int offset_x, int offset_y) const;
  std::string resolve_model_path(const std::string& path) const;
  std::vector<TileRegion> build_tiles(int width, int height) const;
  std::vector<core::Detection> nms_merge(const std::vector<core::Detection>& dets) const;

  std::string model_path_;
  bool loaded_{false};
  double conf_threshold_{0.30};
  double nms_threshold_{0.45};
  TilingConfig tiling_cfg_{};
  mutable uint64_t infer_frame_counter_{0};
  mutable size_t tile_cursor_{0};
  mutable bool priority_roi_valid_{false};
  mutable TileRegion priority_roi_{};
  mutable int priority_roi_ttl_{0};

#ifdef HAVE_OPENCV
  cv::dnn::Net net_;
  int input_width_{640};
  int input_height_{640};
#endif
};

}  // namespace vision
