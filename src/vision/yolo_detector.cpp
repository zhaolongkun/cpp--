#include "vision/yolo_detector.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>

namespace vision {
namespace {

double intersection_area(const YoloDetector::TileRegion& a, const YoloDetector::TileRegion& b) {
  // 用于评估 tile 与 priority ROI 的重叠程度。
  const int x1 = std::max(a.x, b.x);
  const int y1 = std::max(a.y, b.y);
  const int x2 = std::min(a.x + a.w, b.x + b.w);
  const int y2 = std::min(a.y + a.h, b.y + b.h);
  if (x2 <= x1 || y2 <= y1) {
    return 0.0;
  }
  return static_cast<double>(x2 - x1) * static_cast<double>(y2 - y1);
}

double center_distance_norm(const YoloDetector::TileRegion& a, const YoloDetector::TileRegion& b) {
  // 按 tile 尺度归一化中心距离，用于优先挑选离关注区域更近的分块。
  const double acx = static_cast<double>(a.x) + 0.5 * static_cast<double>(a.w);
  const double acy = static_cast<double>(a.y) + 0.5 * static_cast<double>(a.h);
  const double bcx = static_cast<double>(b.x) + 0.5 * static_cast<double>(b.w);
  const double bcy = static_cast<double>(b.y) + 0.5 * static_cast<double>(b.h);
  const double dx = acx - bcx;
  const double dy = acy - bcy;
  const double d = std::sqrt(dx * dx + dy * dy);
  const double norm = std::max(1.0, std::sqrt(static_cast<double>(a.w) * static_cast<double>(a.w) +
                                              static_cast<double>(a.h) * static_cast<double>(a.h)));
  return d / norm;
}

}  // namespace

std::string YoloDetector::resolve_model_path(const std::string& path) const {
  namespace fs = std::filesystem;
  fs::path p(path);

  if (p.extension() == ".onnx" || p.extension() == ".ONNX") {
    return path;
  }

  // 如果配置里填的是 *.pt，则优先尝试同目录下已经导出的 *.onnx，
  // 这样运行时不需要再依赖 PyTorch。
  fs::path onnx = p;
  onnx.replace_extension(".onnx");
  if (fs::exists(onnx)) {
    return onnx.string();
  }

  return path;
}

bool YoloDetector::load(const std::string& path) {
  model_path_ = resolve_model_path(path);

#ifdef HAVE_OPENCV
  try {
    // 当前工程使用 OpenCV DNN + ONNX 的纯 C++ 推理链路。
    net_ = cv::dnn::readNetFromONNX(model_path_);
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    loaded_ = !net_.empty();
    if (!loaded_) {
      std::cerr << "[detector] failed to load ONNX model: " << model_path_ << '\n';
    } else {
      std::cerr << "[detector] loaded ONNX model: " << model_path_ << '\n';
    }
  } catch (const std::exception& e) {
    loaded_ = false;
    std::cerr << "[detector] load error: " << e.what() << '\n';
  }
#else
  loaded_ = false;
  (void)model_path_;
#endif

  return loaded_;
}

void YoloDetector::set_tiling_config(const TilingConfig& cfg) {
  tiling_cfg_ = cfg;
  // 这里统一做一轮参数清洗，防止异常配置导致 tile 尺寸或步长非法。
  tiling_cfg_.tile_size = std::max(160, tiling_cfg_.tile_size);
  tiling_cfg_.tile_overlap = std::clamp(tiling_cfg_.tile_overlap, 0.0, 0.8);
  tiling_cfg_.tile_max_per_frame = std::max(1, tiling_cfg_.tile_max_per_frame);
  tiling_cfg_.tile_global_interval = std::max(1, tiling_cfg_.tile_global_interval);
  tiling_cfg_.priority_topk = std::max(1, tiling_cfg_.priority_topk);
  tiling_cfg_.priority_expand_ratio = std::clamp(tiling_cfg_.priority_expand_ratio, 1.0, 6.0);
  tiling_cfg_.priority_ttl = std::max(1, tiling_cfg_.priority_ttl);
  tile_cursor_ = 0;
  infer_frame_counter_ = 0;
  priority_roi_valid_ = false;
  priority_roi_ttl_ = 0;
}

void YoloDetector::set_priority_roi(const core::BBox& bbox, int frame_width, int frame_height) {
  if (frame_width <= 0 || frame_height <= 0) {
    return;
  }

  // 以当前目标框为中心扩展出一个优先搜索区域，
  // 后续若启用 tiling，会优先跑落在这块区域附近的分块。
  const double cx = bbox.cx();
  const double cy = bbox.cy();
  const double w = std::max(1.0, bbox.width()) * tiling_cfg_.priority_expand_ratio;
  const double h = std::max(1.0, bbox.height()) * tiling_cfg_.priority_expand_ratio;

  int x1 = static_cast<int>(std::floor(cx - 0.5 * w));
  int y1 = static_cast<int>(std::floor(cy - 0.5 * h));
  int x2 = static_cast<int>(std::ceil(cx + 0.5 * w));
  int y2 = static_cast<int>(std::ceil(cy + 0.5 * h));

  x1 = std::clamp(x1, 0, frame_width - 1);
  y1 = std::clamp(y1, 0, frame_height - 1);
  x2 = std::clamp(x2, 0, frame_width);
  y2 = std::clamp(y2, 0, frame_height);
  if (x2 <= x1 || y2 <= y1) {
    return;
  }

  priority_roi_ = TileRegion{x1, y1, x2 - x1, y2 - y1};
  priority_roi_valid_ = true;
  priority_roi_ttl_ = tiling_cfg_.priority_ttl;
}

void YoloDetector::clear_priority_roi() {
  priority_roi_valid_ = false;
  priority_roi_ttl_ = 0;
}

std::vector<YoloDetector::TileRegion> YoloDetector::build_tiles(int width, int height) const {
  std::vector<TileRegion> tiles;
  if (width <= 0 || height <= 0) {
    return tiles;
  }

  // 根据 tile_size 和 overlap 在整幅图上滑窗，生成所有候选子块。
  const int tile = std::min({tiling_cfg_.tile_size, width, height});
  const int stride = std::max(32, static_cast<int>(std::round(tile * (1.0 - tiling_cfg_.tile_overlap))));
  if (tile <= 0 || stride <= 0) {
    return tiles;
  }

  std::vector<int> xs;
  for (int x = 0; x + tile < width; x += stride) {
    xs.push_back(x);
  }
  xs.push_back(std::max(0, width - tile));
  std::sort(xs.begin(), xs.end());
  xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

  std::vector<int> ys;
  for (int y = 0; y + tile < height; y += stride) {
    ys.push_back(y);
  }
  ys.push_back(std::max(0, height - tile));
  std::sort(ys.begin(), ys.end());
  ys.erase(std::unique(ys.begin(), ys.end()), ys.end());

  for (int y : ys) {
    for (int x : xs) {
      tiles.push_back(TileRegion{x, y, tile, tile});
    }
  }

  return tiles;
}

std::vector<core::Detection> YoloDetector::nms_merge(const std::vector<core::Detection>& dets) const {
#ifdef HAVE_OPENCV
  std::vector<core::Detection> out;
  if (dets.empty()) {
    return out;
  }

  // 全局检测和分块检测可能同时命中同一个目标，最后统一再做一次 NMS 去重。
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  boxes.reserve(dets.size());
  scores.reserve(dets.size());
  for (const auto& d : dets) {
    const int x = std::max(0, static_cast<int>(std::round(d.bbox.x1)));
    const int y = std::max(0, static_cast<int>(std::round(d.bbox.y1)));
    const int w = std::max(1, static_cast<int>(std::round(d.bbox.x2 - d.bbox.x1)));
    const int h = std::max(1, static_cast<int>(std::round(d.bbox.y2 - d.bbox.y1)));
    boxes.emplace_back(x, y, w, h);
    scores.push_back(static_cast<float>(d.conf));
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(conf_threshold_), static_cast<float>(nms_threshold_), keep);
  out.reserve(keep.size());
  for (int idx : keep) {
    out.push_back(dets[idx]);
  }
  return out;
#else
  return dets;
#endif
}

std::vector<core::Detection> YoloDetector::infer_single(const cv::Mat& frame, int offset_x, int offset_y) const {
#ifdef HAVE_OPENCV
  std::vector<core::Detection> out_dets;
  if (!loaded_ || frame.empty()) {
    return out_dets;
  }

  // 输入先缩放到模型固定尺寸，再按 1/255 归一化。
  cv::Mat blob;
  cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(input_width_, input_height_), cv::Scalar(), true, false);

  cv::dnn::Net net = net_;
  net.setInput(blob);
  std::vector<cv::Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());
  if (outputs.empty()) {
    return out_dets;
  }

  const cv::Mat& raw = outputs[0];
  cv::Mat pred;

  if (raw.dims == 3) {
    const int d0 = raw.size[0];
    const int d1 = raw.size[1];
    const int d2 = raw.size[2];
    if (d0 == 1 && d1 <= 64 && d2 > d1) {
      // 有些导出模型输出为 [1, C, N]，这里转成统一的 [N, C]。
      cv::Mat c_by_n(d1, d2, CV_32F, const_cast<float*>(raw.ptr<float>()));
      cv::transpose(c_by_n, pred);
    } else if (d0 == 1 && d2 <= 64 && d1 > d2) {
      // 另一类导出模型已经是 [1, N, C]，直接 clone 成连续内存。
      pred = cv::Mat(d1, d2, CV_32F, const_cast<float*>(raw.ptr<float>())).clone();
    } else {
      return out_dets;
    }
  } else if (raw.dims == 2) {
    pred = raw;
  } else {
    return out_dets;
  }

  const float x_gain = static_cast<float>(frame.cols) / static_cast<float>(input_width_);
  const float y_gain = static_cast<float>(frame.rows) / static_cast<float>(input_height_);

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;
  boxes.reserve(pred.rows);
  scores.reserve(pred.rows);
  class_ids.reserve(pred.rows);

  for (int i = 0; i < pred.rows; ++i) {
    const float* row = pred.ptr<float>(i);
    const int cols = pred.cols;
    if (cols < 5) {
      continue;
    }

    const float cx = row[0];
    const float cy = row[1];
    const float w = row[2];
    const float h = row[3];

    float conf = 0.0f;
    int cls = 0;

    if (cols == 5) {
      // 单类别模型：第 5 列直接就是目标置信度。
      conf = row[4];
      cls = 0;
    } else {
      // 多类别模型：objectness * best_class_score 作为最终置信度。
      const float obj = row[4];
      float best_cls = 0.0f;
      int best_id = 0;
      for (int c = 5; c < cols; ++c) {
        if (row[c] > best_cls) {
          best_cls = row[c];
          best_id = c - 5;
        }
      }
      conf = obj * best_cls;
      cls = best_id;
    }

    if (conf < static_cast<float>(conf_threshold_)) {
      continue;
    }

    float x1 = (cx - 0.5f * w) * x_gain;
    float y1 = (cy - 0.5f * h) * y_gain;
    float x2 = (cx + 0.5f * w) * x_gain;
    float y2 = (cy + 0.5f * h) * y_gain;

    x1 = std::max(0.0f, std::min(x1, static_cast<float>(frame.cols - 1)));
    y1 = std::max(0.0f, std::min(y1, static_cast<float>(frame.rows - 1)));
    x2 = std::max(0.0f, std::min(x2, static_cast<float>(frame.cols - 1)));
    y2 = std::max(0.0f, std::min(y2, static_cast<float>(frame.rows - 1)));

    const int bw = std::max(1, static_cast<int>(x2 - x1));
    const int bh = std::max(1, static_cast<int>(y2 - y1));
    boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), bw, bh);
    scores.push_back(conf);
    class_ids.push_back(cls);
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, static_cast<float>(conf_threshold_), static_cast<float>(nms_threshold_), keep);

  out_dets.reserve(keep.size());
  for (int idx : keep) {
    const auto& b = boxes[idx];
    core::Detection d;
    d.bbox.x1 = static_cast<double>(b.x + offset_x);
    d.bbox.y1 = static_cast<double>(b.y + offset_y);
    d.bbox.x2 = static_cast<double>(b.x + b.width + offset_x);
    d.bbox.y2 = static_cast<double>(b.y + b.height + offset_y);
    d.conf = scores[idx];
    d.cls = class_ids[idx];
    // 如果输入是某个 tile，则把局部坐标平移回整图坐标系。
    out_dets.push_back(d);
  }
  return out_dets;
#else
  (void)frame;
  (void)offset_x;
  (void)offset_y;
  return {};
#endif
}

std::vector<core::Detection> YoloDetector::infer(const cv::Mat& frame) const {
#ifdef HAVE_OPENCV
  if (!loaded_ || frame.empty()) {
    return {};
  }

  if (!tiling_cfg_.enable) {
    // 未启用分块时，直接整图推理。
    return infer_single(frame, 0, 0);
  }

  std::vector<core::Detection> all_dets;
  ++infer_frame_counter_;

  const bool run_global = (infer_frame_counter_ % static_cast<uint64_t>(tiling_cfg_.tile_global_interval) == 0ULL);
  if (run_global) {
    // 间隔若干帧做一次整图推理，避免只看局部时彻底丢失新目标。
    const auto global = infer_single(frame, 0, 0);
    all_dets.insert(all_dets.end(), global.begin(), global.end());
  }

  const auto tiles = build_tiles(frame.cols, frame.rows);
  if (tiles.empty()) {
    return nms_merge(all_dets);
  }

  const int k = std::min<int>(tiling_cfg_.tile_max_per_frame, static_cast<int>(tiles.size()));
  if (tile_cursor_ >= tiles.size()) {
    tile_cursor_ = 0;
  }

  std::vector<size_t> selected_indices;
  selected_indices.reserve(static_cast<size_t>(k));

  if (tiling_cfg_.priority_enable && priority_roi_valid_ && priority_roi_ttl_ > 0) {
    struct TileScore {
      size_t idx{0};
      double score{-1e9};
    };
    std::vector<TileScore> scores;
    scores.reserve(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
      const double inter = intersection_area(tiles[i], priority_roi_);
      const double area = std::max(1.0, static_cast<double>(tiles[i].w) * static_cast<double>(tiles[i].h));
      const double overlap_ratio = inter / area;
      const double dist = center_distance_norm(tiles[i], priority_roi_);
      // 分数越高，说明该 tile 越值得优先推理。
      const double score = 5.0 * overlap_ratio - dist;
      scores.push_back(TileScore{i, score});
    }
    std::sort(scores.begin(), scores.end(), [](const TileScore& a, const TileScore& b) { return a.score > b.score; });

    const int want = std::min({k, tiling_cfg_.priority_topk, static_cast<int>(scores.size())});
    for (int i = 0; i < want; ++i) {
      selected_indices.push_back(scores[i].idx);
    }
    --priority_roi_ttl_;
    if (priority_roi_ttl_ <= 0) {
      priority_roi_valid_ = false;
      priority_roi_ttl_ = 0;
    }
  }

  size_t rr = tile_cursor_;
  while (static_cast<int>(selected_indices.size()) < k) {
    const size_t idx = rr % tiles.size();
    if (std::find(selected_indices.begin(), selected_indices.end(), idx) == selected_indices.end()) {
      selected_indices.push_back(idx);
    }
    ++rr;
  }
  tile_cursor_ = (tile_cursor_ + static_cast<size_t>(k)) % tiles.size();

  for (size_t idx : selected_indices) {
    const TileRegion roi = tiles[idx];
    const cv::Rect rect(roi.x, roi.y, roi.w, roi.h);
    const cv::Mat patch = frame(rect);
    // 对选中的局部分块做推理，再把检测结果映射回整图。
    auto dets = infer_single(patch, roi.x, roi.y);
    all_dets.insert(all_dets.end(), dets.begin(), dets.end());
  }

  return nms_merge(all_dets);
#else
  (void)frame;
  return {};
#endif
}

}  // namespace vision
