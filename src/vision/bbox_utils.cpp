#include "vision/bbox_utils.h"

#include <algorithm>

namespace vision {

double iou(const core::BBox& a, const core::BBox& b) {
  // 交并比是检测后处理和跟踪匹配里最常用的相似度指标，
  // 数值越大表示两个框重叠程度越高。
  const double x1 = std::max(a.x1, b.x1);
  const double y1 = std::max(a.y1, b.y1);
  const double x2 = std::min(a.x2, b.x2);
  const double y2 = std::min(a.y2, b.y2);

  const double w = std::max(0.0, x2 - x1);
  const double h = std::max(0.0, y2 - y1);
  const double inter = w * h;
  const double area_a = std::max(0.0, a.width()) * std::max(0.0, a.height());
  const double area_b = std::max(0.0, b.width()) * std::max(0.0, b.height());
  const double uni = area_a + area_b - inter;
  if (uni <= 1e-9) {
    return 0.0;
  }
  return inter / uni;
}

core::BBox cxcywh_to_bbox(double cx, double cy, double w, double h) {
  // YOLO 系模型常输出中心点 + 宽高格式，这里转换成项目内部统一的左上/右下坐标。
  core::BBox b;
  b.x1 = cx - 0.5 * w;
  b.y1 = cy - 0.5 * h;
  b.x2 = cx + 0.5 * w;
  b.y2 = cy + 0.5 * h;
  return b;
}

}  // namespace vision
