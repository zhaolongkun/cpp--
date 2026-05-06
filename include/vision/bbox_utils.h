#pragma once

#include "core/types.h"

namespace vision {

// 计算两个框的交并比。
double iou(const core::BBox& a, const core::BBox& b);
// 将中心点 + 宽高格式转换为左上/右下格式。
core::BBox cxcywh_to_bbox(double cx, double cy, double w, double h);

}  // namespace vision
