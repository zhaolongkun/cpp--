#pragma once

#include <vector>

#include "core/types.h"

namespace control {

class TargetSelector {
 public:
  // 若 preferred_id 仍存在，则继续跟踪；
  // 否则选取离图像中心最近的轨迹作为当前控制目标。
  int select(const std::vector<core::TrackState>& tracks, int preferred_id, double center_x, double center_y) const;
};

}  // namespace control
