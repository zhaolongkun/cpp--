#include "control/target_selector.h"

#include <cmath>
#include <limits>

namespace control {

int TargetSelector::select(const std::vector<core::TrackState>& tracks, int preferred_id, double center_x,
                           double center_y) const {
  // 如果上一次已经锁定了某个轨迹，并且这一帧该轨迹还存在，则继续跟踪它，
  // 这样可以减少目标频繁切换导致的控制抖动。
  for (const auto& t : tracks) {
    if (t.track_id == preferred_id) {
      return preferred_id;
    }
  }

  double best_dist = std::numeric_limits<double>::max();
  int best_id = -1;
  // 如果偏好目标已经丢失，则退化为“选择最靠近图像中心的轨迹”。
  // 这样更符合云台控制场景：通常画面中心附近的目标就是当前关心的目标。
  for (const auto& t : tracks) {
    const double dx = t.cx - center_x;
    const double dy = t.cy - center_y;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_dist) {
      best_dist = d2;
      best_id = t.track_id;
    }
  }
  return best_id;
}

}  // namespace control
