#pragma once

#include <string>
#include <vector>

#include "core/config.h"
#include "core/types.h"
#include "tracking/track.h"

namespace tracking {

// 多目标跟踪器。
// 当前实现支持 ByteTrack 风格双阶段匹配，以及可选的 BoT-SORT 风格代价。
class MultiObjectTracker {
 public:
  explicit MultiObjectTracker(const core::MotConfig& cfg);

  // 输入当前帧检测结果，更新内部轨迹集合。
  void update(const std::vector<core::Detection>& dets, double dt_sec);
  // 导出当前全部有效轨迹。
  std::vector<core::TrackState> get_tracks() const;

 private:
  std::string tracker_type_{"bytetrack"};
  int max_age_{15};
  double iou_gate_{0.25};
  double track_high_thresh_{0.6};
  double track_low_thresh_{0.1};
  double new_track_thresh_{0.7};
  double second_match_iou_{0.15};
  double botsort_iou_weight_{0.75};
  double botsort_center_weight_{0.25};
  double botsort_max_center_dist_ratio_{0.55};
  int next_track_id_{1};
  std::vector<Track> tracks_;
};

}  // namespace tracking
