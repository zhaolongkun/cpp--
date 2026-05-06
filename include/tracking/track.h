#pragma once

#include "core/types.h"
#include "tracking/kalman_bbox.h"

namespace tracking {

// 单条轨迹对象，封装一条目标在时序上的连续状态。
class Track {
 public:
  Track(int track_id, const core::Detection& det);

  // 预测到下一帧时刻。
  void predict(double dt);
  // 用当前帧检测结果更新轨迹。
  void update(const core::Detection& det);
  // 当前帧未匹配到检测时调用。
  void mark_missed();

  bool is_dead(int max_age) const;
  int id() const { return track_id_; }
  int miss_count() const { return miss_count_; }
  int age() const { return age_; }

  core::BBox predicted_bbox() const;
  core::TrackState to_state() const;

 private:
  int track_id_{-1};
  int miss_count_{0};
  int age_{0};
  bool matched_in_frame_{false};
  double last_det_conf_{0.0};
  core::BBox last_det_bbox_{};
  KalmanBBox kf_;
};

}  // namespace tracking
