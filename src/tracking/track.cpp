#include "tracking/track.h"

namespace tracking {

Track::Track(int track_id, const core::Detection& det)
    : track_id_(track_id), last_det_conf_(det.conf), last_det_bbox_(det.bbox) {
  // 新轨迹由一次高置信度检测初始化，内部状态交给 KalmanBBox 管理。
  kf_.init(det);
  age_ = 1;
}

void Track::predict(double dt) {
  // 每到新的一帧，先把轨迹按运动模型向前推一步，
  // 后续再拿预测框去和当前帧检测结果做关联。
  kf_.predict(dt);
  matched_in_frame_ = false;
  ++age_;
}

void Track::update(const core::Detection& det) {
  // 检测成功匹配后，用测量值修正轨迹状态并清空丢失计数。
  kf_.update(det);
  miss_count_ = 0;
  matched_in_frame_ = true;
  last_det_conf_ = det.conf;
  last_det_bbox_ = det.bbox;
}

void Track::mark_missed() {
  // 这一帧没有匹配到任何检测，只累计 miss_count，
  // 是否真正删除轨迹由 MultiObjectTracker 根据 max_age 决定。
  ++miss_count_;
  last_det_conf_ = 0.0;
}

bool Track::is_dead(int max_age) const { return miss_count_ > max_age; }

core::BBox Track::predicted_bbox() const { return kf_.predicted_bbox(); }

core::TrackState Track::to_state() const {
  // 对外导出统一的轨迹状态，供目标选择、日志和控制模块使用。
  const auto x = kf_.state();
  core::TrackState s;
  s.track_id = track_id_;
  s.det_bbox = last_det_bbox_;
  s.det_conf = last_det_conf_;
  s.cx = x[0];
  s.cy = x[1];
  s.w = x[2];
  s.h = x[3];
  s.vx = x[4];
  s.vy = x[5];
  s.vw = x[6];
  s.vh = x[7];
  s.bbox_hat = kf_.predicted_bbox();
  s.miss_count = miss_count_;
  s.age = age_;
  s.matched_in_frame = matched_in_frame_;
  return s;
}

}  // namespace tracking
