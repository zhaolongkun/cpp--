#include "tracking/multi_object_tracker.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <iostream>
#include <numeric>

#include "tracking/hungarian.h"
#include "vision/bbox_utils.h"

namespace tracking {
namespace {

struct MatchOutput {
  std::vector<std::pair<int, int>> matches;  // (track_index, det_index)
  std::vector<int> unmatched_tracks;
  std::vector<int> unmatched_dets;
};

std::string to_lower_copy(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

double center_dist_ratio(const core::BBox& a, const core::BBox& b) {
  // 中心距离按目标尺度归一化，避免“大目标天然距离更大”带来的偏差。
  const double dx = a.cx() - b.cx();
  const double dy = a.cy() - b.cy();
  const double dist = std::sqrt(dx * dx + dy * dy);
  const double da = std::hypot(std::max(1.0, a.width()), std::max(1.0, a.height()));
  const double db = std::hypot(std::max(1.0, b.width()), std::max(1.0, b.height()));
  const double norm = std::max(1.0, 0.5 * (da + db));
  return dist / norm;
}

std::vector<int> all_track_indices(int n) {
  std::vector<int> idx(std::max(0, n));
  std::iota(idx.begin(), idx.end(), 0);
  return idx;
}

MatchOutput match_with_cost(const std::vector<Track>& tracks, const std::vector<int>& track_ids,
                            const std::vector<core::Detection>& dets, const std::vector<int>& det_ids, double iou_gate,
                            double max_cost, bool use_botsort, double botsort_iou_weight,
                            double botsort_center_weight, double botsort_center_gate) {
  MatchOutput out;
  out.unmatched_tracks = track_ids;
  out.unmatched_dets = det_ids;

  if (track_ids.empty() || det_ids.empty()) {
    return out;
  }

  std::vector<std::vector<double>> cost(track_ids.size(), std::vector<double>(det_ids.size(), 1e6));

  for (size_t r = 0; r < track_ids.size(); ++r) {
    const auto pb = tracks[track_ids[r]].predicted_bbox();
    for (size_t c = 0; c < det_ids.size(); ++c) {
      const auto& db = dets[det_ids[c]].bbox;
      const double iou = vision::iou(pb, db);
      if (iou < iou_gate) {
        continue;
      }

      if (!use_botsort) {
        // ByteTrack 代价只基于 IoU，重叠越大代价越小。
        cost[r][c] = 1.0 - iou;
        continue;
      }

      const double cdr = center_dist_ratio(pb, db);
      if (cdr > botsort_center_gate) {
        continue;
      }

      // BoT-SORT 风格代价：IoU 项 + 中心距离项联合决定匹配质量。
      cost[r][c] = botsort_iou_weight * (1.0 - iou) + botsort_center_weight * cdr;
    }
  }

  const auto assign = Hungarian::solve(cost, max_cost);

  out.matches.clear();
  out.matches.reserve(assign.matches.size());
  for (const auto& [r, c] : assign.matches) {
    out.matches.emplace_back(track_ids[r], det_ids[c]);
  }

  out.unmatched_tracks.clear();
  out.unmatched_tracks.reserve(assign.unmatched_rows.size());
  for (int r : assign.unmatched_rows) {
    out.unmatched_tracks.push_back(track_ids[r]);
  }

  out.unmatched_dets.clear();
  out.unmatched_dets.reserve(assign.unmatched_cols.size());
  for (int c : assign.unmatched_cols) {
    out.unmatched_dets.push_back(det_ids[c]);
  }

  return out;
}

}  // namespace

MultiObjectTracker::MultiObjectTracker(const core::MotConfig& cfg)
    : tracker_type_(to_lower_copy(cfg.tracker_type)),
      max_age_(cfg.max_age),
      iou_gate_(cfg.iou_gate),
      track_high_thresh_(cfg.track_high_thresh),
      track_low_thresh_(cfg.track_low_thresh),
      new_track_thresh_(cfg.new_track_thresh),
      second_match_iou_(cfg.second_match_iou),
      botsort_iou_weight_(cfg.botsort_iou_weight),
      botsort_center_weight_(cfg.botsort_center_weight),
      botsort_max_center_dist_ratio_(cfg.botsort_max_center_dist_ratio) {
  if (tracker_type_ != "bytetrack" && tracker_type_ != "botsort") {
    tracker_type_ = cfg.use_bytetrack ? "bytetrack" : "botsort";
  }
  if (track_low_thresh_ > track_high_thresh_) {
    std::swap(track_low_thresh_, track_high_thresh_);
  }
  if (new_track_thresh_ < track_high_thresh_) {
    new_track_thresh_ = track_high_thresh_;
  }

  std::cerr << "[mot] tracker_type=" << tracker_type_ << " high=" << track_high_thresh_ << " low=" << track_low_thresh_
            << " new=" << new_track_thresh_ << '\n';
}

void MultiObjectTracker::update(const std::vector<core::Detection>& dets, double dt_sec) {
  // 先把所有现存轨迹预测到当前帧时刻。
  for (auto& t : tracks_) {
    t.predict(dt_sec);
  }

  std::vector<int> high_dets;
  std::vector<int> low_dets;
  high_dets.reserve(dets.size());
  low_dets.reserve(dets.size());

  for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
    const double conf = dets[i].conf;
    if (conf >= track_high_thresh_) {
      high_dets.push_back(i);
    } else if (conf >= track_low_thresh_) {
      low_dets.push_back(i);
    }
  }

  // 第一阶段：高置信度检测与全部轨迹匹配。
  // 这是 ByteTrack 的主关联阶段，优先保证“强检测”先吃掉已有轨迹。
  const std::vector<int> stage1_tracks = all_track_indices(static_cast<int>(tracks_.size()));
  const bool use_botsort_cost = (tracker_type_ == "botsort");
  const double stage1_max_cost =
      use_botsort_cost ? (botsort_iou_weight_ * (1.0 - iou_gate_) + botsort_center_weight_ * botsort_max_center_dist_ratio_)
                       : (1.0 - iou_gate_);

  const auto stage1 = match_with_cost(tracks_, stage1_tracks, dets, high_dets, iou_gate_, stage1_max_cost,
                                      use_botsort_cost, botsort_iou_weight_, botsort_center_weight_,
                                      botsort_max_center_dist_ratio_);

  for (const auto& [ti, di] : stage1.matches) {
    tracks_[ti].update(dets[di]);
  }

  // 第二阶段：剩余未匹配轨迹再尝试与低置信度检测关联，
  // 用来挽救一些分数略低但时序上连续的目标。
  const auto stage2 =
      match_with_cost(tracks_, stage1.unmatched_tracks, dets, low_dets, second_match_iou_, 1.0 - second_match_iou_,
                      false, 1.0, 0.0, 1.0);

  for (const auto& [ti, di] : stage2.matches) {
    tracks_[ti].update(dets[di]);
  }

  for (int ti : stage2.unmatched_tracks) {
    tracks_[ti].mark_missed();
  }

  // 只有第一阶段剩下的高分检测才允许新建轨迹，
  // 这样可以避免低分噪声检测在画面里大量生灭。
  for (int di : stage1.unmatched_dets) {
    if (dets[di].conf >= new_track_thresh_) {
      tracks_.emplace_back(next_track_id_++, dets[di]);
    }
  }

  // 长时间未匹配的轨迹会被清理掉，避免轨迹集合无限增长。
  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [this](const Track& t) { return t.is_dead(max_age_); }),
                tracks_.end());
}

std::vector<core::TrackState> MultiObjectTracker::get_tracks() const {
  // 导出当前所有有效轨迹的快照。
  std::vector<core::TrackState> out;
  out.reserve(tracks_.size());
  for (const auto& t : tracks_) {
    out.push_back(t.to_state());
  }
  return out;
}

}  // namespace tracking
