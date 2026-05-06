#include "tracking/kalman_bbox.h"

#include <algorithm>

#include "vision/bbox_utils.h"

namespace tracking {

KalmanBBox::KalmanBBox() { x_.fill(0.0); }

void KalmanBBox::init(const core::Detection& det) {
  // 状态向量定义：
  // [cx, cy, w, h, vx, vy, vw, vh]
  // 前四维是框中心和尺度，后四维是它们的一阶变化率。
  x_[0] = det.bbox.cx();
  x_[1] = det.bbox.cy();
  x_[2] = det.bbox.width();
  x_[3] = det.bbox.height();
  x_[4] = 0.0;
  x_[5] = 0.0;
  x_[6] = 0.0;
  x_[7] = 0.0;
  last_dt_ = 1.0 / 30.0;
  initialized_ = true;
}

void KalmanBBox::predict(double dt) {
  if (!initialized_) {
    return;
  }
  // 这里不是完整矩阵版 KF，而是工程上更轻量的常速度外推。
  // 目标是为关联阶段提供一个“下一帧大概在哪”的预测框。
  last_dt_ = std::max(1e-3, dt);

  x_[0] += x_[4] * last_dt_;
  x_[1] += x_[5] * last_dt_;
  x_[2] = std::max(1.0, x_[2] + x_[6] * last_dt_);
  x_[3] = std::max(1.0, x_[3] + x_[7] * last_dt_);
}

void KalmanBBox::update(const core::Detection& det) {
  if (!initialized_) {
    init(det);
    return;
  }

  // 用当前检测框作为测量值，按指数校正的方式更新位置、尺寸和速度。
  // alpha 控制位置跟随强度，beta 控制速度修正强度。
  const double z_cx = det.bbox.cx();
  const double z_cy = det.bbox.cy();
  const double z_w = det.bbox.width();
  const double z_h = det.bbox.height();

  const double alpha = 0.7;
  const double beta = 0.2;

  const double r_cx = z_cx - x_[0];
  const double r_cy = z_cy - x_[1];
  const double r_w = z_w - x_[2];
  const double r_h = z_h - x_[3];

  x_[0] += alpha * r_cx;
  x_[1] += alpha * r_cy;
  x_[2] = std::max(1.0, x_[2] + alpha * r_w);
  x_[3] = std::max(1.0, x_[3] + alpha * r_h);

  x_[4] += beta * r_cx / last_dt_;
  x_[5] += beta * r_cy / last_dt_;
  x_[6] += beta * r_w / last_dt_;
  x_[7] += beta * r_h / last_dt_;
}

core::BBox KalmanBBox::predicted_bbox() const { return vision::cxcywh_to_bbox(x_[0], x_[1], x_[2], x_[3]); }

}  // namespace tracking
