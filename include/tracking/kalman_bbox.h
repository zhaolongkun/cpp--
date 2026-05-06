#pragma once

#include <array>

#include "core/types.h"

namespace tracking {

// 轻量级框状态预测器。
// 状态向量为 [cx, cy, w, h, vx, vy, vw, vh]。
class KalmanBBox {
 public:
  KalmanBBox();

  void init(const core::Detection& det);
  void predict(double dt);
  void update(const core::Detection& det);

  bool initialized() const { return initialized_; }

  std::array<double, 8> state() const { return x_; }
  core::BBox predicted_bbox() const;

 private:
  bool initialized_{false};
  std::array<double, 8> x_{};
  double last_dt_{1.0 / 30.0};
};

}  // namespace tracking
