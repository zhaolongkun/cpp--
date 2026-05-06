#pragma once

#include "core/types.h"

namespace actuator {

// 执行器驱动抽象接口。
class MotorDriver {
 public:
  virtual ~MotorDriver() = default;
  virtual void send(const core::MotorCmd& cmd) = 0;
  virtual void close() {}
  virtual bool takeover_done() const { return true; }
  virtual double get_pitch_rate() const { return 0.0; }
  virtual double get_pitch() const { return 0.0; }
  virtual bool get_pitch_valid() const { return false; }
};

}  // namespace actuator
