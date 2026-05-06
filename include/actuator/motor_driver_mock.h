#pragma once

#include <iostream>

#include "actuator/motor_driver.h"

namespace actuator {

// 纯软件假驱动，用于无硬件联调和日志验证。
class MotorDriverMock final : public MotorDriver {
 public:
  void send(const core::MotorCmd& cmd) override;
};

}  // namespace actuator
