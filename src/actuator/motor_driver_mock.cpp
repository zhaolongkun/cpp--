#include "actuator/motor_driver_mock.h"

namespace actuator {

void MotorDriverMock::send(const core::MotorCmd& cmd) {
  (void)cmd;
  // mock 模式下只吞掉控制命令，不实际驱动硬件。
  // 这样可以在没有真实电机的情况下完成整条控制链联调，
  // 所有发出的命令仍然会体现在日志中，便于离线分析。
}

}  // namespace actuator
