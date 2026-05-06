#include "core/time_utils.h"

#include <thread>

namespace core::time {

uint64_t now_ns() {
  // 统一使用 steady_clock，避免系统时间被校时后出现倒跳。
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

// 控制线程会按固定周期运行，这里直接封装 sleep_until，
// 便于以后替换成平台相关实现。
void sleep_until(std::chrono::steady_clock::time_point tp) { std::this_thread::sleep_until(tp); }

}  // namespace core::time
