#pragma once

#include <chrono>
#include <cstdint>

namespace core::time {

// 单调时钟时间戳，单位纳秒。
uint64_t now_ns();
// 睡眠到指定时刻，主要用于固定频率控制循环。
void sleep_until(std::chrono::steady_clock::time_point tp);

}  // namespace core::time
