#pragma once

#include <fstream>
#include <mutex>
#include <string>

#include "core/types.h"

namespace core {

class CsvLogger {
 public:
  CsvLogger() = default;
  ~CsvLogger();

  // 打开日志文件。可选按 frame_id 去重，避免同一帧被多次写出。
  bool open(const std::string& path, bool dedup_by_frame_id = true, const std::string& profile = "full");
  // 写入一份运行时快照。
  void write(const RuntimeSnapshot& snapshot);
  void close();

 private:
  // 真正执行单行写盘。
  void write_row(const RuntimeSnapshot& snapshot);
  // 去重模式下，把缓存中的最后一份快照刷到磁盘。
  void flush_pending_locked();

  std::mutex mu_;
  std::ofstream ofs_;
  bool header_written_{false};
  bool dedup_by_frame_id_{true};
  std::string profile_{"full"};
  bool has_pending_snapshot_{false};
  RuntimeSnapshot pending_snapshot_{};
  uint64_t last_written_timestamp_ms_{0};
};

}  // namespace core
