// 程序入口文件：
// 负责解析启动参数、加载配置，并把控制权交给 TrackerNode。
#include <fstream>
#include <iostream>
#include <string>

#include "app/tracker_node.h"
#include "core/config.h"

namespace {

// 可选的启动阶段追踪日志，便于排查初始化失败位置。
void startup_trace(const std::string& msg) {
  const char* trace_path = std::getenv("TRACKER_STARTUP_TRACE");
  if (trace_path == nullptr || trace_path[0] == '\0') {
    return;
  }
  std::ofstream ofs(trace_path, std::ios::app);
  if (!ofs.is_open()) {
    return;
  }
  ofs << msg << '\n';
}

// 命令行里只区分 cam 和 replay 两种模式，其他情况默认回放模式。
app::TrackerNode::Mode parse_mode(const std::string& mode) {
  if (mode == "cam") {
    return app::TrackerNode::Mode::Cam;
  }
  return app::TrackerNode::Mode::Replay;
}

}  // namespace

int main(int argc, char** argv) {
  // 未显式指定时，默认采用回放模式和默认配置。
  std::string mode = "replay";
  std::string config_path = "config/tracker.yaml";
  std::string replay_csv = "data/detections.csv";
  uint64_t max_runtime_ms = 0;

  // 解析命令行参数，允许覆盖运行模式、配置路径和回放数据。
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      mode = argv[++i];
    } else if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--replay_csv" && i + 1 < argc) {
      replay_csv = argv[++i];
    } else if (arg == "--max_runtime_ms" && i + 1 < argc) {
      max_runtime_ms = static_cast<uint64_t>(std::stoull(argv[++i]));
    }
  }

  // 先加载配置，再构造总调度节点。
  startup_trace("main:argv_parsed mode=" + mode + " config=" + config_path + " replay=" + replay_csv);
  const auto cfg = core::ConfigLoader::load_or_default(config_path);
  startup_trace("main:config_loaded");
  app::TrackerNode node(cfg, parse_mode(mode), replay_csv, max_runtime_ms);
  startup_trace("main:node_constructed");

  // 初始化失败时直接退出，不进入运行阶段。
  if (!node.init()) {
    startup_trace("main:init_failed");
    std::cerr << "tracker init failed\n";
    return 1;
  }
  startup_trace("main:init_ok");

  // 初始化成功后，进入视觉线程 + 控制线程的主循环。
  node.wait_actuator_ready();
  node.init_pitch_angle();

  std::string input;
  std::cerr << "[startup] Y axis initialized near -25 deg. Enter y to start detection: ";
  while (std::getline(std::cin, input)) {
    if (input == "y" || input == "Y") break;
    std::cerr << "Enter y to start detection: ";
  }

  node.run();
  startup_trace("main:run_returned");
  return 0;
}
