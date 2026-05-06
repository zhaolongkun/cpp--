#include "core/logger.h"

#include <filesystem>
#include <iomanip>
#include <string>

namespace core {

namespace {
std::string sanitize_csv_field(std::string s) {
  // CSV 字段里如果直接出现逗号或换行，会破坏列结构，
  // 这里统一替换成分号，保证日志可以直接被 pandas / Excel 读取。
  for (char& ch : s) {
    if (ch == ',' || ch == '\n' || ch == '\r') {
      ch = ';';
    }
  }
  return s;
}
}  // namespace

CsvLogger::~CsvLogger() { close(); }

bool CsvLogger::open(const std::string& path, bool dedup_by_frame_id, const std::string& profile) {
  std::scoped_lock lock(mu_);
  try {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
    ofs_.open(path, std::ios::out | std::ios::trunc);
    header_written_ = false;
    if (ofs_.is_open()) {
      // 打开 unitbuf 后，每次写入都会自动刷新，
      // 这样一边运行系统一边 tail CSV 时可以实时看到最新记录。
      ofs_ << std::unitbuf;
    }
    dedup_by_frame_id_ = dedup_by_frame_id;
    profile_ = profile;
    has_pending_snapshot_ = false;
    pending_snapshot_ = RuntimeSnapshot{};
    last_written_timestamp_ms_ = 0;
    return ofs_.is_open();
  } catch (...) {
    return false;
  }
}

void CsvLogger::write_row(const RuntimeSnapshot& s) {
  if (!ofs_.is_open()) {
    return;
  }
  if (!header_written_) {
    // 根据日志模式选择表头：
    // brief 适合实时调试，只保留控制链关键字段；
    // full 适合离线分析，记录检测、跟踪、控制、执行器的完整状态。
    if (profile_ == "brief") {
      ofs_ << "run_id,timestamp_ms,frame_id,dt_ms,det_count,track_count,controlled_id,det_conf,"
              "dx_hat,dy_hat,clean_dx,clean_dy,stage1_switch_gate,lost_flag,cmd_base_x,cmd_base_y,alpha_gate,delta_cmd_x,delta_cmd_y,"
              "cmd_sent_x,cmd_sent_y,infer_used_model,fallback_delta_zero,infer_status,note\n";
    } else {
      ofs_ << "run_id,timestamp_ms,frame_id,dt_ms,img_w,img_h,bbox_x1,bbox_y1,bbox_x2,bbox_y2,"
              "bbox_raw_x1,bbox_raw_y1,bbox_raw_x2,bbox_raw_y2,bbox_area_px,det_conf,"
              "dx_raw,dy_raw,dx_hat,dy_hat,clean_dx,clean_dy,vx_hat,vy_hat,lost_flag,is_meas_update,meas_age_ms,"
              "zoom_value,zoom_delta,vision_latency_ms,cmd_base_x,cmd_base_y,"
              "cmd_expert_x,cmd_expert_y,cmd_sent_x,cmd_sent_y,reliability_score,alpha_gate,stage1_switch_gate,delta_cmd_x,delta_cmd_y,"
              "residual_clip_flag,slew_limit_flag,final_sat_flag,infer_used_model,fallback_delta_zero,infer_status,"
              "act_pos_x,act_pos_y,act_vel_x,act_vel_y,"
              "det_count,track_count,controlled_id,coast_count,note\n";
    }
    header_written_ = true;
  }

  ofs_ << s.run_id << ',' << s.timestamp_ms << ',' << s.frame_id << ',' << std::fixed << std::setprecision(6)
       << s.dt_ms;
  if (profile_ == "brief") {
    ofs_ << ',' << s.det_count << ',' << s.track_count << ',' << s.controlled_id << ',' << s.det_conf << ','
         << s.dx_hat << ',' << s.dy_hat << ',' << s.clean_dx << ',' << s.clean_dy << ',' << s.stage1_switch_gate << ','
         << s.lost_flag << ',' << s.cmd_base_x << ',' << s.cmd_base_y << ',' << s.alpha_gate << ',' << s.delta_cmd_x << ',' << s.delta_cmd_y << ',' << s.cmd_sent_x << ','
         << s.cmd_sent_y << ',' << s.infer_used_model << ',' << s.fallback_delta_zero << ','
         << sanitize_csv_field(s.infer_status) << ',' << sanitize_csv_field(s.note) << '\n';
  } else {
    ofs_ << ',' << s.img_w << ',' << s.img_h << ',' << s.bbox.x1 << ',' << s.bbox.y1 << ',' << s.bbox.x2 << ','
         << s.bbox.y2 << ',' << s.bbox_raw.x1 << ',' << s.bbox_raw.y1 << ',' << s.bbox_raw.x2 << ','
         << s.bbox_raw.y2 << ',' << s.bbox_area_px << ',' << s.det_conf << ',' << s.dx_raw << ',' << s.dy_raw
         << ',' << s.dx_hat << ',' << s.dy_hat << ',' << s.clean_dx << ',' << s.clean_dy << ',' << s.vx_hat << ',' << s.vy_hat << ',' << s.lost_flag << ','
         << s.is_meas_update << ',' << s.meas_age_ms << ',' << s.zoom_value << ',' << s.zoom_delta << ','
         << s.vision_latency_ms << ',' << s.cmd_base_x << ',' << s.cmd_base_y << ',' << s.cmd_expert_x << ','
         << s.cmd_expert_y << ',' << s.cmd_sent_x << ',' << s.cmd_sent_y << ',' << s.reliability_score << ','
         << s.alpha_gate << ',' << s.stage1_switch_gate << ',' << s.delta_cmd_x << ',' << s.delta_cmd_y << ',' << s.residual_clip_flag << ','
         << s.slew_limit_flag << ',' << s.final_sat_flag << ',' << s.infer_used_model << ',' << s.fallback_delta_zero
         << ',' << sanitize_csv_field(s.infer_status) << ',' << s.act_pos_x << ',' << s.act_pos_y << ','
         << s.act_vel_x << ',' << s.act_vel_y << ',' << s.det_count << ',' << s.track_count << ','
         << s.controlled_id << ',' << s.coast_count << ',' << sanitize_csv_field(s.note) << '\n';
  }
  ofs_.flush();
  last_written_timestamp_ms_ = s.timestamp_ms;
}

void CsvLogger::flush_pending_locked() {
  if (!has_pending_snapshot_) {
    return;
  }
  // 去重模式下，一个 frame_id 可能被多次更新：
  // 例如先有检测结果，后面又补充执行器状态。
  // 这里在真正落盘前只保留该帧最后一份快照。
  RuntimeSnapshot out = pending_snapshot_;
  if (last_written_timestamp_ms_ > 0 && out.timestamp_ms >= last_written_timestamp_ms_) {
    out.dt_ms = static_cast<double>(out.timestamp_ms - last_written_timestamp_ms_);
  }
  write_row(out);
  has_pending_snapshot_ = false;
}

void CsvLogger::write(const RuntimeSnapshot& s) {
  std::scoped_lock lock(mu_);
  if (!dedup_by_frame_id_) {
    write_row(s);
    return;
  }
  // 开启按帧去重时，不立刻写盘，而是先暂存，
  // 等确认进入下一帧后再把上一帧的最终版本写出。
  if (!has_pending_snapshot_) {
    pending_snapshot_ = s;
    has_pending_snapshot_ = true;
    return;
  }
  if (s.frame_id == pending_snapshot_.frame_id) {
    pending_snapshot_ = s;
    return;
  }
  flush_pending_locked();
  pending_snapshot_ = s;
  has_pending_snapshot_ = true;
}

void CsvLogger::close() {
  std::scoped_lock lock(mu_);
  if (ofs_.is_open()) {
    // 关闭前把最后一帧尚未落盘的缓存刷出去，避免尾帧丢失。
    flush_pending_locked();
    ofs_.flush();
    ofs_.close();
  }
}

}  // namespace core
