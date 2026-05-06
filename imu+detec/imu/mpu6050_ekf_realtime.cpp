#define NOMINMAX
#include <windows.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kAccLsbPerG = 16384.0;
constexpr double kGyroLsbPerDps = 131.0;
constexpr int kPollIntervalMs = 5;
constexpr double kStaleDtFallback = 0.01;
constexpr double kQAngle = 2e-5;
constexpr double kQBias = 1e-6;
constexpr double kRAccAngle = (2.0 * 3.14159265358979323846 / 180.0) *
                              (2.0 * 3.14159265358979323846 / 180.0);

volatile bool g_stop = false;

BOOL WINAPI ConsoleHandler(DWORD ctrl_type) {
  if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT ||
      ctrl_type == CTRL_CLOSE_EVENT || ctrl_type == CTRL_SHUTDOWN_EVENT) {
    g_stop = true;
    return TRUE;
  }
  return FALSE;
}

fs::path resolve_data_dir(const char* argv0) {
  fs::path dir = fs::absolute(fs::path(argv0)).parent_path();
  const std::string leaf = dir.filename().string();
  if ((leaf == "Release" || leaf == "Debug" || leaf == "RelWithDebInfo" || leaf == "MinSizeRel") &&
      fs::exists(dir.parent_path() / "mpu6050_ekf_realtime.cpp")) {
    return dir.parent_path();
  }
  return dir;
}

std::string trim(const std::string& s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return {};
  }
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1);
}

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string token;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        token.push_back('"');
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
    } else if (c == ',' && !in_quotes) {
      out.push_back(token);
      token.clear();
    } else {
      token.push_back(c);
    }
  }
  out.push_back(token);
  return out;
}

struct RawSample {
  std::string pc_time;
  uint64_t pc_monotonic_ns{0};
  int mcu_ms{0};
  int ax{0};
  int ay{0};
  int az{0};
  int gx{0};
  int gy{0};
  int gz{0};
};

std::optional<RawSample> parse_csv_row(const std::string& line) {
  if (line.empty()) {
    return std::nullopt;
  }
  auto row = split_csv_line(line);
  if (row.empty()) {
    return std::nullopt;
  }

  std::string first = trim(row[0]);
  if (!first.empty() && static_cast<unsigned char>(first[0]) == 0xEF) {
    first.erase(0, 3);
  }
  if (first == "PC-time") {
    return std::nullopt;
  }

  try {
    RawSample sample;
    if (row.size() >= 11) {
      sample.pc_time = trim(row[0]);
      sample.pc_monotonic_ns = static_cast<uint64_t>(std::stoull(trim(row[1])));
      sample.mcu_ms = std::stoi(trim(row[2]));
      sample.ax = std::stoi(trim(row[3]));
      sample.ay = std::stoi(trim(row[4]));
      sample.az = std::stoi(trim(row[5]));
      sample.gx = std::stoi(trim(row[6]));
      sample.gy = std::stoi(trim(row[7]));
      sample.gz = std::stoi(trim(row[8]));
      return sample;
    }
    if (row.size() >= 8) {
      sample.pc_time = trim(row[0]);
      sample.pc_monotonic_ns = 0;
      sample.mcu_ms = std::stoi(trim(row[1]));
      sample.ax = std::stoi(trim(row[2]));
      sample.ay = std::stoi(trim(row[3]));
      sample.az = std::stoi(trim(row[4]));
      sample.gx = std::stoi(trim(row[5]));
      sample.gy = std::stoi(trim(row[6]));
      sample.gz = std::stoi(trim(row[7]));
      return sample;
    }
  } catch (...) {
    return std::nullopt;
  }

  return std::nullopt;
}

struct FusedSample {
  double roll_deg{0.0};
  double pitch_deg{0.0};
  double yaw_deg{0.0};
  double acc_roll_deg{0.0};
  double acc_pitch_deg{0.0};
  double gx_dps{0.0};
  double gy_dps{0.0};
  double gz_dps{0.0};
  double bgx_dps{0.0};
  double bgy_dps{0.0};
};

std::array<double, 2> accel_to_roll_pitch(double ax_g, double ay_g, double az_g) {
  const double roll = std::atan2(ay_g, az_g);
  const double pitch = std::atan2(-ax_g, std::sqrt(ay_g * ay_g + az_g * az_g));
  return {roll, pitch};
}

class RollPitchEkf {
 public:
  RollPitchEkf() {
    x_.fill(0.0);
    for (auto& row : P_) {
      row.fill(0.0);
    }
    for (int i = 0; i < 4; ++i) {
      P_[i][i] = 0.1;
    }
  }

  FusedSample step(const RawSample& sample) {
    const double ax_g = static_cast<double>(sample.ax) / kAccLsbPerG;
    const double ay_g = static_cast<double>(sample.ay) / kAccLsbPerG;
    const double az_g = static_cast<double>(sample.az) / kAccLsbPerG;

    const double gx_dps = static_cast<double>(sample.gx) / kGyroLsbPerDps;
    const double gy_dps = static_cast<double>(sample.gy) / kGyroLsbPerDps;
    const double gz_dps = static_cast<double>(sample.gz) / kGyroLsbPerDps;

    const double gx = gx_dps * kDegToRad;
    const double gy = gy_dps * kDegToRad;
    const double gz = gz_dps * kDegToRad;

    double dt = kStaleDtFallback;
    if (last_mcu_ms_ >= 0) {
      dt = static_cast<double>(sample.mcu_ms - last_mcu_ms_) / 1000.0;
      if (dt <= 0.0 || dt > 0.2) {
        dt = kStaleDtFallback;
      }
    }
    last_mcu_ms_ = sample.mcu_ms;

    const auto acc_angles = accel_to_roll_pitch(ax_g, ay_g, az_g);
    const double z_roll = acc_angles[0];
    const double z_pitch = acc_angles[1];

    x_[0] += (gx - x_[2]) * dt;
    x_[1] += (gy - x_[3]) * dt;
    yaw_ += gz * dt;

    Matrix4 F = {};
    for (int i = 0; i < 4; ++i) {
      F[i][i] = 1.0;
    }
    F[0][2] = -dt;
    F[1][3] = -dt;

    Matrix4 FP = mul(F, P_);
    Matrix4 Ft = transpose(F);
    P_ = add(mul(FP, Ft), diag4(kQAngle * std::max(dt, 1e-3),
                                kQAngle * std::max(dt, 1e-3),
                                kQBias * std::max(dt, 1e-3),
                                kQBias * std::max(dt, 1e-3)));

    const double y0 = z_roll - x_[0];
    const double y1 = z_pitch - x_[1];

    double S00 = P_[0][0] + kRAccAngle;
    double S01 = P_[0][1];
    double S10 = P_[1][0];
    double S11 = P_[1][1] + kRAccAngle;

    const double det = S00 * S11 - S01 * S10;
    if (std::abs(det) > 1e-12) {
      const double inv00 = S11 / det;
      const double inv01 = -S01 / det;
      const double inv10 = -S10 / det;
      const double inv11 = S00 / det;

      double K[4][2] = {};
      for (int r = 0; r < 4; ++r) {
        K[r][0] = P_[r][0] * inv00 + P_[r][1] * inv10;
        K[r][1] = P_[r][0] * inv01 + P_[r][1] * inv11;
      }

      for (int r = 0; r < 4; ++r) {
        x_[r] += K[r][0] * y0 + K[r][1] * y1;
      }

      Matrix4 KH = {};
      for (int r = 0; r < 4; ++r) {
        KH[r][0] = K[r][0];
        KH[r][1] = K[r][1];
      }
      Matrix4 I = {};
      for (int i = 0; i < 4; ++i) {
        I[i][i] = 1.0;
      }
      P_ = mul(sub(I, KH), P_);
    }

    FusedSample out;
    out.roll_deg = x_[0] * kRadToDeg;
    out.pitch_deg = x_[1] * kRadToDeg;
    out.yaw_deg = yaw_ * kRadToDeg;
    out.acc_roll_deg = z_roll * kRadToDeg;
    out.acc_pitch_deg = z_pitch * kRadToDeg;
    out.gx_dps = gx_dps;
    out.gy_dps = gy_dps;
    out.gz_dps = gz_dps;
    out.bgx_dps = x_[2] * kRadToDeg;
    out.bgy_dps = x_[3] * kRadToDeg;
    return out;
  }

 private:
  using Matrix4 = std::array<std::array<double, 4>, 4>;

  static constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
  static constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

  static Matrix4 add(const Matrix4& a, const Matrix4& b) {
    Matrix4 out = {};
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out[r][c] = a[r][c] + b[r][c];
      }
    }
    return out;
  }

  static Matrix4 sub(const Matrix4& a, const Matrix4& b) {
    Matrix4 out = {};
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out[r][c] = a[r][c] - b[r][c];
      }
    }
    return out;
  }

  static Matrix4 mul(const Matrix4& a, const Matrix4& b) {
    Matrix4 out = {};
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        double sum = 0.0;
        for (int k = 0; k < 4; ++k) {
          sum += a[r][k] * b[k][c];
        }
        out[r][c] = sum;
      }
    }
    return out;
  }

  static Matrix4 transpose(const Matrix4& a) {
    Matrix4 out = {};
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        out[c][r] = a[r][c];
      }
    }
    return out;
  }

  static Matrix4 diag4(double a0, double a1, double a2, double a3) {
    Matrix4 out = {};
    out[0][0] = a0;
    out[1][1] = a1;
    out[2][2] = a2;
    out[3][3] = a3;
    return out;
  }

  std::array<double, 4> x_{};
  Matrix4 P_{};
  int last_mcu_ms_{-1};
  double yaw_{0.0};
};

void write_csv_header(std::ofstream& ofs) {
  ofs << "PC-time,PC-monotonic(ns),MCU-time(ms),Roll(deg),Pitch(deg),Yaw(deg),"
         "AccRoll(deg),AccPitch(deg),GyroX(dps),GyroY(dps),GyroZ(dps),BiasGx(dps),BiasGy(dps)\n";
}

}  // namespace

int main(int argc, char** argv) {
  SetConsoleCtrlHandler(ConsoleHandler, TRUE);

  fs::path base_dir = resolve_data_dir(argv[0]);
  fs::path input_csv = base_dir / "mpu6050_data.csv";
  fs::path output_csv = base_dir / "mpu_ekf.csv";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) {
      input_csv = fs::path(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_csv = fs::path(argv[++i]);
    }
  }

  fs::create_directories(output_csv.parent_path());

  std::ofstream ofs(output_csv, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    std::cerr << "[ERROR] failed to open output csv: " << output_csv << '\n';
    return 1;
  }
  write_csv_header(ofs);
  ofs.flush();

  RollPitchEkf ekf;
  std::cout << "============================================================\n";
  std::cout << "MPU6050 realtime EKF (C++)\n";
  std::cout << "input  : " << input_csv.string() << '\n';
  std::cout << "output : " << output_csv.string() << '\n';
  std::cout << "Press Ctrl+C to stop.\n";
  std::cout << "============================================================\n";

  std::uintmax_t last_size = 0;
  std::string pending;
  uint64_t row_count = 0;

  while (!g_stop) {
    if (!fs::exists(input_csv)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      continue;
    }

    const auto current_size = fs::file_size(input_csv);
    if (current_size < last_size) {
      last_size = 0;
      pending.clear();
    }

    std::ifstream ifs(input_csv, std::ios::binary);
    if (!ifs.is_open()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      continue;
    }

    ifs.seekg(static_cast<std::streamoff>(last_size), std::ios::beg);
    std::string chunk((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    last_size = current_size;
    if (chunk.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      continue;
    }

    pending += chunk;
    size_t pos = 0;
    while (true) {
      const size_t nl = pending.find('\n', pos);
      if (nl == std::string::npos) {
        pending.erase(0, pos);
        break;
      }
      std::string line = pending.substr(pos, nl - pos);
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      pos = nl + 1;

      auto sample = parse_csv_row(line);
      if (!sample.has_value()) {
        continue;
      }

      const FusedSample fused = ekf.step(*sample);
      ofs << sample->pc_time << ','
          << sample->pc_monotonic_ns << ','
          << sample->mcu_ms << ','
          << std::fixed << std::setprecision(3)
          << fused.roll_deg << ','
          << fused.pitch_deg << ','
          << fused.yaw_deg << ','
          << fused.acc_roll_deg << ','
          << fused.acc_pitch_deg << ','
          << fused.gx_dps << ','
          << fused.gy_dps << ','
          << fused.gz_dps << ','
          << fused.bgx_dps << ','
          << fused.bgy_dps << '\n';
      ofs.flush();

      ++row_count;
      std::cout << '[' << std::setw(4) << std::setfill('0') << row_count << "] "
                << sample->pc_time
                << " | MCU:" << std::setw(6) << sample->mcu_ms << "ms"
                << " | Roll/Pitch/Yaw=("
                << std::fixed << std::setprecision(2)
                << fused.roll_deg << ", " << fused.pitch_deg << ", " << fused.yaw_deg
                << ")\n";
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
  }

  std::cout << "[INFO] total rows: " << row_count << '\n';
  std::cout << "[INFO] csv saved : " << output_csv.string() << '\n';
  return 0;
}
