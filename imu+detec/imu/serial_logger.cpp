#include <windows.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* kDefaultPort = "COM4";
constexpr int kDefaultBaud = 115200;
constexpr int kIdleSleepMs = 5;

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
      fs::exists(dir.parent_path() / "serial_logger.cpp")) {
    return dir.parent_path();
  }
  return dir;
}

uint64_t monotonic_ns() {
  static LARGE_INTEGER freq = {};
  static bool init = QueryPerformanceFrequency(&freq) != 0;
  LARGE_INTEGER now = {};
  QueryPerformanceCounter(&now);
  if (!init || freq.QuadPart <= 0) {
    return 0;
  }
  return static_cast<uint64_t>((1000000000ULL * static_cast<uint64_t>(now.QuadPart)) /
                               static_cast<uint64_t>(freq.QuadPart));
}

std::string wall_clock_string() {
  SYSTEMTIME st = {};
  GetLocalTime(&st);
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(4) << st.wYear << '-'
      << std::setw(2) << st.wMonth << '-'
      << std::setw(2) << st.wDay << ' '
      << std::setw(2) << st.wHour << ':'
      << std::setw(2) << st.wMinute << ':'
      << std::setw(2) << st.wSecond << '.'
      << std::setw(3) << st.wMilliseconds;
  return oss.str();
}

std::string csv_escape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"') {
      escaped += "\"\"";
    } else {
      escaped.push_back(c);
    }
  }
  escaped.push_back('"');
  return escaped;
}

struct ParsedLine {
  std::string mcu_time_ms;
  std::string acc_x;
  std::string acc_y;
  std::string acc_z;
  std::string gyro_x;
  std::string gyro_y;
  std::string gyro_z;
  std::string temp;
  std::string raw_hex;
};

std::optional<ParsedLine> parse_line(const std::string& line) {
  static const std::regex mcu_time_re(R"(\[(\d+)\s*ms\])");
  static const std::regex hex_re(R"(\]\s+([0-9A-F ]+)\s+\|)");
  static const std::regex acc_re(R"(Acc:(-?\d+),(-?\d+),(-?\d+))");
  static const std::regex gyro_re(R"(Gyro:(-?\d+),(-?\d+),(-?\d+))");
  static const std::regex temp_re(R"(T:(-?\d+))");

  if (line.find('[') == std::string::npos || line.find("ms]") == std::string::npos) {
    return std::nullopt;
  }

  std::smatch mcu_match;
  if (!std::regex_search(line, mcu_match, mcu_time_re)) {
    return std::nullopt;
  }

  std::smatch acc_match;
  std::smatch gyro_match;
  std::smatch temp_match;
  std::smatch hex_match;
  std::regex_search(line, acc_match, acc_re);
  std::regex_search(line, gyro_match, gyro_re);
  std::regex_search(line, temp_match, temp_re);
  std::regex_search(line, hex_match, hex_re);

  ParsedLine out;
  out.mcu_time_ms = mcu_match[1].str();
  out.acc_x = acc_match.empty() ? "" : acc_match[1].str();
  out.acc_y = acc_match.empty() ? "" : acc_match[2].str();
  out.acc_z = acc_match.empty() ? "" : acc_match[3].str();
  out.gyro_x = gyro_match.empty() ? "" : gyro_match[1].str();
  out.gyro_y = gyro_match.empty() ? "" : gyro_match[2].str();
  out.gyro_z = gyro_match.empty() ? "" : gyro_match[3].str();
  out.temp = temp_match.empty() ? "" : temp_match[1].str();
  out.raw_hex = hex_match.empty() ? "" : hex_match[1].str();
  return out;
}

std::wstring make_com_device(const std::string& port) {
  std::wstring wide_port(port.begin(), port.end());
  return L"\\\\.\\" + wide_port;
}

class SerialPort {
 public:
  ~SerialPort() { close(); }

  bool open(const std::string& port, int baud_rate) {
    close();
    const std::wstring device = make_com_device(port);
    handle_ = CreateFileW(device.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                          OPEN_EXISTING, 0, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
      handle_ = INVALID_HANDLE_VALUE;
      return false;
    }

    DCB dcb = {};
    dcb.DCBlength = sizeof(DCB);
    if (!GetCommState(handle_, &dcb)) {
      close();
      return false;
    }

    dcb.BaudRate = static_cast<DWORD>(baud_rate);
    dcb.ByteSize = 8;
    dcb.Parity = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary = TRUE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    if (!SetCommState(handle_, &dcb)) {
      close();
      return false;
    }

    COMMTIMEOUTS timeouts = {};
    timeouts.ReadIntervalTimeout = 20;
    timeouts.ReadTotalTimeoutConstant = 20;
    timeouts.ReadTotalTimeoutMultiplier = 2;
    timeouts.WriteTotalTimeoutConstant = 20;
    timeouts.WriteTotalTimeoutMultiplier = 2;
    if (!SetCommTimeouts(handle_, &timeouts)) {
      close();
      return false;
    }

    SetupComm(handle_, 4096, 4096);
    PurgeComm(handle_, PURGE_RXCLEAR | PURGE_TXCLEAR);
    return true;
  }

  void close() {
    if (handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
    }
  }

  bool is_open() const { return handle_ != INVALID_HANDLE_VALUE; }

  bool read_bytes(char* buffer, DWORD buffer_size, DWORD& bytes_read) {
    if (!is_open()) {
      bytes_read = 0;
      return false;
    }
    if (!ReadFile(handle_, buffer, buffer_size, &bytes_read, nullptr)) {
      bytes_read = 0;
      return false;
    }
    return true;
  }

 private:
  HANDLE handle_{INVALID_HANDLE_VALUE};
};

void write_csv_header(std::ofstream& ofs) {
  ofs << "PC-time,PC-monotonic(ns),MCU-time(ms),AccX,AccY,AccZ,GyroX,GyroY,GyroZ,Temp,Raw data\n";
}

}  // namespace

int main(int argc, char** argv) {
  SetConsoleCtrlHandler(ConsoleHandler, TRUE);

  std::string port = kDefaultPort;
  int baud_rate = kDefaultBaud;
  fs::path base_dir = resolve_data_dir(argv[0]);
  fs::path output_csv = base_dir / "mpu6050_data.csv";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--port" && i + 1 < argc) {
      port = argv[++i];
    } else if (arg == "--baud" && i + 1 < argc) {
      baud_rate = std::stoi(argv[++i]);
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

  SerialPort serial;
  if (!serial.open(port, baud_rate)) {
    std::cerr << "[ERROR] failed to open serial port " << port << '\n';
    return 1;
  }

  std::cout << "============================================================\n";
  std::cout << "MPU6050 serial logger (C++)\n";
  std::cout << "port   : " << port << '\n';
  std::cout << "baud   : " << baud_rate << '\n';
  std::cout << "output : " << output_csv.string() << '\n';
  std::cout << "Press Ctrl+C to stop.\n";
  std::cout << "============================================================\n";

  std::string line_buffer;
  uint64_t line_count = 0;
  char buf[512];

  while (!g_stop) {
    DWORD bytes_read = 0;
    if (!serial.read_bytes(buf, static_cast<DWORD>(sizeof(buf)), bytes_read)) {
      std::cerr << "[ERROR] serial read failed\n";
      break;
    }

    if (bytes_read == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kIdleSleepMs));
      continue;
    }

    for (DWORD i = 0; i < bytes_read; ++i) {
      const char c = buf[i];
      if (c == '\r') {
        continue;
      }
      if (c != '\n') {
        line_buffer.push_back(c);
        continue;
      }

      const std::string raw_line = line_buffer;
      line_buffer.clear();

      auto parsed = parse_line(raw_line);
      if (!parsed.has_value()) {
        if (!raw_line.empty()) {
          std::cout << "[INFO] " << raw_line << '\n';
        }
        continue;
      }

      const std::string pc_time = wall_clock_string();
      const uint64_t pc_monotonic = monotonic_ns();

      ofs << csv_escape(pc_time) << ','
          << pc_monotonic << ','
          << parsed->mcu_time_ms << ','
          << parsed->acc_x << ','
          << parsed->acc_y << ','
          << parsed->acc_z << ','
          << parsed->gyro_x << ','
          << parsed->gyro_y << ','
          << parsed->gyro_z << ','
          << parsed->temp << ','
          << csv_escape(parsed->raw_hex) << '\n';
      ofs.flush();

      ++line_count;
      std::cout << '[' << std::setw(4) << std::setfill('0') << line_count << "] "
                << pc_time << " | Mono:" << pc_monotonic
                << " | MCU:" << std::setw(6) << parsed->mcu_time_ms << "ms"
                << " | Acc:(" << std::setw(6) << parsed->acc_x << ','
                << std::setw(6) << parsed->acc_y << ','
                << std::setw(6) << parsed->acc_z << ')'
                << " | Gyro:(" << std::setw(6) << parsed->gyro_x << ','
                << std::setw(6) << parsed->gyro_y << ','
                << std::setw(6) << parsed->gyro_z << ')'
                << " | T:" << std::setw(6) << parsed->temp << '\n';
    }
  }

  serial.close();
  std::cout << "[INFO] total rows: " << line_count << '\n';
  std::cout << "[INFO] csv saved : " << output_csv.string() << '\n';
  return 0;
}
