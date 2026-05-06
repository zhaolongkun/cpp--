cmake_minimum_required(VERSION 3.16)

project(imu_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep the executables next to the source files so the generated CSV files
# are written to D:/.../imu+detec/imu as requested.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_SOURCE_DIR})

add_executable(serial_logger serial_logger.cpp)
add_executable(mpu6050_ekf_realtime mpu6050_ekf_realtime.cpp)

if (MSVC)
  target_compile_options(serial_logger PRIVATE /W4 /utf-8)
  target_compile_options(mpu6050_ekf_realtime PRIVATE /W4 /utf-8)
else()
  target_compile_options(serial_logger PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(mpu6050_ekf_realtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
