cmake_minimum_required(VERSION 3.16)
project(smart_gimbal_tracker_cpp VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(ENABLE_INSTALL "Enable install target" ON)
option(USE_SYSTEM_OPENCV "Use system OpenCV if available" OFF)
option(USE_SYSTEM_YAML_CPP "Use system yaml-cpp if available" OFF)
option(USE_SYSTEM_EIGEN3 "Use system Eigen3 if available" OFF)
option(USE_ONNXRUNTIME "Enable ONNX Runtime for temporal compensation model inference" OFF)
option(MINGW_STATIC_RUNTIME "Link MinGW runtime statically for standalone executable" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(USE_SYSTEM_OPENCV)
  find_package(OpenCV QUIET)
endif()

if(USE_SYSTEM_EIGEN3)
  find_package(Eigen3 QUIET)
endif()

if(USE_SYSTEM_YAML_CPP)
  find_package(yaml-cpp QUIET)
endif()

set(TRACKER_SOURCES
  src/main.cpp
  src/core/time_utils.cpp
  src/core/logger.cpp
  src/core/config.cpp
  src/vision/camera_capture.cpp
  src/vision/yolo_detector.cpp
  src/vision/bbox_utils.cpp
  src/vision/shm_detection_reader.cpp
  src/tracking/kalman_bbox.cpp
  src/tracking/track.cpp
  src/tracking/hungarian.cpp
  src/tracking/multi_object_tracker.cpp
  src/control/pid.cpp
  src/control/slew_limiter.cpp
  src/control/target_selector.cpp
  src/control/temporal_compensator_onnx.cpp
  src/control/controller.cpp
  src/filter/pnr_imm_kf.cpp
  src/actuator/motor_driver.cpp
  src/actuator/motor_driver_mock.cpp
  src/actuator/motor_driver_pyusb.cpp
  src/app/tracker_node.cpp
)

add_executable(tracker ${TRACKER_SOURCES})

target_include_directories(tracker
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(tracker
  PRIVATE
)

if(OpenCV_FOUND)
  if(TARGET opencv_core)
    target_link_libraries(tracker PRIVATE
      opencv_core
      opencv_imgproc
      opencv_highgui
      opencv_videoio
      opencv_dnn
    )
  else()
    target_link_libraries(tracker PRIVATE ${OpenCV_LIBS})
  endif()
  target_compile_definitions(tracker PRIVATE HAVE_OPENCV=1)
endif()

if(Eigen3_FOUND)
  target_link_libraries(tracker PRIVATE Eigen3::Eigen)
  target_compile_definitions(tracker PRIVATE HAVE_EIGEN3=1)
endif()

if(yaml-cpp_FOUND)
  target_link_libraries(tracker PRIVATE yaml-cpp)
  target_compile_definitions(tracker PRIVATE HAVE_YAML_CPP=1)
endif()

if(MSVC)
  target_compile_options(tracker PRIVATE /W4 /utf-8)
else()
  target_compile_options(tracker PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(tracker PRIVATE stdc++fs)
endif()

if(MINGW AND MINGW_STATIC_RUNTIME)
  target_link_options(tracker PRIVATE -static -static-libgcc -static-libstdc++)
endif()

set_target_properties(tracker PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
)

if(USE_ONNXRUNTIME)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime install root")
  if(NOT ONNXRUNTIME_ROOT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.23.2")
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.23.2" CACHE PATH "Path to ONNX Runtime install root" FORCE)
  endif()
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS
      ${ONNXRUNTIME_ROOT}
      ${ONNXRUNTIME_ROOT}/include
      $ENV{ONNXRUNTIME_ROOT}
      $ENV{ONNXRUNTIME_ROOT}/include
    PATH_SUFFIXES include include/onnxruntime
  )
  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS
      ${ONNXRUNTIME_ROOT}
      ${ONNXRUNTIME_ROOT}/lib
      $ENV{ONNXRUNTIME_ROOT}
      $ENV{ONNXRUNTIME_ROOT}/lib
    PATH_SUFFIXES lib lib64
  )
  find_file(ONNXRUNTIME_DLL onnxruntime.dll
    HINTS
      ${ONNXRUNTIME_ROOT}
      ${ONNXRUNTIME_ROOT}/lib
      $ENV{ONNXRUNTIME_ROOT}
      $ENV{ONNXRUNTIME_ROOT}/lib
    PATH_SUFFIXES lib bin
  )
  find_file(ONNXRUNTIME_PROVIDERS_SHARED_DLL onnxruntime_providers_shared.dll
    HINTS
      ${ONNXRUNTIME_ROOT}
      ${ONNXRUNTIME_ROOT}/lib
      $ENV{ONNXRUNTIME_ROOT}
      $ENV{ONNXRUNTIME_ROOT}/lib
    PATH_SUFFIXES lib bin
  )
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    target_include_directories(tracker PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(tracker PRIVATE ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(tracker PRIVATE HAVE_ONNXRUNTIME=1)
    message(STATUS "ONNX Runtime enabled: ${ONNXRUNTIME_LIBRARY}")
    if(ONNXRUNTIME_DLL)
      add_custom_command(TARGET tracker POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${ONNXRUNTIME_DLL}
          $<TARGET_FILE_DIR:tracker>
      )
    endif()
    if(ONNXRUNTIME_PROVIDERS_SHARED_DLL)
      add_custom_command(TARGET tracker POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${ONNXRUNTIME_PROVIDERS_SHARED_DLL}
          $<TARGET_FILE_DIR:tracker>
      )
    endif()
  else()
    message(WARNING "USE_ONNXRUNTIME=ON but ONNX Runtime not found, fallback to disabled backend.")
  endif()
endif()

if(ENABLE_INSTALL)
  include(GNUInstallDirs)
  install(TARGETS tracker
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
  install(DIRECTORY config DESTINATION ${CMAKE_INSTALL_DATADIR}/smart_gimbal_tracker_cpp)
endif()
