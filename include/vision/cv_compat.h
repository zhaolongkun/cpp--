#pragma once

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#else

#include <string>

namespace cv {

// 无 OpenCV 构建时提供最小兼容声明，
// 让非视觉主流程或纯接口编译仍能通过。
constexpr int CAP_PROP_FRAME_WIDTH = 3;
constexpr int CAP_PROP_FRAME_HEIGHT = 4;
constexpr int FONT_HERSHEY_SIMPLEX = 0;

struct Mat {
  bool empty() const { return true; }
};

struct Point {
  int x{0};
  int y{0};
  Point() = default;
  Point(int x_, int y_) : x(x_), y(y_) {}
};

struct Scalar {
  double v0{0.0};
  double v1{0.0};
  double v2{0.0};
  Scalar() = default;
  Scalar(double a, double b, double c) : v0(a), v1(b), v2(c) {}
};

class VideoCapture {
 public:
  bool open(int /*index*/) { return false; }
  bool isOpened() const { return false; }
  bool read(Mat& /*frame*/) { return false; }
  void release() {}
  bool set(int /*prop*/, double /*value*/) { return false; }
};

inline void rectangle(Mat&, const Point&, const Point&, const Scalar&, int) {}
inline void putText(Mat&, const std::string&, const Point&, int, double, const Scalar&, int) {}
inline void circle(Mat&, const Point&, int, const Scalar&, int) {}
inline void imshow(const std::string&, const Mat&) {}
inline int waitKey(int) { return -1; }
inline void destroyAllWindows() {}

}  // namespace cv

#endif
