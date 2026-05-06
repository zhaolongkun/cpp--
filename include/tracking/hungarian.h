#pragma once

#include <utility>
#include <vector>

namespace tracking {

// 线性分配结果。
struct AssignmentResult {
  std::vector<std::pair<int, int>> matches;
  std::vector<int> unmatched_rows;
  std::vector<int> unmatched_cols;
};

class Hungarian {
 public:
  // 对代价矩阵做最优匹配；代价大于 max_cost 的项最终视为未匹配。
  static AssignmentResult solve(const std::vector<std::vector<double>>& cost, double max_cost);
};

}  // namespace tracking
