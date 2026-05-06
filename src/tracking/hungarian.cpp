#include "tracking/hungarian.h"

#include <algorithm>
#include <limits>

namespace tracking {

AssignmentResult Hungarian::solve(const std::vector<std::vector<double>>& cost, double max_cost) {
  AssignmentResult out;
  const int n_rows = static_cast<int>(cost.size());
  const int n_cols = n_rows > 0 ? static_cast<int>(cost[0].size()) : 0;

  if (n_rows == 0) {
    out.unmatched_cols.resize(n_cols);
    for (int j = 0; j < n_cols; ++j) {
      out.unmatched_cols[j] = j;
    }
    return out;
  }
  if (n_cols == 0) {
    out.unmatched_rows.resize(n_rows);
    for (int i = 0; i < n_rows; ++i) {
      out.unmatched_rows[i] = i;
    }
    return out;
  }

  const int n = std::max(n_rows, n_cols);
  const double inf = 1e9;

  // 匈牙利算法要求方阵，这里把原始代价矩阵补成 n x n。
  // 对无效补位使用“大于 max_cost 的代价”，保证后续会被判为未匹配。
  std::vector<std::vector<double>> a(n + 1, std::vector<double>(n + 1, max_cost + 1.0));
  for (int i = 1; i <= n_rows; ++i) {
    for (int j = 1; j <= n_cols; ++j) {
      a[i][j] = cost[i - 1][j - 1];
    }
  }

  std::vector<double> u(n + 1), v(n + 1);
  std::vector<int> p(n + 1), way(n + 1);

  for (int i = 1; i <= n; ++i) {
    // 这是标准 Hungarian 逐行增广过程：
    // 维护顶标 u,v，并不断寻找当前行的最优可增广列。
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(n + 1, inf);
    std::vector<bool> used(n + 1, false);

    do {
      used[j0] = true;
      const int i0 = p[j0];
      double delta = inf;
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[j]) {
          continue;
        }
        const double cur = a[i0][j] - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (int j = 0; j <= n; ++j) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] != 0);

    do {
      const int j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> row_to_col(n_rows, -1);
  for (int j = 1; j <= n; ++j) {
    if (p[j] >= 1 && p[j] <= n_rows && j <= n_cols) {
      row_to_col[p[j] - 1] = j - 1;
    }
  }

  std::vector<bool> col_used(n_cols, false);
  for (int i = 0; i < n_rows; ++i) {
    const int j = row_to_col[i];
    // 即使算法给出了配对，只要代价超过门限，仍然视为未匹配，
    // 这样可以避免强行把明显不相关的目标和检测绑在一起。
    if (j >= 0 && cost[i][j] <= max_cost) {
      out.matches.emplace_back(i, j);
      col_used[j] = true;
    } else {
      out.unmatched_rows.push_back(i);
    }
  }

  for (int j = 0; j < n_cols; ++j) {
    if (!col_used[j]) {
      out.unmatched_cols.push_back(j);
    }
  }

  return out;
}

}  // namespace tracking
