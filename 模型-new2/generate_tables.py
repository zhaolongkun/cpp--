"""
生成论文所需的所有表格（LaTeX格式）
"""
import numpy as np

def generate_table1_dataset_stats():
    """表1: 数据集统计"""
    latex = r"""
\begin{table}[h]
\centering
\caption{Dataset Statistics}
\label{tab:dataset}
\begin{tabular}{lc}
\hline
\textbf{Item} & \textbf{Value} \\
\hline
Total Samples & 2021 \\
Training Set & 1415 (70\%) \\
Validation Set & 303 (15\%) \\
Test Set & 303 (15\%) \\
Sequence Length & 8 frames \\
Feature Dimension & 4 \\
\hline
\end{tabular}
\end{table}
"""
    return latex

def generate_table2_main_comparison():
    """表2: 主方法对比（需要实验结果填充）"""
    latex = r"""
\begin{table*}[t]
\centering
\caption{Main Method Comparison}
\label{tab:main_comparison}
\begin{tabular}{lccccc}
\hline
\textbf{Method} & \textbf{MAE$_{fusion}$} & \textbf{Jitter} & \textbf{Spike Rate} & \textbf{Sign Flip Rate} \\
\hline
Raw & - & - & - & - \\
Ref Only & - & - & - & - \\
Prediction Only & - & - & - & - \\
Fixed Fusion & - & - & - & - \\
Adaptive Fusion (Ours) & - & - & - & - \\
\hline
\end{tabular}
\end{table*}
"""
    return latex

def generate_all_tables():
    """生成所有表格"""
    tables = {
        'table1_dataset.tex': generate_table1_dataset_stats(),
        'table2_main_comparison.tex': generate_table2_main_comparison(),
    }

    output_dir = '../paper/仪器与测量汇刊/new2/picture/'
    for filename, content in tables.items():
        filepath = output_dir + filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已生成: {filepath}")

if __name__ == '__main__':
    generate_all_tables()
    print("\n表格生成完成")
