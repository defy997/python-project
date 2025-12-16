"""
模块名称：analysis.py
模块功能：完成对金融数据的读取、清洗、统计分析以及可视化图表数据的准备。

本模块不直接绘制前端图形，而是生成结构化的字典数据，
由前端 JavaScript + ECharts 进行折线图、柱状图、饼图、散点图、箱线图等可视化展示。
同时，本模块也负责将分析结果导出为 CSV / JSON / TXT 报告以及静态 PNG 图片。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).parent / "outputs"  # 结果输出目录
OUTPUT_DIR.mkdir(exist_ok=True)                # 如果目录不存在则创建


@dataclass
class AnalysisResult:
    """
    数据类：封装一次金融数据分析的核心结果。

    字段说明：
        summary (str)：文字化的分析结论摘要。
        stats (Dict)：用于导出到 JSON / CSV 的统计数据。
        charts (Dict)：前端各类图形所需的数据结构。
        saved_files (List[str])：本地已保存的文件路径列表。
    """

    summary: str
    stats: Dict
    charts: Dict
    saved_files: List[str]


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    函数名称：_ensure_numeric
    函数功能：将指定列转为数值型，无法转换的填充为 NaN。
    参数说明：
        df (pd.DataFrame)：原始数据表。
        cols (List[str])：需要转换为数值型的列名列表。
    返回值：
        pd.DataFrame：转换后的数据表。
    """
    for c in cols:
        if c in df.columns:
            # 使用 pd.to_numeric 强制转换为数值
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_financial_csv(csv_path: Path) -> pd.DataFrame:
    """
    函数名称：load_financial_csv
    函数功能：从给定路径读取 CSV 金融数据，并做简单清洗。
    参数说明：
        csv_path (Path)：CSV 文件路径。
    返回值：
        pd.DataFrame：清洗后的数据框。
    """
    # 尝试使用 UTF-8 编码读取，如果失败则回退到 GBK（适配常见中文 CSV）
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")  # 读取 CSV 文件
    except UnicodeDecodeError:
        # 对于较老版本的 pandas，不支持 errors 参数，这里只切换编码为 GBK
        df = pd.read_csv(csv_path, encoding="gbk")
    df = df.dropna(how="all")   # 删除全为空的行
    return df


def build_charts_from_df(df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
    """
    函数名称：build_charts_from_df
    函数功能：基于给定 DataFrame 和 X/Y 列构建多种图表所需的数据。
    参数说明：
        df (pd.DataFrame)：原始数据表。
        x_col (str)：作为横轴的列名。
        y_col (str)：作为纵轴的列名。
    返回值：
        Dict：包含多种图表配置数据的字典。
    """
    df = _ensure_numeric(df, [y_col])  # 确保纵轴为数值
    df = df.dropna(subset=[y_col])     # 删除纵轴为空的记录

    # 提取基础数组
    x_values = df[x_col].astype(str).tolist()     # X 轴分类或时间
    y_values = df[y_col].astype(float).tolist()   # Y 轴数值

    # 1. 折线图（趋势）
    line_chart = {
        "x": x_values,
        "y": y_values,
    }

    # 2. 柱状图（对比）
    bar_chart = {
        "x": x_values,
        "y": y_values,
    }

    # 3. 饼图（按区间占比）
    if len(y_values) > 0:
        arr = np.array(y_values)
        q1, q2, q3 = np.quantile(arr, [0.25, 0.5, 0.75])  # 四分位数
        buckets = {"低收益": 0, "中等收益": 0, "较高收益": 0, "极高收益": 0}
        # 遍历每个数值划分到不同区间
        for v in y_values:
            if v <= q1:
                buckets["低收益"] += 1
            elif v <= q2:
                buckets["中等收益"] += 1
            elif v <= q3:
                buckets["较高收益"] += 1
            else:
                buckets["极高收益"] += 1
        # 转换为 ECharts 所需格式
        pie_chart = [{"name": k, "value": v} for k, v in buckets.items()]
    else:
        pie_chart = []

    # 4. 散点图（简单用 x 索引）
    scatter_chart = {
        "x": list(range(len(y_values))),  # 使用索引当作横轴
        "y": y_values,
    }

    # 5. 箱线图（收益分布）
    if len(y_values) > 0:
        box_data = [float(np.min(y_values)), q1, q2, q3, float(np.max(y_values))]
    else:
        box_data = [0, 0, 0, 0, 0]

    boxplot_chart = {
        "label": y_col,  # 单个系列的名称
        "data": box_data,
    }

    charts = {
        "line": line_chart,
        "bar": bar_chart,
        "pie": pie_chart,
        "scatter": scatter_chart,
        "boxplot": boxplot_chart,
    }
    return charts


def compute_basic_stats(df: pd.DataFrame, y_col: str) -> Dict:
    """
    函数名称：compute_basic_stats
    函数功能：对单一数值列计算基础统计特征。
    参数说明：
        df (pd.DataFrame)：数据表。
        y_col (str)：数值列列名。
    返回值：
        Dict：描述性统计结果。
    """
    df = _ensure_numeric(df, [y_col])  # 转换为数值
    series = df[y_col].dropna()        # 去掉缺失值
    if series.empty:
        return {}

    stats = {
        "count": int(series.count()),           # 样本数量
        "mean": float(series.mean()),          # 均值
        "std": float(series.std(ddof=1)),      # 标准差
        "min": float(series.min()),            # 最小值
        "max": float(series.max()),            # 最大值
        "q1": float(series.quantile(0.25)),    # 第一四分位数
        "median": float(series.median()),      # 中位数
        "q3": float(series.quantile(0.75)),    # 第三四分位数
    }
    return stats


def _save_text_report(summary: str, stats: Dict, filename: str) -> str:
    """
    函数名称：_save_text_report
    函数功能：将文字摘要与统计结果保存为 TXT 报告文件。
    参数说明：
        summary (str)：文字摘要。
        stats (Dict)：统计结果字典。
        filename (str)：报告文件名（不含路径）。
    返回值：
        str：实际保存的文件路径字符串。
    """
    path = OUTPUT_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        f.write("【金融数据分析报告】\n")
        f.write(summary + "\n\n")
        f.write("【基础统计特征】\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    return str(path)


def _save_stats_csv(stats: Dict, filename: str) -> str:
    """
    函数名称：_save_stats_csv
    函数功能：将统计结果字典保存到 CSV 文件。
    参数说明：
        stats (Dict)：统计结果。
        filename (str)：CSV 文件名。
    返回值：
        str：实际保存路径。
    """
    path = OUTPUT_DIR / filename
    df_stats = pd.DataFrame(list(stats.items()), columns=["metric", "value"])
    df_stats.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _save_hist_png(series: pd.Series, filename: str) -> str:
    """
    函数名称：_save_hist_png
    函数功能：将收益率分布直方图保存为 PNG 图片文件。
    参数说明：
        series (pd.Series)：待绘图的数值序列。
        filename (str)：PNG 文件名。
    返回值：
        str：图片文件路径。
    """
    path = OUTPUT_DIR / filename
    plt.figure(figsize=(6, 4))                     # 设置画布大小
    plt.hist(series, bins=15, color="skyblue")     # 绘制直方图
    plt.title("收益分布直方图")                     # 图标题
    plt.xlabel("收益值")                           # X 轴标签
    plt.ylabel("频数")                             # Y 轴标签
    plt.tight_layout()                             # 布局紧凑
    plt.savefig(path, dpi=120)                     # 保存图片
    plt.close()                                    # 关闭图像避免内存泄露
    return str(path)


def analyze_financial_dataframe(df: pd.DataFrame, x_col: str, y_col: str) -> AnalysisResult:
    """
    函数名称：analyze_financial_dataframe
    函数功能：对给定的金融 DataFrame 做一次完整分析，封装为 AnalysisResult。
    参数说明：
        df (pd.DataFrame)：原始数据。
        x_col (str)：横轴列名。
        y_col (str)：纵轴列名。
    返回值：
        AnalysisResult：包含摘要、统计、图表和导出文件路径的对象。
    """
    stats = compute_basic_stats(df, y_col)         # 计算基础统计
    charts = build_charts_from_df(df, x_col, y_col)  # 构建多种图表数据

    # 根据统计信息生成一段简要文字描述
    if stats:
        summary = (
            f"在选定的金融数据中，目标变量 {y_col} 的平均值为 {stats['mean']:.2f}，"
            f"标准差为 {stats['std']:.2f}，最小值 {stats['min']:.2f}，最大值 {stats['max']:.2f}。"
        )
    else:
        summary = "未能从当前数据中计算出有效的统计特征，请检查数据列选择是否正确。"

    # 保存报告、CSV 与直方图图片
    saved_files: List[str] = []
    if stats:
        # 将 y 列拿出来生成静态图片
        df = _ensure_numeric(df, [y_col])
        series = df[y_col].dropna()
        if not series.empty:
            img_path = _save_hist_png(series, "histogram.png")   # 保存直方图
            saved_files.append(img_path)                         # 记录文件路径

    txt_path = _save_text_report(summary, stats, "analysis_report.txt")
    saved_files.append(txt_path)

    csv_path = _save_stats_csv(stats, "analysis_stats.csv")
    saved_files.append(csv_path)

    return AnalysisResult(summary=summary, stats=stats, charts=charts, saved_files=saved_files)


def load_builtin_dataset(name: str) -> Tuple[pd.DataFrame, str, str]:
    """
    函数名称：load_builtin_dataset
    函数功能：加载内置金融示例数据集（如广告投放与销售、股票收益数据等）。
    参数说明：
        name (str)：数据集名称标识，例如 "ads" 或 "trade"。
    返回值：
        Tuple[pd.DataFrame, str, str]：
            - DataFrame：加载后的数据；
            - x_col：建议用作横轴的列名；
            - y_col：建议用作纵轴的列名。
    """
    # 本项目目录结构：.../school/10 与 .../school/project 同级，因此这里取上一级目录再进入 "10"
    base = Path(__file__).resolve().parent.parent / "10"  # 指向已有的示例数据路径
    if name == "ads":
        csv_path = base / "advertising.csv"    # 广告投放数据
        x_col, y_col = "TV", "Sales"          # 以 TV 投放和销售额为例
    else:
        csv_path = base / "trade.csv"         # 交易数据
        x_col, y_col = "Open", "Close"       # 开盘价与收盘价

    df = load_financial_csv(csv_path)
    return df, x_col, y_col


if __name__ == "__main__":
    # 模块自测代码：加载示例数据并执行一次完整分析。
    data, x, y = load_builtin_dataset("ads")
    result = analyze_financial_dataframe(data, x, y)
    print("=== 分析摘要 ===")
    print(result.summary)
    print("=== 保存文件 ===")
    for p in result.saved_files:
        print(p)


