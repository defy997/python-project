"""
模块名称：app.py
作者信息：沈宏舟（示例），学号：2025000000
模块功能：Flask 后端入口，负责：
    1. 提供前端网页（模板渲染）；
    2. 接收前端上传的金融数据 / 文本；
    3. 调用数据分析模块与 AI 助手模块；
    4. 将图表数据和分析结果以 JSON 形式返回给前端；
    5. 管理分析结果文件的保存与下载。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)
import pandas as pd

from analysis import (
    OUTPUT_DIR,
    AnalysisResult,
    analyze_financial_dataframe,
    load_builtin_dataset,
)
from ai_helper import analyze_sentiment, summarize_financial_text
# 引入爬虫 + MySQL + 机器学习相关函数，方便在同一个 Flask 应用中使用
from crawler_ml import (
    MLResult,
    crawl_financial_news,
    crawl_simple_price_series,
    load_from_mysql,
    run_classification,
    run_clustering,
    run_regression,
    save_to_mysql,
)


BASE_DIR = Path(__file__).parent           # 当前项目根目录
TEMPLATE_DIR = BASE_DIR / "templates"      # HTML 模板目录
STATIC_DIR = BASE_DIR / "static"           # 静态资源目录

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))


@app.route("/", methods=["GET"])
def index() -> str:
    """
    函数名称：index
    函数功能：返回系统首页 HTML 页面。
    参数说明：
        无。
    返回值：
        str：渲染后的 HTML 页面字符串。
    """
    return render_template("index.html")  # 渲染首页模板


@app.route("/crawler_ml", methods=["GET"])
def crawler_ml_index() -> str:
    """
    函数名称：crawler_ml_index
    函数功能：返回“爬虫 + MySQL + 机器学习”实验的前端界面。
    返回值：
        str：渲染后的 HTML 字符串。
    """
    return render_template("crawler_ml.html")


def _build_df_from_request() -> pd.DataFrame:
    """
    函数名称：_build_df_from_request
    函数功能：根据前端传来的表单内容构建 DataFrame。
    优先顺序：上传文件 > 文本粘贴 > 内置示例数据。
    参数说明：
        无（直接从全局 request 对象读取）。
    返回值：
        pd.DataFrame：构建好的数据表。
    """
    # 1. 如果用户上传了 CSV 文件，则优先使用
    if "file" in request.files:
        f = request.files["file"]
        if f and f.filename.endswith(".csv"):
            # 将上传文件读取为 DataFrame，优先尝试 UTF-8，失败时回退到 GBK
            try:
                return pd.read_csv(f, encoding="utf-8-sig")
            except UnicodeDecodeError:
                f.stream.seek(0)  # 重置文件指针
                # 兼容旧版 pandas：不传 errors 参数，只切换为 GBK 编码
                return pd.read_csv(f, encoding="gbk")

    # 2. 如果用户在文本框中粘贴了 CSV 文本
    text_data = request.form.get("text_data", "").strip()
    if text_data:
        from io import StringIO

        # 使用 StringIO 将文本包装成“文件对象”
        return pd.read_csv(StringIO(text_data))

    # 3. 否则尝试加载内置示例数据
    dataset = request.form.get("dataset", "ads")  # 默认使用 advertising 示例
    df, _, _ = load_builtin_dataset(dataset)
    return df


@app.route("/analyze", methods=["POST"])
def analyze() -> Any:
    """
    函数名称：analyze
    函数功能：处理前端的“开始分析”请求，执行数据分析和 AI 文本分析，并返回 JSON 结构。
    参数说明：
        无（从 request.form / request.files 读取）。
    返回值：
        Flask Response：包含分析结果的 JSON 响应。
    """
    # 从请求中获取用户选定的列信息
    x_col = request.form.get("x_col", "")
    y_col = request.form.get("y_col", "")

    # 构建 DataFrame
    df = _build_df_from_request()

    # 如果未指定列名，简单取前两列作为 X/Y
    if not x_col or x_col not in df.columns:
        x_col = df.columns[0]
    if not y_col or y_col not in df.columns:
        y_col = df.columns[1]

    # 调用数据分析模块
    result: AnalysisResult = analyze_financial_dataframe(df, x_col, y_col)

    # 文本与 AI 分析相关
    raw_text = request.form.get("doc_text", "").strip()
    if raw_text:
        summary = summarize_financial_text(raw_text)      # AI 摘要
        sentiment = analyze_sentiment(raw_text)           # 情感分析
    else:
        summary = ""
        sentiment = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

    # 将结果打包为 JSON
    payload: Dict[str, Any] = {
        "status": "ok",               # 状态码
        "x_col": x_col,               # 实际使用的 X 列
        "y_col": y_col,               # 实际使用的 Y 列
        "summary": result.summary,    # 数值分析摘要
        "stats": result.stats,        # 统计结果
        "charts": result.charts,      # 多种图表数据
        "files": result.saved_files,  # 已保存的本地文件
        "ai_summary": summary,        # 文本 AI 摘要
        "sentiment": sentiment,       # 情感分析分布
    }

    return jsonify(payload)  # 返回 JSON 响应


@app.route("/files/<path:filename>", methods=["GET"])
def download_file(filename: str):
    """
    函数名称：download_file
    函数功能：向前端提供输出结果文件的下载功能。
    参数说明：
        filename (str)：文件名相对路径。
    返回值：
        Flask Response：文件下载响应。
    """
    return send_from_directory(str(OUTPUT_DIR), filename, as_attachment=True)


@app.route("/api/crawl_news", methods=["POST"])
def api_crawl_news():
    """
    函数名称：api_crawl_news
    函数功能：根据关键字爬取金融新闻列表并保存到 MySQL。
    前端控件传入：
        keyword：新闻搜索关键字。
        table_name：保存到 MySQL 的表名。
    返回值：
        JSON：包含爬取数量和预览数据。
    """
    keyword = request.form.get("keyword", "").strip() or "金融"
    table_name = request.form.get("table_name", "news_data")

    df_news = crawl_financial_news(keyword)  # 爬取新闻
    if not df_news.empty:
        save_to_mysql(df_news, table_name)   # 保存到 MySQL

    preview = df_news.head(10).to_dict(orient="records")
    return jsonify(
        {
            "status": "ok",
            "source": "news",
            "keyword": keyword,
            "table": table_name,
            "rows": int(len(df_news)),
            "preview": preview,
        }
    )


@app.route("/api/crawl_price", methods=["POST"])
def api_crawl_price():
    """
    函数名称：api_crawl_price
    函数功能：根据用户输入的 API 地址爬取价格时间序列，并保存到 MySQL。
    前端控件传入：
        api_url：返回 JSON 的接口地址；
        table_name：保存到 MySQL 的表名。
    返回值：
        JSON：包含爬取数量和预览数据。
    """
    api_url = request.form.get("api_url", "").strip()
    table_name = request.form.get("table_name", "financial_series")

    if not api_url:
        return jsonify({"status": "error", "msg": "api_url 不能为空"})

    df_price = crawl_simple_price_series(api_url)
    if not df_price.empty:
        save_to_mysql(df_price, table_name)

    preview = df_price.head(10).to_dict(orient="records")
    return jsonify(
        {
            "status": "ok",
            "source": "price",
            "api_url": api_url,
            "table": table_name,
            "rows": int(len(df_price)),
            "preview": preview,
        }
    )


@app.route("/api/run_ml", methods=["POST"])
def api_run_ml():
    """
    函数名称：api_run_ml
    函数功能：从 MySQL 中读出指定表，调用三类机器学习算法进行建模或预测。
    前端控件传入：
        table_name：数据表名；
        feature_cols：特征列名，逗号分隔；
        target_col：目标列名（供回归和分类使用）。
    返回值：
        JSON：包含三种模型的指标和预测预览。
    """
    table_name = request.form.get("table_name", "financial_series")
    feature_cols_str = request.form.get("feature_cols", "").strip()
    target_col = request.form.get("target_col", "").strip()

    if not feature_cols_str:
        return jsonify({"status": "error", "msg": "feature_cols 不能为空"})

    feature_cols = [c.strip() for c in feature_cols_str.split(",") if c.strip()]

    df = load_from_mysql(table_name)  # 从 MySQL 读取数据

    results: Dict[str, Dict] = {}

    # 1. 回归
    if target_col and target_col in df.columns:
        reg_res: MLResult = run_regression(df, feature_cols, target_col)
        results["regression"] = {
            "metrics": reg_res.metrics,
            "preview": reg_res.preview.to_dict(orient="records"),
        }

        # 2. 分类（基于同一个连续目标）
        cls_res: MLResult = run_classification(df, feature_cols, target_col)
        results["classification"] = {
            "metrics": cls_res.metrics,
            "preview": cls_res.preview.to_dict(orient="records"),
        }

    # 3. 聚类（只用特征列）
    clu_res: MLResult = run_clustering(df, feature_cols)
    results["clustering"] = {
        "metrics": clu_res.metrics,
        "preview": clu_res.preview.to_dict(orient="records"),
    }

    return jsonify(
        {
            "status": "ok",
            "table": table_name,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "models": results,
        }
    )


if __name__ == "__main__":
    # 仅用于本地开发调试时启动 Flask 应用
    app.run(debug=True)


