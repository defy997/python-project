"""
模块名称：crawler_ml_app.py
作者信息：沈宏舟（示例），学号：2025000000
模块功能：金融爬虫 + MySQL + 机器学习 综合实验后端。
    - 提供前端页面（/crawler_ml）；
    - 根据不同控件输入执行爬虫任务，将数据保存到 MySQL；
    - 从 MySQL 读取数据并调用三类机器学习算法进行预测或建模；
    - 以 JSON 形式把结果返回给前端展示。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

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


BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))


@app.route("/crawler_ml", methods=["GET"])
def crawler_ml_index() -> str:
    """
    函数名称：crawler_ml_index
    函数功能：返回“爬虫 + MySQL + 机器学习”实验的前端界面。
    返回值：
        str：渲染后的 HTML 字符串。
    """
    return render_template("crawler_ml.html")


@app.route("/api/crawl_news", methods=["POST"])
def api_crawl_news() -> Any:
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
def api_crawl_price() -> Any:
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
def api_run_ml() -> Any:
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
    # 仅用于单独调试本实验后端时使用
    app.run(port=5001, debug=True)


