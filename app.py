"""
模块名称：app.py
模块功能：Flask 后端入口，负责：
    1. 提供前端网页（模板渲染）；
    2. 接收前端上传的金融数据 / 文本；
    3. 调用数据分析模块与 AI 助手模块；
    4. 将图表数据和分析结果以 JSON 形式返回给前端；
    5. 管理分析结果文件的保存与下载。
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import pymysql
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

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
from utils import handle_api_errors, logger, validate_dataframe
from db_config import DB_CONFIG

try:
    import tushare as ts
except ImportError:  # pragma: no cover
    ts = None


BASE_DIR = Path(__file__).parent           # 当前项目根目录
TEMPLATE_DIR = BASE_DIR / "templates"      # HTML 模板目录
STATIC_DIR = BASE_DIR / "static"           # 静态资源目录

# 默认使用用户提供的 TuShare Token，可通过环境变量覆盖
TUSHARE_TOKEN = os.getenv(
    "TUSHARE_TOKEN",
    "9d41733e4997a12f4eac28b57f9d1337eacad86f5f980ebe5370162f",
)

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))


def get_mysql_connection():
    """
    函数名称：get_mysql_connection
    函数功能：根据 DB_CONFIG 创建一个 MySQL 连接。
    返回值：
        pymysql.Connection：数据库连接对象。
    """
    return pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        charset=DB_CONFIG["charset"],
        cursorclass=pymysql.cursors.DictCursor,
    )


def _extract_text_list_from_df(df: pd.DataFrame) -> List[str]:
    """将 DataFrame 中的文本列拼接成列表，供词云/情感分析使用。"""
    texts: List[str] = []
    for col in df.columns:
        if df[col].dtype == object:
            texts.extend(df[col].astype(str).tolist())
    return [t for t in texts if isinstance(t, str) and t.strip()]


def _build_texts_from_request() -> Tuple[List[str], str]:
    """
    根据前端提交内容获取文本数据列表。
    优先顺序：上传文件 > 文本粘贴 > MySQL 表（可选） > 内置示例。
    返回 (texts, source_desc)
    """
    # 1) 上传 CSV
    if "file" in request.files:
        f = request.files["file"]
        if f and f.filename.endswith(".csv"):
            try:
                df = pd.read_csv(f, encoding="utf-8-sig")
            except UnicodeDecodeError:
                f.stream.seek(0)
                df = pd.read_csv(f, encoding="gbk")
            return _extract_text_list_from_df(df), "上传文件"

    # 2) 文本框
    raw_text = request.form.get("text_data", "").strip()
    if raw_text:
        return raw_text.splitlines(), "文本框"

    # 3) MySQL 表
    table_name = request.form.get("table_name", "").strip()
    if table_name:
        try:
            df = load_from_mysql(table_name)
            return _extract_text_list_from_df(df), f"MySQL:{table_name}"
        except Exception:
            pass

    # 4) 内置示例：使用新闻关键词“金融”
    df_news = crawl_financial_news("金融")
    return _extract_text_list_from_df(df_news), "示例新闻"


def _tokenize_texts(texts: List[str]) -> Counter:
    """
    简单分词：按非中英文/数字拆分，再统计词频。
    若有 jieba 可替换为更优分词，这里保持零依赖。
    """
    counter: Counter = Counter()
    pattern = re.compile(r"[A-Za-z0-9\u4e00-\u9fa5]+")
    for t in texts:
        for m in pattern.findall(t):
            # 过滤过短词
            if len(m.strip()) >= 2:
                counter[m.strip().lower()] += 1
    return counter


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
@handle_api_errors
def analyze() -> Any:
    """
    函数名称：analyze
    函数功能：处理前端的"开始分析"请求，执行数据分析和 AI 文本分析，并返回 JSON 结构。
    参数说明：
        无（从 request.form / request.files 读取）。
    返回值：
        Flask Response：包含分析结果的 JSON 响应。
    """
    logger.info("收到分析请求")
    
    # 从请求中获取用户选定的列信息
    x_col = request.form.get("x_col", "").strip()
    y_col = request.form.get("y_col", "").strip()

    # 构建 DataFrame
    try:
        df = _build_df_from_request()
    except Exception as e:
        logger.error(f"构建 DataFrame 失败: {e}")
        return jsonify({"status": "error", "msg": f"数据读取失败: {str(e)}"}), 400

    # 验证数据
    validation = validate_dataframe(df, min_rows=2)
    if not validation["valid"]:
        return jsonify({"status": "error", "msg": validation["msg"]}), 400

    # 如果未指定列名，简单取前两列作为 X/Y
    if not x_col or x_col not in df.columns:
        x_col = df.columns[0]
        logger.info(f"自动选择 X 轴列: {x_col}")
    if not y_col or y_col not in df.columns:
        y_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        logger.info(f"自动选择 Y 轴列: {y_col}")

    # 调用数据分析模块
    try:
        result: AnalysisResult = analyze_financial_dataframe(df, x_col, y_col)
    except Exception as e:
        logger.error(f"数据分析失败: {e}")
        return jsonify({"status": "error", "msg": f"数据分析失败: {str(e)}"}), 500

    # 文本与 AI 分析相关
    raw_text = request.form.get("doc_text", "").strip()
    if raw_text:
        try:
            summary = summarize_financial_text(raw_text)      # AI 摘要
            sentiment = analyze_sentiment(raw_text)           # 情感分析
        except Exception as e:
            logger.warning(f"AI 分析失败: {e}")
            summary = "AI 分析暂时不可用"
            sentiment = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}
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

    logger.info("分析完成，返回结果")
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
@handle_api_errors
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
    table_name = request.form.get("table_name", "news_data").strip() or "news_data"
    
    if not keyword:
        return jsonify({"status": "error", "msg": "关键字不能为空"}), 400
    
    logger.info(f"开始爬取新闻，关键字: {keyword}, 表名: {table_name}")

    try:
        df_news = crawl_financial_news(keyword)  # 爬取新闻
        if not df_news.empty:
            save_to_mysql(df_news, table_name)   # 保存到 MySQL
            logger.info(f"成功爬取 {len(df_news)} 条新闻并保存到 {table_name}")
        else:
            logger.warning(f"未爬取到任何新闻数据")
    except Exception as e:
        logger.error(f"爬取新闻失败: {e}")
        return jsonify({"status": "error", "msg": f"爬取失败: {str(e)}"}), 500

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


@app.route("/api/crawl_news_v2", methods=["POST"])
def api_crawl_news_v2():
    """
    增强版新闻爬虫接口：允许指定关键字与表名，返回更多预览。
    若外网失败，回落到内置示例。
    """
    keyword = request.form.get("keyword", "").strip() or "金融"
    table_name = request.form.get("table_name", "news_data")

    try:
        df_news = crawl_financial_news(keyword)
    except Exception:
        df_news = pd.DataFrame()

    if df_news.empty:
        df_news = pd.DataFrame(
            {
                "keyword": [keyword] * 3,
                "title": [
                    f"{keyword} 政策对市场影响待评估",
                    f"{keyword} 数据公布，波动加剧",
                    f"机构解读：{keyword} 相关资产走势",
                ],
                "url": [
                    "https://example.com/a",
                    "https://example.com/b",
                    "https://example.com/c",
                ],
            }
        )

    save_to_mysql(df_news, table_name)
    preview = df_news.head(10).to_dict(orient="records")
    return jsonify(
        {
            "status": "ok",
            "keyword": keyword,
            "table": table_name,
            "rows": int(len(df_news)),
            "preview": preview,
        }
    )


@app.route("/api/wordcloud", methods=["POST"])
def api_wordcloud():
    """
    生成词云数据，支持来源：上传 CSV、文本框、MySQL 表、示例新闻。
    返回词频前 80 个。
    """
    texts, source = _build_texts_from_request()
    counter = _tokenize_texts(texts)
    top_words = counter.most_common(80)
    return jsonify(
        {
            "status": "ok",
            "source": source,
            "words": [{"name": w, "value": int(c)} for w, c in top_words],
        }
    )


@app.route("/api/sentiment_score", methods=["POST"])
def api_sentiment_score():
    """
    对文本集合做情感评分，并附加简单热度/风险分。
    """
    texts, source = _build_texts_from_request()
    combined = "\n".join(texts)[:2000]  # 防止过长
    sentiment = analyze_sentiment(combined if combined else "中性")

    # 热度分：按文本条数归一到 0~1
    heat = min(1.0, len(texts) / 50.0)

    # 简单风险分：负向概率加权
    risk = round(sentiment.get("negative", 0.33) * 10, 2)

    return jsonify(
        {
            "status": "ok",
            "source": source,
            "sentiment": sentiment,
            "heat": heat,
            "risk": risk,
        }
    )


@app.route("/api/stock_predict", methods=["POST"])
def api_stock_predict():
    """
    简单股票预测：若提供外部接口则优先使用，否则回退示例 trade.csv。
    返回历史与未来预测序列。
    """
    api_url = request.form.get("api_url", "").strip()
    horizon = int(request.form.get("horizon", 8) or 8)
    horizon = max(3, min(horizon, 30))

    df: pd.DataFrame
    try:
        if api_url:
            resp = requests.get(api_url, timeout=10)
            data = resp.json()
            records = data.get("data", data.get("prices", data))
            df = pd.DataFrame(records)
        else:
            df, _, _ = load_builtin_dataset("trade")
    except Exception:
        df, _, _ = load_builtin_dataset("trade")

    # 选择 close/price 列
    candidates = ["close", "Close", "price", "Price"]
    y_col = next((c for c in candidates if c in df.columns), df.columns[-1])
    df = df.dropna(subset=[y_col]).reset_index(drop=True)
    y = df[y_col].astype(float).values
    x = np.arange(len(y))

    if len(y) < 5:
        return jsonify({"status": "error", "msg": "数据量过少，无法预测"})

    # 一阶线性拟合
    coef = np.polyfit(x, y, deg=1)
    poly = np.poly1d(coef)

    future_x = np.arange(len(y), len(y) + horizon)
    future_y = poly(future_x)

    history = [{"x": int(i), "y": float(v)} for i, v in zip(x.tolist(), y.tolist())]
    forecast = [{"x": int(i), "y": float(v)} for i, v in zip(future_x.tolist(), future_y.tolist())]

    return jsonify(
        {
            "status": "ok",
            "y_col": y_col,
            "history": history,
            "forecast": forecast,
        }
    )


def _fetch_tushare_indices() -> Dict[str, Any]:
    """
    函数名称：_fetch_tushare_indices
    函数功能：从 TuShare 获取若干常见指数的近 60 个交易日行情，生成多系列折线图所需数据。
    返回值：
        Dict：包含 series 列表、x 轴日期、最新报价表等。
    """
    try:
        if ts is None:
            raise RuntimeError("当前环境未安装 tushare 库，请先 pip install tushare")

        if not TUSHARE_TOKEN:
            raise RuntimeError("未配置 TuShare Token，请设置环境变量 TUSHARE_TOKEN")

        pro = ts.pro_api(TUSHARE_TOKEN)

        # 指数代码映射（TuShare index_daily 的 ts_code）
        indices = {
            "上证指数": "000001.SH",
            "深证成指": "399001.SZ",
            "创业板指": "399006.SZ",
            "沪深300": "000300.SH",
            "中证500": "000905.SH",
        }

        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)
        end_str = end_date.strftime("%Y%m%d")
        start_str = start_date.strftime("%Y%m%d")

        series_list: List[Dict[str, Any]] = []
        latest_rows: List[Dict[str, Any]] = []

        ref_x: List[str] = []
        missing: List[str] = []

        for name, code in indices.items():
            # 若临时网络抖动 / TuShare 偶发返回空数据，这里做多次尝试
            df_idx = pd.DataFrame()
            max_attempts = 10
            for attempt in range(1, max_attempts + 1):
                try:
                    df_idx = pro.index_daily(ts_code=code, start_date=start_str, end_date=end_str)
                    if not df_idx.empty:
                        break
                    logger.warning(f"指数 {code} 第 {attempt} 次返回空数据，准备重试")
                except Exception as e:
                    logger.warning(f"获取指数 {code} 第 {attempt} 次失败: {e}")

                # 简单退避，避免瞬间连发
                if attempt < max_attempts:
                    time.sleep(0.6 * attempt)

            if df_idx.empty:
                missing.append(name)
                continue

            # 按日期升序
            df_idx = df_idx.sort_values("trade_date")
            x = df_idx["trade_date"].tolist()
            y = df_idx["close"].astype(float).round(3).tolist()

            # 记录参考 x 轴
            if len(x) > len(ref_x):
                ref_x = x

            series_list.append(
                {
                    "name": name,
                    "type": "line",
                    "smooth": True,
                    "data": y,
                }
            )
            last_row = df_idx.iloc[-1]
            prev_row = df_idx.iloc[-2] if len(df_idx) > 1 else last_row
            pct_chg = round((last_row["close"] - prev_row["close"]) / prev_row["close"] * 100, 2) if prev_row["close"] else 0
            latest_rows.append(
                {
                    "name": name,
                    "price": round(float(last_row["close"]), 3),
                    "chg": pct_chg,
                    "date": last_row["trade_date"],
                }
            )

        # 如果部分指数拉取失败，用平滑占位数据填充，避免前端只显示等待
        if missing and ref_x:
            for m in missing:
                # 构造一条平滑的占位线，以 ref_x 长度为准
                base = float(latest_rows[0]["price"]) if latest_rows else 1000.0
                noise = np.linspace(-5, 5, num=len(ref_x))
                y_fake = (base + noise).round(3).tolist()
                series_list.append(
                    {
                        "name": f"{m}(占位)",
                        "type": "line",
                        "smooth": True,
                        "data": y_fake,
                    }
                )
                latest_rows.append(
                    {
                        "name": f"{m}(占位)",
                        "price": round(base, 3),
                        "chg": 0.0,
                        "date": ref_x[-1],
                    }
                )

        # 如果完全没有从 TuShare 拿到任何指数，则退回到示例数据（避免 500）
        if not series_list:
            logger.warning("未从 TuShare 获取到任何指数数据，使用本地示例数据代替。")
            # 使用 trade.csv 构造两条示例指数
            df_trade, _, _ = load_builtin_dataset("trade")
            df_trade = df_trade.head(60)  # 取前 60 条
            ref_x = list(range(len(df_trade)))
            close = df_trade["Close"].astype(float).tolist()
            open_ = df_trade["Open"].astype(float).tolist()
            series_list = [
                {"name": "示例指数A", "type": "line", "smooth": True, "data": close},
                {"name": "示例指数B", "type": "line", "smooth": True, "data": open_},
            ]
            latest_rows = [
                {"name": "示例指数A", "price": round(float(close[-1]), 3), "chg": 0.0, "date": "示例"},
                {"name": "示例指数B", "price": round(float(open_[-1]), 3), "chg": 0.0, "date": "示例"},
            ]
            missing = ["全部真实指数"]

        return {
            "x": ref_x or x,  # 优先使用参考 x 轴
            "series": series_list,
            "latest": latest_rows,
            "missing": missing,
        }

    except Exception as e:
        # 最后一层兜底：任何异常都不抛到外面，直接返回简单示例数据
        logger.error(f"_fetch_tushare_indices 出现异常，使用最终兜底示例数据: {e}")
        xs = list(range(30))
        y1 = (1000 + np.linspace(-20, 20, 30)).round(2).tolist()
        y2 = (800 + np.linspace(10, -10, 30)).round(2).tolist()
        return {
            "x": xs,
            "series": [
                {"name": "示例指数1", "type": "line", "smooth": True, "data": y1},
                {"name": "示例指数2", "type": "line", "smooth": True, "data": y2},
            ],
            "latest": [
                {"name": "示例指数1", "price": y1[-1], "chg": 0.0, "date": "示例"},
                {"name": "示例指数2", "price": y2[-1], "chg": 0.0, "date": "示例"},
            ],
            "missing": ["TuShare 指数异常，已使用示例数据"],
        }


@app.route("/api/ts_market", methods=["GET"])
@handle_api_errors
def api_ts_market():
    """
    函数名称：api_ts_market
    函数功能：拉取 TuShare 指数行情，返回折线图和最新报价数据。
    返回值：
        JSON：包含 x 轴日期、series 数组（多指数折线）、latest 表格数据。
    """
    logger.info("开始拉取 TuShare 指数行情")
    data = _fetch_tushare_indices()
    return jsonify({"status": "ok", **data})


def _build_mock_fx() -> Dict[str, Any]:
    """
    函数名称：_build_mock_fx
    函数功能：在外汇数据无法从数据库或 TuShare 获取时，构造一组示例货币数据（不包含比特币）。
    返回值：
        Dict：包含 currencies 列表。
    """
    today = datetime.today().strftime("%Y-%m-%d")
    return {
        "currencies": [
            {"name": "英镑", "symbol": "GBP/USD", "price": 1.1825, "chg": -0.0005, "pct": 0.0, "date": today},
            {"name": "日元", "symbol": "JPY/USD", "price": 136.1876, "chg": -0.0715, "pct": -0.7, "date": today},
            {"name": "欧元", "symbol": "EUR/USD", "price": 0.9982, "chg": -0.0016, "pct": -0.2, "date": today},
        ]
    }


def _fetch_eastmoney_fx() -> Dict[str, Any]:
    """
    函数名称：_fetch_eastmoney_fx
    函数功能：从东方财富外汇频道（https://forex.eastmoney.com/）爬取“人民币中间价”相关外汇数据。
    说明：
        - 使用 requests 获取网页 HTML；
        - 使用 pandas.read_html 解析页面中的表格；
        - 通过包含“人民币”和“最新价”的表格作为候选，构造 currencies 列表。
    返回值：
        Dict：包含 currencies 列表。
    """
    url = "https://forex.eastmoney.com/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0 Safari/537.36"
        ),
        "Referer": "https://forex.eastmoney.com/",
    }

    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding

    # 从 HTML 中解析所有表格
    tables: List[pd.DataFrame] = pd.read_html(resp.text)

    target_df: pd.DataFrame | None = None

    # 通过包含“人民币”和列名中有“最新价”的表格，大概率就是“人民币中间价”
    for df in tables:
        col_str = "".join(map(str, df.columns))
        if "最新价" not in col_str:
            continue

        # 将整个表展开为字符串，便于判断是否涉及“人民币”
        try:
            content_str = col_str + "".join(df.astype(str).fillna("").values.ravel())
        except Exception:
            content_str = col_str

        if "人民币" in content_str:
            target_df = df
            break

    if target_df is None or target_df.empty:
        raise RuntimeError("未在东方财富页面中识别到人民币中间价表格")

    # 期望的列：代码 / 名称 / 最新价 / 涨跌额 / 涨跌幅 ...
    # 适配一下列名，避免页面有轻微调整导致 KeyError
    col_map = {}
    for col in target_df.columns:
        col_str = str(col)
        if "代码" in col_str and "code" not in col_map:
            col_map["code"] = col
        elif ("名称" in col_str or "币种" in col_str) and "name" not in col_map:
            col_map["name"] = col
        elif "最新价" in col_str and "price" not in col_map:
            col_map["price"] = col
        elif "涨跌额" in col_str and "chg" not in col_map:
            col_map["chg"] = col
        elif "涨跌幅" in col_str and "pct" not in col_map:
            col_map["pct"] = col

    required_keys = ["name", "price"]
    if not all(k in col_map for k in required_keys):
        raise RuntimeError(f"人民币中间价表格列名不符合预期: {target_df.columns}")

    currencies: List[Dict[str, Any]] = []
    today = datetime.today().strftime("%Y-%m-%d")

    for _, row in target_df.iterrows():
        try:
            name = str(row[col_map.get("name", "")]).strip()
            if not name:
                continue

            code = str(row[col_map.get("code", "")]).strip() if "code" in col_map else ""
            symbol = code if code else name

            # 最新价
            price_raw = row[col_map["price"]]
            price = float(str(price_raw).replace(",", "").strip()) if price_raw not in ("", None) else 0.0

            # 涨跌额
            chg_raw = row[col_map.get("chg", col_map["price"])]
            try:
                chg = float(str(chg_raw).replace(",", "").strip())
            except Exception:
                chg = 0.0

            # 涨跌幅（百分比，形如 "0.25%"）
            pct_raw = row[col_map.get("pct", col_map["price"])]
            pct_str = str(pct_raw).replace("%", "").replace(",", "").strip()
            try:
                pct = float(pct_str)
            except Exception:
                pct = 0.0

            currencies.append(
                {
                    "name": name,
                    "symbol": symbol,
                    "price": round(price, 6),
                    "chg": round(chg, 6),
                    "pct": round(pct, 2),
                    "date": today,
                }
            )
        except Exception:
            # 单行出错时跳过，继续处理后续行
            continue

    if not currencies:
        raise RuntimeError("未能从人民币中间价表格中解析出有效数据行")

    return {"currencies": currencies}


def _fetch_tushare_fx() -> Dict[str, Any]:
    """
    函数名称：_fetch_tushare_fx
    函数功能：通过 TuShare 获取部分主要货币的行情数据。
    说明：
        TuShare 外汇接口 fx_daily 以 ts_code 标识货币对，如 USDCNY.FX。
        这里选取几种常见货币对，并计算涨跌额和涨跌幅。
        若获取失败，将在上层回退到 _build_mock_fx。
    返回值：
        Dict：包含 currencies 列表。
    """
    if ts is None:
        raise RuntimeError("当前环境未安装 tushare 库")
    if not TUSHARE_TOKEN:
        raise RuntimeError("未配置 TuShare Token")

    pro = ts.pro_api(TUSHARE_TOKEN)

    # TuShare 外汇代码示例：USDCNY.FX、USDJPY.FX 等（具体以 TuShare 文档为准）
    symbols = {
        "英镑": ("GBPUSD.FX", "GBP/USD"),
        "日元": ("USDJPY.FX", "USD/JPY"),
        "欧元": ("EURUSD.FX", "EUR/USD"),
    }

    end_date = datetime.today()
    start_date = end_date - timedelta(days=5)
    end_str = end_date.strftime("%Y%m%d")
    start_str = start_date.strftime("%Y%m%d")

    rows: List[Dict[str, Any]] = []

    for name, (ts_code, display) in symbols.items():
        try:
            df_fx = pro.fx_daily(ts_code=ts_code, start_date=start_str, end_date=end_str)
        except Exception as e:
            logger.warning(f"获取外汇 {ts_code} 失败: {e}")
            continue

        if df_fx.empty:
            logger.warning(f"外汇 {ts_code} 在区间 {start_str}-{end_str} 无数据")
            continue

        df_fx = df_fx.sort_values("trade_date")
        last = df_fx.iloc[-1]
        prev = df_fx.iloc[-2] if len(df_fx) > 1 else last
        price = float(last.get("close", last.get("rate", 0.0)))
        prev_price = float(prev.get("close", prev.get("rate", price)))
        chg = round(price - prev_price, 6)
        pct = round((price / prev_price - 1.0) * 100, 2) if prev_price else 0.0
        rows.append(
            {
                "name": name,
                "symbol": display,
                "price": round(price, 6),
                "chg": chg,
                "pct": pct,
                "date": last.get("trade_date", ""),
            }
        )

    if not rows:
        raise RuntimeError("未从 TuShare 获取到外汇数据")

    return {"currencies": rows}


def _fx_table_mapping() -> Dict[str, Dict[str, str]]:
    """
    函数名称：_fx_table_mapping
    函数功能：返回外汇货币对与 MySQL 表名/展示名称之间的映射。
    返回值：
        Dict：key 为简写货币对（如 USDCNY），value 为包含表名和展示名的字典。
    """
    return {
        # 直接使用表中存储的 USD/CNY 作为美元对人民币
        "USDCNY": {
            "mode": "direct",
            "table": "fx_usdcny",
            "name": "美元/人民币",
            "symbol": "USD/CNY",
        },
        # 通过交叉汇率，将 USD/CNY 和 USD/GBP 转成 GBP/CNY = (USD/CNY) / (USD/GBP)
        "GBPCNY": {
            "mode": "cross",
            "table": "fx_usdgbp",  # 存储的是 USD/GBP
            "name": "英镑/人民币",
            "symbol": "GBP/CNY",
        },
        # 通过交叉汇率，将 USD/CNY 和 USD/JPY 转成 JPY/CNY = (USD/CNY) / (USD/JPY)
        "JPYCNY": {
            "mode": "cross",
            "table": "fx_usdjpy",  # 存储的是 USD/JPY
            "name": "日元/人民币",
            "symbol": "JPY/CNY",
        },
    }


def _fetch_db_fx_overview() -> Dict[str, Any]:
    """
    函数名称：_fetch_db_fx_overview
    函数功能：从 MySQL 中读取各外汇表最近两个交易日的数据，计算最新价格和涨跌幅。
    返回值：
        Dict：包含 currencies 列表（不含比特币）。
    """
    mapping = _fx_table_mapping()
    conn = get_mysql_connection()
    cursor = conn.cursor()

    currencies: List[Dict[str, Any]] = []

    try:
        # 先取出 USD/CNY 的最近两天，作为交叉汇率的基础
        cursor.execute("SELECT date, close FROM `fx_usdcny` ORDER BY date DESC LIMIT 2")
        base_rows = cursor.fetchall()
        if not base_rows:
            raise RuntimeError("fx_usdcny 表中无数据")

        base_latest = base_rows[0]
        base_prev = base_rows[1] if len(base_rows) > 1 else base_rows[0]
        base_latest_price = float(base_latest["close"])
        base_prev_price = float(base_prev["close"]) if base_prev["close"] is not None else base_latest_price

        # 1) 美元/人民币：直接使用 fx_usdcny
        date_val = base_latest["date"]
        if isinstance(date_val, datetime):
            date_str = date_val.strftime("%Y-%m-%d")
        else:
            date_str = str(date_val)

        usd_info = mapping["USDCNY"]
        usd_chg = round(base_latest_price - base_prev_price, 6)
        usd_pct = round((base_latest_price / base_prev_price - 1.0) * 100, 2) if base_prev_price else 0.0

        currencies.append(
            {
                "name": usd_info["name"],
                "symbol": usd_info["symbol"],
                "price": round(base_latest_price, 6),
                "chg": usd_chg,
                "pct": usd_pct,
                "date": date_str,
                "pair": "USDCNY",
            }
        )

        # 2) 其他币种相对人民币，通过交叉汇率计算（GBP/CNY、JPY/CNY）
        for pair in ["GBPCNY", "JPYCNY"]:
            info = mapping[pair]
            table = info["table"]
            # 对应 fx_usdgbp / fx_usdjpy 的最近两天
            sql = f"SELECT date, close FROM `{table}` ORDER BY date DESC LIMIT 2"
            cursor.execute(sql)
            rows = cursor.fetchall()
            if not rows:
                continue

            cross_latest = rows[0]
            cross_prev = rows[1] if len(rows) > 1 else rows[0]
            cross_latest_price = float(cross_latest["close"])
            cross_prev_price = float(cross_prev["close"]) if cross_prev["close"] is not None else cross_latest_price

            # 使用交叉汇率：X/CNY = (USD/CNY) / (USD/X)
            latest_rate = base_latest_price / cross_latest_price if cross_latest_price else 0.0
            prev_rate = base_prev_price / cross_prev_price if cross_prev_price else latest_rate
            chg = round(latest_rate - prev_rate, 6)
            pct = round((latest_rate / prev_rate - 1.0) * 100, 2) if prev_rate else 0.0

            currencies.append(
                {
                    "name": info["name"],
                    "symbol": info["symbol"],
                    "price": round(latest_rate, 6),
                    "chg": chg,
                    "pct": pct,
                    "date": date_str,
                    "pair": pair,
                }
            )
    finally:
        conn.close()

    if not currencies:
        raise RuntimeError("数据库中未找到任何外汇数据")

    return {"currencies": currencies}


@app.route("/api/ts_fx", methods=["GET"])
@handle_api_errors
def api_ts_fx():
    """
    函数名称：api_ts_fx
    函数功能：返回若干主要货币的价格与涨跌幅（数据来源：本地 MySQL 外汇表）。
    说明：
        1. 优先从 MySQL 中已保存的 fx_usdcny / fx_usdgbp / fx_usdjpy 等表读取最近交易日数据；
        2. 若数据库中暂无数据，则回退到 TuShare 或示例数据。
    返回值：
        JSON：包含 currencies 列表。
    """
    try:
        data = _fetch_db_fx_overview()
    except Exception as e:
        logger.warning(f"从数据库读取外汇数据失败，尝试 TuShare: {e}")
        try:
            data = _fetch_tushare_fx()
        except Exception as e2:
            logger.warning(f"TuShare 外汇获取失败，使用示例数据: {e2}")
            data = _build_mock_fx()
    return jsonify({"status": "ok", **data})


@app.route("/api/fx_history", methods=["GET"])
@handle_api_errors
def api_fx_history():
    """
    函数名称：api_fx_history
    函数功能：根据指定外汇货币对，从 MySQL 中返回最近一段时间的“相对人民币”的收盘价序列，用于前端折线图展示。
    前端可传入参数（query string）：
        pair：货币对简写，如 USDCNY / GBPCNY / JPYCNY（默认 USDCNY）；
        days：向前追溯的天数，默认 30 天。
    返回值：
        JSON：包含 dates、closes 等信息。
    """
    pair = request.args.get("pair", "USDCNY").upper().replace("/", "")
    try:
        days = int(request.args.get("days", "30") or 30)
    except ValueError:
        days = 30
    days = max(1, min(days, 365))

    mapping = _fx_table_mapping()
    if pair not in mapping:
        return jsonify({"status": "error", "msg": f"不支持的货币对: {pair}"}), 400

    name = mapping[pair]["name"]
    symbol = mapping[pair]["symbol"]

    conn = get_mysql_connection()
    cursor = conn.cursor()

    start_date = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        if pair == "USDCNY":
            # 直接从 fx_usdcny 读取
            sql = "SELECT date, close FROM `fx_usdcny` WHERE date >= %s ORDER BY date ASC"
            cursor.execute(sql, (start_date,))
            rows = cursor.fetchall()
        elif pair in ("GBPCNY", "JPYCNY"):
            # 通过交叉汇率：X/CNY = (USD/CNY) / (USD/X)，按日期做内连接
            if pair == "GBPCNY":
                cross_table = "fx_usdgbp"
            else:
                cross_table = "fx_usdjpy"

            sql = f"""
                SELECT a.date AS date,
                       a.close AS usdcny_close,
                       b.close AS cross_close
                FROM `fx_usdcny` a
                JOIN `{cross_table}` b ON a.date = b.date
                WHERE a.date >= %s
                ORDER BY a.date ASC
            """
            cursor.execute(sql, (start_date,))
            rows = cursor.fetchall()
        else:
            rows = []
    finally:
        conn.close()

    if not rows:
        return jsonify({"status": "error", "msg": f"{pair} 在最近 {days} 天内无数据"}), 400

    dates: List[str] = []
    closes: List[float] = []

    if pair == "USDCNY":
        for r in rows:
            d = r["date"]
            if isinstance(d, datetime):
                dates.append(d.strftime("%Y-%m-%d"))
            else:
                dates.append(str(d))
            closes.append(float(r["close"]) if r["close"] is not None else 0.0)
    else:
        for r in rows:
            d = r["date"]
            if isinstance(d, datetime):
                dates.append(d.strftime("%Y-%m-%d"))
            else:
                dates.append(str(d))
            usdcny_close = float(r["usdcny_close"])
            cross_close = float(r["cross_close"])
            rate = usdcny_close / cross_close if cross_close else 0.0
            closes.append(rate)

    return jsonify(
        {
            "status": "ok",
            "pair": pair,
            "name": name,
            "symbol": symbol,
            "dates": dates,
            "closes": closes,
        }
    )


@app.route("/api/ts_stock_range", methods=["POST"])
@handle_api_errors
def api_ts_stock_range():
    """
    函数名称：api_ts_stock_range
    函数功能：查询单只股票在指定日期区间内的收盘价和累计涨幅。
    前端传入：
        ts_code：股票代码（TuShare ts_code，如 600519.SH）；
        start_date：开始日期（YYYYMMDD）；
        end_date：结束日期（YYYYMMDD）。
    返回值：
        JSON：包含 dates、closes、cum_change（累计涨幅百分比）。
    """
    if ts is None:
        return jsonify({"status": "error", "msg": "当前环境未安装 tushare 库"}), 500

    if not TUSHARE_TOKEN:
        return jsonify({"status": "error", "msg": "未配置 TuShare Token"}), 500

    ts_code = request.form.get("ts_code", "").strip()
    start_date = request.form.get("start_date", "").strip()
    end_date = request.form.get("end_date", "").strip()

    if not ts_code or not start_date or not end_date:
        return jsonify({"status": "error", "msg": "股票代码与起止日期均不能为空"}), 400

    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pd.DataFrame()
    max_attempts = 10
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if not df.empty:
                break
            logger.warning(f"股票 {ts_code} 第 {attempt} 次返回空数据，准备重试")
        except Exception as e:
            last_err = e
            logger.warning(f"拉取股票 {ts_code} 区间数据第 {attempt} 次失败: {e}")

        if attempt < max_attempts:
            time.sleep(0.6 * attempt)

    if df.empty and last_err is not None:
        logger.error(f"拉取股票 {ts_code} 区间数据多次失败，最后一次错误: {last_err}")
        return jsonify({"status": "error", "msg": f"拉取股票数据失败: {str(last_err)}"}), 500

    if df.empty:
        return jsonify({"status": "error", "msg": "指定区间内无交易数据"}), 400

    # 按日期升序
    df = df.sort_values("trade_date")
    dates = df["trade_date"].tolist()
    closes = df["close"].astype(float).round(3).tolist()

    base = closes[0]
    cum_change = [round((c / base - 1.0) * 100, 2) for c in closes]

    return jsonify(
        {
            "status": "ok",
            "ts_code": ts_code,
            "dates": dates,
            "closes": closes,
            "cum_change": cum_change,
        }
    )


@app.route("/api/run_ml", methods=["POST"])
@handle_api_errors
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
    table_name = request.form.get("table_name", "financial_series").strip() or "financial_series"
    feature_cols_str = request.form.get("feature_cols", "").strip()
    target_col = request.form.get("target_col", "").strip()

    if not feature_cols_str:
        return jsonify({"status": "error", "msg": "feature_cols 不能为空"}), 400

    feature_cols = [c.strip() for c in feature_cols_str.split(",") if c.strip()]
    if not feature_cols:
        return jsonify({"status": "error", "msg": "特征列名无效"}), 400

    logger.info(f"开始运行机器学习模型，表: {table_name}, 特征: {feature_cols}, 目标: {target_col}")

    try:
        df = load_from_mysql(table_name)  # 从 MySQL 读取数据
    except Exception as e:
        logger.error(f"从 MySQL 读取数据失败: {e}")
        return jsonify({"status": "error", "msg": f"读取数据库失败: {str(e)}"}), 500

    # 验证数据
    validation = validate_dataframe(df, min_rows=10, required_cols=feature_cols)
    if not validation["valid"]:
        return jsonify({"status": "error", "msg": validation["msg"]}), 400

    results: Dict[str, Dict] = {}

    # 1. 回归
    if target_col and target_col in df.columns:
        try:
            reg_res: MLResult = run_regression(df, feature_cols, target_col)
            results["regression"] = {
                "metrics": reg_res.metrics,
                "preview": reg_res.preview.to_dict(orient="records"),
            }
            logger.info(f"回归模型完成，R²: {reg_res.metrics.get('r2', 'N/A')}")
        except Exception as e:
            logger.error(f"回归模型失败: {e}")
            results["regression"] = {"error": str(e)}

        # 2. 分类（基于同一个连续目标）
        try:
            cls_res: MLResult = run_classification(df, feature_cols, target_col)
            results["classification"] = {
                "metrics": cls_res.metrics,
                "preview": cls_res.preview.to_dict(orient="records"),
            }
            logger.info(f"分类模型完成，准确率: {cls_res.metrics.get('accuracy', 'N/A')}")
        except Exception as e:
            logger.error(f"分类模型失败: {e}")
            results["classification"] = {"error": str(e)}
    else:
        logger.warning(f"目标列 {target_col} 不存在，跳过回归和分类")

    # 3. 聚类（只用特征列）
    try:
        clu_res: MLResult = run_clustering(df, feature_cols)
        results["clustering"] = {
            "metrics": clu_res.metrics,
            "preview": clu_res.preview.to_dict(orient="records"),
        }
        logger.info(f"聚类模型完成，簇数: {clu_res.metrics.get('n_clusters', 'N/A')}")
    except Exception as e:
        logger.error(f"聚类模型失败: {e}")
        results["clustering"] = {"error": str(e)}

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


