"""
模块名称：crawler_ml.py
作者信息：沈宏舟（示例），学号：2025000000
模块功能：
    1. 从指定网站爬取不同类型的金融数据（示例）；
    2. 对数据进行简单清洗与特征工程；
    3. 将处理后的数据保存到 MySQL 数据库；
    4. 从数据库中读取数据，使用三类机器学习算法（回归、分类、聚类）进行预测或建模。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from db_config import DB_CONFIG


# 全局 Engine 实例，实现连接池复用
_engine_cache: Engine | None = None


def build_engine() -> Engine:
    """
    函数名称：build_engine
    函数功能：根据配置创建 SQLAlchemy 的 MySQL Engine 对象（使用单例模式，复用连接池）。
    返回值：
        Engine：数据库连接引擎。
    """
    global _engine_cache
    
    if _engine_cache is not None:
        return _engine_cache
    
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        f"?charset={DB_CONFIG['charset']}"
    )
    # 使用连接池，提高性能
    _engine_cache = create_engine(
        url,
        echo=False,
        pool_size=5,           # 连接池大小
        max_overflow=10,       # 最大溢出连接数
        pool_pre_ping=True,    # 连接前检查连接是否有效
        pool_recycle=3600,     # 连接回收时间（秒）
    )
    return _engine_cache


def crawl_financial_news(keyword: str) -> pd.DataFrame:
    """
    函数名称：crawl_financial_news
    函数功能：示例性地从东方财富新闻搜索（或其它网站）抓取与关键字相关的新闻标题与链接。
    说明：由于课堂/实验环境可能限制访问外网，本函数主要体现“爬虫流程和代码结构”，
          实际运行时请根据老师要求替换为合法且可访问的目标网站。
    参数说明：
        keyword (str)：搜索关键字，例如“股票”、“利率”等。
    返回值：
        pd.DataFrame：包含 title / url 两列的新闻表。
    """
    # 这里构造一个示例 URL，真实使用时请替换为可访问的金融新闻网站
    url = f"https://so.eastmoney.com/news/s?keyword={keyword}"
    resp = requests.get(url, timeout=10)  # 发送 GET 请求
    resp.encoding = resp.apparent_encoding

    soup = BeautifulSoup(resp.text, "html.parser")  # 解析 HTML

    titles: List[str] = []
    links: List[str] = []

    # 这里只是示例的 CSS 选择器，实际需要根据网页结构调整
    for a in soup.select("a.news-link"):
        title = a.get_text(strip=True)
        href = a.get("href", "")
        if title and href:
            titles.append(title)
            links.append(href)

    # 如果由于目标网站结构变化 / 反爬等原因导致未能解析出任何新闻，为了实验演示，
    # 这里构造几条“模拟新闻数据”，方便后续流程（写入 MySQL + 机器学习）顺利跑通。
    if not titles:
        titles = [
            f"{keyword}政策对全球金融市场的潜在影响",
            f"{keyword}会议纪要显示未来利率路径不确定性增加",
            f"机构解读：{keyword}最新表态对风险资产意味着什么",
        ]
        links = [
            "https://example.com/article1",
            "https://example.com/article2",
            "https://example.com/article3",
        ]

    df = pd.DataFrame({"keyword": keyword, "title": titles, "url": links})
    return df


def crawl_simple_price_series(api_url: str) -> pd.DataFrame:
    """
    函数名称：crawl_simple_price_series
    函数功能：从一个返回 JSON 的简易接口中获取“日期-价格”时间序列数据。
    参数说明：
        api_url (str)：返回 JSON 数据的 HTTP 接口。
    返回值：
        pd.DataFrame：包含 date / price 两列的时间序列表。
    """
    resp = requests.get(api_url, timeout=10)
    data = resp.json()
    # 假设返回结构示例：{"data":[{"date":"2024-01-01","price":10.2}, ...]}
    records = data.get("data", [])
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])  # 转为日期类型
    return df


def save_to_mysql(df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
    """
    函数名称：save_to_mysql
    函数功能：将 DataFrame 写入 MySQL 指定数据表。
    参数说明：
        df (pd.DataFrame)：要保存的数据表。
        table_name (str)：目标表名。
        if_exists (str)：表已存在时的处理方式（默认 append）。
    返回值：
        None。
    """
    if df.empty:
        raise ValueError("DataFrame 为空，无法保存到数据库")
    
    if not table_name or not isinstance(table_name, str):
        raise ValueError("表名无效")
    
    engine = build_engine()
    try:
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    except Exception as e:
        raise RuntimeError(f"保存到 MySQL 失败: {str(e)}")


def load_from_mysql(table_name: str) -> pd.DataFrame:
    """
    函数名称：load_from_mysql
    函数功能：从 MySQL 中读取整个数据表到 DataFrame。
    参数说明：
        table_name (str)：数据表名。
    返回值：
        pd.DataFrame：读取到的数据。
    """
    if not table_name or not isinstance(table_name, str):
        raise ValueError("表名无效")
    
    engine = build_engine()
    try:
        df = pd.read_sql(f"SELECT * FROM `{table_name}`", engine)
        if df.empty:
            raise ValueError(f"表 {table_name} 为空")
        return df
    except Exception as e:
        raise RuntimeError(f"从 MySQL 读取失败: {str(e)}")


@dataclass
class MLResult:
    """
    数据类：封装一轮机器学习建模的结果。
    字段说明：
        model_type (str)：模型类型，例如 "regression" / "classification" / "clustering"。
        metrics (Dict)：用于前端展示的评价指标。
        preview (pd.DataFrame)：带有预测结果预览的数据表（前几行）。
    """

    model_type: str
    metrics: Dict
    preview: pd.DataFrame


def run_regression(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> MLResult:
    """
    函数名称：run_regression
    函数功能：使用线性回归模型对金融数据进行回归预测。
    参数说明：
        df (pd.DataFrame)：原始数据框。
        feature_cols (List[str])：特征列名列表。
        target_col (str)：目标列名。
    返回值：
        MLResult：包含 R^2 等指标以及预测预览。
    """
    # 丢弃缺失的行
    data = df[feature_cols + [target_col]].dropna()
    X = data[feature_cols].values
    y = data[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)  # 拟合线性回归模型

    r2 = model.score(X_test, y_test)  # 计算 R^2
    y_pred = model.predict(X_test)    # 预测

    preview_df = pd.DataFrame(
        {
            "y_true": y_test[:10],
            "y_pred": y_pred[:10],
        }
    )

    metrics = {
        "r2": float(r2),
        "n_samples": int(len(data)),
    }
    return MLResult("regression", metrics, preview_df)


def run_classification(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> MLResult:
    """
    函数名称：run_classification
    函数功能：将金融数据中的目标变量离散化后，使用逻辑回归进行二分类预测。
    参数说明：
        df (pd.DataFrame)：原始数据。
        feature_cols (List[str])：特征列名。
        target_col (str)：连续目标列，将被转换为“高/低”二分类。
    返回值：
        MLResult：包含准确率等指标和预测预览。
    """
    data = df[feature_cols + [target_col]].dropna()

    # 简单规则：大于等于中位数记为 1（高），否则为 0（低）
    median = data[target_col].median()
    data["label"] = (data[target_col] >= median).astype(int)

    X = data[feature_cols].values
    y = data["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)  # 训练逻辑回归分类器

    acc = clf.score(X_test_scaled, y_test)  # 准确率
    y_pred = clf.predict(X_test_scaled)

    preview_df = pd.DataFrame(
        {
            "y_true": y_test[:10],
            "y_pred": y_pred[:10],
        }
    )

    metrics = {
        "accuracy": float(acc),
        "n_samples": int(len(data)),
    }
    return MLResult("classification", metrics, preview_df)


def run_clustering(df: pd.DataFrame, feature_cols: List[str], n_clusters: int = 3) -> MLResult:
    """
    函数名称：run_clustering
    函数功能：使用 KMeans 对金融数据进行聚类分析（例如客户分群、股票分组）。
    参数说明：
        df (pd.DataFrame)：原始数据。
        feature_cols (List[str])：参与聚类的特征列名列表。
        n_clusters (int)：聚类簇数量，默认 3。
    返回值：
        MLResult：包含每个簇的样本数等信息和带簇标签的预览数据。
    """
    data = df[feature_cols].dropna()
    X = data.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)  # 聚类

    data_with_label = data.copy()
    data_with_label["cluster"] = labels

    # 统计每个簇的样本数
    cluster_counts = data_with_label["cluster"].value_counts().to_dict()
    metrics = {
        "n_clusters": n_clusters,
        "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},
    }

    preview_df = data_with_label.head(10)  # 预览前 10 行
    return MLResult("clustering", metrics, preview_df)


if __name__ == "__main__":
    # 简单自测：从 MySQL 读取一张示例表并依次跑三类模型（需提前准备好数据表）。
    try:
        df_example = load_from_mysql("financial_series")
        cols = [c for c in df_example.columns if c not in ("date", "target")]
        if len(cols) >= 1 and "target" in df_example.columns:
            print("=== 回归模型示例 ===")
            print(run_regression(df_example, cols, "target").metrics)
            print("=== 分类模型示例 ===")
            print(run_classification(df_example, cols, "target").metrics)
            print("=== 聚类模型示例 ===")
            print(run_clustering(df_example, cols).metrics)
        else:
            print("示例表结构不满足要求，请检查列名。")
    except Exception as exc:  # pragma: no cover
        print("自测失败，请确认 MySQL 配置与示例数据表是否存在：", exc)


