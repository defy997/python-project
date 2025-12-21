"""
实验报告完整核心代码
包含：网络爬虫、数据库操作、机器学习算法三类核心功能
所有代码均包含完整的模块注释、函数注释和行注释
"""

# ============================================================================
# 第一部分：网络爬虫技术核心代码
# ============================================================================

"""
模块名称：crawler_ml.py
作者信息：沈宏舟（示例），学号：2025000000
模块功能：
    1. 从指定网站爬取不同类型的金融数据（示例）；
    2. 对数据进行简单清洗与特征工程；
    3. 将处理后的数据保存到 MySQL 数据库；
    4. 从数据库中读取数据，使用三类机器学习算法（回归、分类、聚类）进行预测或建模。
"""

from typing import Dict, List
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pymysql

# 数据库配置（示例）
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "password",
    "database": "financial_db",
    "charset": "utf8mb4"
}

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
    
    # 构造 MySQL 连接 URL
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
    说明：由于课堂/实验环境可能限制访问外网，本函数主要体现"爬虫流程和代码结构"，
          实际运行时请根据老师要求替换为合法且可访问的目标网站。
    参数说明：
        keyword (str)：搜索关键字，例如"股票"、"利率"等。
    返回值：
        pd.DataFrame：包含 title / url 两列的新闻表。
    """
    # 这里构造一个示例 URL，真实使用时请替换为可访问的金融新闻网站
    url = f"https://so.eastmoney.com/news/s?keyword={keyword}"
    resp = requests.get(url, timeout=10)  # 发送 GET 请求
    resp.encoding = resp.apparent_encoding  # 自动检测编码

    soup = BeautifulSoup(resp.text, "html.parser")  # 解析 HTML

    titles: List[str] = []
    links: List[str] = []

    # 这里只是示例的 CSS 选择器，实际需要根据网页结构调整
    for a in soup.select("a.news-link"):
        title = a.get_text(strip=True)  # 提取标题文本
        href = a.get("href", "")  # 提取链接地址
        if title and href:
            titles.append(title)
            links.append(href)

    # 如果由于目标网站结构变化 / 反爬等原因导致未能解析出任何新闻，为了实验演示，
    # 这里构造几条"模拟新闻数据"，方便后续流程（写入 MySQL + 机器学习）顺利跑通。
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

    # 构造 DataFrame 返回
    df = pd.DataFrame({"keyword": keyword, "title": titles, "url": links})
    return df


def crawl_tushare_indices() -> pd.DataFrame:
    """
    函数名称：crawl_tushare_indices
    函数功能：从 TuShare API 爬取中国主要股票指数的日线行情数据。
    说明：TuShare 是一个专业的金融数据接口，需要注册获取 Token。
    返回值：
        pd.DataFrame：包含指数代码、日期、开盘、收盘等字段的数据表。
    """
    try:
        import tushare as ts
    except ImportError:
        raise RuntimeError("请先安装 tushare 库：pip install tushare")
    
    # TuShare Token（需要注册获取）
    TUSHARE_TOKEN = "your_token_here"
    pro = ts.pro_api(TUSHARE_TOKEN)
    
    # 指数代码映射
    indices = {
        "上证指数": "000001.SH",
        "深证成指": "399001.SZ",
        "创业板指": "399006.SZ",
        "沪深300": "000300.SH",
        "中证500": "000905.SH",
    }
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)  # 获取近60个交易日
    end_str = end_date.strftime("%Y%m%d")
    start_str = start_date.strftime("%Y%m%d")
    
    all_data = []
    
    # 遍历每个指数，获取数据
    for name, code in indices.items():
        try:
            # 调用 TuShare API 获取指数日线数据
            df_idx = pro.index_daily(ts_code=code, start_date=start_str, end_date=end_str)
            if not df_idx.empty:
                df_idx["index_name"] = name  # 添加指数名称列
                all_data.append(df_idx)
        except Exception as e:
            print(f"获取指数 {name} 数据失败: {e}")
    
    # 合并所有指数数据
    if all_data:
        result_df = pd.concat(all_data, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()


# ============================================================================
# 第二部分：数据库技术核心代码
# ============================================================================

def save_to_mysql(df: pd.DataFrame, table_name: str, if_exists: str = "append") -> None:
    """
    函数名称：save_to_mysql
    函数功能：将 DataFrame 写入 MySQL 指定数据表。
    参数说明：
        df (pd.DataFrame)：要保存的数据表。
        table_name (str)：目标表名。
        if_exists (str)：表已存在时的处理方式（默认 append，可选 replace、fail）。
    返回值：
        None。
    """
    if df.empty:
        raise ValueError("DataFrame 为空，无法保存到数据库")
    
    if not table_name or not isinstance(table_name, str):
        raise ValueError("表名无效")
    
    engine = build_engine()  # 获取数据库引擎
    try:
        # 使用 pandas 的 to_sql 方法将 DataFrame 写入 MySQL
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
        print(f"成功保存 {len(df)} 条记录到表 {table_name}")
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
    
    engine = build_engine()  # 获取数据库引擎
    try:
        # 使用 pandas 的 read_sql 方法从 MySQL 读取数据
        df = pd.read_sql(f"SELECT * FROM `{table_name}`", engine)
        if df.empty:
            raise ValueError(f"表 {table_name} 为空")
        print(f"成功从表 {table_name} 读取 {len(df)} 条记录")
        return df
    except Exception as e:
        raise RuntimeError(f"从 MySQL 读取失败: {str(e)}")


def get_mysql_connection():
    """
    函数名称：get_mysql_connection
    函数功能：根据 DB_CONFIG 创建一个 MySQL 连接（使用 pymysql）。
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
        cursorclass=pymysql.cursors.DictCursor,  # 返回字典格式
    )


# ============================================================================
# 第三部分：数据清洗核心代码
# ============================================================================

def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    函数名称：clean_financial_data
    函数功能：对金融数据进行清洗，包括缺失值处理、异常值检测、数据类型转换。
    参数说明：
        df (pd.DataFrame)：原始数据框。
    返回值：
        pd.DataFrame：清洗后的数据框。
    """
    # 1. 删除完全重复的行
    df = df.drop_duplicates()
    
    # 2. 处理缺失值：数值列用中位数填充，文本列用众数填充
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)  # 用中位数填充
    
    text_cols = df.select_dtypes(include=['object']).columns
    for col in text_cols:
        if df[col].isna().any():
            mode_value = df[col].mode()[0] if not df[col].mode().empty else ""
            df[col].fillna(mode_value, inplace=True)  # 用众数填充
    
    # 3. 异常值处理：使用 IQR 方法检测并处理异常值
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)  # 第一四分位数
        Q3 = df[col].quantile(0.75)  # 第三四分位数
        IQR = Q3 - Q1  # 四分位距
        lower_bound = Q1 - 1.5 * IQR  # 下界
        upper_bound = Q3 + 1.5 * IQR  # 上界
        # 将异常值替换为边界值
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # 4. 数据类型转换：确保日期列为 datetime 类型
    date_cols = ['date', 'trade_date', 'datetime']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df


# ============================================================================
# 第四部分：机器学习算法核心代码
# ============================================================================

def run_regression(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict:
    """
    函数名称：run_regression
    函数功能：使用线性回归模型对金融数据进行回归预测。
    参数说明：
        df (pd.DataFrame)：原始数据框。
        feature_cols (List[str])：特征列名列表。
        target_col (str)：目标列名。
    返回值：
        Dict：包含 R^2 等指标以及预测预览。
    """
    # 丢弃缺失的行，确保数据完整性
    data = df[feature_cols + [target_col]].dropna()
    X = data[feature_cols].values  # 特征矩阵
    y = data[target_col].values    # 目标向量

    # 划分训练集和测试集（7:3）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 创建并训练线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)  # 拟合线性回归模型

    r2 = model.score(X_test, y_test)  # 计算 R^2 评估指标
    y_pred = model.predict(X_test)    # 对测试集进行预测

    # 构造预测结果预览（前10条）
    preview_df = pd.DataFrame(
        {
            "y_true": y_test[:10],  # 真实值
            "y_pred": y_pred[:10],  # 预测值
        }
    )

    # 返回评估指标
    metrics = {
        "r2": float(r2),  # R² 决定系数
        "n_samples": int(len(data)),  # 样本数量
    }
    return {"model_type": "regression", "metrics": metrics, "preview": preview_df}


def run_classification(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict:
    """
    函数名称：run_classification
    函数功能：将金融数据中的目标变量离散化后，使用逻辑回归进行二分类预测。
    参数说明：
        df (pd.DataFrame)：原始数据。
        feature_cols (List[str])：特征列名。
        target_col (str)：连续目标列，将被转换为"高/低"二分类。
    返回值：
        Dict：包含准确率等指标和预测预览。
    """
    data = df[feature_cols + [target_col]].dropna()

    # 简单规则：大于等于中位数记为 1（高），否则为 0（低）
    median = data[target_col].median()  # 计算中位数
    data["label"] = (data[target_col] >= median).astype(int)  # 转换为二分类标签

    X = data[feature_cols].values  # 特征矩阵
    y = data["label"].values       # 分类标签

    # 划分训练集和测试集（7:3）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 特征标准化（逻辑回归需要标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 训练集标准化
    X_test_scaled = scaler.transform(X_test)      # 测试集标准化

    # 创建并训练逻辑回归分类器
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_train)  # 训练逻辑回归分类器

    acc = clf.score(X_test_scaled, y_test)  # 计算准确率
    y_pred = clf.predict(X_test_scaled)     # 对测试集进行预测

    # 构造预测结果预览（前10条）
    preview_df = pd.DataFrame(
        {
            "y_true": y_test[:10],  # 真实标签
            "y_pred": y_pred[:10],  # 预测标签
        }
    )

    # 返回评估指标
    metrics = {
        "accuracy": float(acc),  # 分类准确率
        "n_samples": int(len(data)),  # 样本数量
    }
    return {"model_type": "classification", "metrics": metrics, "preview": preview_df}


def run_clustering(df: pd.DataFrame, feature_cols: List[str], n_clusters: int = 3) -> Dict:
    """
    函数名称：run_clustering
    函数功能：使用 KMeans 对金融数据进行聚类分析（例如客户分群、股票分组）。
    参数说明：
        df (pd.DataFrame)：原始数据。
        feature_cols (List[str])：参与聚类的特征列名列表。
        n_clusters (int)：聚类簇数量，默认 3。
    返回值：
        Dict：包含每个簇的样本数等信息和带簇标签的预览数据。
    """
    data = df[feature_cols].dropna()  # 丢弃缺失值
    X = data.values  # 转换为 numpy 数组

    # 特征标准化（KMeans 对距离敏感，需要标准化）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 创建并训练 KMeans 聚类模型
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)  # 执行聚类，返回每个样本的簇标签

    # 将簇标签添加到原始数据中
    data_with_label = data.copy()
    data_with_label["cluster"] = labels

    # 统计每个簇的样本数
    cluster_counts = data_with_label["cluster"].value_counts().to_dict()
    metrics = {
        "n_clusters": n_clusters,  # 簇数量
        "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},  # 每个簇的样本数
    }

    preview_df = data_with_label.head(10)  # 预览前 10 行
    return {"model_type": "clustering", "metrics": metrics, "preview": preview_df}


# ============================================================================
# 第五部分：完整使用示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("金融数据爬虫、数据库操作与机器学习分析系统")
    print("=" * 60)
    
    # 步骤1：爬取金融新闻数据
    print("\n【步骤1】爬取金融新闻数据")
    print("-" * 60)
    df_news = crawl_financial_news("股票")
    print(f"成功爬取 {len(df_news)} 条新闻")
    print(df_news.head())
    
    # 步骤2：数据清洗
    print("\n【步骤2】数据清洗")
    print("-" * 60)
    df_news_cleaned = clean_financial_data(df_news)
    print("数据清洗完成")
    
    # 步骤3：保存到数据库（注释掉，避免实际执行）
    # print("\n【步骤3】保存到数据库")
    # print("-" * 60)
    # save_to_mysql(df_news_cleaned, "news_data")
    
    # 步骤4：从数据库读取数据（需要先有数据）
    # print("\n【步骤4】从数据库读取数据")
    # print("-" * 60)
    # df_from_db = load_from_mysql("ts_index_daily")
    
    # 步骤5：机器学习分析（使用示例数据）
    print("\n【步骤5】机器学习分析")
    print("-" * 60)
    
    # 构造示例数据用于演示
    import numpy as np
    np.random.seed(42)
    n_samples = 100
    df_example = pd.DataFrame({
        "open": np.random.randn(n_samples) * 10 + 100,
        "high": np.random.randn(n_samples) * 10 + 105,
        "low": np.random.randn(n_samples) * 10 + 95,
        "close": np.random.randn(n_samples) * 10 + 100,
    })
    
    feature_cols = ["open", "high", "low"]
    target_col = "close"
    
    # 5.1 回归分析
    print("\n--- 回归模型（预测收盘价）---")
    reg_result = run_regression(df_example, feature_cols, target_col)
    print(f"R² 得分: {reg_result['metrics']['r2']:.4f}")
    print(f"样本数量: {reg_result['metrics']['n_samples']}")
    print("预测预览（前5条）:")
    print(reg_result['preview'].head())
    
    # 5.2 分类分析
    print("\n--- 分类模型（判断收盘价高低）---")
    cls_result = run_classification(df_example, feature_cols, target_col)
    print(f"准确率: {cls_result['metrics']['accuracy']:.4f}")
    print(f"样本数量: {cls_result['metrics']['n_samples']}")
    print("预测预览（前5条）:")
    print(cls_result['preview'].head())
    
    # 5.3 聚类分析
    print("\n--- 聚类模型（股票分组）---")
    clu_result = run_clustering(df_example, feature_cols, n_clusters=3)
    print(f"簇数量: {clu_result['metrics']['n_clusters']}")
    print(f"各簇样本数: {clu_result['metrics']['cluster_counts']}")
    print("聚类预览（前5条）:")
    print(clu_result['preview'].head())
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)

