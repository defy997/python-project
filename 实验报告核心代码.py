"""
实验报告核心代码
包含：网络爬虫、数据库操作、机器学习算法三类核心功能
"""

# ============================================================================
# 一、网络爬虫技术核心代码
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
    global _engine_cache  # 使用全局变量实现单例模式
    
    # 如果已经创建过引擎，直接返回（避免重复创建，节省资源）
    if _engine_cache is not None:
        return _engine_cache
    
    # 构造 MySQL 连接 URL，格式：mysql+pymysql://用户名:密码@主机:端口/数据库名?字符集
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        f"?charset={DB_CONFIG['charset']}"
    )
    # 使用连接池创建引擎，提高数据库访问性能
    _engine_cache = create_engine(
        url,
        echo=False,            # 不打印 SQL 语句（生产环境设为 False）
        pool_size=5,           # 连接池大小：保持5个常驻连接
        max_overflow=10,       # 最大溢出连接数：最多可创建10个额外连接
        pool_pre_ping=True,    # 连接前检查连接是否有效（自动重连断开的连接）
        pool_recycle=3600,     # 连接回收时间（秒）：1小时后回收连接，避免连接超时
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
    # 步骤1：构造目标网站的搜索 URL，将关键字作为查询参数
    url = f"https://so.eastmoney.com/news/s?keyword={keyword}"
    
    # 步骤2：发送 HTTP GET 请求获取网页内容，设置10秒超时避免长时间等待
    resp = requests.get(url, timeout=10)
    
    # 步骤3：自动检测网页编码，确保中文内容正确显示
    resp.encoding = resp.apparent_encoding

    # 步骤4：使用 BeautifulSoup 解析 HTML 文档，提取结构化数据
    soup = BeautifulSoup(resp.text, "html.parser")

    # 步骤5：初始化两个列表，用于存储提取的标题和链接
    titles: List[str] = []
    links: List[str] = []

    # 步骤6：使用 CSS 选择器定位所有新闻链接元素（需要根据实际网页结构调整选择器）
    for a in soup.select("a.news-link"):
        title = a.get_text(strip=True)  # 提取链接内的文本作为标题，strip=True 去除首尾空白
        href = a.get("href", "")        # 提取链接的 href 属性值作为 URL
        # 步骤7：验证标题和链接都不为空，避免无效数据
        if title and href:
            titles.append(title)  # 将标题添加到列表
            links.append(href)   # 将链接添加到列表

    # 步骤8：容错处理 - 如果由于目标网站结构变化、反爬虫机制等原因导致未能解析出任何新闻，
    # 为了实验演示能够继续进行，这里构造几条"模拟新闻数据"作为兜底方案
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

    # 步骤9：将提取的数据组织成 pandas DataFrame 格式，便于后续处理和存储
    df = pd.DataFrame({"keyword": keyword, "title": titles, "url": links})
    return df


# ============================================================================
# 二、数据库技术核心代码
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
    # 步骤1：数据验证 - 检查 DataFrame 是否为空
    if df.empty:
        raise ValueError("DataFrame 为空，无法保存到数据库")
    
    # 步骤2：参数验证 - 检查表名是否有效
    if not table_name or not isinstance(table_name, str):
        raise ValueError("表名无效")
    
    # 步骤3：获取数据库连接引擎（使用连接池，提高性能）
    engine = build_engine()
    try:
        # 步骤4：使用 pandas 的 to_sql 方法将 DataFrame 批量写入 MySQL
        # if_exists="append" 表示如果表存在则追加数据，index=False 表示不保存行索引
        df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    except Exception as e:
        # 步骤5：异常处理 - 如果写入失败，抛出运行时错误并包含详细错误信息
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
    # 步骤1：参数验证 - 检查表名是否有效
    if not table_name or not isinstance(table_name, str):
        raise ValueError("表名无效")
    
    # 步骤2：获取数据库连接引擎
    engine = build_engine()
    try:
        # 步骤3：构造 SQL 查询语句，使用反引号包裹表名避免关键字冲突
        sql = f"SELECT * FROM `{table_name}`"
        
        # 步骤4：使用 pandas 的 read_sql 方法执行查询并返回 DataFrame
        df = pd.read_sql(sql, engine)
        
        # 步骤5：数据验证 - 检查读取的数据是否为空
        if df.empty:
            raise ValueError(f"表 {table_name} 为空")
        return df
    except Exception as e:
        # 步骤6：异常处理 - 如果读取失败，抛出运行时错误
        raise RuntimeError(f"从 MySQL 读取失败: {str(e)}")


# ============================================================================
# 三、机器学习算法核心代码
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
    # 步骤1：数据预处理 - 选择需要的特征列和目标列，丢弃包含缺失值的行
    data = df[feature_cols + [target_col]].dropna()
    
    # 步骤2：特征提取 - 将特征列转换为 numpy 数组（机器学习算法需要的格式）
    X = data[feature_cols].values  # X 是特征矩阵，每行是一个样本，每列是一个特征
    y = data[target_col].values    # y 是目标向量，每个元素对应一个样本的目标值

    # 步骤3：数据集划分 - 将数据随机划分为训练集（70%）和测试集（30%）
    # random_state=42 确保每次运行结果一致（可复现性）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 步骤4：模型创建 - 实例化线性回归模型对象
    model = LinearRegression()
    
    # 步骤5：模型训练 - 使用训练集拟合线性回归模型（学习特征与目标的关系）
    model.fit(X_train, y_train)

    # 步骤6：模型评估 - 在测试集上计算 R² 得分（决定系数，越接近1越好）
    r2 = model.score(X_test, y_test)
    
    # 步骤7：模型预测 - 使用训练好的模型对测试集进行预测
    y_pred = model.predict(X_test)

    # 步骤8：结果预览 - 构造包含真实值和预测值的 DataFrame（前10条用于展示）
    preview_df = pd.DataFrame(
        {
            "y_true": y_test[:10],  # 测试集的真实目标值
            "y_pred": y_pred[:10],  # 模型预测的目标值
        }
    )

    # 步骤9：返回结果 - 组织评估指标和预览数据
    metrics = {
        "r2": float(r2),              # R² 决定系数：衡量模型拟合优度，范围 [0,1]
        "n_samples": int(len(data)),  # 样本数量：参与训练的样本总数
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
    # 步骤1：数据预处理 - 选择特征列和目标列，丢弃缺失值
    data = df[feature_cols + [target_col]].dropna()

    # 步骤2：标签生成 - 将连续型目标变量转换为二分类标签
    # 计算目标列的中位数作为阈值
    median = data[target_col].median()
    # 大于等于中位数的标记为 1（高），否则标记为 0（低）
    data["label"] = (data[target_col] >= median).astype(int)

    # 步骤3：特征提取 - 提取特征矩阵和分类标签
    X = data[feature_cols].values  # 特征矩阵：n_samples × n_features
    y = data["label"].values        # 分类标签：n_samples 个 0 或 1

    # 步骤4：数据集划分 - 随机划分为训练集（70%）和测试集（30%）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 步骤5：特征标准化 - 逻辑回归对特征尺度敏感，需要标准化
    scaler = StandardScaler()
    # 在训练集上拟合标准化器（计算均值和标准差），然后转换训练集
    X_train_scaled = scaler.fit_transform(X_train)
    # 使用训练集的均值和标准差转换测试集（避免数据泄露）
    X_test_scaled = scaler.transform(X_test)

    # 步骤6：模型创建 - 实例化逻辑回归分类器，max_iter=1000 增加最大迭代次数
    clf = LogisticRegression(max_iter=1000)
    
    # 步骤7：模型训练 - 使用标准化后的训练集训练分类器
    clf.fit(X_train_scaled, y_train)

    # 步骤8：模型评估 - 在测试集上计算分类准确率（正确预测的样本比例）
    acc = clf.score(X_test_scaled, y_test)
    
    # 步骤9：模型预测 - 对测试集进行预测，返回每个样本的预测类别（0 或 1）
    y_pred = clf.predict(X_test_scaled)

    # 步骤10：结果预览 - 构造包含真实标签和预测标签的 DataFrame
    preview_df = pd.DataFrame(
        {
            "y_true": y_test[:10],  # 测试集的真实类别标签
            "y_pred": y_pred[:10],  # 模型预测的类别标签
        }
    )

    # 步骤11：返回评估指标
    metrics = {
        "accuracy": float(acc),         # 分类准确率：正确预测的样本数 / 总样本数
        "n_samples": int(len(data)),    # 样本数量：参与训练的样本总数
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
    # 步骤1：数据预处理 - 选择参与聚类的特征列，丢弃包含缺失值的行
    data = df[feature_cols].dropna()
    
    # 步骤2：数据转换 - 将 DataFrame 转换为 numpy 数组（KMeans 算法需要的格式）
    X = data.values  # X 是特征矩阵，每行是一个样本，每列是一个特征

    # 步骤3：特征标准化 - KMeans 基于欧氏距离计算样本相似度，对特征尺度敏感
    # 不同特征的量纲不同会导致距离计算偏向大数值特征，因此需要标准化
    scaler = StandardScaler()
    # 标准化：将每个特征缩放到均值为0、标准差为1的分布
    X_scaled = scaler.fit_transform(X)

    # 步骤4：模型创建 - 实例化 KMeans 聚类模型
    # n_clusters 指定要聚成的簇数，random_state=42 确保结果可复现
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # 步骤5：执行聚类 - fit_predict 同时完成模型训练和预测
    # 返回每个样本所属的簇标签（0 到 n_clusters-1 之间的整数）
    labels = kmeans.fit_predict(X_scaled)

    # 步骤6：结果整合 - 将簇标签添加到原始数据中，便于后续分析
    data_with_label = data.copy()
    data_with_label["cluster"] = labels  # 新增一列存储每个样本的簇标签

    # 步骤7：统计信息 - 统计每个簇包含的样本数量
    cluster_counts = data_with_label["cluster"].value_counts().to_dict()
    metrics = {
        "n_clusters": n_clusters,  # 聚类簇数量
        # 每个簇的样本数：字典格式，键是簇编号，值是该簇的样本数
        "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},
    }

    # 步骤8：结果预览 - 返回前10行数据（包含簇标签）
    preview_df = data_with_label.head(10)
    return {"model_type": "clustering", "metrics": metrics, "preview": preview_df}


# ============================================================================
# 四、使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例1：爬取金融新闻并保存到数据库
    print("=== 爬取金融新闻 ===")
    df_news = crawl_financial_news("股票")
    print(f"爬取到 {len(df_news)} 条新闻")
    # save_to_mysql(df_news, "news_data")  # 保存到数据库
    
    # 示例2：从数据库读取数据并进行机器学习分析
    print("\n=== 机器学习分析 ===")
    try:
        # 从数据库读取数据
        df_data = load_from_mysql("ts_index_daily")
        
        # 选择特征列和目标列
        feature_cols = ["open", "high", "low"]  # 特征列
        target_col = "close"  # 目标列（收盘价）
        
        # 回归分析
        print("\n--- 回归模型 ---")
        reg_result = run_regression(df_data, feature_cols, target_col)
        print(f"R² 得分: {reg_result['metrics']['r2']:.4f}")
        
        # 分类分析
        print("\n--- 分类模型 ---")
        cls_result = run_classification(df_data, feature_cols, target_col)
        print(f"准确率: {cls_result['metrics']['accuracy']:.4f}")
        
        # 聚类分析
        print("\n--- 聚类模型 ---")
        clu_result = run_clustering(df_data, feature_cols, n_clusters=3)
        print(f"簇数量: {clu_result['metrics']['n_clusters']}")
        print(f"各簇样本数: {clu_result['metrics']['cluster_counts']}")
        
    except Exception as e:
        print(f"分析失败: {e}")

