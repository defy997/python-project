"""
模块名称：db_config.py
模块功能：集中管理 MySQL 数据库连接参数，便于在一个地方修改配置。
说明：请根据自己电脑上的 MySQL 配置修改下面的参数。
"""

DB_CONFIG = {
    "host": "127.0.0.1",   # MySQL 服务器地址
    "port": 3306,          # 端口
    "user": "root",        # 用户名（请改成你自己的）
    "password": "264510",  # 密码（请改成你自己的）
    "database": "financial_analysis",  # 数据库名称（需要提前在 MySQL 中创建）
    "charset": "utf8mb4",  # 字符集
}


