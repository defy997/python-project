"""
测试程序：检查后端是否能正常连接到 MySQL，并完成简单的写入与读取。

运行方式：
    1）确认已在 MySQL 中创建好 db_config.py 里指定的数据库（例如 finance_db）
    2）python test_db_connection.py
"""

from sqlalchemy import text
from crawler_ml import build_engine  # 复用项目里的数据库引擎构建函数


def test_db_connection():
    """测试数据库连接、建表、插入、查询是否正常"""
    try:
        engine = build_engine()
        print("成功创建 Engine，对应连接字符串：", engine.url)

        with engine.connect() as conn:
            # 1. 测试简单 SELECT 1
            result = conn.execute(text("SELECT 1"))
            print("SELECT 1 结果：", list(result))

            # 2. 创建一个简单测试表（如已存在则先删除再建）
            conn.execute(text("DROP TABLE IF EXISTS db_connect_test"))
            conn.execute(
                text(
                    """
                    CREATE TABLE db_connect_test (
                        id INT PRIMARY KEY AUTO_INCREMENT,
                        msg VARCHAR(100)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
            )
            print("已创建测试表 db_connect_test")

            # 3. 插入一条测试数据
            conn.execute(
                text("INSERT INTO db_connect_test (msg) VALUES (:m)"),
                {"m": "hello mysql from python"},
            )
            conn.commit()
            print("已插入一条测试数据")

            # 4. 查询刚插入的数据
            result = conn.execute(
                text("SELECT id, msg FROM db_connect_test ORDER BY id DESC LIMIT 1")
            )
            row = result.fetchone()
            print("查询到的数据：", row)

        print("数据库读写测试 **成功** ✅")

    except Exception as e:
        print("数据库连接或读写测试 **失败** ❌")
        print("错误信息：", e)


if __name__ == "__main__":
    test_db_connection()