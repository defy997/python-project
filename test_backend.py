"""
测试程序：简单测试 Flask 后端是否正常联通及主要接口是否可用
运行方式：python test_backend.py
"""

import json
import requests

BASE_URL = "http://127.0.0.1:5000"


def test_index():
    """测试主页是否可以正常访问"""
    url = f"{BASE_URL}/"
    resp = requests.get(url)
    print("GET / ->", resp.status_code)
    print("主页内容前 100 字节：", resp.text[:100], "\n")


def test_analyze_with_builtin():
    """测试 /analyze 接口（使用内置示例数据集）"""
    url = f"{BASE_URL}/analyze"
    data = {
        "dataset": "ads",  # 使用内置广告数据集
        "x_col": "",       # 留空让后端自动选择
        "y_col": "",
        "doc_text": "测试一下金融文本摘要与情感分析功能。",
    }
    resp = requests.post(url, data=data)
    print("POST /analyze ->", resp.status_code)
    try:
        j = resp.json()
        print("返回 JSON 字段：", list(j.keys()))
        print("summary:", j.get("summary", "")[:80], "...")
        print("ai_summary:", j.get("ai_summary", "")[:80], "...")
    except json.JSONDecodeError as e:
        print("解析 JSON 失败：", e)
        print("原始响应前 200 字符：", resp.text[:200])
    print()


def test_crawl_news():
    """测试 /api/crawl_news 接口（爬虫 + MySQL 写入）"""
    url = f"{BASE_URL}/api/crawl_news"
    data = {
        "keyword": "股票",       # 爬取关键字
        "table_name": "news_data_test",  # 测试表名
    }
    resp = requests.post(url, data=data)
    print("POST /api/crawl_news ->", resp.status_code)
    try:
        j = resp.json()
        print("返回 JSON：")
        print(json.dumps(j, ensure_ascii=False, indent=2)[:400], "...")
    except json.JSONDecodeError as e:
        print("解析 JSON 失败：", e)
        print("原始响应前 200 字符：", resp.text[:200])
    print()


if __name__ == "__main__":
    print("=== 测试主页 ===")
    test_index()
    print("=== 测试 /analyze 接口 ===")
    test_analyze_with_builtin()
    print("=== 测试 /api/crawl_news 接口 ===")
    test_crawl_news()