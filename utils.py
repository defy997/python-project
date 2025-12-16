"""
模块名称：utils.py
作者信息：沈宏舟（示例），学号：2025000000
模块功能：提供项目通用的工具函数，包括错误处理装饰器、日志记录、数据验证等。
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict

from flask import jsonify

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def handle_api_errors(func: Callable) -> Callable:
    """
    函数名称：handle_api_errors
    函数功能：统一的 API 错误处理装饰器，捕获异常并返回友好的 JSON 错误响应。
    参数说明：
        func (Callable)：被装饰的函数。
    返回值：
        Callable：装饰后的函数。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"参数错误: {e}")
            return jsonify({"status": "error", "msg": f"参数错误: {str(e)}"}), 400
        except FileNotFoundError as e:
            logger.error(f"文件未找到: {e}")
            return jsonify({"status": "error", "msg": f"文件未找到: {str(e)}"}), 404
        except Exception as e:
            logger.exception(f"未预期的错误: {e}")
            return jsonify({"status": "error", "msg": f"服务器内部错误: {str(e)}"}), 500
    return wrapper


def validate_dataframe(df, min_rows: int = 1, required_cols: list = None) -> Dict[str, Any]:
    """
    函数名称：validate_dataframe
    函数功能：验证 DataFrame 是否符合基本要求。
    参数说明：
        df：待验证的 DataFrame。
        min_rows (int)：最小行数要求，默认 1。
        required_cols (list)：必需的列名列表，默认 None。
    返回值：
        Dict：包含验证结果的字典，{"valid": bool, "msg": str}。
    """
    import pandas as pd
    
    if not isinstance(df, pd.DataFrame):
        return {"valid": False, "msg": "数据不是有效的 DataFrame"}
    
    if df.empty:
        return {"valid": False, "msg": "数据表为空"}
    
    if len(df) < min_rows:
        return {"valid": False, "msg": f"数据行数不足，至少需要 {min_rows} 行"}
    
    if required_cols:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return {"valid": False, "msg": f"缺少必需的列: {', '.join(missing)}"}
    
    return {"valid": True, "msg": "验证通过"}


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    函数名称：safe_float
    函数功能：安全地将值转换为浮点数，转换失败时返回默认值。
    参数说明：
        value：待转换的值。
        default (float)：转换失败时的默认值。
    返回值：
        float：转换后的浮点数。
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def truncate_text(text: str, max_length: int = 200) -> str:
    """
    函数名称：truncate_text
    函数功能：截断文本到指定长度，超出部分用省略号表示。
    参数说明：
        text (str)：原始文本。
        max_length (int)：最大长度。
    返回值：
        str：截断后的文本。
    """
    if not isinstance(text, str):
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

