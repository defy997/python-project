"""
模块名称：ai_helper.py
作者信息：沈宏舟（示例），学号：2025000000
模块功能：封装与 AI 大模型交互的相关函数，用于金融文本的摘要和情感分析。

说明：
本模块中默认实现为“伪 AI 调用”（本地规则与简单模型），
方便在无外网或无真实大模型接口时完成课程实验。
如果需要，可在 `call_llm_api` 函数中接入真实大模型 HTTP 接口。
"""

from __future__ import annotations

from typing import Dict

import textwrap


def call_llm_api(prompt: str) -> str:
    """
    函数名称：call_llm_api
    函数功能：根据输入的提示词 prompt，调用（或模拟调用）大语言模型并返回生成文本。
    参数说明：
        prompt (str)：用户构造的提示词，包含金融问题或文献内容。
    返回值：
        str：模型生成的文本（此处为模拟结果）。
    """
    # 这里是“伪大模型”实现，仅用于课程实验演示。
    # 如需真实接入，可使用 requests.post 调用在线大模型 API。
    # 简单规则：根据关键词返回不同的固定回复。
    lowered = prompt.lower()  # 将提示词转换为小写，便于关键词判断

    if "risk" in lowered or "风险" in lowered:
        # 返回一段关于风险分析的示例文本
        return (
            "本段文本为模拟 AI 模型输出：\n"
            "综合历史收益率波动、最大回撤以及相关性等指标，可以看出该投资组合整体风险处于中等偏上水平。"
        )
    if "return" in lowered or "收益" in lowered:
        return (
            "本段文本为模拟 AI 模型输出：\n"
            "在观察期内，资产平均日收益率为 0.6%，夏普比率在同类产品中处于较好水平。"
        )

    # 默认返回通用回复
    wrapped = textwrap.fill(prompt[:200], width=36)  # 将部分输入文本换行，模拟“引用”
    return (
        "本段文本为模拟 AI 模型输出：\n"
        "系统已接收您的金融文本或问题，以下为自动生成的简要回应示例：\n"
        f"{wrapped}\n"
        "（在真实环境中，这里可以展示由在线大模型生成的摘要或分析结论。）"
    )


def summarize_financial_text(text: str) -> str:
    """
    函数名称：summarize_financial_text
    函数功能：对输入的金融文献或新闻文本进行摘要（模拟），返回简要总结。
    参数说明：
        text (str)：原始金融文本内容。
    返回值：
        str：AI 生成的摘要文本。
    """
    # 在提示词中包含清晰的指令
    prompt = f"请用中文对以下金融文本做简要总结，控制在 3~5 句话之内：\n{text}"
    # 调用上面的“伪大模型”接口
    summary = call_llm_api(prompt)
    return summary  # 返回摘要结果


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    函数名称：analyze_sentiment
    函数功能：对输入文本进行简单情感分析（模拟），输出正向、中性、负向情感的概率。
    参数说明：
        text (str)：需要分析情感倾向的文本。
    返回值：
        Dict[str, float]：包含 positive / neutral / negative 三个键的字典。
    """
    lowered = text.lower()  # 小写化便于关键字匹配

    # 初始化默认情感分布
    positive = 0.33
    neutral = 0.34
    negative = 0.33

    # 根据常见关键词做非常简单的规则判断
    if any(k in lowered for k in ["good", "增长", "盈利", "利好"]):
        positive, neutral, negative = 0.7, 0.2, 0.1
    elif any(k in lowered for k in ["down", "下跌", "亏损", "利空"]):
        positive, neutral, negative = 0.1, 0.2, 0.7

    # 保证概率和为 1
    total = positive + neutral + negative
    positive /= total
    neutral /= total
    negative /= total

    return {
        "positive": positive,  # 正向情感概率
        "neutral": neutral,    # 中性情感概率
        "negative": negative,  # 负向情感概率
    }


if __name__ == "__main__":
    # 模块自测代码：仅在直接运行本文件时触发，不影响被导入时的使用。
    sample_text = "本基金在过去一年实现了稳健增长，但短期内存在一定回撤风险。"
    print("=== 模拟摘要 ===")
    print(summarize_financial_text(sample_text))  # 打印模拟摘要结果
    print("=== 模拟情感分析 ===")
    print(analyze_sentiment(sample_text))         # 打印模拟情感分布


