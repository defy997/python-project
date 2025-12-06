## 金融数据与文献分析一体化系统

本项目为课程实验设计的前后端交互金融分析小系统，基于 Flask + HTML/JavaScript + ECharts + Bootstrap 实现。

### 功能概述

- 通过前端表单选择金融分析类型（如股票收益分析、广告投放与销售分析等）。
- 支持三类数据来源：
  - 上传本地 CSV 金融数据文件；
  - 直接在前端文本框粘贴表格数据；
  - 选择示例内置数据集（广告投放与销售数据、股票交易数据）。
- 后端使用 Pandas/Numpy 对数据进行处理和统计分析。
- 借助 AI 大模型接口（预留函数）对金融文本进行摘要与情感分析。
- 前端使用 ECharts 展示不少于 5 类图形：
  - 折线图、柱状图、饼图、散点图、箱线图（可扩展雷达图等）。
- 分析结果可导出到多种文件：
  - CSV、JSON、TXT 文本报告，以及 Matplotlib 静态图像（PNG）。

### 运行环境

- Python 3.10+
- 建议使用虚拟环境：

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 启动项目

```bash
set FLASK_APP=app.py
flask run --reload
```

浏览器访问 `http://127.0.0.1:5000` 即可使用系统。

### 目录结构

- `app.py`：Flask 入口与路由控制。
- `analysis.py`：金融数据分析与图表数据生成模块。
- `ai_helper.py`：AI 大模型调用封装（可接入在线大模型或本地模型）。
- `static/`：前端静态资源（CSS、JS）。
- `templates/`：HTML 模板。
- `outputs/`：分析结果导出文件保存目录。


