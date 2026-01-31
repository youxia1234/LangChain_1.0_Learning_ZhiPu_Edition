# 模块 21：混合模态

## 🎯 学习目标

学习如何在单个工作流中结合文本、图像和结构化数据，构建真正的多模态 AI 应用。

## 📚 核心概念

### 什么是混合模态？

混合模态（Mixed Modality）是指在同一个 AI 工作流中处理多种类型的数据：
- **文本**：自然语言描述、提问、指令
- **图像**：照片、图表、截图
- **结构化数据**：JSON、表格、数据库记录
- **文档**：PDF、Word、Markdown

### 应用场景

1. **报告生成**：分析数据图表 + 生成文字报告
2. **内容审核**：同时检查文字和图片内容
3. **智能文档处理**：理解带图文的文档
4. **数据可视化解读**：分析图表并提取数据

## 🔑 关键模式

### 图文结合分析

```python
message = HumanMessage(content=[
    {"type": "text", "text": "分析这个销售报告"},
    {"type": "image_url", "image_url": {"url": chart_url}},
    {"type": "text", "text": f"相关数据：{json.dumps(sales_data)}"}
])
```

### 结构化输出

```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    key_findings: list[str]
    data_points: dict

agent = create_agent(
    model="openai:gpt-4o",
    response_format=AnalysisResult
)
```

## 📝 本模块示例

1. **图表+数据分析**：结合图表图像和原始数据
2. **文档理解**：处理带图文的文档
3. **多源数据融合**：整合多种来源的信息
