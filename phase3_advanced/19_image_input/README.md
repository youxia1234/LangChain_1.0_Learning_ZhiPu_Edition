# 模块 19：图像输入

## 🎯 学习目标

学习如何使用视觉模型（Vision Models）处理图像输入，实现多模态 AI 应用。

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
```

获取 API Key: https://open.bigmodel.cn/usercenter/apikeys

**注意**：本模块使用智谱 AI 的 `glm-4v` 视觉模型，需要单独的 API 密钥。

## 📚 核心概念

### 多模态支持

LangChain 1.0 原生支持多模态输入：
- **文本**：传统的文字输入
- **图像**：照片、截图、图表等
- **文件**：PDF、文档等

### 支持视觉的模型

| 模型 | 图像支持 | 特点 |
|------|----------|------|
| **glm-4v** | ✅ | 智谱 AI 视觉模型，中文支持优秀 |
| GPT-4o | ✅ | 强大的多模态理解 |
| GPT-4o-mini | ✅ | 性价比高 |
| Claude 3.5 | ✅ | 出色的图像理解 |
| Gemini Pro | ✅ | Google 的多模态模型 |

### 图像输入方式

```python
from langchain_core.messages import HumanMessage

# 方式 1：URL
message = HumanMessage(content=[
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])

# 方式 2：Base64 编码
import base64

with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = HumanMessage(content=[
    {"type": "text", "text": "这是什么？"},
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
])
```

## 🔑 关键 API

### 使用智谱 AI 视觉模型

```python
from langchain_openai import ChatOpenAI
import os

# 初始化智谱 AI 视觉模型
model = ChatOpenAI(
    model="glm-4v",  # 视觉模型
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 发送带图像的消息
response = model.invoke([message_with_image])
```

### 图像处理工具

```python
from langchain_core.tools import tool
import base64

@tool
def analyze_image(image_path: str) -> str:
    """分析图像并返回描述"""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode()
    
    message = HumanMessage(content=[
        {"type": "text", "text": "详细描述这张图片"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
    ])
    
    return model.invoke([message]).content
```

## 📝 本模块示例

1. **图像描述**：让模型描述图片内容
2. **图像问答**：基于图片回答问题
3. **OCR 文字识别**：从图像中提取文字
4. **图表分析**：理解图表数据

## ⚠️ 注意事项

1. 图像大小有限制，建议压缩大图片
2. Base64 编码会增加 payload 大小约 33%
3. 不同模型的图像理解能力差异较大
4. 注意 token 消耗，图像会消耗较多 token

---

## ❓ 常见问题

### Q1: Windows 上运行时 emoji 显示乱码怎么办？

**A:** 这是 Windows 终端 GBK 编码问题。在代码开头添加：

```python
import sys
import io

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

### Q2: 为什么使用智谱 AI 的 glm-4v 而不是 OpenAI？

**A:**

| 特性 | OpenAI GPT-4o | 智谱 AI glm-4v |
|-----|---------------|---------------|
| 费用 | 较高 | **有免费额度** |
| 中文支持 | 良好 | **优秀** |
| 视觉能力 | 强大 | **强大，中文场景更佳** |
| 国内网络 | 需代理 | **直接访问** |

### Q3: glm-4v 和 glm-4-flash 有什么区别？

**A:**

| 模型 | 用途 | 视觉支持 |
|------|------|----------|
| glm-4-flash | 通用文本模型 | ❌ 不支持 |
| glm-4v | 视觉多模态模型 | ✅ 支持 |
| glm-4-plus | 通用增强模型 | ⚠️ 部分支持 |

### Q4: 图片大小有限制吗？

**A:** 是的，建议：
- 压缩图片到 5MB 以下
- 分辨率建议 2048x2048 以内
- 使用 JPEG 格式可以减小文件大小
- Base64 编码后大小会增加约 33%

### Q5: 如何提高 OCR 识别准确率？

**A:** 几个技巧：
```python
# 1. 提高图片分辨率
# 2. 确保文字清晰可见
# 3. 使用明确的提示词
message = create_image_message(
    text="""请仔细识别图片中的所有文字，包括：
    - 标题和副标题
    - 正文内容
    - 图片中的标注或说明

    请按从上到下、从左到右的顺序列出所有文字。""",
    image_path=image_path
)
```
