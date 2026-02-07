# 模块 21：混合模态

## 🎯 学习目标

学习如何在单个工作流中结合文本、图像和结构化数据，构建真正的多模态 AI 应用。

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
```

获取 API Key: https://open.bigmodel.cn/usercenter/apikeys

**注意**：本模块使用智谱 AI 的 `glm-4v` 视觉模型，需要单独的 API 密钥。

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
    model="glm-4v",  # 智谱 AI 视觉模型
    response_format=AnalysisResult
)
```

### 智谱 AI 模型选择

| 模型 | 视觉支持 | 适用场景 |
|------|----------|----------|
| glm-4-flash | ❌ | 通用文本对话 |
| **glm-4v** | ✅ | 图像理解、多模态处理 |
| glm-4-plus | ⚠️ 部分支持 | 高级多模态任务 |

## 📝 本模块示例

1. **图表+数据分析**：结合图表图像和原始数据
2. **文档理解**：处理带图文的文档
3. **多源数据融合**：整合多种来源的信息

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

### Q3: 混合模态消息的构建顺序有讲究吗？

**A:** 建议的顺序：

```python
content = [
    # 1. 先发送上下文说明
    {"type": "text", "text": "我将分析以下图表..."},

    # 2. 然后发送图像
    {"type": "image_url", "image_url": {"url": "..."}},

    # 3. 最后发送具体问题
    {"type": "text", "text": "请告诉我..."}

    # 多张图片时，建议按逻辑顺序排列
]
```

### Q4: 如何提高多图像分析的效果？

**A:** 几个技巧：

```python
# 1. 为每张图片添加说明
content = [
    {"type": "text", "text": "图片1是去年数据，图片2是今年数据"},
    create_image_content("last_year.jpg"),
    create_image_content("this_year.jpg"),
    {"type": "text", "text": "请对比两张图片的差异"}
]

# 2. 限制单次处理的图片数量（建议不超过 3-4 张）

# 3. 对于复杂对比，可以使用多轮对话
```

### Q5: 如何处理超大图片？

**A:** 图片过大会导致处理失败或响应缓慢：

```python
from PIL import Image
import io

def resize_image_if_needed(image_path: str, max_size: int = 2048) -> str:
    """如果图片过大则压缩"""
    img = Image.open(image_path)

    # 计算缩放比例
    ratio = min(max_size / img.width, max_size / img.height)

    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

        # 保存到临时文件
        temp_path = "resized_image.jpg"
        img.save(temp_path, "JPEG", quality=85)
        return temp_path

    return image_path
```
