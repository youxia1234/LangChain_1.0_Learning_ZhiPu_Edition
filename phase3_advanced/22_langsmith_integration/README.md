# 模块 22：LangSmith 集成

## 🎯 学习目标

学习如何使用 LangSmith 进行 LLM 应用的追踪、监控和调试。

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # 可选，用于追踪功能
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-study
```

获取 API Keys:
- Zhipu AI: https://open.bigmodel.cn/usercenter/apikeys
- LangSmith: https://smith.langchain.com

## 📚 核心概念

### 什么是 LangSmith？

LangSmith 是 LangChain 官方提供的**可观测性平台**，用于：
- 🔍 **追踪**：记录每次 LLM 调用的详细信息
- 📊 **监控**：实时查看应用性能
- 🐛 **调试**：排查问题和优化性能
- 📈 **评估**：系统化测试 LLM 应用

### 核心功能

| 功能 | 描述 |
|------|------|
| Traces | 记录完整的执行链路 |
| Runs | 单次 LLM 调用的详细记录 |
| Feedback | 用户反馈收集 |
| Datasets | 测试数据集管理 |
| Evaluation | 自动化评估 |

## 🔑 配置步骤

### 1. 获取 API Key

1. 访问 [smith.langchain.com](https://smith.langchain.com)
2. 创建账号并获取 API Key
3. 配置环境变量

### 2. 环境变量配置

```bash
# .env 文件
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
LANGSMITH_API_KEY=your_api_key_here  # 可选
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-project-name
```

### 3. 代码配置

```python
import os
from langchain_openai import ChatOpenAI

# 初始化智谱 AI 模型
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 启用 LangSmith 追踪
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-project"

# LangChain 会自动发送追踪数据
response = model.invoke("Hello!")
```

## 🔧 追踪示例

### 自动追踪

```python
from langchain_openai import ChatOpenAI

# 启用追踪后，所有调用自动记录
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
response = model.invoke("Hello!")
# -> 自动发送到 LangSmith
```

### 手动标记

```python
from langsmith import traceable

@traceable(name="my_function", tags=["production"])
def my_custom_function(input_data):
    # 你的代码
    return result
```

### 添加元数据

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    metadata={
        "user_id": "user_123",
        "session_id": "sess_456"
    },
    tags=["production", "v2"]
)

response = model.invoke("Hello!", config=config)
```

## 📝 本模块示例

1. **基本追踪**：自动记录 LLM 调用
2. **自定义追踪**：添加自定义元数据和标签
3. **性能监控**：记录延迟和 token 使用
4. **错误追踪**：捕获和记录错误

## ⚠️ 注意事项

1. 生产环境记得设置合适的采样率
2. 敏感数据可能需要脱敏处理
3. 注意 API 调用配额
4. 建议为不同环境使用不同项目

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

### Q2: 为什么使用智谱 AI 而不是 Groq？

**A:**

| 特性 | Groq | 智谱 AI |
|-----|------|---------|
| 费用 | 完全免费 | 有免费额度 |
| 速度 | 极快 | 快 |
| 中文支持 | 一般 | **优秀** |
| LangSmith 追踪 | 支持 | **完全支持** |
| 国内网络 | 需代理 | **直接访问** |

### Q3: LangSmith 是免费的吗？

**A:** LangSmith 有免费额度：
- **免费层**：每月一定数量的追踪记录
- **付费层**：更高的配额和更多功能

对于学习和开发，免费层通常足够使用。

### Q4: 如何设置追踪采样率？

**A:** 在生产环境中，为了避免过多的追踪数据，可以设置采样率：

```python
import os

# 采样率：0.0 到 1.0 之间
# 1.0 = 追踪所有请求
# 0.1 = 只追踪 10% 的请求
os.environ["LANGCHAIN_TRACING_SAMPLING_RATE"] = "0.1"
```

### Q5: 如何脱敏敏感数据？

**A:** 使用 LangSmith 的数据脱敏功能或手动处理：

```python
# 方法 1：环境变量配置
os.environ["LANGSMITH_HIDE_INPUTS"] = "true"  # 隐藏输入
os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"  # 隐藏输出

# 方法 2：手动脱敏
def sanitize_input(text: str) -> str:
    """移除敏感信息"""
    import re
    # 移除邮箱
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    # 移除手机号
    text = re.sub(r'1[3-9]\d{9}', '[PHONE]', text)
    return text

# 使用脱敏后的输入
clean_input = sanitize_input(user_input)
response = model.invoke(clean_input)
```
