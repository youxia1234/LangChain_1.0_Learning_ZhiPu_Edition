# 模块 23：错误处理

## 🎯 学习目标

学习如何在 LangChain 应用中实现健壮的错误处理和恢复机制。

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
```

获取 API Key: https://open.bigmodel.cn/usercenter/apikeys

## 📚 核心概念

### 常见错误类型

| 错误类型 | 原因 | 处理策略 |
|----------|------|----------|
| RateLimitError | API 调用频率过高 | 指数退避重试 |
| AuthenticationError | API Key 无效 | 检查配置 |
| InvalidRequestError | 请求参数错误 | 验证输入 |
| TimeoutError | 响应超时 | 设置超时重试 |
| OutputParserError | 输出解析失败 | 提供默认值/重试 |

### 错误处理策略

1. **重试机制**：自动重试失败的请求
2. **回退策略**：失败时使用备用方案
3. **优雅降级**：部分功能不可用时继续运行
4. **错误边界**：隔离错误防止级联失败

## 🔑 关键 API

### 使用 with_retry

```python
from langchain_core.runnables import RunnableConfig

# 配置重试
model_with_retry = model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)
```

### 使用 with_fallbacks

```python
from langchain_openai import ChatOpenAI

# 配置回退模型
primary_model = ChatOpenAI(model="glm-4-flash", ...)
fallback_model = ChatOpenAI(model="glm-4-flash", ...)

robust_model = primary_model.with_fallbacks([fallback_model])
```

### 自定义错误处理

```python
from langchain_core.runnables import RunnableLambda

def safe_invoke(input_data):
    try:
        return model.invoke(input_data)
    except Exception as e:
        return f"Error: {e}"

safe_chain = RunnableLambda(safe_invoke)
```

## 📝 本模块示例

1. **重试机制**：实现指数退避重试
2. **模型回退**：主模型失败时切换备用
3. **输出验证**：验证和修复 LLM 输出
4. **全局错误处理**：统一的错误处理框架

## ⚠️ 最佳实践

1. 始终为生产代码添加错误处理
2. 记录错误日志便于排查
3. 设置合理的重试次数和超时
4. 向用户提供友好的错误信息

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
| 错误处理场景 | 良好 | **更适合中文错误消息** |
| 国内网络 | 需代理 | **直接访问** |

### Q3: 如何设置合适的重试次数和延迟？

**A:** 根据场景选择：

```python
# 开发环境：快速反馈
max_retries=2
base_delay=0.5

# 生产环境：更耐心
max_retries=5
base_delay=1.0

# 对时间敏感的操作
max_retries=3
base_delay=2.0

# 计算公式（指数退避）
delay = min(base_delay * (2 ** attempt), max_delay)
```

### Q4: 如何处理 JSON 解析错误？

**A:** 使用安全的 JSON 解析函数：

```python
import json
import re

def safe_parse_json(text: str, default: dict = None) -> dict:
    """安全地解析JSON文本"""
    if default is None:
        default = {}
    
    content = text.strip()
    
    # 移除 Markdown 代码块
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
    
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return default
```

### Q5: 如何记录错误日志？

**A:** 使用 Python 的 logging 模块：

```python
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app_errors.log'
)

logger = logging.getLogger(__name__)

try:
    result = model.invoke(prompt)
except Exception as e:
    logger.error(f"模型调用失败: {e}", exc_info=True)
    logger.error(f"输入: {prompt}")
    # 返回友好的错误消息
```
