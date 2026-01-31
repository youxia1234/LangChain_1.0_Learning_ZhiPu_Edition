# 模块 23：错误处理

## 🎯 学习目标

学习如何在 LangChain 应用中实现健壮的错误处理和恢复机制。

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
# 配置回退模型
primary_model = init_chat_model("openai:gpt-4o")
fallback_model = init_chat_model("openai:gpt-4o-mini")

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
