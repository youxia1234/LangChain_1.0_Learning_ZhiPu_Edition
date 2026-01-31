# 08 - Context Management (上下文管理)

## 核心概念

**问题**：对话历史会无限增长 → 超 token、成本高、响应慢

**解决**：使用中间件自动管理上下文长度

## SummarizationMiddleware（推荐）

### 基本用法

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.3-70b-versatile",
            max_tokens_before_summary=500  # 超过 500 tokens 触发摘要
        )
    ]
)
```

### 工作原理

```
对话历史: [消息1, 消息2, ..., 消息20]  (超过 500 tokens)
    ↓
SummarizationMiddleware 自动触发
    ↓
摘要旧消息: "用户是张三，在北京工作，喜欢编程..."
    ↓
新历史: [摘要, 最近消息]  (减少到 300 tokens)
```

### 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `model` | 生成摘要的模型（可用便宜模型） | 必需 |
| `max_tokens_before_summary` | 触发摘要的 token 阈值 | 1000 |

## trim_messages（手动修剪）

### 基本用法

```python
from langchain_core.messages import trim_messages

# 只保留最近 N 条消息
trimmed = trim_messages(
    messages,
    max_tokens=100,
    strategy="last",  # 保留最后的
    token_counter=len
)
```

### 适用场景

- 只需要最近几轮对话
- 不需要保留旧信息
- 简单直接

## 策略对比

| 策略 | 优点 | 缺点 | 适用 |
|-----|------|------|------|
| **不处理** | 完整历史 | 超 token | 短对话 |
| **SummarizationMiddleware** | 自动化、保留信息 | 摘要成本 | 长对话（推荐）|
| **trim_messages** | 简单、精确 | 丢失旧信息 | 只要最近 N 轮 |

## 实际应用

### 客服机器人

```python
agent = create_agent(
    model=model,
    tools=[查询订单, 查询物流],
    system_prompt="客服助手",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.3-70b-versatile",
            max_tokens_before_summary=800  # 适合客服
        )
    ]
)
```

### 长期对话助手

```python
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.3-70b-versatile",
            max_tokens_before_summary=1000
        )
    ],
    checkpointer=InMemorySaver()
)
```

## 常见问题

### 1. 摘要会丢失信息吗？

会有一些细节丢失，但：
- 重要信息会保留（姓名、关键事实）
- 最近的消息完整保留
- 对于大部分场景足够

### 2. 如何选择 max_tokens_before_summary？

```python
# 模型上下文窗口 4k → 设置 3000
# 模型上下文窗口 8k → 设置 6000
# 模型上下文窗口 16k → 设置 12000

# 留一些余量给工具调用和系统提示
```

### 3. 摘要成本高吗？

- 摘要只在超过阈值时触发
- 可以使用便宜的模型（如 gpt-3.5）
- 相比传输全部历史，通常更便宜

### 4. 能自定义摘要提示词吗？

目前 `SummarizationMiddleware` 使用默认提示词。
如需自定义，可以实现自己的中间件（Module 10 会学）。

## 最佳实践

```python
# 1. 根据场景选择阈值
agent = create_agent(
    model=model,
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="groq:llama-3.3-70b-versatile",
            max_tokens_before_summary=500  # 短对话
            # max_tokens_before_summary=2000  # 长对话
        )
    ],
    checkpointer=InMemorySaver()
)

# 2. 使用便宜模型摘要（降低成本）
SummarizationMiddleware(
    model="groq:llama-3.3-70b-versatile",  # 用便宜模型摘要
    max_tokens_before_summary=1000
)

# 3. 监控摘要触发频率
# 如果频繁触发 → 提高阈值
# 如果从不触发 → 降低阈值

```AI给的总结使用示例：
  核心 API

  SummarizationMiddleware

  SummarizationMiddleware(
      model=model_instance,        # 摘要模型实例   
      trigger=('tokens', 500),     # 触发条件       
      # trigger=('messages', 10),  # 或按消息数触发 
  )

  trim_messages

  trim_messages(
      messages,
      max_tokens=500,              # token 数限制   
      strategy="last",             # "first" 或 "last"
      token_counter=len,           # 计数器（len=消 
息数）
      start_on="human",            # 从 HumanMessage 开始
      include_system=True          # 保留 SystemMessage
  )

## 核心要点

1. **默认问题**：对话历史无限增长
2. **推荐方案**：`SummarizationMiddleware`
3. **配置位置**：`middleware=[]` 参数
4. **触发条件**：`max_tokens_before_summary`
5. **自动化**：无需手动管理

## 下一步

**09_checkpointing** - 学习如何持久化对话状态（SQLite）
