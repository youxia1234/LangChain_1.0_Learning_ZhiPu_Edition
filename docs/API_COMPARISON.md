# LangChain 1.0 Agent API 指南

## ✅ 正确的 `create_agent` 用法

根据 LangChain 1.0 官方文档，正确的使用方式如下：

```python
from langchain.agents import create_agent

# 创建 Agent
agent = create_agent(
    model="groq:llama-3.3-70b-versatile",  # 模型（必需）
    tools=[my_tool],                         # 工具列表（可选）
    system_prompt="你是一个有帮助的助手。"    # 系统提示（建议提供）
)

# 调用 Agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "你好"}]
})

# 获取回复
answer = result["messages"][-1].content
```

---

## 📝 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model` | str 或 ChatModel | ✅ | 模型标识符或实例 |
| `tools` | List | ❌ | 工具列表，默认 `None` |
| `system_prompt` | str | ❌ | 系统提示，建议提供 |
| `middleware` | List | ❌ | 中间件列表 |
| `checkpointer` | Checkpointer | ❌ | 状态持久化 |

---

## ⚠️ 常见错误

### 错误 1：缺少关键字参数
```python
# ❌ 错误
agent = create_agent(model, tools)

# ✅ 正确
agent = create_agent(model=model, tools=tools)
```

### 错误 2：把 system_prompt 拼接到消息中
```python
# ❌ 错误（不要这样做）
agent = create_agent(model=model, tools=tools)
agent.invoke({
    "messages": [{"role": "user", "content": f"{system_prompt}\n{user_msg}"}]
})

# ✅ 正确（在创建时传入 system_prompt）
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)
agent.invoke({
    "messages": [{"role": "user", "content": user_msg}]
})
```

### 错误 3：使用已弃用的 API
```python
# ❌ 已弃用（将在 V2.0 移除）
from langgraph.prebuilt import create_react_agent

# ✅ 正确
from langchain.agents import create_agent
```

---

## 🔧 JSON 解析最佳实践

LLM 返回的 JSON 可能包含 Markdown 代码块，需要安全解析：

```python
def safe_parse_json(text: str, default: dict = None) -> dict:
    """安全地解析JSON文本"""
    if default is None:
        default = {}
    
    content = text.strip()
    
    # 移除 Markdown 代码块
    if "```json" in content:
        try:
            content = content.split("```json")[1].split("```")[0]
        except IndexError:
            pass
    elif "```" in content:
        try:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1]
        except IndexError:
            pass
    
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        return default
```

---

## 🔧 高级用法

### 带记忆的 Agent
```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是助手",
    checkpointer=InMemorySaver()  # 添加内存
)

# 使用 thread_id 保持对话上下文
config = {"configurable": {"thread_id": "user_123"}}
result = agent.invoke({"messages": [...]}, config=config)
```

### 带中间件的 Agent
```python
from langchain.agents.middleware import AgentMiddleware

class MyMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        print("模型调用前")
        return None

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="你是助手",
    middleware=[MyMiddleware()]
)
```

---

## 📚 参考链接

- [LangChain 1.0 Agents 文档](https://docs.langchain.com/oss/python/langchain/agents)
- [LangChain API 参考](https://reference.langchain.com/python/langchain/agents)
