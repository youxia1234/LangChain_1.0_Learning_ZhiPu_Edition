# 07 - Memory Basics (内存管理基础)

## 核心概念

**内存 = Agent 记住对话历史的能力**

默认情况下，每次调用 `agent.invoke()` 都是全新的开始，不记得之前的对话。使用 `InMemorySaver` 可以让 Agent 记住历史。

## 基本用法

### 没有内存（默认）

```python
from langchain.agents import create_agent

agent = create_agent(model=model, tools=[])

# 第一轮
agent.invoke({"messages": [{"role": "user", "content": "我叫张三"}]})

# 第二轮 - 不记得第一轮！
response = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]})
# AI 会说"不知道"
```

### 添加内存

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 1. 创建 Agent 时添加 checkpointer
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver()  # 添加内存
)

# 2. 调用时指定 thread_id
config = {"configurable": {"thread_id": "conversation_1"}}

# 第一轮
agent.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    config=config
)

# 第二轮 - 记得第一轮！
response = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么？"}]},
    config=config
)
# AI 会说"你叫张三"
```

## 关键参数

### checkpointer

**作用**：为 Agent 添加内存管理能力

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=tools,
    checkpointer=InMemorySaver()  # InMemorySaver = 短期内存
)
```

**注意**：
- `InMemorySaver` - 内存中保存（进程结束就丢失）
- 后续会学习持久化（SQLite、Postgres）

### thread_id

**作用**：区分不同的会话

```python
# 会话 1
config1 = {"configurable": {"thread_id": "user_alice"}}
agent.invoke({...}, config=config1)

# 会话 2（完全独立）
config2 = {"configurable": {"thread_id": "user_bob"}}
agent.invoke({...}, config=config2)
```

**thread_id 的选择：**
- 聊天应用：使用用户 ID 或会话 ID
- 多轮任务：使用任务 ID
- 测试：使用描述性字符串（如 "test_1"）

## 工作原理

### 内存保存了什么？

```python
agent.invoke({"messages": [{"role": "user", "content": "你好"}]}, config)
# InMemorySaver 保存：
# {
#     "thread_id": "xxx",
#     "messages": [
#         HumanMessage("你好"),
#         AIMessage("你好！有什么可以帮助你的吗？")
#     ]
# }

agent.invoke({"messages": [{"role": "user", "content": "天气"}]}, config)
# InMemorySaver 更新：
# {
#     "thread_id": "xxx",
#     "messages": [
#         HumanMessage("你好"),
#         AIMessage("你好！有什么可以帮助你的吗？"),
#         HumanMessage("天气"),
#         AIMessage("...")
#     ]
# }
```

### 自动追加历史

```python
# 你只需要传新消息
agent.invoke(
    {"messages": [{"role": "user", "content": "新问题"}]},
    config
)

# checkpointer 自动：
# 1. 读取之前的历史
# 2. 追加新消息
# 3. 调用模型（传入完整历史）
# 4. 保存新的历史
```

## 多会话管理

### 场景：多用户聊天

```python
agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver()
)

# 用户 Alice
config_alice = {"configurable": {"thread_id": "user_alice"}}
agent.invoke({"messages": [...]}, config_alice)

# 用户 Bob
config_bob = {"configurable": {"thread_id": "user_bob"}}
agent.invoke({"messages": [...]}, config_bob)

# 两个会话完全独立
```

### 场景：同一用户的不同任务

```python
# 任务 1：写代码
config_task1 = {"configurable": {"thread_id": "task_coding"}}
agent.invoke({"messages": [...]}, config_task1)

# 任务 2：写文档
config_task2 = {"configurable": {"thread_id": "task_docs"}}
agent.invoke({"messages": [...]}, config_task2)
```

## 内存 + 工具

Agent 会记住工具调用的结果：

```python
@tool
def search(query: str) -> str:
    """搜索工具"""
    return f"关于 {query} 的结果..."

agent = create_agent(
    model=model,
    tools=[search],
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "session_1"}}

# 第一轮：使用工具
agent.invoke({"messages": [{"role": "user", "content": "搜索 Python"}]}, config)
# Agent 调用 search("Python")

# 第二轮：引用之前的结果
response = agent.invoke(
    {"messages": [{"role": "user", "content": "刚才搜索的结果是什么？"}]},
    config
)
# Agent 记得工具返回的结果，无需重新调用
```

## 查看内存状态

```
  # 用户输入
  {"role": "user", "content": "你好"}
      ↓ 转换为
  HumanMessage(content="你好")

  # AI 回复
  {"role": "assistant", "content": "你好！"}
      ↓ 转换为
  AIMessage(content="你好！")

  # 系统指令
  {"role": "system", "content": "你是助手"}
      ↓ 转换为
  SystemMessage(content="你是助手")


```

```python
response = agent.invoke({"messages": [...]}, config)

# 查看完整的对话历史
print("消息数量:", len(response['messages']))

# 查看最近的消息
for msg in response['messages'][-5:]:
    print(f"{msg.__class__.__name__}: {msg.content}")
```

## 常见问题

### 1. 为什么 Agent 不记得？

**检查：**
- ✅ 是否添加了 `checkpointer=InMemorySaver()`？
- ✅ 是否传入了 `config` 参数？
- ✅ 两次调用的 `thread_id` 是否相同？

```python
# ❌ 错误：没有 checkpointer
agent = create_agent(model=model, tools=[])
agent.invoke({...})  # 不会记住

# ❌ 错误：没有 config
agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
agent.invoke({...})  # 不会记住

# ❌ 错误：thread_id 不同
agent.invoke({...}, {"configurable": {"thread_id": "1"}})
agent.invoke({...}, {"configurable": {"thread_id": "2"}})  # 不同会话

# ✅ 正确
agent = create_agent(model=model, tools=[], checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "1"}}
agent.invoke({...}, config)
agent.invoke({...}, config)  # 记得！
```

### 2. InMemorySaver 会丢失数据吗？

**会！** InMemorySaver 只保存在内存中：
- ✅ 同一进程内有效
- ❌ 程序重启后丢失
- ❌ 不同进程无法共享

**解决方案**：Module 09 会学习持久化（SQLite）

### 3. 内存会无限增长吗？

**会！** 默认情况下，InMemorySaver 会保存所有消息。

**问题**：
- 消息越来越多
- 超过模型的 token 限制
- 响应变慢、成本增加

**解决方案**：Module 08 会学习上下文管理（修剪、摘要）

### 4. 如何清空某个会话的历史？

目前 `InMemorySaver` 没有提供删除 API。

**临时方案**：
- 使用新的 `thread_id`
- 或重新创建 Agent

## 实际应用场景

### 1. 聊天机器人

```python
def handle_user_message(user_id: str, message: str):
    config = {"configurable": {"thread_id": f"user_{user_id}"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config
    )

    return response['messages'][-1].content
```

### 2. 多轮任务助手

```python
def process_task(task_id: str, user_input: str):
    config = {"configurable": {"thread_id": f"task_{task_id}"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config
    )

    return response['messages'][-1].content
```

### 3. 客服系统

```python
agent = create_agent(
    model=model,
    tools=[查询订单, 查询物流],
    system_prompt="你是客服助手，记住用户的订单号",
    checkpointer=InMemorySaver()
)

def customer_service(session_id: str, message: str):
    config = {"configurable": {"thread_id": session_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config
    )
    return response['messages'][-1].content
```

## 运行示例

```bash
# 运行主程序
python main.py

# 测试
python test.py
```

## 核心要点

1. **默认无内存**：每次 `invoke` 是全新开始
2. **添加内存**：`checkpointer=InMemorySaver()`
3. **会话管理**：`config={"configurable": {"thread_id": "xxx"}}`
4. **自动保存**：checkpointer 自动管理历史
5. **多会话**：不同 thread_id = 不同会话
6. **记住工具**：也会记住工具调用结果

## 限制

- ❌ 进程重启后丢失
- ❌ 无限增长（需要管理上下文）
- ❌ 不支持跨进程共享

## 下一步

**08_context_management** - 学习如何管理上下文长度（修剪、摘要）
