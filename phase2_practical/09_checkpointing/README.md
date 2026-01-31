# 09 - Checkpointing (检查点持久化)

## 核心概念

**Checkpointing = 将对话状态持久化到数据库**

- `InMemorySaver` → 内存中（程序退出即丢失）
- `SqliteSaver` → SQLite 数据库（持久化存储）

## 基本用法

### InMemorySaver 的限制

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=model,
    tools=[],
    checkpointer=InMemorySaver()
)

# 限制：
# ❌ 程序重启后丢失
# ❌ 无法跨进程共享
# ❌ 不适合生产环境
```

### SqliteSaver（推荐生产使用）

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 创建持久化 checkpointer（使用 with 语句）
with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer  # 使用 SQLite
    )

    config = {"configurable": {"thread_id": "user_123"}}

    # 第一次运行
    agent.invoke({"messages": [...]}, config)

# 程序重启后，对话仍然保留！
with SqliteSaver.from_conn_string("sqlite:///checkpoints.sqlite") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer)
    agent.invoke({"messages": [...]}, config)
```

**重要：** `SqliteSaver.from_conn_string()` 返回上下文管理器，必须使用 `with` 语句！

## 工作原理

### 数据保存位置

```
InMemorySaver:
    对话历史 → 内存（变量）→ 程序退出即丢失

SqliteSaver:
    对话历史 → SQLite 文件 → 持久化存储
        ↓
    checkpoints.sqlite
    ├── thread_id: user_123
    │   ├── checkpoint_1
    │   ├── checkpoint_2
    │   └── checkpoint_3
    └── thread_id: user_456
        ├── checkpoint_1
        └── checkpoint_2
```

### 跨进程访问

```python
# 进程 A（Web 服务器）
with SqliteSaver.from_conn_string("shared.sqlite") as checkpointer:
    agent_a = create_agent(model=model, checkpointer=checkpointer)
    agent_a.invoke({...}, config={"configurable": {"thread_id": "user_1"}})

# 进程 B（后台任务）
with SqliteSaver.from_conn_string("shared.sqlite") as checkpointer:
    agent_b = create_agent(model=model, checkpointer=checkpointer)
    # 可以访问进程 A 创建的对话
    agent_b.invoke({...}, config={"configurable": {"thread_id": "user_1"}})
```

## 参数说明

### SqliteSaver.from_conn_string()

| 参数 | 说明 | 示例 |
|-----|------|------|
| `conn_string` | 数据库文件路径（不要加 `sqlite:///` 前缀） | `"checkpoints.sqlite"` |

### 路径格式

```python
# 相对路径（当前目录） - 推荐
with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer)

# 绝对路径（生产环境）
with SqliteSaver.from_conn_string("C:/data/checkpoints.sqlite") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer)

# 内存数据库（测试用）
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer)
```

**重要：**
- ✅ 直接传文件路径，不要加 `sqlite:///` 前缀
- ✅ 相对路径会在当前目录创建数据库
- ✅ Windows 路径使用正斜杠 `/` 或双反斜杠 `\\`

## 对比 InMemorySaver

| 特性 | InMemorySaver | SqliteSaver |
|-----|--------------|-------------|
| **持久化** | ❌ 程序退出即丢失 | ✅ 持久化到文件 |
| **跨进程** | ❌ 无法共享 | ✅ 可以共享 |
| **性能** | ⚡ 快（内存） | 🐢 慢一点（磁盘 I/O）|
| **适用** | 开发、测试 | 生产环境 |

## 实际应用

### 客服系统

```python
# 客户今天上午咨询
with SqliteSaver.from_conn_string("customer_service.sqlite") as checkpointer:
    agent = create_agent(model=model, tools=[查询订单], checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "customer_zhang"}}
    agent.invoke({"messages": [{"role": "user", "content": "订单 12345 在哪？"}]}, config)

# 下午客户再次咨询（即使服务重启）
with SqliteSaver.from_conn_string("customer_service.sqlite") as checkpointer:
    agent = create_agent(model=model, tools=[查询订单], checkpointer=checkpointer)
    agent.invoke({"messages": [{"role": "user", "content": "到了吗？"}]}, config)
    # Agent 记得上午查询的订单号！
```

### 多用户聊天

```python
with SqliteSaver.from_conn_string("chat.sqlite") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer)

    # 用户 A
    agent.invoke({...}, config={"configurable": {"thread_id": "user_alice"}})

    # 用户 B
    agent.invoke({...}, config={"configurable": {"thread_id": "user_bob"}})

    # 所有用户的对话都持久化在 chat.sqlite 中
```

## 常见问题

### 1. 数据库文件在哪？

```python
# 相对路径 → 当前工作目录
SqliteSaver.from_conn_string("sqlite:///checkpoints.sqlite")
# 文件位置：当前目录/checkpoints.sqlite

# 绝对路径 → 指定位置
SqliteSaver.from_conn_string("sqlite:///C:/data/checkpoints.sqlite")
# 文件位置：C:/data/checkpoints.sqlite
```

### 2. 如何清空某个用户的历史？

目前需要手动操作数据库：

```python
import sqlite3

conn = sqlite3.connect("checkpoints.sqlite")
cursor = conn.cursor()

# 删除特定 thread_id 的记录
cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", ("user_123",))
conn.commit()
conn.close()
```

### 3. 数据库会无限增长吗？

会！需要定期清理：

**策略：**
- 定期删除旧对话（如 30 天前）
- 限制每个 thread 的 checkpoint 数量
- 定期备份和归档

### 4. 性能影响？

- SQLite 比内存慢，但影响不大
- 适合中小型应用（< 10000 并发用户）
- 大规模应用考虑 PostgreSQL（LangGraph 也支持）

## 最佳实践

```python
# 1. 生产环境使用绝对路径 + with 语句
with SqliteSaver.from_conn_string("C:/production/data/checkpoints.sqlite") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer)

# 2. 开发环境使用相对路径
with SqliteSaver.from_conn_string("dev_checkpoints.sqlite") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer)

# 3. 测试环境使用内存数据库
with SqliteSaver.from_conn_string(":memory:") as checkpointer:
    agent = create_agent(model=model, checkpointer=checkpointer)

# 4. 定期备份数据库文件
# 使用系统任务定期复制 checkpoints.sqlite

# 5. 监控数据库大小
import os
db_size = os.path.getsize("checkpoints.sqlite")
print(f"数据库大小: {db_size / 1024 / 1024:.2f} MB")
```

## 核心要点

1. **InMemorySaver**：内存存储，程序退出即丢失
2. **SqliteSaver**：持久化到 SQLite 文件
3. **创建方式**：`with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:`
4. **路径格式**：直接传文件路径，不要加 `sqlite:///` 前缀
5. **跨进程**：多个进程可访问同一数据库
6. **生产推荐**：使用 SqliteSaver + with 语句

## 下一步

**10_middleware_basics** - 学习如何创建自定义中间件
