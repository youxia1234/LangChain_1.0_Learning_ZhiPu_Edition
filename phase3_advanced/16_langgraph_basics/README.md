# 模块 16：LangGraph 基础

## 🎯 学习目标

本模块将帮助你理解 LangGraph 1.0 的核心概念，学会创建状态图来构建复杂的 AI 工作流。

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
```

获取 API Key: https://open.bigmodel.cn/usercenter/apikeys

## 📚 核心概念

### 什么是 LangGraph？

LangGraph 是一个用于构建**状态化、多步骤 AI 应用**的框架。它使用**图（Graph）** 的概念来组织工作流：

- **节点（Nodes）**：图中的处理单元，可以是 LLM 调用、工具执行或自定义函数
- **边（Edges）**：连接节点的路径，定义执行顺序
- **状态（State）**：在节点之间传递的数据结构

### LangGraph vs LangChain

| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| 抽象级别 | 高级 | 低级 |
| 适用场景 | 快速构建标准 Agent | 复杂自定义工作流 |
| 控制粒度 | 通过中间件 | 完全控制每个节点 |
| 状态管理 | 自动 | 手动但灵活 |

### 核心组件

```python
# 1. 定义状态 - 使用 TypedDict
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，自动累加
    current_step: str                         # 当前步骤

# 2. 定义节点 - 接收和返回状态的函数
def my_node(state: State) -> dict:
    # 处理逻辑
    return {"current_step": "completed"}

# 3. 创建图
from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node("my_node", my_node)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)

# 4. 编译并运行
app = graph.compile()
result = app.invoke({"messages": [], "current_step": "start"})
```

## 🔑 关键 API

### 模型初始化（智谱 AI）

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
```

### StateGraph

```python
from langgraph.graph import StateGraph, START, END

# 创建图
graph = StateGraph(State)

# 添加节点
graph.add_node("node_name", node_function)

# 添加边
graph.add_edge("from_node", "to_node")  # 普通边
graph.add_edge(START, "first_node")      # 从入口开始
graph.add_edge("last_node", END)         # 到出口结束

# 添加条件边
graph.add_conditional_edges(
    "from_node",
    condition_function,  # 返回下一个节点名称
    {"option1": "node1", "option2": "node2"}
)
```

### add_messages 注解

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    # add_messages 确保消息被追加而不是替换
    messages: Annotated[list, add_messages]
```

### 编译选项

```python
from langgraph.checkpoint.memory import MemorySaver

# 添加内存检查点（用于持久化）
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 使用 thread_id 进行会话管理
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke(input_data, config=config)
```

## 📝 本模块示例

### main.py

包含三个递进的示例：
1. **简单顺序图**：展示基本的节点和边
2. **条件分支图**：根据条件选择不同路径
3. **带内存的对话图**：实现多轮对话

## 🧪 练习

1. 修改 `simple_workflow` 添加一个新的处理节点
2. 在 `conditional_workflow` 中添加第三个分支
3. 扩展 `conversation_workflow` 支持更多类型的用户意图

## 📖 延伸阅读

- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph)
- [LangGraph 概念指南](https://docs.langchain.com/oss/python/langgraph/concepts)
- [图的可视化](https://docs.langchain.com/oss/python/langgraph/visualization)

## ⚠️ 注意事项

1. 状态更新是**合并式**的，只需返回要更新的字段
2. 使用 `add_messages` 注解时，消息会自动追加
3. 条件函数必须返回有效的节点名称字符串
4. 编译后的图是不可变的，需要修改时重新编译

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
| LangGraph 兼容性 | 良好 | **完全兼容** |
| 国内网络 | 需代理 | **直接访问** |

### Q3: LangGraph 和 LangChain Agent 的区别？

**A:**

| 特性 | LangChain Agent | LangGraph |
|------|-----------------|-----------|
| 抽象级别 | 高级（开箱即用）| 低级（完全控制）|
| 使用场景 | 标准 Agent 应用 | 复杂自定义工作流 |
| 控制粒度 | 通过 `create_agent` | 每个节点自定义 |
| 学习曲线 | 简单 | 较陡 |

**选择建议**：
- 简单需求 → `create_agent`（模块 5、15）
- 复杂工作流 → `StateGraph`（本模块）
