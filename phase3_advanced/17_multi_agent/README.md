# 模块 17：多 Agent 协作

## 🎯 学习目标

本模块将帮助你理解如何在 LangGraph 中创建多个专业化 Agent，并让它们协作完成复杂任务。

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
```

获取 API Key: https://open.bigmodel.cn/usercenter/apikeys

## 📚 核心概念

### 为什么需要多 Agent？

单个 Agent 在处理复杂任务时可能存在以下问题：
- **上下文过载**：单个 Agent 需要处理所有类型的任务
- **专业性不足**：难以在所有领域都表现出色
- **维护困难**：单一庞大的 Agent 难以调试和优化

多 Agent 架构通过**分而治之**的策略解决这些问题。

### 常见的多 Agent 模式

#### 1. 监督者模式（Supervisor Pattern）

```
                    ┌──────────────┐
                    │  Supervisor  │
                    │   (协调者)    │
                    └──────┬───────┘
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Agent A    │ │  Agent B    │ │  Agent C    │
    │  (研究员)    │ │  (编辑)     │ │  (审核员)   │
    └─────────────┘ └─────────────┘ └─────────────┘
```

- **Supervisor**：接收任务，决定分配给哪个 Agent
- **Worker Agents**：专注于特定类型的任务

#### 2. 协作模式（Collaborative Pattern）

```
    ┌─────────────┐     ┌─────────────┐
    │  Agent A    │────▶│  Agent B    │
    │  (写初稿)    │     │  (审核修改)  │
    └─────────────┘     └──────┬──────┘
                               │
                               ▼
                        ┌─────────────┐
                        │  Agent C    │
                        │  (最终确认)  │
                        └─────────────┘
```

- Agent 按顺序处理任务
- 每个 Agent 的输出是下一个 Agent 的输入

#### 3. 层级模式（Hierarchical Pattern）

```
                    ┌──────────────┐
                    │   Manager    │
                    └──────┬───────┘
           ┌───────────────┴───────────────┐
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │  Team Lead A│                 │  Team Lead B│
    └──────┬──────┘                 └──────┬──────┘
      ┌────┴────┐                     ┌────┴────┐
      ▼         ▼                     ▼         ▼
   Agent 1   Agent 2              Agent 3   Agent 4
```

### 实现多 Agent 的关键组件

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 初始化模型（智谱 AI）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 2. 定义共享状态
class TeamState(TypedDict):
    task: str
    current_agent: str
    messages: list
    final_result: str

# 3. 定义专业化 Agent 节点
def researcher(state: TeamState) -> dict:
    """研究员 Agent：收集和整理信息"""
    messages = [
        SystemMessage(content="你是一个研究员，专门收集和整理信息。"),
        HumanMessage(content=state["task"])
    ]
    response = model.invoke(messages)
    return {"research_result": response.content}

def writer(state: TeamState) -> dict:
    """作家 Agent：撰写内容"""
    messages = [
        SystemMessage(content="你是一个作家，擅长将信息组织成清晰的文章。"),
        HumanMessage(content=state["research_result"])
    ]
    response = model.invoke(messages)
    return {"draft": response.content}

# 4. 创建监督者逻辑
def supervisor(state: TeamState) -> str:
    """决定下一个执行的 Agent"""
    if "需要研究" in state["task"]:
        return "researcher"
    elif "需要写作" in state["task"]:
        return "writer"
    else:
        return "end"
```

## 🔑 关键 API

### 使用 send() 进行动态分发

```python
from langgraph.types import Send

def supervisor(state: State):
    """将任务分发给多个 Agent"""
    return [
        Send("agent_a", {"task": "子任务1"}),
        Send("agent_b", {"task": "子任务2"})
    ]
```

### Agent 间通信

```python
# 通过状态传递信息
class SharedState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_outputs: dict  # 存储各 Agent 的输出
```

## 📝 本模块示例

### main.py

实现了一个**内容创作团队**：
1. **研究员 Agent**：收集相关信息
2. **作家 Agent**：撰写内容
3. **编辑 Agent**：审核和优化
4. **监督者**：协调整个流程

## 🧪 练习

1. 添加一个"翻译 Agent"，将最终内容翻译成英文
2. 实现并行执行：让研究员和作家同时工作
3. 添加人工审核节点（Human-in-the-loop）

## 📖 延伸阅读

- [LangGraph 多 Agent 教程](https://docs.langchain.com/oss/python/langgraph/tutorials/multi_agent)
- [Agent 协作模式](https://blog.langchain.com/multi-agent-collaboration/)

## ⚠️ 注意事项

1. 合理划分 Agent 职责，避免功能重叠
2. 设置合理的迭代上限，防止无限循环
3. 监控 token 使用，多 Agent 会消耗更多 token
4. 考虑添加错误处理和回退机制

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
| 多 Agent 场景 | 良好 | **更适合中文协作场景** |
| 国内网络 | 需代理 | **直接访问** |

### Q3: 多 Agent 系统如何避免无限循环？

**A:** 三种防护策略：

```python
# 1. 监督者设置状态标记
def supervisor(state):
    if state.get("completed"):
        return "end"  # 明确的退出条件

# 2. 设置迭代上限
max_iterations = 10

# 3. 状态检测
if not state.get("new_data"):
    return "end"  # 无新数据时退出
```
