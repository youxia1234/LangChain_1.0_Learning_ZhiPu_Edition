# 05 - Simple Agent (简单 Agent)

## 核心概念

**Agent = 模型 + 工具 + 自动决策**

Agent 的关键能力：
- 理解用户问题
- 自动判断是否需要工具
- 选择合适的工具
- 基于工具结果生成回答

## create_agent 基本用法

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model("groq:llama-3.3-70b-versatile"),
    tools=[tool1, tool2],
    system_prompt="Agent 的行为指令"  # 可选
)

response = agent.invoke({
    "messages": [{"role": "user", "content": "问题"}]
})
```

### 参数说明

| 参数 | 说明 | 必需 |
|-----|------|------|
| `model` | 语言模型 | ✅ |
| `tools` | 工具列表 | ✅ |
| `system_prompt` | 系统提示，定义 Agent 行为 | ❌ |

## Agent 执行循环

```
用户问题
   ↓
[Agent 分析]
   ↓
需要工具？ ─── 否 ──→ 直接回答
   ↓ 是
调用工具
   ↓
获取结果
   ↓
生成回答
```

### 完整流程示例

```python
# 问题：北京天气如何？

# 步骤1：用户提问
messages = [HumanMessage("北京天气如何？")]

# 步骤2：AI 决定调用工具
# AIMessage(tool_calls=[{
#     "name": "get_weather",
#     "args": {"city": "北京"}
# }])

# 步骤3：执行工具
# ToolMessage("晴天，温度 15°C")

# 步骤4：AI 生成最终答案
# AIMessage("北京今天是晴天，温度 15°C")
```

## 多工具选择

Agent 如何选择工具？

**依据：工具的 docstring**

```python
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""  # ← AI 读这个！
    ...

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """执行基本的数学计算"""  # ← AI 也读这个！
    ...
```

AI 会根据：
1. 问题内容
2. 每个工具的描述
3. 自动选择最匹配的工具

## 多轮对话

**关键：传入历史消息**

```python
# 第一轮
response1 = agent.invoke({
    "messages": [{"role": "user", "content": "10 + 5"}]
})

# 第二轮（带历史）
response2 = agent.invoke({
    "messages": response1['messages'] + [
        {"role": "user", "content": "再乘以 3"}
    ]
})
```

## 常见问题

### 1. Agent 不调用工具？

**原因：**
- 工具的 docstring 不清晰
- 问题表述不明确
- 模型认为不需要工具

**解决：**
```python
# ❌ 不好
@tool
def tool1(x: str) -> str:
    """做一些事情"""  # 太模糊

# ✅ 好
@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的实时天气信息

    参数:
        city: 城市名称，如"北京"、"上海"
    """
```

### 2. Agent 选错工具？

**原因：**
- 多个工具的功能描述相似
- 工具太多导致混淆

**解决：**
- 只给必要的工具
- 工具描述要有明确区分
- 在 system_prompt 中说明工具使用场景

### 3. Agent 返回什么？

```python
response = agent.invoke({"messages": [...]})

# response 是字典
{
    "messages": [
        HumanMessage(...),      # 用户问题
        AIMessage(...),          # AI 工具调用
        ToolMessage(...),        # 工具结果
        AIMessage(...)           # 最终回答 ← 通常取这个
    ]
}

# 获取最终回答
final_answer = response['messages'][-1].content
```

## 最佳实践

### 1. 工具配置
```python
# ✅ 好：只给需要的工具
agent = create_agent(
    model=model,
    tools=[get_weather, calculator]  # 2-5 个工具最佳
)

# ❌ 不好：工具太多
agent = create_agent(
    model=model,
    tools=[tool1, tool2, ..., tool20]  # 会混淆
)
```

### 2. System Prompt
```python
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="""你是天气助手。

工作流程：
1. 理解用户的城市查询
2. 使用 get_weather 工具获取数据
3. 简洁清晰地回答

输出格式：
- 天气状况
- 温度
- 注意事项（如有）
"""
)
```

### 3. 错误处理
```python
try:
    response = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })
    answer = response['messages'][-1].content
except Exception as e:
    print(f"Agent 错误：{e}")
```

## 运行示例

```bash
# 确保已安装依赖
pip install langchain langchain-groq python-dotenv

# 设置 API Key（.env 文件）
GROQ_API_KEY=your_key_here

# 运行
python main.py
```

## 下一步

**06_agent_loop** - 深入理解 Agent 执行循环的底层机制
