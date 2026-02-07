# 模块 18：条件路由

## 🎯 学习目标

掌握 LangGraph 中的条件路由机制，实现动态工作流控制。

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
```

获取 API Key: https://open.bigmodel.cn/usercenter/apikeys

## 📚 核心概念

### 什么是条件路由？

条件路由允许你根据**运行时的状态**动态决定下一步执行哪个节点。这是构建智能工作流的关键。

### 路由类型

1. **静态边（Static Edge）**：总是执行固定的下一个节点
2. **条件边（Conditional Edge）**：根据条件函数的返回值选择下一个节点

```python
# 静态边
graph.add_edge("node_a", "node_b")  # 总是 A -> B

# 条件边
graph.add_conditional_edges(
    "node_a",                    # 起始节点
    condition_function,          # 返回下一个节点名的函数
    {"option1": "node_b", "option2": "node_c"}  # 映射
)
```

### 条件函数的写法

```python
from typing import Literal

def my_router(state: MyState) -> Literal["next_a", "next_b", "end"]:
    """路由函数必须返回节点名称"""
    if state["score"] > 80:
        return "next_a"
    elif state["score"] > 50:
        return "next_b"
    else:
        return "end"
```

## 🔑 关键模式

### 1. 循环控制

```python
def should_continue(state) -> Literal["continue", "end"]:
    if state["iteration"] < state["max_iterations"]:
        return "continue"
    return "end"

graph.add_conditional_edges("process", should_continue, {
    "continue": "process",  # 回到自己
    "end": END
})
```

### 2. 错误处理路由

```python
def error_router(state) -> Literal["retry", "fallback", "success"]:
    if state.get("error"):
        if state["retry_count"] < 3:
            return "retry"
        return "fallback"
    return "success"
```

### 3. 多条件组合

```python
def complex_router(state) -> str:
    # 可以组合多个条件
    if state["is_urgent"] and state["has_permission"]:
        return "fast_track"
    elif state["needs_review"]:
        return "review"
    else:
        return "standard"
```

## 📝 本模块示例

实现了：
1. **评分路由**：根据分数选择不同的处理流程
2. **重试机制**：失败时自动重试
3. **复杂决策树**：多条件组合路由

## ⚠️ 注意事项

1. 条件函数必须是**纯函数**，不应有副作用
2. 返回值必须是映射中定义的有效节点名
3. 设置最大迭代次数防止无限循环

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
| 条件路由场景 | 良好 | **更适合中文条件判断** |
| 国内网络 | 需代理 | **直接访问** |

### Q3: 路由函数返回值必须是什么类型？

**A:** 路由函数必须返回字符串类型的节点名称。使用 `Literal` 类型注解可以提供类型检查：

```python
from typing import Literal

def my_router(state) -> Literal["node_a", "node_b", END]:
    # 类型检查器会验证返回值是否有效
    return "node_a"  # ✅
    # return "invalid"  # ❌ 类型错误
```

### Q4: 如何防止条件路由进入无限循环？

**A:** 三种策略：

```python
# 1. 设置迭代计数器
class State(TypedDict):
    iteration: int
    max_iterations: int

def should_continue(state) -> Literal["continue", "end"]:
    if state["iteration"] < state["max_iterations"]:
        return {"iteration": state["iteration"] + 1}, "continue"
    return "end"

# 2. 检测状态变化
def should_continue(state) -> Literal["continue", "end"]:
    if state.get("new_data"):
        return "continue"
    return "end"  # 无新数据时退出

# 3. 在图中设置全局最大步数
app = graph.compile()
result = app.invoke(initial_state, config={"recursion_limit": 100})
```
