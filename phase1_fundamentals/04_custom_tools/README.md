# 04 - Custom Tools (自定义工具)

## 核心概念

**工具 (Tool) = 给 AI 的函数**

使用 `@tool` 装饰器，让 AI 能调用你的 Python 函数。

## @tool 基本用法

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"

    返回:
        天气信息字符串
    """
    # 你的实现
    return "晴天，温度 15°C"
```

### 关键要点

| 必需项 | 说明 |
|-------|------|
| `@tool` 装饰器 | 声明这是一个工具 |
| **docstring** | AI 读这个来理解工具用途 ⚠️ 非常重要！ |
| 类型注解 | 参数和返回值的类型 |
| 返回 `str` | 工具应该返回字符串（AI 最容易理解） |

## 工具的 docstring

**AI 依赖 docstring 来理解工具！**

```python
@tool
def my_tool(param: str) -> str:
    """
    工具的简短描述（AI 读这个！）

    参数:
        param: 参数说明

    返回:
        返回值说明
    """
    ...
```

### 好的 vs 不好的 docstring

```python
# ❌ 不好：太模糊
@tool
def tool1(x: str) -> str:
    """做一些事情"""
    ...

# ✅ 好：清晰明确
@tool
def search_products(query: str) -> str:
    """
    在产品数据库中搜索产品

    参数:
        query: 搜索关键词，如"笔记本电脑"、"手机"

    返回:
        产品列表的 JSON 字符串
    """
    ...
```

## 参数类型

### 1. 单参数
```python
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    ...
```

### 2. 多参数
```python
@tool
def calculator(operation: str, a: float, b: float) -> str:
    """
    执行数学计算

    参数:
        operation: "add", "subtract", "multiply", "divide"
        a: 第一个数字
        b: 第二个数字
    """
    ...
```

### 3. 可选参数
```python
from typing import Optional

@tool
def web_search(query: str, num_results: Optional[int] = 3) -> str:
    """
    搜索网页

    参数:
        query: 搜索关键词
        num_results: 返回结果数量，默认 3
    """
    ...
```

## 调用工具

工具有两种调用方式：

### 1. 直接调用（测试用）
```python
# 使用 .invoke() 方法
result = get_weather.invoke({"city": "北京"})
print(result)  # "晴天，温度 15°C"
```

### 2. 绑定到模型（让 AI 调用）
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("groq:llama-3.3-70b-versatile")

# 绑定工具
model_with_tools = model.bind_tools([get_weather])

# AI 可以决定是否调用工具
response = model_with_tools.invoke("北京天气如何？")

# 检查 AI 是否要调用工具
if response.tool_calls:
    print("AI 想调用工具：", response.tool_calls)
else:
    print("AI 直接回答：", response.content)
```

## 工具属性

创建工具后，可以查看其属性：

```python
@tool
def my_tool(param: str) -> str:
    """工具描述"""
    ...

print(my_tool.name)         # "my_tool"
print(my_tool.description)  # "工具描述"
print(my_tool.args)         # 参数模式
```

## 最佳实践

### 1. 清晰的描述
```python
# ✅ 好
@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """
    搜索航班信息

    参数:
        origin: 出发城市，如"北京"
        destination: 目的地城市，如"上海"
        date: 出发日期，格式 YYYY-MM-DD

    返回:
        可用航班的 JSON 列表
    """
```

### 2. 功能单一
```python
# ❌ 不好：一个工具做太多事
@tool
def do_everything(action: str, data: str) -> str:
    """做各种事情"""
    if action == "weather": ...
    elif action == "calculate": ...
    elif action == "search": ...

# ✅ 好：每个工具做一件事
@tool
def get_weather(city: str) -> str:
    """获取天气"""
    ...

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """计算"""
    ...
```

### 3. 错误处理
```python
@tool
def divide(a: float, b: float) -> str:
    """
    除法计算

    参数:
        a: 被除数
        b: 除数
    """
    try:
        if b == 0:
            return "错误：除数不能为零"
        result = a / b
        return f"{a} / {b} = {result}"
    except Exception as e:
        return f"计算错误：{e}"
```

### 4. 返回字符串
```python
# ✅ 好：返回字符串
@tool
def get_user_info(user_id: str) -> str:
    """获取用户信息"""
    user = {"id": user_id, "name": "张三"}
    return json.dumps(user, ensure_ascii=False)  # 转成 JSON 字符串

# ❌ 不好：返回字典（某些情况可能有问题）
@tool
def get_user_info(user_id: str) -> dict:
    """获取用户信息"""
    return {"id": user_id, "name": "张三"}
```

## 测试工具

每个工具文件都可以直接运行测试：

```python
# 在文件末尾添加
if __name__ == "__main__":
    print("测试工具：")
    print(my_tool.invoke({"param": "test"}))
```

运行测试：
```bash
python tools/weather.py
```

## 项目结构

```
04_custom_tools/
├── main.py                # 6 个示例
├── README.md              # 本文件
└── tools/                 # 工具目录
    ├── weather.py         # 天气工具
    ├── calculator.py      # 计算器工具
    └── web_search.py      # 搜索工具
```

## 运行示例

```bash
# 测试单个工具
python tools/weather.py

# 运行所有示例
python main.py
```

## 下一步

**05_simple_agent** - 使用 `create_agent` 让 AI 自动调用这些工具
