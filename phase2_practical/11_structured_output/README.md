# 11 - Structured Output (结构化输出)

## 核心概念

**Structured Output = 将 LLM 的自然语言输出转为结构化 Python 对象**

在 LangChain 1.0 中，理论上可以使用 `with_structured_output()` 方法结合 Pydantic 模型，确保 LLM 返回符合预定义模式的数据。

**重要说明**：
- 智谱 AI（本模块使用的模型）**不直接支持** `with_structured_output()` 方法
- 本模块通过 **JSON 解析方式** 实现结构化输出，功能完全相同
- 如果你使用 OpenAI GPT-4、Anthropic Claude 等支持原生结构化输出的模型，可以直接使用 `with_structured_output()`
- 本模块的 `structured_output()` 函数可以自动处理这两种情况

## 基本用法

### 定义 Pydantic 模型

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """人物信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    occupation: str = Field(description="职业")
```

### 使用结构化输出

**方法1：使用 `with_structured_output()`（支持原生结构化输出的模型）**

```python
from langchain.chat_models import init_chat_model

# 使用 OpenAI GPT-4
model = init_chat_model("openai:gpt-4")

# 创建结构化输出的 LLM
structured_llm = model.with_structured_output(Person)

# 调用
result = structured_llm.invoke("张三是一名 30 岁的软件工程师")

# result 是 Person 实例
print(result.name)       # "张三"
print(result.age)        # 30
print(result.occupation) # "软件工程师"
```

**方法2：使用 JSON 解析方式（本模块的方法，兼容所有模型）**

```python
from langchain_openai import ChatOpenAI

# 使用智谱 AI（不支持原生结构化输出）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key="your_api_key",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# 使用本模块提供的 structured_output 函数（main.py 中定义）
# 这是自定义的辅助函数，通过 JSON 解析实现结构化输出
result = structured_output("张三是一名 30 岁的软件工程师", Person)

# result 仍然是 Person 实例
print(result.name)       # "张三"
print(result.age)        # 30
print(result.occupation) # "软件工程师"
```

**注意**：`structured_output(prompt, Model)` 函数是本模块 `main.py` 中提供的自定义辅助函数，不是 LangChain 的内置方法。如果你使用支持原生结构化输出的模型（如 GPT-4），可以直接使用 `model.with_structured_output(Model)`。

## 核心组件

### 1. Pydantic BaseModel

所有结构化输出的数据模型都必须继承 `BaseModel`：

```python
from pydantic import BaseModel

class MyModel(BaseModel):
    field1: str
    field2: int
```

### 2. Field 描述

使用 `Field()` 添加字段描述，帮助 LLM 理解：

```python
from pydantic import Field

class Book(BaseModel):
    title: str = Field(description="书名")
    author: str = Field(description="作者")
    year: int = Field(description="出版年份")
```

**重要**：`description` 会传递给 LLM，帮助它正确填充字段。

### 3. 类型注解

Pydantic 支持丰富的类型：

```python
from typing import Optional, List

class Product(BaseModel):
    name: str                    # 必填字符串
    price: float                 # 必填浮点数
    description: Optional[str]   # 可选字符串
    tags: List[str]              # 字符串列表
```

## 高级特性

### 可选字段

```python
class User(BaseModel):
    username: str
    email: Optional[str] = None  # 可以为 None
    age: Optional[int] = None
```

### 默认值

```python
class Config(BaseModel):
    timeout: int = 30         # 默认 30
    retry: bool = True        # 默认 True
    max_attempts: int = Field(3, description="最大重试次数")
```

### 枚举类型

```python
from enum import Enum

class Priority(str, Enum):
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"

class Task(BaseModel):
    title: str
    priority: Priority  # 只能是 LOW/MEDIUM/HIGH
```

### 列表提取

```python
class Person(BaseModel):
    name: str
    age: int

class PeopleList(BaseModel):
    people: List[Person]  # 多个 Person 对象

structured_llm = model.with_structured_output(PeopleList)
result = structured_llm.invoke("张三 30岁，李四 25岁")
# result.people = [Person(name="张三", age=30), Person(name="李四", age=25)]
```

### 嵌套模型

```python
class Address(BaseModel):
    city: str
    district: str

class Company(BaseModel):
    name: str
    address: Address  # 嵌套模型

structured_llm = model.with_structured_output(Company)
result = structured_llm.invoke("阿里巴巴在杭州滨江区")
# result.address.city = "杭州"
# result.address.district = "滨江区"
```

## 工作原理

### 传统方式 vs 结构化输出

**传统方式（繁琐）**：
```python
# 1. 提示词要求 JSON
prompt = "以JSON格式返回：{name, age, occupation}"
response = model.invoke(prompt)

# 2. 手动解析
import json
data = json.loads(response.content)

# 3. 手动验证类型
if not isinstance(data['age'], int):
    raise ValueError("age must be int")

# 4. 手动创建对象
person = Person(**data)
```

**结构化输出（简洁）**：
```python
# 一步到位
result = structured_output("张三是一名 30 岁的软件工程师", Person)
# ✅ 自动解析、验证、创建对象
```

### 两种实现方式

**方式1：原生 `with_structured_output()`（支持原生结构化输出的模型）**

```
1. Pydantic 模型 → JSON Schema
   Person → {
     "type": "object",
     "properties": {
       "name": {"type": "string", "description": "姓名"},
       "age": {"type": "integer", "description": "年龄"}
     }
   }

2. JSON Schema → LLM (通过函数调用/工具调用)
   LLM 被强制返回符合 schema 的 JSON

3. JSON → Pydantic 对象
   自动验证类型并创建 Person 实例
```

**方式2：JSON 解析方式（本模块实现，兼容所有模型）**

```
1. Pydantic 模型 → JSON Schema
   同上

2. JSON Schema → 提示词模板
   生成包含字段说明、类型、示例的详细提示词

3. LLM 返回 JSON 文本
   通过精心设计的提示词引导 LLM 返回正确格式

4. JSON 解析 → Pydantic 对象
   清理 Markdown 格式、解析 JSON、验证类型、创建对象
```

## 实际应用

### 1. 客户信息提取

```python
class CustomerInfo(BaseModel):
    name: str = Field(description="客户姓名")
    phone: str = Field(description="电话号码")
    email: Optional[str] = Field(None, description="邮箱")
    issue: str = Field(description="问题描述")

structured_llm = model.with_structured_output(CustomerInfo)

conversation = """
客户: 我是李明，电话 138-1234-5678，订单没发货
"""

info = structured_llm.invoke(f"提取客户信息：{conversation}")
# info.name = "李明"
# info.phone = "138-1234-5678"
# info.issue = "订单没发货"
```

**应用**：
- 自动填充 CRM 系统
- 工单自动分类
- 客服辅助

### 2. 产品评论分析

```python
class Review(BaseModel):
    product: str
    rating: int = Field(description="评分 1-5")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")

structured_llm = model.with_structured_output(Review)

review = structured_llm.invoke("""
iPhone 15 很棒！摄像头强大，手感好。但是价格贵，没有充电器。4分。
""")

# review.product = "iPhone 15"
# review.rating = 4
# review.pros = ["摄像头强大", "手感好"]
# review.cons = ["价格贵", "没有充电器"]
```

**应用**：
- 批量处理用户评论
- 自动生成分析报告
- 发现产品改进点

### 3. 文档信息提取

```python
class Invoice(BaseModel):
    invoice_number: str
    date: str
    total_amount: float
    items: List[str]

structured_llm = model.with_structured_output(Invoice)

invoice_text = """
发票号: INV-2024-001
日期: 2024-01-15
总金额: 1299.00
商品: MacBook Pro, AppleCare+
"""

invoice = structured_llm.invoke(f"提取发票信息：{invoice_text}")
# invoice.invoice_number = "INV-2024-001"
# invoice.total_amount = 1299.00
```

**应用**：
- 自动化财务处理
- OCR 后结构化
- 数据录入

## 常见问题

### 1. LLM 未填充某些字段怎么办？

使用 `Optional` 和默认值：

```python
class Data(BaseModel):
    required_field: str              # 必填
    optional_field: Optional[str] = None  # 可选
    with_default: int = 100          # 有默认值
```

### 2. 如何限制字段的可选值？

使用枚举：

```python
from enum import Enum

class Status(str, Enum):
    ACTIVE = "激活"
    INACTIVE = "未激活"

class User(BaseModel):
    status: Status  # 只能是 ACTIVE 或 INACTIVE
```

### 3. 复杂嵌套结构会出错吗？

LLM 能力有限，建议：
- 嵌套层级 ≤ 3 层
- 使用清晰的 `description`
- 必要时拆分成多个调用

### 4. 如何验证提取的准确性？

见下一章 `12_validation_retry` - 验证和重试机制。

### 5. 所有模型都支持吗？

**原生 `with_structured_output()` 支持情况**：

- ✅ OpenAI (gpt-4, gpt-3.5-turbo, gpt-4o)
- ✅ Anthropic (claude-3-opus, claude-3-sonnet, claude-3-haiku)
- ✅ Groq (llama-3.1, llama-3.3)
- ✅ Azure OpenAI
- ✅ Google Gemini
- ❌ 智谱 AI（不支持原生结构化输出）
- ❌ 某些旧模型不支持

**不支持的模型怎么办？**

- 使用本模块提供的 `structured_output()` 函数，它通过 JSON 解析方式实现结构化输出
- 功能完全相同，兼容所有支持文本生成的模型
- 本模块的所有示例都使用这种方式，确保在智谱 AI 上正常运行

## 最佳实践

```python
# 1. 使用清晰的字段描述
class Good(BaseModel):
    created_at: str = Field(description="创建时间，格式 YYYY-MM-DD")

class Bad(BaseModel):
    created_at: str  # 没有描述，LLM 可能格式错误

# 2. 合理使用 Optional
class Good(BaseModel):
    email: Optional[str] = None  # 邮箱可能没有

class Bad(BaseModel):
    email: str  # 强制必填，可能导致提取失败

# 3. 使用枚举限制值
class Good(BaseModel):
    status: Status  # 枚举

class Bad(BaseModel):
    status: str  # 可能返回任意字符串

# 4. 列表设置合理的描述
class Good(BaseModel):
    tags: List[str] = Field(description="产品标签，如 '电子产品', '手机'")

class Bad(BaseModel):
    tags: List[str]  # LLM 不知道该提取什么

# 5. 嵌套模型保持简单
class Good(BaseModel):
    user: User      # 1 层嵌套
    settings: dict  # 复杂数据用 dict

class Bad(BaseModel):
    user: User
        company: Company
            address: Address
                country: Country  # 4 层嵌套，容易出错
```

## 核心要点

### 实现方式

1. **`with_structured_output(Model)`** - 原生方法（支持 GPT-4、Claude 等）
2. **`structured_output(prompt, Model)`** - JSON 解析方式（兼容所有模型，本模块使用）

### 数据建模

3. **Pydantic BaseModel** - 定义数据模式
4. **Field(description=...)** - 帮助 LLM 理解字段含义
5. **Optional[T]** - 可选字段
6. **List[T]** - 列表类型
7. **Enum** - 限制可选值
8. **嵌套模型** - 处理复杂结构（≤3 层）
9. **自动验证** - Pydantic 自动检查类型

### 重要提示

- 智谱 AI 不支持原生 `with_structured_output()`
- 本模块使用 JSON 解析方式，功能完全相同
- 所有示例在智谱 AI 上经过验证，可正常运行

## 下一步

**12_validation_retry** - 学习如何验证提取结果并处理错误重试
