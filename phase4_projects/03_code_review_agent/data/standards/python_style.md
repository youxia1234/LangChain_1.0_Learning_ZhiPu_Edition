# Python 代码风格规范 (PEP 8 核心规则)

本文档定义了 Python 代码审查中遵循的代码风格标准，基于 PEP 8 并结合实际项目需求进行了扩展。
每条规则均包含正确与错误的代码示例，用于指导自动化代码审查 Agent 的检测逻辑。

---

## 目录

1. [命名规范](#1-命名规范)
2. [导入排序](#2-导入排序)
3. [行宽限制](#3-行宽限制)
4. [缩进规则](#4-缩进规则)
5. [空行规则](#5-空行规则)
6. [注释规范](#6-注释规范)
7. [文档字符串规范 (Google 风格)](#7-文档字符串规范-google-风格)
8. [代码格式化最佳实践](#8-代码格式化最佳实践)

---

## 1. 命名规范

Python 中有三种主要的命名风格，必须根据上下文正确使用。

### 1.1 命名风格总览

| 类型 | 风格 | 示例 | 适用场景 |
|------|------|------|----------|
| snake_case | 小写 + 下划线 | `user_name`, `get_data()` | 变量、函数、方法、模块 |
| camelCase | 驼峰（首字母小写） | `userName` | **Python 中不推荐使用** |
| PascalCase | 驼峰（首字母大写） | `UserName`, `HttpClient` | 类、异常、类型别名 |
| UPPER_SNAKE_CASE | 全大写 + 下划线 | `MAX_RETRIES`, `PI` | 常量 |
| _leading_underscore | 前置下划线 | `_internal_var` | 模块/类内部使用（私有） |
| __double_underscore | 双前置下划线 | `__private_method` | 名称改写（防止继承冲突） |

### 1.2 变量和函数命名（snake_case）

**错误示例：**

```python
# 错误：使用驼峰命名
def getUserName(userId):
    userName = fetchFromDb(userId)
    return userName

# 错误：缩写不清晰
def get_unm(uid):
    n = db.get(uid)
    return n
```

**正确示例：**

```python
# 正确：使用 snake_case，命名清晰表达意图
def get_user_name(user_id: int) -> str:
    """根据用户 ID 获取用户名。"""
    user_name = fetch_from_database(user_id)
    return user_name
```

### 1.3 类命名（PascalCase）

**错误示例：**

```python
# 错误：类名使用 snake_case
class user_manager:
    pass

# 错误：类名使用 camelCase
class httpClient:
    pass
```

**正确示例：**

```python
# 正确：类名使用 PascalCase
class UserManager:
    """管理用户数据的类。"""
    pass

class HttpClient:
    """HTTP 客户端封装类。"""
    pass
```

### 1.4 常量命名（UPPER_SNAKE_CASE）

**错误示例：**

```python
# 错误：常量使用小写
maxRetries = 3
defaultTimeout = 30
apiBaseUrl = "https://api.example.com"
```

**正确示例：**

```python
# 正确：常量使用全大写 + 下划线
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"
```

### 1.5 私有成员命名

**错误示例：**

```python
class DataProcessor:
    def process(self):
        # 错误：没有约定私有方法
        self.helper_method()

    def helper_method(self):
        pass
```

**正确示例：**

```python
class DataProcessor:
    """数据处理器。"""

    def process(self):
        """公开接口：处理数据。"""
        self._validate_input()
        self._transform_data()

    def _validate_input(self):
        """内部方法：验证输入数据（单下划线表示内部使用）。"""
        pass

    def _transform_data(self):
        """内部方法：转换数据格式。"""
        pass
```

### 1.6 避免使用的命名

**错误示例：**

```python
# 错误：使用内置名称覆盖
list = [1, 2, 3]        # 覆盖了内置 list
dict = {"a": 1}         # 覆盖了内置 dict
id = 123                # 覆盖了内置 id()
input = "hello"         # 覆盖了内置 input()
type = "user"           # 覆盖了内置 type()
str = "text"            # 覆盖了内置 str

# 错误：使用单字母命名（循环计数器除外）
def calculate(x, y, z):
    a = x + y
    b = a * z
    return b
```

**正确示例：**

```python
# 正确：避免遮蔽内置名称
user_list = [1, 2, 3]
config_dict = {"a": 1}
user_id = 123
user_input = "hello"
user_type = "user"
text_content = "text"

# 正确：使用有意义的变量名
def calculate_total(base_price: float, tax_rate: float, quantity: int) -> float:
    subtotal = base_price + tax_rate
    total = subtotal * quantity
    return total

# 正确：循环计数器可以使用单字母
for i in range(10):
    print(i)
```

---

## 2. 导入排序

导入语句应按以下顺序分组，每组之间用空行分隔：
1. 标准库模块（stdlib）
2. 第三方库（third-party）
3. 本地模块（local）

### 2.1 导入排序规则

**错误示例：**

```python
# 错误：导入没有分组排序
import os
from myapp.models import User
import json
from langchain.agents import create_agent
import sys
from .utils import helper
from datetime import datetime
```

**正确示例：**

```python
# 正确：按 stdlib → third-party → local 分组，组内按字母排序
# 标准库
import json
import os
import sys
from datetime import datetime

# 第三方库
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import requests

# 本地模块
from myapp.models import User
from myapp.services import DataService
from .utils import helper
```

### 2.2 导入风格规则

**错误示例：**

```python
# 错误：使用通配符导入
from os import *

# 错误：在 from 导入中使用逗号合并不同模块
from langchain.agents import create_agent, AgentExecutor

# 错误：相对导入和绝对导入混用不一致
from myapp.utils import helper
from .config import settings
from myapp.db import connect
```

**正确示例：**

```python
# 正确：显式导入需要的名称
from os import path, environ

# 正确：同一模块的多个名称可以合并
from langchain.agents import AgentExecutor, create_agent

# 正确：统一使用绝对导入（推荐）或相对导入
from myapp.config import settings
from myapp.db import connect
from myapp.utils import helper
```

### 2.3 避免循环导入

**错误示例：**

```python
# module_a.py
from module_b import process_b  # 循环导入

def process_a():
    return process_b()

# module_b.py
from module_a import process_a  # 循环导入

def process_b():
    return process_a()
```

**正确示例：**

```python
# module_a.py
def process_a():
    """延迟导入避免循环依赖。"""
    from module_b import process_b
    return process_b()

# 或者更好的方式：提取共享逻辑到独立模块
# shared.py
def shared_logic():
    pass
```

---

## 3. 行宽限制

### 3.1 基本规则

- PEP 8 标准行宽：79 个字符
- 项目推荐行宽：120 个字符（现代显示器适配）
- 文档字符串和注释：72 个字符（PEP 8 建议）

**错误示例：**

```python
# 错误：行过长，难以阅读
result = some_function_with_long_name(first_argument, second_argument, third_argument, fourth_argument, fifth_argument, sixth_argument)

# 错误：URL 或路径直接放在代码中导致行过长
url = "https://api.example.com/v1/users/profile/data?user_id=12345&include_details=true&format=json&expand=all"
```

**正确示例：**

```python
# 正确：使用括号隐式续行
result = some_function_with_long_name(
    first_argument,
    second_argument,
    third_argument,
    fourth_argument,
    fifth_argument,
    sixth_argument,
)

# 正确：长字符串使用括号拼接
url = (
    "https://api.example.com/v1/users/profile/data"
    "?user_id=12345"
    "&include_details=true"
    "&format=json"
    "&expand=all"
)
```

### 3.2 链式调用换行

**错误示例：**

```python
# 错误：链式调用挤在一行
result = data.filter(lambda x: x.active).map(lambda x: x.name).sort().distinct().to_list()
```

**正确示例：**

```python
# 正确：链式调用逐行换行，点号对齐
result = (
    data
    .filter(lambda x: x.active)
    .map(lambda x: x.name)
    .sort()
    .distinct()
    .to_list()
)
```

---

## 4. 缩进规则

### 4.1 使用 4 个空格缩进

**错误示例：**

```python
# 错误：使用 Tab 缩进
def greet(name):
	if name:
		print(f"Hello, {name}")

# 错误：使用 2 个空格缩进
def greet(name):
  if name:
    print(f"Hello, {name}")

# 错误：Tab 和空格混用
def greet(name):
    if name:
	    print(f"Hello, {name}")
```

**正确示例：**

```python
# 正确：始终使用 4 个空格缩进
def greet(name: str) -> None:
    if name:
        print(f"Hello, {name}")
```

### 4.2 续行缩进

**错误示例：**

```python
# 错误：续行没有额外缩进，难以区分
def calculate(base_price, tax_rate, discount,
quantity, shipping_fee):
    pass

# 错误：续行缩进层级混乱
def calculate(
    base_price,
        tax_rate,
    discount,
            quantity,
    shipping_fee):
    pass
```

**正确示例：**

```python
# 正确方式一：参数与左括号对齐
def calculate(base_price, tax_rate, discount,
              quantity, shipping_fee):
    pass

# 正确方式二（推荐）：挂行缩进，多加一层缩进
def calculate(
    base_price,
    tax_rate,
    discount,
    quantity,
    shipping_fee,
):
    pass
```

### 4.3 条件表达式缩进

**错误示例：**

```python
# 错误：if 条件续行缩进不明确
if (some_condition and
another_condition and
yet_another_condition):
    do_something()
```

**正确示例：**

```python
# 正确：条件续行使用额外缩进
if (
    some_condition
    and another_condition
    and yet_another_condition
):
    do_something()

# 或者使用运算符结尾的风格
if (some_condition and
        another_condition and
        yet_another_condition):
    do_something()
```

---

## 5. 空行规则

### 5.1 顶层定义之间

**错误示例：**

```python
import os
def func_a():
    pass
def func_b():
    pass
class MyClass:
    def method_a(self):
        pass
    def method_b(self):
        pass
```

**正确示例：**

```python
import os


def func_a():
    """函数 A。"""
    pass


def func_b():
    """函数 B。"""
    pass


class MyClass:
    """示例类。"""

    def method_a(self):
        """方法 A。"""
        pass

    def method_b(self):
        """方法 B。"""
        pass
```

### 5.2 空行规则总结

| 位置 | 空行数量 |
|------|----------|
| 顶层函数/类定义之间 | 2 个空行 |
| 类中方法定义之间 | 1 个空行 |
| 函数/方法内部逻辑段落之间 | 1 个空行 |
| 导入语句之后 | 2 个空行 |
| 文件末尾 | 1 个换行符 |

### 5.3 函数内部空行

**错误示例：**

```python
def process_data(data):
    validated = validate(data)
    cleaned = clean(validated)
    transformed = transform(cleaned)
    result = save(transformed)
    return result
```

**正确示例：**

```python
def process_data(data: dict) -> dict:
    """处理数据的完整流程。"""
    # 验证阶段
    validated = validate(data)

    # 清洗阶段
    cleaned = clean(validated)

    # 转换阶段
    transformed = transform(cleaned)

    # 持久化阶段
    result = save(transformed)
    return result
```

---

## 6. 注释规范

### 6.1 注释类型与使用场景

| 类型 | 格式 | 使用场景 |
|------|------|----------|
| 块注释 | `# 注释内容` | 解释后续的一段代码 |
| 行内注释 | `code  # 注释内容` | 解释单行代码 |
| TODO 注释 | `# TODO(作者): 描述` | 标记待办事项 |
| FIXME 注释 | `# FIXME: 描述` | 标记已知问题 |

### 6.2 注释规则

**错误示例：**

```python
# 错误：注释描述的是"做了什么"而非"为什么"
x = x + 1  # x 加 1

# 错误：注释与代码矛盾
total = price * quantity  # 计算折扣后的价格

# 错误：无意义的注释
# 循环遍历列表
for item in items:
    # 打印元素
    print(item)

# 错误：使用中文注释但格式不规范
#TODO 修复这个bug
```

**正确示例：**

```python
# 正确：注释解释"为什么"
x = x + 1  # 补偿数组从 0 开始索引的偏移量

# 正确：注释解释业务逻辑原因
# 使用稳定排序以保持相同分数学生的原始顺序
students.sort(key=lambda s: s.score)

# 正确：块注释完整句首字母大写
# 由于数据库连接池的限制，每次请求完成后必须显式关闭连接，
# 避免连接泄漏导致服务不可用。
db_connection.close()

# 正确：TODO 注释格式规范
# TODO(zhangsan): 添加对分页查询的支持，当前版本仅返回前 100 条

# 正确：FIXME 注释格式规范
# FIXME: 当输入为空列表时会抛出 IndexError，需要添加边界检查
```

### 6.3 行内注释规范

**错误示例：**

```python
result=compute(x,y) #计算结果
```

**正确示例：**

```python
result = compute(x, y)  # 计算结果，注意与行内注释之间至少两个空格
```

---

## 7. 文档字符串规范 (Google 风格)

### 7.1 模块级文档字符串

```python
"""用户管理模块。

本模块提供用户注册、登录、权限管理等功能。
所有用户相关的数据库操作都通过本模块的 UserService 类进行。

典型用法:
    service = UserService(db_connection)
    user = service.get_user(user_id=42)

注意:
    本模块依赖 database 模块提供的连接池功能。
"""
```

### 7.2 函数/方法文档字符串

**错误示例：**

```python
def get_user(id):
    # 没有文档字符串
    return db.query(id)

def calculate(a, b):
    """计算。"""  # 描述不充分
    return a + b
```

**正确示例：**

```python
def get_user(user_id: int) -> dict:
    """根据用户 ID 获取用户信息。

    从数据库中查询指定 ID 的用户信息，如果用户不存在则返回 None。
    查询结果会缓存 5 分钟以减少数据库压力。

    Args:
        user_id (int): 用户的唯一标识符，必须为正整数。

    Returns:
        dict: 包含用户信息的字典，键包括 'id', 'name', 'email'。
            如果用户不存在则返回 None。

    Raises:
        ValueError: 当 user_id 不是正整数时抛出。
        DatabaseError: 当数据库连接失败时抛出。

    Examples:
        >>> user = get_user(42)
        >>> print(user['name'])
        '张三'
    """
    if user_id <= 0:
        raise ValueError(f"user_id 必须为正整数，收到: {user_id}")
    return db.query(user_id)
```

### 7.3 类文档字符串

**错误示例：**

```python
class User:
    pass  # 没有文档字符串
```

**正确示例：**

```python
class User:
    """表示系统中的用户实体。

    本类封装了用户的基本信息和权限管理功能。
    用户创建后会自动分配默认角色。

    Attributes:
        user_id (int): 用户的唯一标识符。
        name (str): 用户的显示名称。
        email (str): 用户的电子邮箱地址。
        roles (list[str]): 用户拥有的角色列表。
        is_active (bool): 用户是否处于活跃状态。

    Examples:
        >>> user = User(user_id=1, name="张三", email="zhang@example.com")
        >>> user.has_permission("read")
        True
    """

    def __init__(
        self,
        user_id: int,
        name: str,
        email: str,
        roles: list[str] | None = None,
    ) -> None:
        """初始化用户实例。

        Args:
            user_id: 用户的唯一标识符。
            name: 用户的显示名称，不能为空。
            email: 用户的邮箱地址，必须符合邮箱格式。
            roles: 用户的角色列表，默认为 ['viewer']。
        """
        self.user_id = user_id
        self.name = name
        self.email = email
        self.roles = roles or ["viewer"]
        self.is_active = True
```

### 7.4 特殊方法文档字符串

```python
def __init__(self, config: dict) -> None:
    """初始化服务实例。"""
    self.config = config

def __repr__(self) -> str:
    """返回实例的正式字符串表示。"""
    return f"User(id={self.user_id}, name={self.name!r})"

def __str__(self) -> str:
    """返回实例的可读字符串表示。"""
    return f"用户: {self.name} ({self.email})"
```

---

## 8. 代码格式化最佳实践

### 8.1 推荐工具

| 工具 | 用途 | 配置文件 |
|------|------|----------|
| Black | 代码格式化 | `pyproject.toml` |
| isort | 导入排序 | `pyproject.toml` |
| flake8 | 风格检查 | `.flake8` |
| mypy | 类型检查 | `pyproject.toml` |
| pylint | 深度代码分析 | `.pylintrc` |
| ruff | 集成格式化+检查 | `pyproject.toml` |

### 8.2 Black 兼容的配置示例

```toml
# pyproject.toml
[tool.black]
line-length = 120
target-version = ['py311']
skip-string-normalization = true

[tool.isort]
profile = "black"
line_length = 120
known_first_party = ["myapp"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

### 8.3 二元运算符周围的空格

**错误示例：**

```python
# 错误：运算符周围缺少空格
x=y+1
result=compute(x,y)
if x>0 and y<10:
    pass

# 错误：关键字参数等号周围有空格
def func(name="default"):
    pass
```

**正确示例：**

```python
# 正确：运算符周围有空格
x = y + 1
result = compute(x, y)
if x > 0 and y < 10:
    pass

# 正确：关键字参数等号周围不加空格
def func(name="default"):
    pass

# 正确：默认参数值不加空格
def connect(host="localhost", port=3306):
    pass
```

### 8.4 逗号规则

**错误示例：**

```python
# 错误：多行结构中末尾缺少逗号
items = [
    "apple",
    "banana",
    "cherry"
]

# 错误：单行元素之间逗号后缺少空格
items = ["apple","banana","cherry"]
```

**正确示例：**

```python
# 正确：多行结构末尾加逗号（便于 git diff 和增删元素）
items = [
    "apple",
    "banana",
    "cherry",
]

# 正确：逗号后有空格
items = ["apple", "banana", "cherry"]
```

### 8.5 字符串引号使用

**错误示例：**

```python
# 错误：引号使用不一致
message = 'Hello'
name = "World"
greeting = f"Hi, {name}"
```

**正确示例：**

```python
# 正确：项目内统一使用双引号（Black 默认行为）
message = "Hello"
name = "World"
greeting = f"Hi, {name}"

# 正确：字符串内包含双引号时使用单引号
dialog = '他说："你好"'

# 正确：三引号用于多行字符串
doc = """这是一个
多行字符串的示例。"""
```

### 8.6 类型注解规范

**错误示例：**

```python
# 错误：缺少类型注解
def process(data, options):
    result = transform(data)
    return result

# 错误：复杂类型未导入
def get_users() -> List[Dict[str, Any]]:
    pass
```

**正确示例：**

```python
from typing import Any

# 正确：使用完整的类型注解
def process(data: dict[str, Any], options: list[str]) -> dict[str, Any]:
    """处理数据并返回结果。"""
    result = transform(data)
    return result

# 正确：使用现代 Python 类型注解语法（3.10+）
def get_users() -> list[dict[str, Any]]:
    """获取所有用户信息。"""
    pass

# 正确：使用 Optional 或 | 语法表示可选参数
def find_user(user_id: int | None = None) -> dict | None:
    """查找用户。"""
    pass
```

### 8.7 pre-commit 配置示例

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: '24.4.2'
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: '5.13.2'
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: '7.0.0'
    hooks:
      - id: flake8
        args: ['--max-line-length=120', '--extend-ignore=E203,E501']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: 'v1.10.0'
    hooks:
      - id: mypy
        additional_dependencies: [types-requests]
```

---

## 审查检测规则总结

代码审查 Agent 在检测代码风格时，应按以下优先级报告问题：

| 优先级 | 规则类别 | 检测方法 |
|--------|----------|----------|
| P0 (必须修复) | 命名遮蔽内置函数、Tab/空格混用、导入错误 | AST 分析 + tokenize |
| P1 (强烈建议) | 命名规范违反、缺少文档字符串、行宽超限 | AST 分析 + 正则匹配 |
| P2 (建议改进) | 注释质量、空行规范、格式统一性 | 启发式规则 + 模式匹配 |
| P3 (可选优化) | 类型注解完善、代码组织优化 | AST 分析 + 类型推断 |
