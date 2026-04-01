# Python 性能优化编码规范

本文档总结了 Python 开发中常见的性能反模式及其优化方案。
每种模式均包含慢代码和快代码的对比，用于指导代码审查 Agent 识别性能问题并提出优化建议。

---

## 目录

1. [N+1 数据库查询](#1-n1-数据库查询)
2. [不必要的列表拷贝](#2-不必要的列表拷贝)
3. [字符串拼接](#3-字符串拼接)
4. [全局变量查找开销](#4-全局变量查找开销)
5. [阻塞式 IO](#5-阻塞式-io)
6. [内存泄漏模式](#6-内存泄漏模式)
7. [低效循环](#7-低效循环)
8. [集合操作优化](#8-集合操作优化)
9. [生成器 vs 列表](#9-生成器-vs-列表)

---

## 1. N+1 数据库查询

**问题：** 在循环中逐条查询数据库，导致查询次数为 N+1（1 次主查询 + N 次关联查询），而非一次性批量获取。
**影响：** 数据量大时性能严重下降，网络往返延迟累积。

### 1.1 慢代码：N+1 查询模式

```python
from typing import Any
import sqlite3


def get_users_with_orders() -> list[dict[str, Any]]:
    """获取所有用户及其订单信息 - N+1 查询模式。"""
    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 第 1 次查询：获取所有用户
    users = cursor.execute("SELECT id, name FROM users").fetchall()

    result = []
    for user in users:
        # 每个用户执行 1 次查询（N 次查询）
        orders = cursor.execute(
            "SELECT id, total FROM orders WHERE user_id = ?",
            (user["id"],)
        ).fetchall()

        result.append({
            "user_id": user["id"],
            "user_name": user["name"],
            "orders": [dict(o) for o in orders],
        })

    # 总查询次数：1 + len(users) 次
    # 如果有 1000 个用户，就执行 1001 次查询
    return result


def get_articles_with_tags(article_ids: list[int]) -> list[dict]:
    """获取文章及其标签 - N+1 查询模式。"""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    result = []
    for article_id in article_ids:
        # 每篇文章执行 1 次标签查询
        tags = cursor.execute(
            "SELECT tag_name FROM article_tags WHERE article_id = ?",
            (article_id,)
        ).fetchall()
        result.append({"article_id": article_id, "tags": tags})

    return result
```

### 1.2 快代码：批量查询模式

```python
from typing import Any
import sqlite3


def get_users_with_orders() -> list[dict[str, Any]]:
    """获取所有用户及其订单信息 - 批量查询模式。"""
    conn = sqlite3.connect("app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 第 1 次查询：获取所有用户
    users = cursor.execute("SELECT id, name FROM users").fetchall()
    user_ids = [user["id"] for user in users]

    # 第 2 次查询：一次性获取所有用户的订单
    placeholders = ",".join("?" * len(user_ids))
    orders = cursor.execute(
        f"SELECT user_id, id, total FROM orders WHERE user_id IN ({placeholders})",
        user_ids,
    ).fetchall()

    # 在内存中按 user_id 分组
    orders_by_user: dict[int, list] = {}
    for order in orders:
        uid = order["user_id"]
        if uid not in orders_by_user:
            orders_by_user[uid] = []
        orders_by_user[uid].append(dict(order))

    result = []
    for user in users:
        result.append({
            "user_id": user["id"],
            "user_name": user["name"],
            "orders": orders_by_user.get(user["id"], []),
        })

    # 总查询次数：仅 2 次，无论有多少用户
    return result


def get_articles_with_tags(article_ids: list[int]) -> list[dict]:
    """获取文章及其标签 - 批量查询模式。"""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # 一次查询获取所有文章的标签
    placeholders = ",".join("?" * len(article_ids))
    tags = cursor.execute(
        f"SELECT article_id, tag_name FROM article_tags WHERE article_id IN ({placeholders})",
        article_ids,
    ).fetchall()

    # 内存分组
    tags_by_article: dict[int, list[str]] = {}
    for tag in tags:
        aid = tag[0]
        if aid not in tags_by_article:
            tags_by_article[aid] = []
        tags_by_article[aid].append(tag[1])

    return [
        {"article_id": aid, "tags": tags_by_article.get(aid, [])}
        for aid in article_ids
    ]
```

### 1.3 ORM 中的预加载 (SQLAlchemy)

```python
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload
from models import User, Order


# 慢：N+1 查询
def get_users_slow(session: Session) -> list[User]:
    users = session.execute(select(User)).scalars().all()
    for user in users:
        # 每次访问 user.orders 都触发一次查询
        _ = user.orders  # Lazy loading 导致 N+1
    return users


# 快：使用 selectinload 预加载
def get_users_fast(session: Session) -> list[User]:
    stmt = select(User).options(selectinload(User.orders))
    return session.execute(stmt).scalars().all()
```

### 1.4 检测方法

```python
DETECTION_RULES = {
    "n_plus_one_query": {
        "patterns": [
            "循环内执行数据库查询",
            "for 循环或列表推导内包含 cursor.execute()",
            "ORM 中遍历关联属性但未使用 eager loading",
        ],
        "severity": "high",
    }
}
```

---

## 2. 不必要的列表拷贝

**问题：** 在不需要修改列表的情况下创建了不必要的副本，浪费内存和 CPU 时间。
**影响：** 大列表拷贝代价高昂，尤其在循环中重复拷贝时。

### 2.1 慢代码：不必要的拷贝

```python
from typing import Any


def process_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """处理数据列表 - 不必要的拷贝。"""
    # 慢：创建了一个不必要的完整副本
    working_copy = list(items)

    # 慢：对列表排序时创建副本
    sorted_copy = sorted(working_copy, key=lambda x: x["id"])

    # 慢：切片创建了完整副本
    subset = sorted_copy[:]

    # 慢：传递给函数时创建副本
    result = validate(list(subset))
    return result


def filter_active_users(users: list[dict]) -> list[dict]:
    """筛选活跃用户 - 不必要的拷贝。"""
    # 慢：先拷贝再过滤
    all_users = list(users)
    active = list(filter(lambda u: u["active"], all_users))
    return active


def merge_lists(list_a: list, list_b: list) -> list:
    """合并列表 - 不必要的拷贝。"""
    # 慢：创建两个副本再合并
    copy_a = list_a.copy()
    copy_b = list_b.copy()
    copy_a.extend(copy_b)
    return copy_a
```

### 2.2 快代码：避免不必要的拷贝

```python
from typing import Any


def process_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """处理数据列表 - 直接操作。"""
    # 快：直接使用原列表的引用（如果不需要修改原列表）
    # 原地排序（如果允许修改原列表）
    items.sort(key=lambda x: x["id"])

    # 快：切片只在需要子集时使用
    result = validate(items)
    return result


def filter_active_users(users: list[dict]) -> list[dict]:
    """筛选活跃用户 - 直接过滤。"""
    # 快：列表推导直接创建过滤后的新列表，无需先拷贝
    return [u for u in users if u["active"]]


def merge_lists(list_a: list, list_b: list) -> list:
    """合并列表 - 高效方式。"""
    # 快：使用 + 运算符直接创建合并列表
    return list_a + list_b
```

### 2.3 需要拷贝 vs 不需要拷贝的判断

```python
# 需要拷贝的场景：
# 1. 需要保留原列表不被修改
original = [1, 2, 3]
backup = original.copy()  # 必要的拷贝
original.sort()           # 修改不影响 backup

# 2. 函数需要修改传入的列表但不影响调用方
def add_default(config: dict) -> dict:
    result = config.copy()  # 必要的拷贝
    result.setdefault("timeout", 30)
    return result

# 不需要拷贝的场景：
# 1. 只读取不修改
def count_items(items: list) -> int:
    return len(items)  # 不需要拷贝

# 2. 创建新列表的过滤操作本身就会产生新列表
filtered = [x for x in data if x > 0]  # 已经是新列表，不需要再拷贝
```

### 2.4 检测方法

```python
DETECTION_RULES = {
    "unnecessary_copy": {
        "patterns": [
            "list() 包装已经是列表的对象",
            ".copy() 后未修改原列表",
            "切片 [:] 创建完整副本但无修改需求",
        ],
        "severity": "low",
    }
}
```

---

## 3. 字符串拼接

**问题：** 在循环中使用 `+` 或 f-string 拼接字符串，每次拼接都创建新的字符串对象。
**影响：** 大量字符串拼接时，时间复杂度为 O(n^2)，性能显著下降。

### 3.1 慢代码：循环中使用 + 拼接

```python
def build_csv_header(columns: list[str]) -> str:
    """构建 CSV 表头 - 使用 + 拼接。"""
    # 慢：每次 + 都创建新的字符串对象
    result = ""
    for col in columns:
        result = result + col + ","
    return result.rstrip(",")


def generate_report(data: list[dict]) -> str:
    """生成报告 - 循环内 f-string 拼接。"""
    # 慢：循环中不断创建新的字符串
    report = ""
    for item in data:
        report += f"名称: {item['name']}, 数量: {item['count']}\n"
    return report


def concat_paths(parts: list[str]) -> str:
    """拼接路径 - 循环拼接。"""
    # 慢：路径拼接使用 +
    path = ""
    for part in parts:
        path = path + "/" + part
    return path


def build_html(items: list[str]) -> str:
    """构建 HTML 列表。"""
    # 慢：大量 HTML 字符串拼接
    html = "<ul>"
    for item in items:
        html = html + "<li>" + item + "</li>"
    html = html + "</ul>"
    return html
```

### 3.2 快代码：使用 join 和列表收集

```python
def build_csv_header(columns: list[str]) -> str:
    """构建 CSV 表头 - 使用 join。"""
    # 快：join 一次性分配内存并拼接
    return ",".join(columns)


def generate_report(data: list[dict]) -> str:
    """生成报告 - 使用列表收集 + join。"""
    # 快：先收集到列表，最后一次 join
    lines = [
        f"名称: {item['name']}, 数量: {item['count']}"
        for item in data
    ]
    return "\n".join(lines)


def concat_paths(parts: list[str]) -> str:
    """拼接路径 - 使用 join。"""
    # 快：join 拼接
    return "/".join(parts)


def build_html(items: list[str]) -> str:
    """构建 HTML 列表 - 使用列表推导 + join。"""
    # 快：列表推导 + join
    list_items = [f"<li>{item}</li>" for item in items]
    return f"<ul>{''.join(list_items)}</ul>"
```

### 3.3 性能对比

```python
import timeit

# 测试数据
items = [str(i) for i in range(10000)]

# + 拼接：约 0.5 秒
def concat_plus():
    result = ""
    for item in items:
        result += item
    return result

# join 拼接：约 0.0005 秒（快 1000 倍）
def concat_join():
    return "".join(items)

# 时间对比
time_plus = timeit.timeit(concat_plus, number=100)
time_join = timeit.timeit(concat_join, number=100)
# join 比 + 快约 100-1000 倍（取决于字符串长度和数量）
```

### 3.4 检测方法

```python
DETECTION_RULES = {
    "string_concat_in_loop": {
        "patterns": [
            "for/while 循环内使用 += 拼接字符串",
            "for/while 循环内使用 + 拼接字符串并赋值",
            "循环内累积字符串变量",
        ],
        "severity": "medium",
    }
}
```

---

## 4. 全局变量查找开销

**问题：** Python 中局部变量查找比全局变量快得多（局部变量使用数组索引，全局变量使用字典查找）。
**影响：** 在高频调用的函数中使用全局变量，累积性能损失明显。

### 4.1 慢代码：频繁访问全局变量

```python
import math

# 全局常量
MAX_RETRIES = 3
TIMEOUT = 30
API_BASE_URL = "https://api.example.com"

# 全局缓存
_cache: dict = {}


def calculate_distance(points: list[tuple[float, float]]) -> float:
    """计算距离 - 频繁访问全局 math 模块。"""
    total = 0.0
    for i in range(len(points) - 1):
        # 慢：每次循环都查找全局 math.sqrt
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total += math.sqrt(dx * dx + dy * dy)  # 全局查找
    return total


def process_with_retry(data: dict) -> dict | None:
    """带重试的处理 - 频繁访问全局变量。"""
    for attempt in range(MAX_RETRIES):  # 全局查找 MAX_RETRIES
        try:
            result = call_api(
                API_BASE_URL,      # 全局查找
                data,
                timeout=TIMEOUT,   # 全局查找
            )
            return result
        except Exception:
            continue
    return None


def fibonacci(n: int) -> int:
    """斐波那契 - 全局缓存查找。"""
    # 慢：每次递归都查找全局 _cache
    if n in _cache:
        return _cache[n]
    if n <= 1:
        return n
    result = fibonacci(n - 1) + fibonacci(n - 2)
    _cache[n] = result  # 全局查找 _cache
    return result
```

### 4.2 快代码：局部变量缓存

```python
import math
from functools import lru_cache

# 全局常量（只读，开销可接受）
MAX_RETRIES = 3
TIMEOUT = 30
API_BASE_URL = "https://api.example.com"


def calculate_distance(points: list[tuple[float, float]]) -> float:
    """计算距离 - 局部缓存全局函数。"""
    # 快：将全局 math.sqrt 赋值给局部变量
    _sqrt = math.sqrt
    total = 0.0
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        total += _sqrt(dx * dx + dy * dy)  # 局部查找，更快
    return total


def process_with_retry(data: dict) -> dict | None:
    """带重试的处理 - 局部缓存常量。"""
    # 快：将全局变量缓存为局部变量
    _max_retries = MAX_RETRIES
    _base_url = API_BASE_URL
    _timeout = TIMEOUT

    for attempt in range(_max_retries):
        try:
            result = call_api(_base_url, data, timeout=_timeout)
            return result
        except Exception:
            continue
    return None


# 快：使用 lru_cache 装饰器代替手动全局缓存
@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    """斐波那契 - 使用 lru_cache 装饰器。"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

### 4.3 查找速度对比

```python
# Python 变量查找速度层级（从快到慢）：
# 1. 局部变量 (LOAD_FAST)          - 数组索引访问
# 2. 自由变量 (LOAD_DEREF)         - 单元对象访问
# 3. 全局变量 (LOAD_GLOBAL)        - 字典查找
# 4. 内置名称 (LOAD_BUILTIN)       - 字典查找
# 5. 对象属性 (LOAD_ATTR)          - 属性描述符查找
# 6. 对象方法 (LOAD_ATTR + CALL)   - 属性查找 + 绑定方法创建

# 在高频循环中，将全局/属性查找提升为局部变量可以显著提升性能
```

### 4.4 检测方法

```python
DETECTION_RULES = {
    "global_variable_lookup": {
        "patterns": [
            "紧凑循环内频繁访问全局变量或模块属性",
            "循环内重复调用 math/JSON 等模块函数",
        ],
        "severity": "low",
        "note": "仅在循环执行次数很大时才值得优化",
    }
}
```

---

## 5. 阻塞式 IO

**问题：** 使用同步 HTTP 客户端（如 requests）在需要并发请求的场景下，一个请求必须等待上一个请求完成。
**影响：** 并发 10 个请求时，同步方式需要 10 倍的延迟，异步方式仅需 1 倍延迟。

### 5.1 慢代码：同步阻塞请求

```python
import requests
from typing import Any


def fetch_user_data(user_ids: list[int]) -> list[dict[str, Any]]:
    """批量获取用户数据 - 同步阻塞。"""
    results = []
    for uid in user_ids:
        # 慢：每个请求都阻塞等待，假设每个请求 200ms
        # 100 个用户 = 100 * 200ms = 20 秒
        response = requests.get(f"https://api.example.com/users/{uid}")
        if response.status_code == 200:
            results.append(response.json())
    return results


def fetch_multiple_apis(api_urls: list[str]) -> list[dict]:
    """调用多个 API - 同步阻塞。"""
    results = []
    for url in api_urls:
        # 慢：依次调用，延迟累加
        resp = requests.get(url, timeout=10)
        results.append(resp.json())
    return results


def process_files(file_urls: list[str]) -> list[str]:
    """下载多个文件 - 同步阻塞。"""
    contents = []
    for url in file_urls:
        # 慢：每个文件下载都阻塞
        resp = requests.get(url)
        contents.append(resp.text)
    return contents
```

### 5.2 快代码：异步非阻塞请求

```python
import asyncio
import aiohttp
from typing import Any


async def fetch_one(session: aiohttp.ClientSession, url: str) -> dict[str, Any]:
    """异步获取单个资源。"""
    async with session.get(url) as response:
        return await response.json()


async def fetch_user_data_async(user_ids: list[int]) -> list[dict[str, Any]]:
    """批量获取用户数据 - 异步并发。"""
    # 快：所有请求同时发出，假设每个请求 200ms
    # 100 个用户 ≈ 200ms（并发执行）
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_one(session, f"https://api.example.com/users/{uid}")
            for uid in user_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]


async def fetch_multiple_apis_async(api_urls: list[str]) -> list[dict]:
    """调用多个 API - 异步并发。"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in api_urls]
        return await asyncio.gather(*tasks)


async def fetch_with_semaphore(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """带并发限制的异步获取。"""
    async with semaphore:
        async with session.get(url) as response:
            return await response.text()


async def process_files_async(file_urls: list[str]) -> list[str]:
    """下载多个文件 - 异步并发 + 信号量限制。"""
    # 快：并发下载，但限制最大并发数为 10
    semaphore = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_with_semaphore(session, url, semaphore)
            for url in file_urls
        ]
        return await asyncio.gather(*tasks)


# 使用 concurrent.futures 进行多线程（适合 IO 密集但无法改写为 async 的场景）
import concurrent.futures
import requests


def fetch_user_data_threaded(user_ids: list[int]) -> list[dict[str, Any]]:
    """使用线程池并发请求。"""
    def fetch(uid: int) -> dict:
        resp = requests.get(f"https://api.example.com/users/{uid}")
        return resp.json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch, user_ids))
    return results
```

### 5.3 检测方法

```python
DETECTION_RULES = {
    "blocking_io": {
        "patterns": [
            "for 循环内使用 requests.get/post 发起 HTTP 请求",
            "循环内使用 urllib.request 发起请求",
            "循环内使用 open() 读取多个文件（可用 aiofiles 替代）",
        ],
        "severity": "medium",
        "note": "仅在需要并发 IO 时才建议改为异步",
    }
}
```

---

## 6. 内存泄漏模式

**问题：** Python 的垃圾回收机制不能处理所有内存泄漏场景，特别是循环引用和全局缓存无限增长。
**影响：** 长期运行的服务进程内存持续增长，最终导致 OOM (Out of Memory)。

### 6.1 慢代码（泄漏）：循环引用和无限缓存

```python
from typing import Any, Callable
import weakref


# 泄漏模式 1：全局缓存无限增长
class Cache:
    """全局缓存 - 无限增长。"""
    _store: dict[str, Any] = {}  # 危险：只增不减

    @classmethod
    def set(cls, key: str, value: Any) -> None:
        cls._store[key] = value  # 永远不会清理

    @classmethod
    def get(cls, key: str) -> Any:
        return cls._store.get(key)


# 泄漏模式 2：循环引用
class Node:
    """链表节点 - 循环引用。"""
    def __init__(self, value: int):
        self.value = value
        self.parent: "Node | None" = None
        self.children: list["Node"] = []

    def add_child(self, child: "Node") -> None:
        # 危险：子节点引用父节点，形成循环引用
        self.children.append(child)
        child.parent = self  # 循环引用


# 泄漏模式 3：闭包捕获大对象
def create_handlers(data: list[dict]) -> list[Callable]:
    """创建处理器列表 - 闭包泄漏。"""
    handlers = []
    for item in data:
        # 危险：每个闭包都捕获了整个 data 列表的引用
        def handler(x: int) -> dict:
            return data[x]  # 闭包引用整个 data
        handlers.append(handler)
    return handlers


# 泄漏模式 4：事件监听器未注销
class EventBus:
    """事件总线 - 监听器泄漏。"""
    _listeners: dict[str, list[Callable]] = {}

    @classmethod
    def on(cls, event: str, callback: Callable) -> None:
        if event not in cls._listeners:
            cls._listeners[event] = []
        cls._listeners[event].append(callback)
        # 危险：只添加，从不移除


# 泄漏模式 5：__del__ 阻止垃圾回收
class Resource:
    """资源管理 - __del__ 阻止 GC。"""
    def __init__(self, name: str):
        self.name = name

    def __del__(self):
        # 危险：__del__ 方法中的循环引用无法被 GC 回收
        print(f"释放资源: {self.name}")
```

### 6.2 快代码（无泄漏）：弱引用和有界缓存

```python
from typing import Any, Callable
from collections import OrderedDict
import weakref


# 修复 1：有界缓存 (LRU Cache)
class BoundedCache:
    """有界 LRU 缓存 - 自动淘汰旧数据。"""
    def __init__(self, max_size: int = 1000):
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._max_size = max_size

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        # 超过容量时淘汰最旧的条目
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def get(self, key: str) -> Any:
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        return None


# 修复 2：使用弱引用打破循环引用
class SafeNode:
    """链表节点 - 使用弱引用。"""
    def __init__(self, value: int):
        self.value = value
        self._parent_ref: weakref.ref | None = None
        self.children: list["SafeNode"] = []

    @property
    def parent(self) -> "SafeNode | None":
        if self._parent_ref is not None:
            return self._parent_ref()
        return None

    def add_child(self, child: "SafeNode") -> None:
        self.children.append(child)
        # 安全：使用弱引用，不阻止 GC 回收父节点
        child._parent_ref = weakref.ref(self)


# 修复 3：闭包只捕获必要的数据
def create_handlers_safe(data: list[dict]) -> list[Callable]:
    """创建处理器列表 - 只捕获单个元素。"""
    handlers = []
    for item in data:
        # 安全：闭包只捕获当前 item，不引用整个 data
        def handler(x: int, _item: dict = item) -> dict:
            return _item
        handlers.append(handler)
    return handlers


# 修复 4：使用 context manager 代替 __del__
class SafeResource:
    """资源管理 - 使用上下文管理器。"""
    def __init__(self, name: str):
        self.name = name

    def __enter__(self) -> "SafeResource":
        return self

    def __exit__(self, *args) -> None:
        print(f"释放资源: {self.name}")

    # 安全：不定义 __del__，避免阻止 GC


# 修复 5：使用 functools.lru_cache 代替手动缓存
from functools import lru_cache

@lru_cache(maxsize=256)
def expensive_computation(n: int) -> int:
    """使用 lru_cache 自动管理缓存大小。"""
    return n * n
```

### 6.3 检测方法

```python
DETECTION_RULES = {
    "memory_leak": {
        "patterns": [
            "类变量或模块级字典无限增长（无淘汰策略）",
            "循环引用（父子互相引用但未使用弱引用）",
            "闭包捕获外部大对象",
            "事件监听器只注册不注销",
            "定义 __del__ 方法且可能产生循环引用",
        ],
        "severity": "medium",
    }
}
```

---

## 7. 低效循环

**问题：** 使用 for 循环处理可以通过列表推导、map 或内置函数更高效完成的操作。
**影响：** Python 的 for 循环比 C 层实现的内置函数慢数倍。

### 7.1 慢代码：低效循环模式

```python
# 慢：for 循环构建列表
def square_numbers(numbers: list[int]) -> list[int]:
    result = []
    for n in numbers:
        result.append(n * n)
    return result


# 慢：for 循环过滤
def filter_positive(numbers: list[int]) -> list[int]:
    result = []
    for n in numbers:
        if n > 0:
            result.append(n)
    return result


# 慢：for 循环求和
def total_price(items: list[dict]) -> float:
    total = 0.0
    for item in items:
        total += item["price"] * item["quantity"]
    return total


# 慢：for 循环查找
def find_user_by_id(users: list[dict], user_id: int) -> dict | None:
    for user in users:
        if user["id"] == user_id:
            return user
    return None


# 慢：for 循环计数
def count_words(text: str) -> int:
    words = text.split()
    count = 0
    for word in words:
        count += 1
    return count
```

### 7.2 快代码：列表推导和内置函数

```python
# 快：列表推导
def square_numbers(numbers: list[int]) -> list[int]:
    return [n * n for n in numbers]


# 快：列表推导 + 条件过滤
def filter_positive(numbers: list[int]) -> list[int]:
    return [n for n in numbers if n > 0]


# 快：生成器表达式 + sum（避免创建中间列表）
def total_price(items: list[dict]) -> float:
    return sum(item["price"] * item["quantity"] for item in items)


# 快：next + 生成器表达式（找到第一个匹配即停止）
def find_user_by_id(users: list[dict], user_id: int) -> dict | None:
    return next((u for u in users if u["id"] == user_id), None)


# 快：内置 len 函数
def count_words(text: str) -> int:
    return len(text.split())


# 快：map 和 filter（适用于简单转换）
def get_all_names(users: list[dict]) -> list[str]:
    return list(map(lambda u: u["name"], users))


# 快：all/any 替代循环检查
def all_active(users: list[dict]) -> bool:
    return all(u["active"] for u in users)

def has_admin(users: list[dict]) -> bool:
    return any(u["role"] == "admin" for u in users)
```

### 7.3 性能对比

```python
import timeit

data = list(range(10000))

# for 循环 + append：约 1.2ms
def for_loop():
    result = []
    for x in data:
        result.append(x * x)
    return result

# 列表推导：约 0.6ms（快 2 倍）
def list_comp():
    return [x * x for x in data]

# map：约 0.5ms（快 2.4 倍）
def map_func():
    return list(map(lambda x: x * x, data))

# numpy（大数据场景）：约 0.05ms（快 24 倍）
import numpy as np
def numpy_func():
    arr = np.array(data)
    return arr * arr
```

### 7.4 检测方法

```python
DETECTION_RULES = {
    "inefficient_loop": {
        "patterns": [
            "for 循环仅做 list.append()（应使用列表推导）",
            "for 循环仅做累加（应使用 sum()）",
            "for 循环仅做查找（应使用 next() 或 dict 查找）",
            "for 循环做 all/any 检查（应使用内置函数）",
        ],
        "severity": "low",
    }
}
```

---

## 8. 集合操作优化

**问题：** 在 list 上进行成员检查（`in` 操作），时间复杂度为 O(n)，而 set 为 O(1)。
**影响：** 当列表较大且查询频繁时，性能差距可达数百倍。

### 8.1 慢代码：在 list 上做查找

```python
from typing import Any


def filter_valid_items(items: list[str], valid_items: list[str]) -> list[str]:
    """过滤有效项目 - list 查找 O(n)。"""
    result = []
    for item in items:
        # 慢：list 的 in 操作是 O(n)
        # 假设 items 有 10000 个，valid_items 有 5000 个
        # 总比较次数：10000 * 5000 = 50,000,000
        if item in valid_items:
            result.append(item)
    return result


def remove_duplicates(items: list[int]) -> list[int]:
    """去重 - 手动实现。"""
    # 慢：使用 list 检查是否已存在
    result = []
    for item in items:
        if item not in result:  # O(n) 查找
            result.append(item)
    return result


def find_common_elements(list_a: list[int], list_b: list[int]) -> list[int]:
    """查找交集 - 嵌套循环。"""
    # 慢：嵌套循环 O(n*m)
    result = []
    for a in list_a:
        for b in list_b:
            if a == b and a not in result:
                result.append(a)
    return result


def count_occurrences(items: list[str]) -> dict[str, int]:
    """统计出现次数 - 手动实现。"""
    # 慢：手动遍历统计
    result = {}
    for item in items:
        if item in result:
            result[item] += 1
        else:
            result[item] = 1
    return result
```

### 8.2 快代码：使用 set 和 collections

```python
from collections import Counter


def filter_valid_items(items: list[str], valid_items: list[str]) -> list[str]:
    """过滤有效项目 - set 查找 O(1)。"""
    # 快：先转换为 set，in 操作变成 O(1)
    valid_set = set(valid_items)
    return [item for item in items if item in valid_set]
    # 总查找次数：10000 * O(1) ≈ 10000 次 hash 查找


def remove_duplicates(items: list[int]) -> list[int]:
    """去重 - 使用 set。"""
    # 快：set 自动去重，然后转回 list
    # 注意：这不保持原始顺序
    return list(set(items))


def remove_duplicates_ordered(items: list[int]) -> list[int]:
    """去重并保持顺序。"""
    # 快：使用 dict 保持顺序（Python 3.7+）
    return list(dict.fromkeys(items))


def find_common_elements(list_a: list[int], list_b: list[int]) -> list[int]:
    """查找交集 - 使用 set 操作。"""
    # 快：set 交集操作 O(min(n, m))
    return list(set(list_a) & set(list_b))


def count_occurrences(items: list[str]) -> dict[str, int]:
    """统计出现次数 - 使用 Counter。"""
    # 快：Counter 使用 C 实现，比手动循环快
    return dict(Counter(items))
```

### 8.3 数据结构选择指南

| 操作 | list | set | dict | 推荐选择 |
|------|------|-----|------|----------|
| 成员检查 (x in ...) | O(n) | O(1) | O(1) | set/dict |
| 添加元素 | O(1) | O(1) | O(1) | 均可 |
| 访问第 n 个元素 | O(1) | 不支持 | 不支持 | list |
| 去重 | O(n^2) | O(n) | O(n) | set |
| 交集/并集 | O(n*m) | O(min(n,m)) | - | set |
| 键值映射 | 不适用 | 不适用 | O(1) | dict |
| 保持顺序 | 是 | 否（3.7+ dict 保序） | 是（3.7+） | list/dict |

### 8.4 检测方法

```python
DETECTION_RULES = {
    "collection_optimization": {
        "patterns": [
            "在循环内对 list 使用 in 操作（应转为 set）",
            "手动实现去重逻辑（应使用 set）",
            "手动实现计数逻辑（应使用 Counter）",
            "嵌套循环查找交集（应使用 set 操作）",
        ],
        "severity": "medium",
    }
}
```

---

## 9. 生成器 vs 列表

**问题：** 一次性创建包含所有结果的列表，而实际上只需要逐个处理，浪费内存。
**影响：** 处理大数据集时，列表版本可能耗尽内存，生成器版本只需常数内存。

### 9.1 慢代码（高内存）：使用列表

```python
from typing import Generator


def read_large_csv(filepath: str) -> list[dict]:
    """读取大型 CSV - 一次性加载到列表。"""
    # 高内存：将所有行存储在列表中
    # 1000 万行 * 每行 200 字节 ≈ 2GB 内存
    results = []
    with open(filepath, "r") as f:
        headers = f.readline().strip().split(",")
        for line in f:
            values = line.strip().split(",")
            results.append(dict(zip(headers, values)))
    return results


def process_range(start: int, end: int) -> list[int]:
    """处理范围 - 使用 range 创建列表。"""
    # 高内存：创建包含所有元素的列表
    # range(10000000) 创建 1000 万个整数的列表
    numbers = list(range(start, end))
    return [n * 2 for n in numbers]


def filter_log_entries(entries: list[str]) -> list[str]:
    """过滤日志 - 列表推导创建完整列表。"""
    # 高内存：即使只需要逐行处理，也创建了完整列表
    return [
        entry for entry in entries
        if "ERROR" in entry or "CRITICAL" in entry
    ]


def chain_operations(data: list[int]) -> list[int]:
    """链式操作 - 创建多个中间列表。"""
    # 高内存：每一步都创建新的中间列表
    step1 = [x * 2 for x in data]          # 中间列表 1
    step2 = [x for x in step1 if x > 10]   # 中间列表 2
    step3 = [x + 1 for x in step2]         # 中间列表 3
    return step3
```

### 9.2 快代码（低内存）：使用生成器

```python
from typing import Generator


def read_large_csv(filepath: str) -> Generator[dict, None, None]:
    """读取大型 CSV - 生成器逐行返回。"""
    # 低内存：每次只处理一行，内存占用恒定
    with open(filepath, "r") as f:
        headers = f.readline().strip().split(",")
        for line in f:
            values = line.strip().split(",")
            yield dict(zip(headers, values))


def process_range(start: int, end: int) -> Generator[int, None, None]:
    """处理范围 - 使用生成器表达式。"""
    # 低内存：不创建列表，逐个生成
    return (n * 2 for n in range(start, end))


def filter_log_entries(entries: list[str]) -> Generator[str, None, None]:
    """过滤日志 - 生成器表达式。"""
    # 低内存：逐个过滤，不创建中间列表
    return (
        entry for entry in entries
        if "ERROR" in entry or "CRITICAL" in entry
    )


def chain_operations(data: list[int]) -> Generator[int, None, None]:
    """链式操作 - 生成器管道。"""
    # 低内存：整条管道不创建任何中间列表
    step1 = (x * 2 for x in data)           # 生成器
    step2 = (x for x in step1 if x > 10)    # 生成器
    step3 = (x + 1 for x in step2)          # 生成器
    return step3


# 实际使用示例：逐行处理大文件
def count_errors(log_file: str) -> int:
    """统计日志中的错误数量 - 流式处理。"""
    count = 0
    for entry in read_large_csv(log_file):
        if entry.get("level") in ("ERROR", "CRITICAL"):
            count += 1
    return count
```

### 9.3 何时使用列表 vs 生成器

| 场景 | 推荐 | 原因 |
|------|------|------|
| 需要多次遍历 | 列表 | 生成器只能遍历一次 |
| 需要随机访问 (data[i]) | 列表 | 生成器不支持索引 |
| 需要获取长度 (len()) | 列表 | 生成器没有长度 |
| 数据量大，只需逐个处理 | 生成器 | 节省内存 |
| 只需要前 N 个结果 | 生成器 | 避免计算全部结果 |
| 管道式链式操作 | 生成器 | 不创建中间列表 |
| 函数返回值给外部使用 | 视情况 | 小数据用列表，大数据用生成器 |

### 9.4 检测方法

```python
DETECTION_RULES = {
    "list_vs_generator": {
        "patterns": [
            "大文件读取后一次性加载到列表",
            "链式列表推导创建多个中间列表",
            "range() 被包装在 list() 中但只需迭代",
            "列表推导结果只被迭代一次",
        ],
        "severity": "low",
        "note": "仅在数据量较大时才建议改为生成器",
    }
}
```

---

## 性能检测优先级总结

代码审查 Agent 在检测性能问题时，应按以下优先级报告：

| 优先级 | 问题类型 | 预估性能影响 |
|--------|----------|------------|
| P0 (严重) | N+1 数据库查询 | 10x - 1000x |
| P1 (高) | 阻塞式 IO 并发场景 | 10x - 100x |
| P2 (中) | 集合操作优化 (list vs set) | 10x - 100x |
| P2 (中) | 字符串循环拼接 | 10x - 100x |
| P2 (中) | 内存泄漏 | 长期影响，可导致 OOM |
| P3 (低) | 低效循环 | 2x - 5x |
| P3 (低) | 全局变量查找 | 1.2x - 2x |
| P3 (低) | 不必要拷贝 | 1.5x - 3x |
| P3 (低) | 列表 vs 生成器 | 内存影响为主 |
