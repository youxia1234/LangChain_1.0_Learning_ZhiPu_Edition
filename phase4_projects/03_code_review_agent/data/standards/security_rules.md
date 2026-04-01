# Python 安全编码规范

本文档基于 OWASP Top 10 安全风险，针对 Python 开发中的常见安全漏洞提供编码规范。
每类漏洞均包含危险代码示例、安全替代方案和自动化检测方法，用于指导代码审查 Agent 的安全检测逻辑。

---

## 目录

1. [SQL 注入](#1-sql-注入)
2. [跨站脚本攻击 (XSS)](#2-跨站脚本攻击-xss)
3. [命令注入](#3-命令注入)
4. [硬编码密钥和 API Key](#4-硬编码密钥和-api-key)
5. [不安全的反序列化](#5-不安全的反序列化)
6. [路径遍历](#6-路径遍历)
7. [不安全的随机数](#7-不安全的随机数)
8. [敏感信息日志泄露](#8-敏感信息日志泄露)
9. [安全检测规则总结](#9-安全检测规则总结)

---

## 1. SQL 注入

**风险等级：** 严重 (CVSS 9.8)
**OWASP 分类：** A03:2021 - Injection

SQL 注入是 Python Web 应用中最常见的安全漏洞之一。攻击者通过构造恶意输入来操纵 SQL 查询，可能导致数据泄露、数据篡改或数据库服务器被控制。

### 1.1 危险代码：f-string 拼接 SQL

```python
import sqlite3


def get_user(username: str) -> dict:
    """根据用户名查询用户信息。"""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # 危险：直接使用 f-string 拼接用户输入
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)

    # 攻击示例：username = "admin' OR '1'='1"
    # 实际执行：SELECT * FROM users WHERE username = 'admin' OR '1'='1'
    # 结果：返回所有用户数据
    return cursor.fetchone()


def search_products(keyword: str) -> list:
    """搜索产品。"""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # 危险：字符串格式化拼接
    query = "SELECT * FROM products WHERE name LIKE '%%%s%%'" % keyword
    cursor.execute(query)

    # 攻击示例：keyword = "'; DROP TABLE products; --"
    # 可能导致整个产品表被删除
    return cursor.fetchall()


def delete_user(user_id: str) -> None:
    """删除用户。"""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # 危险：使用 + 号拼接
    query = "DELETE FROM users WHERE id = " + user_id
    cursor.execute(query)
    conn.commit()
```

### 1.2 安全代码：参数化查询

```python
import sqlite3
from typing import Any


def get_user(username: str) -> dict | None:
    """根据用户名安全地查询用户信息。

    使用参数化查询防止 SQL 注入。
    """
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # 安全：使用参数化查询（占位符 ?）
    query = "SELECT * FROM users WHERE username = ?"
    cursor.execute(query, (username,))

    return cursor.fetchone()


def search_products(keyword: str) -> list:
    """安全地搜索产品。"""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # 安全：参数化查询，LIKE 语句也使用占位符
    query = "SELECT * FROM products WHERE name LIKE ?"
    cursor.execute(query, (f"%{keyword}%",))

    return cursor.fetchall()


def delete_user(user_id: int) -> None:
    """安全地删除用户。"""
    conn = sqlite3.connect("app.db")
    cursor = conn.cursor()

    # 安全：参数化查询 + 类型验证
    if not isinstance(user_id, int) or user_id <= 0:
        raise ValueError(f"无效的用户 ID: {user_id}")

    query = "DELETE FROM users WHERE id = ?"
    cursor.execute(query, (user_id,))
    conn.commit()
```

### 1.3 使用 ORM 的安全实践

```python
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from models import User


def get_user_by_name(session: Session, username: str) -> User | None:
    """使用 SQLAlchemy ORM 安全查询。"""
    # 安全：ORM 自动参数化
    stmt = select(User).where(User.username == username)
    return session.execute(stmt).scalar_one_or_none()


# 危险：即使在 ORM 中也要避免原生 SQL 拼接
def get_user_unsafe(session: Session, username: str) -> User | None:
    # 危险：使用 text() 拼接字符串
    from sqlalchemy import text
    query = text(f"SELECT * FROM users WHERE username = '{username}'")
    return session.execute(query).fetchone()


# 安全：text() 配合参数绑定
def get_user_safe(session: Session, username: str) -> User | None:
    from sqlalchemy import text
    query = text("SELECT * FROM users WHERE username = :username")
    return session.execute(query, {"username": username}).fetchone()
```

### 1.4 检测方法

```python
# 代码审查 Agent 应检测以下模式：
DETECTION_RULES = {
    "sql_injection": {
        "patterns": [
            r"f[\"']SELECT.*\{.*\}.*[\"']",
            r"f[\"']INSERT.*\{.*\}.*[\"']",
            r"f[\"']UPDATE.*\{.*\}.*[\"']",
            r"f[\"']DELETE.*\{.*\}.*[\"']",
            r"\"SELECT.*%s\"",
            r"\".*WHERE.*\"\s*\+\s*\w+",
        ],
        "ast_check": "检测 execute() 调用中的字符串拼接操作",
        "severity": "critical",
    }
}
```

---

## 2. 跨站脚本攻击 (XSS)

**风险等级：** 高 (CVSS 7.1)
**OWASP 分类：** A03:2021 - Injection

### 2.1 危险代码：未转义输出

```python
from flask import Flask, request, make_response

app = Flask(__name__)


@app.route("/greet")
def greet():
    """危险：直接将用户输入嵌入 HTML。"""
    name = request.args.get("name", "")
    # 危险：未转义用户输入直接输出到 HTML
    html = f"<h1>你好, {name}!</h1>"

    # 攻击示例：name = "<script>alert('XSS')</script>"
    # 结果：恶意脚本在用户浏览器中执行
    return make_response(html)


@app.route("/profile")
def profile():
    """危险：使用 render_template_string 拼接用户输入。"""
    from flask import render_template_string
    username = request.args.get("username", "")

    # 危险：Jinja2 模板字符串中直接插入用户输入
    template = f"<div class='profile'>欢迎回来, {username}</div>"
    return render_template_string(template)


@app.route("/search")
def search():
    """危险：在 JSON 响应中包含未转义 HTML。"""
    query = request.args.get("q", "")
    # 危险：返回的 HTML 内容未转义
    result_html = f'<div class="result">搜索 {query} 的结果</div>'
    return {"html": result_html, "query": query}
```

### 2.2 安全代码：模板自动转义

```python
from flask import Flask, request, render_template, escape, jsonify

app = Flask(__name__)
app.jinja_env.autoescape = True  # 确保自动转义开启


@app.route("/greet")
def greet():
    """安全：使用模板引擎自动转义。"""
    name = request.args.get("name", "")
    return render_template("greet.html", name=name)


@app.route("/greet-simple")
def greet_simple():
    """安全：手动使用 escape 函数。"""
    name = request.args.get("name", "")
    # 安全：使用 html.escape() 转义特殊字符
    safe_name = escape(name)
    html = f"<h1>你好, {safe_name}!</h1>"
    return make_response(html)


@app.route("/search")
def search():
    """安全：返回纯文本或使用 Content-Type 标记。"""
    query = request.args.get("q", "")
    return jsonify({
        "query": query,  # JSON 自动转义 < > & 等字符
        "results": [],
    })
```

### 2.3 Jinja2 模板中的安全实践

```html
<!-- 安全：Jinja2 默认自动转义 -->
<h1>你好, {{ name }}!</h1>

<!-- 危险：使用 |safe 过滤器绕过转义，仅在确认安全时使用 -->
<div>{{ user_bio | safe }}</div>

<!-- 安全：使用 |e 显式转义 -->
<p>搜索关键词: {{ query | e }}</p>

<!-- 安全：URL 属性中使用 |urlencode -->
<a href="/search?q={{ query | urlencode }}">搜索</a>
```

### 2.4 检测方法

```python
DETECTION_RULES = {
    "xss": {
        "patterns": [
            r"render_template_string\s*\(\s*f[\"']",
            r"make_response\s*\(\s*f[\"'].*<",
            r"\|\s*safe\s*}}",
            r"autoescape\s*=\s*False",
        ],
        "ast_check": "检测未转义的用户输入直接嵌入 HTML 的情况",
        "severity": "high",
    }
}
```

---

## 3. 命令注入

**风险等级：** 严重 (CVSS 9.8)
**OWASP 分类：** A03:2021 - Injection

### 3.1 危险代码：os.system 和 shell=True

```python
import os
import subprocess


def ping_host(host: str) -> str:
    """危险：使用 os.system 执行命令。"""
    # 危险：用户输入直接拼接到 shell 命令
    result = os.system(f"ping -c 4 {host}")
    return str(result)
    # 攻击示例：host = "8.8.8.8; rm -rf /"
    # 实际执行：ping -c 4 8.8.8.8; rm -rf /


def get_file_info(filename: str) -> str:
    """危险：使用 subprocess + shell=True。"""
    # 危险：shell=True 允许 shell 解释特殊字符
    result = subprocess.check_output(
        f"ls -la {filename}",
        shell=True,
        text=True,
    )
    return result
    # 攻击示例：filename = "test.txt; cat /etc/passwd"


def convert_image(input_path: str, output_format: str) -> str:
    """危险：拼接复杂命令。"""
    cmd = f"convert {input_path} output.{output_format}"
    # 危险：用户控制的输入用于构建命令
    return os.popen(cmd).read()
```

### 3.2 安全代码：subprocess 参数列表

```python
import subprocess
import shlex
import re


def ping_host(host: str) -> str:
    """安全：使用参数列表 + 输入验证。"""
    # 安全：验证输入格式
    if not re.match(r"^[\d.]+$|^[a-zA-Z0-9.-]+$", host):
        raise ValueError(f"无效的主机地址: {host}")

    # 安全：使用列表传参，不经过 shell 解释
    result = subprocess.run(
        ["ping", "-c", "4", host],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


def get_file_info(filename: str) -> str:
    """安全：使用参数列表避免 shell 注入。"""
    # 安全：验证文件名不包含路径分隔符
    safe_name = os.path.basename(filename)

    result = subprocess.run(
        ["ls", "-la", safe_name],
        capture_output=True,
        text=True,
    )
    return result.stdout


def convert_image(input_path: str, output_format: str) -> str:
    """安全：白名单验证 + 参数列表。"""
    # 安全：白名单验证输出格式
    allowed_formats = {"png", "jpg", "gif", "webp"}
    if output_format not in allowed_formats:
        raise ValueError(f"不支持的格式: {output_format}")

    safe_input = os.path.basename(input_path)
    output_path = f"output.{output_format}"

    result = subprocess.run(
        ["convert", safe_input, output_path],
        capture_output=True,
        text=True,
    )
    return result.stdout
```

### 3.3 检测方法

```python
DETECTION_RULES = {
    "command_injection": {
        "patterns": [
            r"os\.system\s*\(",
            r"os\.popen\s*\(",
            r"subprocess\.\w+\s*\(.*shell\s*=\s*True",
            r"subprocess\.\w+\s*\(\s*f[\"']",
        ],
        "ast_check": "检测 shell=True 参数和字符串拼接的命令",
        "severity": "critical",
    }
}
```

---

## 4. 硬编码密钥和 API Key

**风险等级：** 高 (CVSS 7.5)
**OWASP 分类：** A02:2021 - Cryptographic Failures

### 4.1 危险代码：硬编码凭据

```python
# 危险：直接在代码中写入 API Key
import openai

openai.api_key = "sk-abc123def456ghi789jkl012mno345pqr678"

# 危险：数据库密码硬编码
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "admin",
    "password": "MyS3cr3tP@ssw0rd!",  # 硬编码密码
}

# 危险：AWS 凭据硬编码
AWS_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# 危险：加密密钥硬编码
SECRET_KEY = "django-insecure-xyz123abc456"
ENCRYPTION_KEY = b"0123456789abcdef0123456789abcdef"

# 危险：JWT 密钥硬编码
JWT_SECRET = "my-super-secret-jwt-key-2024"

# 危险：OAuth client_secret 硬编码
GOOGLE_CLIENT_ID = "123456789.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "GOCSPX-abcdefghijklmnopqrstuvwxyz"
```

### 4.2 安全代码：使用环境变量

```python
import os
from functools import lru_cache

# 安全：从环境变量读取敏感信息
from dotenv import load_dotenv

load_dotenv()

# 安全：使用环境变量
openai_api_key = os.environ["OPENAI_API_KEY"]

# 安全：数据库配置从环境变量读取
def get_db_config() -> dict:
    """从环境变量获取数据库配置。"""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }

# 安全：使用配置中心或密钥管理服务
@lru_cache
def get_secret(secret_name: str) -> str:
    """从密钥管理服务获取密钥。"""
    # 从 AWS Secrets Manager / HashiCorp Vault 等获取
    import boto3
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return response["SecretString"]
```

### 4.3 .env 文件管理

```bash
# .env.example（提交到版本控制，不含真实值）
OPENAI_API_KEY=your-api-key-here
DB_HOST=localhost
DB_PORT=5432
DB_USER=your-db-user
DB_PASSWORD=your-db-password
JWT_SECRET=your-jwt-secret-here

# .env（不提交到版本控制，包含真实值）
# .gitignore 中必须包含 .env
```

```gitignore
# .gitignore
.env
.env.local
.env.production
*.pem
*.key
credentials.json
```

### 4.4 检测方法

```python
DETECTION_RULES = {
    "hardcoded_secrets": {
        "patterns": [
            r"(?:api_key|apikey|api_secret|secret_key|access_key)\s*=\s*[\"'][^\"']{8,}",
            r"(?:password|passwd|pwd)\s*=\s*[\"'][^\"']{4,}",
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API Key 格式
            r"AKIA[A-Z0-9]{16}",     # AWS Access Key 格式
            r"ghp_[a-zA-Z0-9]{36}",  # GitHub Token 格式
            r"gho_[a-zA-Z0-9]{36}",  # GitHub OAuth Token
            r"glpat-[a-zA-Z0-9\-]{20,}",  # GitLab Token
        ],
        "entropy_check": "检测高熵值字符串（可能是密钥）",
        "severity": "high",
    }
}
```

---

## 5. 不安全的反序列化

**风险等级：** 严重 (CVSS 9.8)
**OWASP 分类：** A08:2021 - Software and Data Integrity Failures

### 5.1 危险代码：pickle 和 yaml.load

```python
import pickle
import yaml


def load_user_data(data: bytes) -> object:
    """危险：使用 pickle 反序列化不受信任的数据。"""
    # 危险：pickle 可以执行任意 Python 代码
    return pickle.loads(data)

    # 攻击示例：
    # import pickle
    # import os
    # class Exploit:
    #     def __reduce__(self):
    #         return (os.system, ("id",))
    # payload = pickle.dumps(Exploit())
    # load_user_data(payload)  # 执行 os.system("id")


def load_config(filename: str) -> dict:
    """危险：使用 yaml.load 不指定 Loader。"""
    with open(filename, "r") as f:
        # 危险：yaml.load() 可以执行任意 Python 对象
        return yaml.load(f)
        # 攻击示例 YAML：
        # !!python/object/apply:os.system ["rm -rf /"]


def load_session(data: str) -> dict:
    """危险：使用 eval 反序列化。"""
    # 危险：eval 可以执行任意 Python 表达式
    return eval(data)
```

### 5.2 安全代码：安全的替代方案

```python
import json
import yaml


def load_user_data(data: str) -> dict:
    """安全：使用 json 反序列化。"""
    # 安全：JSON 只能表示基本数据类型，无法执行代码
    return json.loads(data)


def load_config(filename: str) -> dict:
    """安全：使用 yaml.safe_load。"""
    with open(filename, "r") as f:
        # 安全：safe_load 只解析基本 YAML 类型，不构造 Python 对象
        return yaml.safe_load(f)


def save_user_data(data: dict) -> bytes:
    """安全：使用 json 序列化。"""
    return json.dumps(data).encode("utf-8")


# 如果必须使用 pickle，至少限制可反序列化的类型
import pickle
import io


class SafeUnpickler(pickle.Unpickler):
    """限制 pickle 反序列化的类型白名单。"""

    ALLOWED_CLASSES = {
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "str"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "bool"),
        ("builtins", "tuple"),
    }

    def find_class(self, module: str, name: str) -> type:
        if (module, name) not in self.ALLOWED_CLASSES:
            raise pickle.UnpicklingError(
                f"不允许的类型: {module}.{name}"
            )
        return super().find_class(module, name)


def safe_load_pickle(data: bytes) -> object:
    """受限的 pickle 反序列化。"""
    return SafeUnpickler(io.BytesIO(data)).load()
```

### 5.3 检测方法

```python
DETECTION_RULES = {
    "insecure_deserialization": {
        "patterns": [
            r"pickle\.loads?\s*\(",
            r"yaml\.load\s*\([^)]*\)",       # 没有指定 Loader
            r"eval\s*\(",                      # eval 反序列化
            r"marshal\.loads?\s*\(",
            r"shelve\.open\s*\(",
        ],
        "negative_patterns": [
            r"yaml\.safe_load\s*\(",           # 安全，不需要警告
            r"yaml\.load\s*\([^)]*Loader\s*=\s*yaml\.SafeLoader",  # 安全
        ],
        "severity": "critical",
    }
}
```

---

## 6. 路径遍历

**风险等级：** 高 (CVSS 7.5)
**OWASP 分类：** A01:2021 - Broken Access Control

### 6.1 危险代码：未验证的文件路径

```python
import os


def read_file(filename: str) -> str:
    """危险：直接使用用户输入作为文件路径。"""
    # 危险：攻击者可以使用 ../ 遍历目录
    with open(filename, "r") as f:
        return f.read()
    # 攻击示例：filename = "../../etc/passwd"


def download_file(filepath: str) -> bytes:
    """危险：直接拼接用户输入的路径。"""
    base_dir = "/var/www/uploads"
    # 危险：os.path.join 无法阻止绝对路径
    # 如果 filepath 以 / 开头，会忽略 base_dir
    full_path = os.path.join(base_dir, filepath)
    with open(full_path, "rb") as f:
        return f.read()
    # 攻击示例：filepath = "/etc/shadow"
    # os.path.join("/var/www/uploads", "/etc/shadow") => "/etc/shadow"


def save_upload(file, upload_dir: str, filename: str) -> str:
    """危险：直接使用上传文件名。"""
    # 危险：文件名可能包含路径分隔符
    save_path = os.path.join(upload_dir, filename)
    with open(save_path, "wb") as f:
        f.write(file.read())
    return save_path
    # 攻击示例：filename = "../../../tmp/malicious.py"
```

### 6.2 安全代码：路径验证和规范化

```python
import os
import uuid
from pathlib import Path


def read_file(filename: str, base_dir: str = "/var/www/data") -> str:
    """安全：验证解析后的路径是否在允许的目录内。"""
    # 安全：规范化路径，解析 .. 和符号链接
    base = Path(base_dir).resolve()
    target = (base / filename).resolve()

    # 安全：确保目标路径在基础目录内
    if not str(target).startswith(str(base)):
        raise ValueError(f"非法路径访问: {filename}")

    if not target.exists():
        raise FileNotFoundError(f"文件不存在: {filename}")

    return target.read_text()


def download_file(filepath: str, base_dir: str = "/var/www/uploads") -> bytes:
    """安全：使用 resolve 防止路径遍历。"""
    base = Path(base_dir).resolve()
    target = (base / filepath).resolve()

    # 安全：双重检查
    if not target.is_relative_to(base):
        raise PermissionError(f"路径遍历攻击检测: {filepath}")

    return target.read_bytes()


def save_upload(file, upload_dir: str, filename: str) -> str:
    """安全：使用安全的文件名。"""
    # 安全：只保留文件名部分，去除任何路径
    safe_name = os.path.basename(filename)

    # 安全：可选：使用随机文件名避免冲突和攻击
    ext = os.path.splitext(safe_name)[1]
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".pdf"}
    if ext.lower() not in allowed_extensions:
        raise ValueError(f"不支持的文件类型: {ext}")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(upload_dir, unique_name)

    with open(save_path, "wb") as f:
        f.write(file.read())

    return save_path
```

### 6.3 检测方法

```python
DETECTION_RULES = {
    "path_traversal": {
        "patterns": [
            r"open\s*\(\s*\w+\s*\)",                    # 未验证的 open()
            r"os\.path\.join\s*\([^)]*\+\s*\w+",        # 拼接路径
        ],
        "negative_patterns": [
            r"os\.path\.basename\s*\(",
            r"\.resolve\s*\(\)",
            r"is_relative_to\s*\(",
            r"startswith\s*\(",
        ],
        "severity": "high",
    }
}
```

---

## 7. 不安全的随机数

**风险等级：** 中 (CVSS 5.3)
**OWASP 分类：** A02:2021 - Cryptographic Failures

### 7.1 危险代码：使用 random 模块

```python
import random
import string


def generate_password(length: int = 12) -> str:
    """危险：使用 random 模块生成密码。"""
    # 危险：random 模块使用 Mersenne Twister 算法，可预测
    chars = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.choice(chars) for _ in range(length))


def generate_token(length: int = 32) -> str:
    """危险：使用 random 生成认证令牌。"""
    # 危险：random 不是密码学安全的
    return "".join(random.choices("0123456789abcdef", k=length))


def generate_session_id() -> str:
    """危险：使用 random 生成会话 ID。"""
    # 危险：攻击者可以预测会话 ID
    return str(random.randint(100000, 999999))


def shuffle_deck(deck: list) -> list:
    """安全：random.shuffle 用于非安全场景是合适的。"""
    # 非安全场景使用 random 是正确的
    random.shuffle(deck)
    return deck
```

### 7.2 安全代码：使用 secrets 模块

```python
import secrets
import string


def generate_password(length: int = 16) -> str:
    """安全：使用 secrets 模块生成密码。"""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    # 安全：secrets 使用操作系统提供的密码学安全随机源
    return "".join(secrets.choice(alphabet) for _ in range(length))


def generate_token(length: int = 32) -> str:
    """安全：使用 secrets 生成令牌。"""
    # 安全：secrets.token_hex 生成密码学安全的十六进制令牌
    return secrets.token_hex(length // 2)


def generate_session_id() -> str:
    """安全：使用 secrets 生成会话 ID。"""
    # 安全：secrets.token_urlsafe 生成 URL 安全的令牌
    return secrets.token_urlsafe(32)


def generate_api_key() -> str:
    """安全：生成 API 密钥。"""
    return f"sk-{secrets.token_urlsafe(40)}"


def generate_verification_code(length: int = 6) -> str:
    """安全：生成数字验证码。"""
    # 安全：secrets.randbelow 生成密码学安全的随机数
    code = "".join(str(secrets.randbelow(10)) for _ in range(length))
    return code
```

### 7.3 random vs secrets 使用场景

| 场景 | 推荐模块 | 原因 |
|------|----------|------|
| 密码生成 | `secrets` | 必须不可预测 |
| 令牌/Session ID | `secrets` | 必须不可预测 |
| API Key | `secrets` | 必须不可预测 |
| 验证码 | `secrets` | 必须不可预测 |
| 抽奖/洗牌（娱乐） | `random` | 无安全需求 |
| 蒙特卡洛模拟 | `random` | 无安全需求，需要性能 |
| 测试数据生成 | `random` | 需要可重现（seed） |

### 7.4 检测方法

```python
DETECTION_RULES = {
    "insecure_random": {
        "patterns": [
            r"random\.\w+\s*\(.*(?:password|token|secret|session|key|auth)",
            r"random\.choice\s*\(.*(?:password|token|secret)",
            r"random\.randint\s*\(.*(?:session|id|code)",
        ],
        "context_keywords": [
            "password", "token", "secret", "session",
            "key", "auth", "verify", "csrf", "nonce",
        ],
        "severity": "medium",
    }
}
```

---

## 8. 敏感信息日志泄露

**风险等级：** 中 (CVSS 5.3)
**OWASP 分类：** A09:2021 - Security Logging and Monitoring Failures

### 8.1 危险代码：日志中记录敏感信息

```python
import logging

logger = logging.getLogger(__name__)


def login(username: str, password: str) -> bool:
    """危险：在日志中记录密码。"""
    # 危险：密码被记录到日志文件
    logger.info(f"用户登录: username={username}, password={password}")
    logger.debug(f"登录请求参数: {locals()}")  # 危险：locals() 包含密码

    result = authenticate(username, password)
    return result


def process_payment(card_number: str, cvv: str, amount: float) -> dict:
    """危险：记录信用卡信息。"""
    # 危险：信用卡号和 CVV 被记录
    logger.info(
        f"处理支付: card={card_number}, cvv={cvv}, amount={amount}"
    )
    return {"status": "success"}


def handle_request(request) -> None:
    """危险：记录完整的 HTTP 请求（可能包含敏感头信息）。"""
    # 危险：请求头中可能包含 Authorization token
    logger.debug(f"完整请求: headers={request.headers}")
    logger.debug(f"请求体: {request.body}")

    # 危险：记录完整的异常信息（可能包含数据库连接串）
    try:
        process(request)
    except Exception as e:
        logger.error(f"处理失败: {e}", exc_info=True)
```

### 8.2 安全代码：脱敏和过滤日志

```python
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def mask_sensitive(value: str, visible_chars: int = 4) -> str:
    """对敏感信息进行脱敏处理。"""
    if not value or len(value) <= visible_chars:
        return "****"
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]


def sanitize_log_data(data: dict) -> dict:
    """移除或脱敏日志中的敏感字段。"""
    SENSITIVE_KEYS = {
        "password", "passwd", "pwd", "secret",
        "token", "api_key", "apikey", "authorization",
        "credit_card", "card_number", "cvv", "ssn",
    }

    sanitized = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            sanitized[key] = "****"
        elif isinstance(value, str) and len(value) > 10:
            sanitized[key] = mask_sensitive(value)
        else:
            sanitized[key] = value
    return sanitized


def login(username: str, password: str) -> bool:
    """安全：日志中不记录密码。"""
    # 安全：只记录用户名，不记录密码
    logger.info(f"用户登录尝试: username={username}")

    result = authenticate(username, password)

    if result:
        logger.info(f"用户登录成功: username={username}")
    else:
        logger.warning(f"用户登录失败: username={username}")

    return result


def process_payment(card_number: str, cvv: str, amount: float) -> dict:
    """安全：脱敏信用卡信息后记录日志。"""
    # 安全：只显示卡号后四位
    masked_card = mask_sensitive(card_number)
    logger.info(f"处理支付: card={masked_card}, amount={amount}")
    # 日志输出: "处理支付: card=************4242, amount=99.99"

    return {"status": "success"}


class SensitiveDataFilter(logging.Filter):
    """日志过滤器：自动过滤敏感信息。"""

    SENSITIVE_PATTERNS = [
        (r"password[\"']?\s*[:=]\s*[\"']?(\S+)", "password=****"),
        (r"token[\"']?\s*[:=]\s*[\"']?(\S+)", "token=****"),
        (r"(?:Bearer\s+)\S+", "Bearer ****"),
        (r"api[_-]?key[\"']?\s*[:=]\s*[\"']?(\S+)", "api_key=****"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            record.msg = re.sub(pattern, replacement, str(record.msg), flags=re.IGNORECASE)
        return True


# 配置日志过滤器
logger.addFilter(SensitiveDataFilter())
```

### 8.3 检测方法

```python
DETECTION_RULES = {
    "sensitive_logging": {
        "patterns": [
            r"log(?:ger)?\.\w+\(.*(?:password|passwd|pwd)",
            r"log(?:ger)?\.\w+\(.*(?:secret|token|api_key)",
            r"log(?:ger)?\.\w+\(.*(?:credit_card|card_number|cvv)",
            r"log(?:ger)?\.\w+\(.*locals\(\)",
            r"log(?:ger)?\.\w+\(.*request\.body",
            r"log(?:ger)?\.\w+\(.*request\.headers",
        ],
        "severity": "medium",
    }
}
```

---

## 9. 安全检测规则总结

代码审查 Agent 在进行安全审查时，应按以下优先级和方式组织检测：

### 9.1 漏洞严重等级

| 等级 | 漏洞类型 | 检测方式 |
|------|----------|----------|
| 严重 | SQL 注入、命令注入、不安全反序列化 | AST 分析 + 正则匹配 |
| 高 | 硬编码密钥、XSS、路径遍历 | 正则匹配 + 熵值分析 |
| 中 | 不安全随机数、日志泄露 | 上下文感知模式匹配 |
| 低 | 缺少安全头、不安全的 CORS | 配置文件检查 |

### 9.2 检测策略

1. **静态模式匹配**：通过正则表达式检测已知危险模式
2. **AST 分析**：解析代码结构，检测数据流中的安全问题
3. **上下文分析**：结合变量名、函数名判断是否存在安全风险
4. **白名单比对**：对比已知的安全 API 和不安全 API
5. **熵值检测**：通过信息熵分析识别可能的硬编码密钥

### 9.3 通用安全编码原则

- 永远不要信任用户输入，所有外部数据必须验证和清理
- 使用参数化查询代替字符串拼接
- 敏感信息通过环境变量或密钥管理服务获取
- 使用安全的默认配置（如 yaml.safe_load）
- 遵循最小权限原则
- 保持依赖库及时更新
- 实施适当的日志记录和监控
- 定期进行安全审查和渗透测试
