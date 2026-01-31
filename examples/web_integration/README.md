# Web 应用集成指南

本文档说明如何将 LangChain Agent 集成到 Web 应用中。

## 目录

1. [架构概述](#架构概述)
2. [快速开始](#快速开始)
3. [集成方式详解](#集成方式详解)
4. [生产环境配置](#生产环境配置)
5. [最佳实践](#最佳实践)

---

## 架构概述

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户浏览器                                │
│                    (React/Vue/原生 JS)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP/HTTPS
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Web 服务器/API 层                           │
│                    (FastAPI/Flask/Django)                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  功能:                                                    │  │
│  │  - 身份验证                                               │  │
│  │  - 请求路由                                               │  │
│  │  - 会话管理                                               │  │
│  │  - 限流/安全                                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │ 内部调用
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LangChain Agent 层                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  组件:                                                    │  │
│  │  - LLM 模型 (GLM-4/GPT/etc.)                              │  │
│  │  - 工具 (数据库/API 调用/计算)                             │  │
│  │  - 记忆管理 (会话历史)                                     │  │
│  │  - 业务逻辑                                                │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │ 数据访问
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                         外部资源                                  │
│  - 向量数据库 (ChromaDB/Pinecone)  - 知识库                     │
│  - SQL/NoSQL 数据库                - 业务数据                   │
│  - 外部 API                        - 第三方服务                 │
│  - 文件系统                        - 本地资源                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 安装依赖

```bash
cd examples/web_integration

# 安装后端依赖
pip install fastapi uvicorn pydantic python-dotenv langchain langchain-openai

# 前端无需安装（纯 HTML/JS）
```

### 2. 配置环境变量

确保项目根目录的 `.env` 文件包含：

```bash
ZHIPUAI_API_KEY=your_api_key_here
```

### 3. 启动后端服务

```bash
# 方式 1: 使用 uvicorn
uvicorn api_server:app --reload --port 8000

# 方式 2: 直接运行 Python
python api_server.py
```

服务启动后访问 http://localhost:8000/docs 查看 API 文档。

### 4. 打开前端

在浏览器中打开 `frontend/index.html` 文件。

---

## 集成方式详解

### 方式 1: REST API (推荐用于生产环境)

#### 优点
- 完全控制请求/响应
- 易于调试和监控
- 支持任何前端框架
- 灵活的认证集成

#### 缺点
- 需要手动处理流式输出
- 需要自己实现会话管理

#### 示例代码

**后端 (FastAPI):**
```python
from fastapi import FastAPI
from langchain.agents import create_agent

app = FastAPI()
agent = create_agent(model=model, tools=tools)

@app.post("/chat")
async def chat(message: str):
    response = agent.invoke({
        "messages": [{"role": "user", "content": message}]
    })
    return {"answer": response["messages"][-1].content}
```

**前端 (JavaScript):**
```javascript
async function sendMessage(message) {
    const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message})
    });
    const data = await response.json();
    return data.answer;
}
```

---

### 方式 2: LangServe (官方方案)

LangServe 是 LangChain 官方提供的部署方案。

#### 安装
```bash
pip install langserve
```

#### 示例代码

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain.agents import create_agent

app = FastAPI()

# 创建 Agent
agent = create_agent(
    model=ChatOpenAI(model="glm-4-flash"),
    tools=[tool1, tool2]
)

# 添加 LangServe 路由
add_routes(
    app,
    agent,
    path="/agent",
    input_type={"messages": list},
    output_type={"messages": list}
)
```

#### 优点
- 开箱即用的 API
- 自动生成文档
- 支持流式输出
- 类型安全

#### 缺点
- 定制化程度较低
- 学习曲线稍陡

---

### 方式 3: 云服务部署

#### 选项
- **LangSmith Platform**: 托管的 LangChain 服务
- **AWS Lambda**: 无服务器部署
- **Google Cloud Run**: 容器化部署
- **Azure Functions**: 函数计算

#### 示例 (AWS Lambda)

```python
import json
from langchain.agents import create_agent

agent = create_agent(model=model, tools=tools)

def lambda_handler(event, context):
    message = event.get("message")
    response = agent.invoke({
        "messages": [{"role": "user", "content": message}]
    })
    return {
        "statusCode": 200,
        "body": json.dumps({
            "answer": response["messages"][-1].content
        })
    }
```

---

## 生产环境配置

### 1. 会话管理

#### 内存存储 (仅开发)
```python
from collections import defaultdict
sessions = defaultdict(list)
```

#### Redis 存储 (推荐生产)
```python
import redis
import json

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def save_session(session_id: str, messages: list):
    r.set(f"session:{session_id}", json.dumps(messages))

def load_session(session_id: str) -> list:
    data = r.get(f"session:{session_id}")
    return json.loads(data) if data else []
```

#### 数据库存储
```python
# SQLAlchemy 示例
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ChatSession(Base):
    __tablename__ = 'chat_sessions'
    session_id = Column(String, primary_key=True)
    messages = Column(Text)

    def get_messages(self):
        return json.loads(self.messages)

    def set_messages(self, msgs):
        self.messages = json.dumps(msgs)
```

### 2. 异步处理

```python
from fastapi import FastAPI
from langchain.agents import create_agent
import asyncio

app = FastAPI()

@app.post("/chat")
async def chat(message: str):
    # 异步调用 Agent
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: agent.invoke({
            "messages": [{"role": "user", "content": message}]
        })
    )
    return {"answer": response["messages"][-1].content}
```

### 3. 错误处理

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"内部错误: {str(exc)}"}
    )

@app.post("/chat")
async def chat(message: str):
    try:
        response = agent.invoke({"messages": [...]})
        return {"answer": response["messages"][-1].content}
    except Exception as e:
        logger.error(f"Agent 错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="处理失败")
```

### 4. 限流和认证

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")
VALID_API_KEYS = {"your-secret-key"}

async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key"
        )
    return api_key

@app.post("/chat")
async def chat(
    message: str,
    api_key: str = Depends(verify_api_key)
):
    # 处理请求
    ...
```

### 5. 监控和日志

```python
import logging
from langchain.callbacks import LangChainTracer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 使用 LangSmith 追踪
tracer = LangChainTracer(project_name="my-app")

@app.post("/chat")
async def chat(message: str):
    logger.info(f"收到请求: {message[:50]}...")

    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"callbacks": [tracer]}
        )
        logger.info("请求成功")
        return {"answer": response["messages"][-1].content}
    except Exception as e:
        logger.error(f"请求失败: {e}", exc_info=True)
        raise
```

---

## 最佳实践

### 1. 安全性

- [ ] 使用 HTTPS 部署
- [ ] 实现 API Key 或 OAuth 认证
- [ ] 验证和清理所有输入
- [ ] 限流防止滥用
- [ ] 不要在日志中记录敏感信息
- [ ] 使用环境变量管理密钥

### 2. 性能优化

```python
# 连接池
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.7,
    max_retries=3,  # 重试机制
    request_timeout=30  # 超时设置
)

# 缓存常见查询
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(query_hash: str):
    return agent.invoke({"messages": [{"role": "user", "content": query}]})
```

### 3. 用户体验

```python
# 添加打字机效果的前端代码
function typeWriter(text, element, speed = 20) {
    let i = 0;
    element.textContent = '';
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    type();
}

// 显示加载状态
function showLoading() {
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}
```

### 4. 成本控制

```python
# 限制 Token 使用
from langchain_core.messages import HumanMessage

def count_tokens(messages):
    return sum(len(msg.content) for msg in messages) // 4

@app.post("/chat")
async def chat(message: str):
    messages = [{"role": "user", "content": message}]

    if count_tokens(messages) > 4000:
        raise HTTPException(
            status_code=400,
            detail="消息过长，请缩短后重试"
        )

    response = agent.invoke({"messages": messages})
    return {"answer": response["messages"][-1].content}
```

### 5. 测试

```python
import pytest
from fastapi.testclient import TestClient

client = TestClient(app)

def test_chat_endpoint():
    response = client.post("/chat", json={"message": "你好"})
    assert response.status_code == 200
    assert "answer" in response.json()

def test_stream_endpoint():
    response = client.post("/chat/stream", json={"message": "计算 1+1"})
    assert response.status_code == 200
```

---

## 部署清单

### 开发环境
- [ ] 本地运行 API 服务器
- [ ] 测试所有端点
- [ ] 验证流式输出
- [ ] 检查会话管理

### 生产环境
- [ ] 配置 HTTPS (使用 Let's Encrypt 或云服务)
- [ ] 设置环境变量
- [ ] 配置 Redis/数据库
- [ ] 设置监控和日志
- [ ] 实现限流和认证
- [ ] 配置 CDN (如需要)
- [ ] 设置自动备份

---

## 常见问题

### Q: 如何处理并发请求？
A: 使用异步框架 (FastAPI) 和连接池：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

@app.post("/chat")
async def chat(message: str):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        executor,
        lambda: agent.invoke({"messages": [...]})
    )
    return response
```

### Q: 如何降低延迟？
A:
1. 使用更快的模型
2. 启用缓存
3. 使用 CDN
4. 优化网络拓扑

### Q: 如何扩展到多个 Agent？
A: 使用路由模式：

```python
from langchain_core.runnables import RouterRunnable

agents = {
    "search": search_agent,
    "calculate": calc_agent,
    "general": general_agent
}

router = RouterRunnable(
    runnables=agents,
    key=lambda x: x.get("agent_type", "general")
)
```

---

## 下一步学习

- **Phase 2**: 学习内存管理、中间件
- **Phase 3**: 学习 LangGraph 高级架构
- **Phase 4**: 完整的生产级项目
