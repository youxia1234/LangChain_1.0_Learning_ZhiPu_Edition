# 多代理智能客服系统 Web 应用 - 技术设计文档

## 📋 项目概述

将 02_multi_agent_support 改造为 Web 应用，支持：
1. 通过 Web 界面进行多代理问答
2. 上传文档动态扩充知识库
3. 集成 RAG 技术提升问答质量
4. 分离模拟数据为独立知识库模块

## 🎯 技术栈

### 后端
- **Web 框架**: FastAPI 0.104+
- **LLM**: Zhipu AI (glm-4-flash)
- **Embeddings**: HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
- **向量数据库**: Pinecone (免费 tier)
- **文档处理**:
  - LangChain DocumentLoaders (PyPDFLoader, TextLoader)
  - RecursiveCharacterTextSplitter

### 前端
- **框架**: Streamlit 1.28+
  - 快速开发，适合 AI 应用
  - 内置文件上传组件
  - 原生 Python 集成
  - 实时更新支持

### 为什么选择 Streamlit 而不是 React/Vue？

| 特性 | Streamlit | React/Vue |
|------|-----------|-----------|
| 开发速度 | ⚡ 极快（纯 Python） | 🐌 慢（前后端分离） |
| 学习曲线 | 📈 低 | 📈 陡 |
| 文件上传 | ✅ 内置 | ❌ 需要自己实现 |
| AI 集成 | ✅ 原生支持 | ⚠️ 需要 API |
| 实时更新 | ✅ 自动 | ❌ 需要 WebSocket |
| 部署复杂度 | 🟢 简单 | 🟡 中等 |

对于 AI 原型/演示项目，Streamlit 是最佳选择。

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端 (Streamlit)                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────┐  │
│  │ 聊天界面    │  │ 文档上传   │  │ 知识库管理 │  │ 系统日志 │  │
│  └────────────┘  └────────────┘  └────────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       后端 (FastAPI)                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                     API Router                              │ │
│  │  /api/chat - 聊天接口                                       │ │
│  │  /api/upload - 文档上传                                     │ │
│  │  /api/knowledge - 知识库管理                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌───────────────┐    ┌───────────────┐    ┌──────────────────┐ │
│  │ Multi-Agent   │    │  RAG Engine   │    │ Knowledge Base   │ │
│  │   System      │    │               │    │    Manager       │ │
│  │               │    │               │    │                  │ │
│  │ - Intent      │    │ - Vector      │    │ - Document       │ │
│  │   Classifier  │    │   Store       │    │   Loader         │ │
│  │ - Tech        │    │ - Retriever   │    │ - Chunker        │ │
│  │   Support     │    │ - Generator   │    │ - Embedder       │ │
│  │ - Order       │    │               │    │ - Pinecone       │ │
│  │   Service     │    │               │    │   Interface      │ │
│  │ - Product     │    │               │    │                  │ │
│  │   Consult     │    │               │    │                  │ │
│  └───────────────┘    └───────────────┘    └──────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │                    External Services                         ││
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  ││
│  │  │Zhipu AI    │  │  Pinecone  │  │ HuggingFace          │  ││
│  │  │(LLM)       │  │(Vector DB) │  │(Embeddings)          │  ││
│  │  └────────────┘  └────────────┘  └──────────────────────┘  ││
│  └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 项目结构

```
02_multi_agent_support/
├── backend/                    # FastAPI 后端
│   ├── main.py                 # FastAPI 主程序
│   ├── api/                    # API 路由
│   │   ├── __init__.py
│   │   ├── chat.py             # 聊天接口
│   │   ├── knowledge.py        # 知识库管理接口
│   │   └── upload.py           # 文档上传接口
│   ├── core/                   # 核心业务逻辑
│   │   ├── __init__.py
│   │   ├── agents.py           # 多代理系统
│   │   ├── rag.py              # RAG 引擎
│   │   ├── knowledge.py         # 知识库管理
│   │   └── config.py           # 配置管理
│   ├── models/                 # 数据模型
│   │   ├── __init__.py
│   │   ├── chat.py             # 聊天请求/响应模型
│   │   └── knowledge.py        # 知识库模型
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── document.py         # 文档处理
│       └── vector.py           # 向量存储
│
├── frontend/                   # Streamlit 前端
│   ├── main.py                 # Streamlit 主程序
│   ├── pages/                  # 页面组件
│   │   ├── __init__.py
│   │   ├── chat.py             # 聊天页面
│   │   ├── upload.py           # 上传页面
│   │   └── knowledge.py        # 知识库管理页面
│   └── components/            # UI 组件
│       ├── __init__.py
│       └── chat_widget.py      # 聊天组件
│
├── data/                       # 数据目录
│   ├── uploads/                # 上传的文档
│   ├── knowledge/              # 知识库数据
│   │   ├── products/           # 产品文档
│   │   ├── technical/          # 技术文档
│   │   └── faq/                # FAQ 文档
│   └── vector_store/           # 向量存储缓存
│
├── tests/                      # 测试
│   ├── test_api.py
│   └── test_agents.py
│
├── requirements.txt            # Python 依赖
├── .env                        # 环境变量
├── README.md                   # 项目说明
└── TECH_DESIGN.md              # 本文档
```

---

## 🔧 核心模块设计

### 1. 多代理系统 (backend/core/agents.py)

```python
"""
多代理系统 - 使用 LangChain 1.0 + LangGraph

代理类型：
- IntentClassifier: 意图分类
- TechSupportAgent: 技术支持（集成 RAG）
- OrderServiceAgent: 订单服务
- ProductConsultAgent: 产品咨询（集成 RAG）
- QualityChecker: 质量检查
"""

class CustomerServiceSystem:
    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        # ... 初始化代理
```

### 2. RAG 引擎 (backend/core/rag.py)

```python
"""
RAG 引擎 - 集成 Pinecone 向量数据库

功能：
- 文档加载和分块
- 向量化（HuggingFaceEmbeddings）
- 存储到 Pinecone
- 语义检索
"""
```

### 3. 知识库管理 (backend/core/knowledge.py)

```python
"""
知识库管理器

功能：
- 上传文档到指定知识库
- 解析文档（PDF/TXT/MD）
- 分块和向量化
- 存储到 Pinecone
- 查询知识库
"""
```

---

## 📡 API 接口设计

### 1. 聊天接口

```http
POST /api/chat
Content-Type: application/json

{
  "message": "我的蓝牙耳机连不上手机",
  "session_id": "optional-session-id"
}

Response:
{
  "response": "客服回复内容",
  "intent": "tech_support",
  "confidence": 0.95,
  "sources": [
    {
      "type": "knowledge_base",
      "content": "来自知识库的内容",
      "metadata": {...}
    }
  ],
  "escalated": false
}
```

### 2. 文档上传接口

```http
POST /api/upload
Content-Type: multipart/form-data

file: <文档文件>
category: products|technical|faq

Response:
{
  "status": "success",
  "document_id": "doc-123",
  "chunks_created": 15,
  "message": "文档已成功添加到知识库"
}
```

### 3. 知识库查询接口

```http
GET /api/knowledge?category=products&query=蓝牙耳机

Response:
{
  "documents": [
    {
      "id": "doc-123",
      "filename": "product_manual.pdf",
      "chunks": 15,
      "uploaded_at": "2024-12-20T10:00:00Z"
    }
  ],
  "total": 1
}
```

### 4. 知识库删除接口

```http
DELETE /api/knowledge/{document_id}

Response:
{
  "status": "success",
  "message": "文档已删除"
}
```

---

## 🎨 前端页面设计

### 页面 1: 聊天界面 (默认页面)

```
┌────────────────────────────────────────────────────────────────┐
│  🤖 多代理智能客服系统                               [知识库管理] │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  对话历史                                                 │ │
│  │                                                           │ │
│  │  👤 用户: 我的蓝牙耳机连不上手机                           │ │
│  │  🤖 客服: [技术支持代理] 让我帮您解决这个问题...           │ │
│  │                                                           │ │
│  │  👤 用户: 好的，谢谢                                       │ │
│  │  🤖 客服: 不客气，还有其他问题吗？                         │ │
│  │                                                           │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  输入框                                                   │ │
│  │  [请输入您的问题...                           ] [发送]   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  💡 当前代理: 技术支持 | 置信度: 0.95 | 质量评分: 0.92          │
└────────────────────────────────────────────────────────────────┘
```

### 页面 2: 文档上传

```
┌────────────────────────────────────────────────────────────────┐
│  📚 知识库管理                                    [返回聊天]     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  上传新文档                                               │ │
│  │                                                           │ │
│  │  选择类别:                                                │ │
│  │  ○ 产品文档  ○ 技术文档  ○ FAQ                           │ │
│  │                                                           │ │
│  │  选择文件: [浏览...]                                      │ │
│  │  支持 PDF, TXT, Markdown 格式                            │ │
│  │                                                           │ │
│  │  [上传文档]                                               │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  当前知识库                                               │ │
│  │                                                           │ │
│  │  📁 产品文档 (5 个文档)                                   │ │
│  │     - product_manual.pdf (120 chunks)                    │ │
│  │     - price_list.pdf (45 chunks)                         │ │
│  │     [...]                                                │ │
│  │                                                           │ │
│  │  📁 技术文档 (3 个文档)                                   │ │
│  │  📁 FAQ (10 个文档)                                        │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔑 环境变量配置

```bash
# .env 文件

# Zhipu AI (必需)
ZHIPUAI_API_KEY=your_zhipuai_api_key

# Pinecone (必需)
# 获取地址: https://www.pinecone.io/
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=gcp-starter  # 或 us-east-1-aws
PINECONE_INDEX_NAME=customer-service-kb

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 路径配置
UPLOAD_DIR=./data/uploads
KNOWLEDGE_DIR=./data/knowledge

# Embeddings 配置
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_ENDPOINT=https://hf-mirror.com  # 国内镜像
```

---

## 📦 依赖包

```txt
# requirements.txt

# FastAPI 核心依赖
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6  # 文件上传

# LangChain 核心依赖
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
langchain-pinecone==0.0.1
langchain-huggingface==0.0.1
langchain-text-splitters==0.0.1

# LangGraph
langgraph==0.0.20

# 向量存储
pinecone-client==2.2.4
sentence-transformers==2.2.2

# 文档处理
pypdf==3.17.0
python-docx==1.1.0
python-pptx==0.6.23

# Streamlit 前端
streamlit==1.28.1
streamlit-chat==0.1.1
requests==2.31.0

# 工具库
python-dotenv==1.0.0
aiofiles==23.2.1  # 异步文件操作
```

---

## 🚀 部署方案

### 开发环境

```bash
# 启动后端
cd backend
uvicorn main:app --reload --port 8000

# 启动前端
cd frontend
streamlit run main.py --server.port 8501
```

### 生产环境

```bash
# 使用 Docker Compose
docker-compose up -d
```

---

## 📝 开发计划

### Phase 1: 基础架构 (1-2天)
- [x] 技术文档编写
- [ ] FastAPI 项目结构搭建
- [ ] 基础 API 框架
- [ ] Streamlit 页面框架

### Phase 2: 核心功能 (3-4天)
- [ ] 多代理系统迁移
- [ ] RAG 引擎实现
- [ ] Pinecone 集成
- [ ] 文档上传和处理

### Phase 3: 前端开发 (2-3天)
- [ ] 聊天界面
- [ ] 文档上传界面
- [ ] 知识库管理界面
- [ ] API 集成

### Phase 4: 测试和优化 (1-2天)
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能优化
- [ ] 错误处理

### Phase 5: 部署和文档 (1天)
- [ ] 部署脚本
- [ ] 用户文档
- [ ] API 文档

---

## 🎯 关键技术决策

### 1. 为什么选择 Pinecone？

✅ **优势**：
- 免费 tier 支持 1 个索引
- 托管服务，无需运维
- 性能优秀
- API 简单

❌ **替代方案**：
- Chroma（本地，适合小规模）
- Weaviate（功能更强但复杂）
- Qdrant（开源，需自己部署）

### 2. 为什么选择 HuggingFace Embeddings？

✅ **优势**：
- 完全免费
- 可本地部署
- 中文支持较好（all-MiniLM-L6-v2）
- 无 API 调用限制

❌ **替代方案**：
- OpenAI Embeddings（需要付费，但效果更好）
- Jina AI（有免费额度，但有限制）

### 3. 为什么分离前端和后端？

✅ **优势**：
- 前后端独立开发
- API 可复用
- 便于部署

⚠️ **简化方案**（初期）：
- Streamlit 直接调用后端逻辑
- 无需 FastAPI
- 更快原型开发

---

## 📊 数据流设计

### 聊天流程

```
用户输入
    │
    ▼
Streamlit 前端
    │
    │ HTTP POST
    ▼
FastAPI /api/chat
    │
    ▼
意图分类
    │
    │ 条件路由
    ▼
专业代理 + RAG 检索
    │
    ├─────► Pinecone 向量检索
    │
    ├─────► 生成回复
    │
    ▼
质量检查
    │
    ▼
返回结果
    │
    │ HTTP Response
    ▼
Streamlit 显示
```

### 文档上传流程

```
用户选择文件
    │
    ▼
Streamlit 上传组件
    │
    │ POST multipart/form-data
    ▼
FastAPI /api/upload
    │
    ▼
保存文件到 uploads/
    │
    ▼
文档解析 (PDF/TXT)
    │
    ▼
文本分块
    │
    ▼
Embeddings 向量化
    │
    │─────────────┬─────────────┐
    │             │             │
    ▼             ▼             ▼
产品知识库     技术知识库     FAQ 知识库
(Pinecone)   (Pinecone)   (Pinecone)
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
返回成功响应
```

---

## 🎨 UI/UX 设计要点

### 聊天界面

1. **实时显示**：显示当前代理类型、置信度
2. **消息类型**：区分用户消息和不同代理的回复
3. **来源追溯**：显示知识库来源（如果使用了 RAG）
4. **升级提示**：需要人工时突出显示

### 上传界面

1. **进度条**：显示文档处理进度
2. **分类选择**：单选按钮选择知识库类别
3. **文件列表**：显示已上传的文档
4. **删除功能**：支持删除已上传的文档

### 知识库管理

1. **统计信息**：文档数量、chunk 数量
2. **搜索功能**：在知识库中搜索
3. **预览功能**：预览文档内容

---

## 🧪 测试策略

### 单元测试

```python
# tests/test_agents.py
def test_intent_classification():
    classifier = IntentClassifier()
    result = classifier.classify("我的订单在哪？")
    assert result["intent"] == "order_service"
    assert result["confidence"] > 0.8

# tests/test_rag.py
def test_knowledge_retrieval():
    rag = RAGEngine()
    results = rag.search("蓝牙耳机连接")
    assert len(results) > 0
```

### 集成测试

```python
# tests/test_api.py
def test_chat_api():
    response = client.post("/api/chat", json={
        "message": "我的蓝牙耳机连不上"
    })
    assert response.status_code == 200
    assert "response" in response.json()
```

---

## 📚 参考资料

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Streamlit 官方文档](https://docs.streamlit.io/)
- [Pinecone 文档](https://docs.pinecone.io/)
- [LangChain 文档](https://python.langchain.com/)
- [LangGraph 文档](https://python.langchain.com/docs/langgraph)

---

## ✅ 验收标准

### 功能验收

- [ ] 用户可以通过 Web 界面进行多轮对话
- [ ] 系统能正确识别意图并路由到对应代理
- [ ] 用户可以上传文档到知识库
- [ ] 上传的文档能被正确解析和向量化
- [ ] RAG 检索能返回相关结果
- [ ] 代理能使用 RAG 检索结果生成回复
- [ ] 显示知识库来源

### 性能验收

- [ ] 聊天响应时间 < 5 秒
- [ ] 文档上传处理时间 < 10 秒（10 页 PDF）
- [ ] RAG 检索时间 < 2 秒
- [ ] 系统稳定运行，无崩溃

---

**文档版本**: 1.0
**创建时间**: 2024-12-20
**最后更新**: 2024-12-20
