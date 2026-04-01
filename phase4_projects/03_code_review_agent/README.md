# AI Code Review Agent — 智能代码审查助手

基于 **LangChain 1.0** 和 **LangGraph 1.0** 构建的多Agent智能代码审查系统。自动分析 Git 仓库的代码变更，从风格、安全、性能、架构四个维度并行审查，生成结构化审查报告。

## 技术架构

```
用户输入仓库路径
       │
       ▼
  ┌──────────┐     ┌──────────────────────┐
  │ 解析 Diff │ ──→ │  RAG 检索编码规范     │
  └──────────┘     └──────────────────────┘
       │                     │
       ▼                     ▼
  ┌──────────────────────────────────────┐
  │      LangGraph 并行审查工作流         │
  │                                      │
  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│
  │  │ 风格  │ │ 安全  │ │ 性能  │ │ 架构  ││
  │  │Agent │ │Agent │ │Agent │ │Agent ││
  │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘│
  │     └────────┼────────┼────────┘    │
  │              ▼                       │
  │        ┌──────────┐                  │
  │        │  报告综合  │                  │
  │        └──────────┘                  │
  └──────────────────────────────────────┘
       │
       ▼
  结构化审查报告
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **多Agent并行审查** | 4个专业Agent并行工作（Fan-out/Fan-in模式） |
| **RAG规范检索** | 从知识库检索编码规范，审查有据可依 |
| **MCP工具集成** | 通过MCP协议连接文件系统，标准化工具调用 |
| **LangGraph编排** | 状态图管理，并行节点自动结果累积 |
| **结构化报告** | 按严重程度分级（Critical/Warning/Suggestion/Positive） |
| **Git Diff分析** | 自动解析Git变更，精准定位代码行 |

## 审查维度

- **代码风格**: 命名规范、格式化、注释质量、导入管理、PEP 8 合规
- **安全漏洞**: SQL注入、命令注入、硬编码密钥、不安全反序列化、路径遍历（OWASP Top 10）
- **性能优化**: N+1查询、内存泄漏、低效算法、阻塞IO、数据结构选择
- **架构设计**: SOLID原则、代码耦合度、设计模式、错误处理、可测试性

## 快速开始

### 1. 安装依赖

```bash
cd phase4_projects/03_code_review_agent
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 ZHIPUAI_API_KEY
```

### 3. 启动后端

```bash
cd backend
python main.py
# 后端运行在 http://127.0.0.1:8001
```

### 4. 启动前端

```bash
# 新终端
cd frontend
streamlit run main.py
# 前端运行在 http://localhost:8501
```

### 5. 开始审查

1. 在Web界面输入Git仓库路径
2. 选择审查维度
3. 点击"开始审查"
4. 查看结构化报告，下载JSON/Markdown格式

## 项目结构

```
03_code_review_agent/
├── backend/
│   ├── main.py                     # FastAPI 应用入口
│   ├── core/
│   │   ├── state.py                # LangGraph 状态定义
│   │   ├── workflow.py             # 审查工作流（fan-out/fan-in）
│   │   ├── agents/
│   │   │   ├── style_reviewer.py   # 风格审查 Agent
│   │   │   ├── security_reviewer.py# 安全审查 Agent
│   │   │   ├── performance_reviewer.py  # 性能审查 Agent
│   │   │   └── architecture_reviewer.py # 架构审查 Agent
│   │   ├── tools/
│   │   │   ├── git_tools.py        # Git 操作工具
│   │   │   └── mcp_tools.py        # MCP 工具集成
│   │   └── rag/
│   │       ├── embeddings.py       # 向量化配置
│   │       └── standards_store.py  # 规范知识库管理
│   └── models/
│       └── schemas.py              # Pydantic 数据模型
├── frontend/
│   └── main.py                     # Streamlit Web 界面
├── data/
│   └── standards/                  # 内置编码规范
│       ├── python_style.md
│       ├── security_rules.md
│       ├── performance_patterns.md
│       └── architecture_principles.md
├── .env.example
├── requirements.txt
└── README.md
```

## 技术栈

| 技术 | 用途 |
|------|------|
| **LangChain 1.0** | LLM 接口、工具定义、文档处理、RAG |
| **LangGraph 1.0** | 多Agent编排、状态图、并行节点 |
| **Zhipu AI (glm-4-flash)** | LLM 推理引擎 |
| **ChromaDB** | 规范知识库向量存储 |
| **langchain-mcp-adapters** | MCP 协议工具集成 |
| **Streamlit** | Web 界面 |
| **FastAPI** | 后端 API |
| **GitPython** | Git 仓库操作 |

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/review` | 提交代码审查请求 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/standards/upload` | 上传自定义规范 |
| GET | `/api/standards/list` | 列出规范库 |
| DELETE | `/api/standards/{source}` | 删除规范文档 |

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ZHIPUAI_API_KEY` | 智谱AI API Key（必填） | - |
| `CHROMA_PERSIST_DIR` | ChromaDB 持久化目录 | `./data/chroma` |
| `EMBEDDINGS_MODEL` | 向量模型 | `sentence-transformers/all-MiniLM-L6-v2` |
| `MCP_FILESYSTEM_ENABLED` | 启用MCP工具 | `true` |
| `PORT` | 后端端口 | `8001` |
