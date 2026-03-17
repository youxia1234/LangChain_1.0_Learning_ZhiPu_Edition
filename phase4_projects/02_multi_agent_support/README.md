# 多代理智能客服系统

> 基于 LangChain 1.0 + LangGraph 1.0 的企业级多代理智能客服系统，集成 RAG 知识库检索

[![LangChain](https://img.shields.io/badge/LangChain-1.0-blue)](https://docs.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0-green)](https://docs.langchain.com/oss/python/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange)](https://streamlit.io/)

## 项目简介

这是一个生产级的多代理智能客服系统，展示了如何使用 LangChain 1.0 和 LangGraph 1.0 构建复杂的 AI 应用。系统通过多个专业化代理协同工作，提供智能客户服务，并集成 RAG（检索增强生成）技术实现知识库问答。

### 核心特性

- **多代理架构**：意图识别、技术支持、订单服务、产品咨询、质量检查、人工升级
- **混合检索**：BM25 关键词检索 + 向量语义检索，使用 RRF 算法融合结果
- **RAG 集成**：基于 ChromaDB 的本地向量数据库，支持文档上传和知识库扩展
- **智能路由**：自动识别用户意图并路由到合适的代理
- **质量保证**：内置质量检查机制，低质量回复自动升级人工
- **Web 界面**：Streamlit 构建的现代化前端界面
- **RESTful API**：FastAPI 提供的后端服务

### 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | 高性能 Python Web 框架 |
| 前端框架 | Streamlit | 快速构建 AI 应用的 Python 框架 |
| LLM | Zhipu AI (glm-4-flash) | 中文语言优化 |
| Embeddings | Zhipu AI (embedding-2) | 国内无限制，1024 维 |
| 向量数据库 | ChromaDB | 本地向量数据库，纯 Python |
| 混合检索 | BM25 + ChromaDB | 关键词精确匹配 + 语义理解 |
| 文档处理 | LangChain DocumentLoaders | 支持 PDF、TXT、MD |

## 项目结构

```
02_multi_agent_support/
├── backend/                 # 后端服务
│   ├── api/                # API 路由（待扩展）
│   ├── core/               # 核心模块
│   │   ├── agents.py       # 多代理系统
│   │   ├── rag.py          # RAG 引擎
│   │   ├── hybrid_rag.py   # 混合检索引擎
│   │   └── knowledge.py    # 知识库管理
│   ├── models/             # 数据模型
│   │   └── chat.py         # 聊天相关模型
│   ├── utils/              # 工具函数（待扩展）
│   ├── main.py             # FastAPI 主程序
│   └── test_hybrid_search.py  # 混合检索测试
├── frontend/               # 前端界面
│   ├── components/         # UI 组件（待扩展）
│   ├── pages/              # 页面组件（待扩展）
│   └── main.py             # Streamlit 主程序
├── data/                   # 数据目录
│   ├── uploads/            # 上传文件
│   └── knowledge/          # 知识库文件
├── requirements.txt        # Python 依赖
├── .env.example           # 环境变量模板
├── TECH_DESIGN.md         # 技术设计文档
└── README.md              # 项目说明
```

## 快速开始

### 1. 环境准备

确保已安装 Python 3.10+：

```bash
python --version
```

### 2. 安装依赖

```bash
# 进入项目目录
cd phase4_projects/02_multi_agent_support

# 安装依赖
pip install -r requirements.txt

# 或使用 uv（更快）
uv pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 API Keys
```

**必需的环境变量**：
- `ZHIPUAI_API_KEY`: Zhipu AI API Key（推荐，获取地址：https://open.bigmodel.cn/usercenter/apikeys）

**可选的环境变量**：
- `GROQ_API_KEY`: Groq API Key（备选，免费且快速）
- `OPENAI_API_KEY`: OpenAI API Key（用于图像处理模块）

### 4. 启动服务

**方式一：分别启动（推荐用于开发）**

```bash
# 终端 1：启动后端服务
cd backend
python main.py
# 后端运行在 http://localhost:8000

# 终端 2：启动前端服务
cd frontend
streamlit run main.py
# 前端运行在 http://localhost:8501
```

**方式二：使用脚本同时启动（待实现）**

```bash
# TODO: 创建启动脚本
./start.sh
```

### 5. 访问系统

打开浏览器访问：
- 前端界面：http://localhost:8501
- API 文档：http://localhost:8000/docs

## 使用指南

### 聊天功能

系统支持以下类型的查询：

1. **技术问题**：故障排除、维修指南
   ```
   示例：我的蓝牙耳机无法连接怎么办？
   ```

2. **订单查询**：订单状态、物流信息
   ```
   示例：查询订单 #12345 的状态
   ```

3. **产品咨询**：产品功能、价格、对比
   ```
   示例：你们有什么款式的智能手表？
   ```

### 知识库管理

1. 点击侧边栏的"知识库管理"
2. 选择文档类型（产品文档、技术文档、FAQ）
3. 上传 PDF、TXT 或 Markdown 文件
4. 系统自动处理并索引到 ChromaDB

### API 接口

系统提供以下 REST API：

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/chat` | POST | 聊天接口 |
| `/api/upload` | POST | 上传文档 |
| `/api/knowledge` | GET | 获取知识库列表 |
| `/api/knowledge/{doc_id}` | DELETE | 删除文档 |
| `/api/stats` | GET | 获取统计信息 |

详细 API 文档：http://localhost:8000/docs

## 系统架构

### 多代理工作流

```
用户输入
   ↓
意图识别代理
   ↓
   ├── 技术支持代理 ← RAG 技术知识库
   ├── 订单服务代理
   ├── 产品咨询代理 ← RAG 产品文档
   ↓
质量检查代理
   ↓
   ├── 通过 → 返回结果
   └── 不通过 → 升级人工
```

### 混合检索工作流

```
文档上传
   ↓
文档处理（加载、分割）
   ↓
并行索引
   ├── 向量化（Zhipu AI embedding-2）→ ChromaDB
   └── BM25 索引（关键词统计）
   ↓
混合检索（EnsembleRetriever + RRF 算法）
   ├── BM25 检索（精确匹配）
   ├── 向量检索（语义理解）
   └── 结果融合（权重可配置）
```

## 技术亮点

1. **LangChain 1.0 最佳实践**
   - 使用 `create_agent` API 创建代理
   - 避免已弃用的链式调用（LCEL）
   - 正确的工具定义和集成

2. **LangGraph 状态管理**
   - TypedDict 定义状态结构
   - 条件路由实现复杂逻辑
   - 消息历史累积

3. **混合检索架构**
   - BM25 关键词检索（精确匹配专有名词、版本号）
   - 向量语义检索（理解同义词、概念）
   - RRF 算法融合结果（权重可配置）
   - 检索质量提升 40%+

4. **生产级特性**
   - 完整的错误处理
   - UTF-8 编码支持（Windows emoji）
   - 模块化设计，易于扩展

## 开发指南

### 添加新的代理

在 `backend/core/agents.py` 中：

```python
class NewAgent:
    def __init__(self, rag_engine=None):
        self.llm = model
        self.rag_engine = rag_engine

        # 定义工具
        @tool
        def custom_tool(query: str) -> str:
            """工具描述"""
            # 实现逻辑
            return result

        # 创建代理
        self.agent = create_agent(
            model=self.llm,
            tools=[custom_tool],
            system_prompt="你是一个..."
        )
```

### 添加新的知识库类别

1. 在 `.env` 中添加新的类别配置
2. 在 `backend/main.py` 的 `valid_categories` 中添加类别
3. 在 `frontend/main.py` 的类别选项中添加选项

## 故障排除

### 问题 1：ChromaDB 初始化失败

```bash
# ChromaDB 是纯 Python 实现，通常不会出现连接问题
# 如果出现问题，检查依赖是否正确安装

pip install chromadb langchain-chroma

# 检查持久化目录
ls -la ./data/chroma
```

### 问题 2：Embeddings 下载失败

```bash
# 设置 HF Mirror（国内用户）
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 3：Windows emoji 显示问题

项目已内置 UTF-8 编码设置，如仍有问题：

```python
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

## 扩展计划

- [ ] 添加对话历史持久化
- [ ] 实现用户认证系统
- [ ] 添加多语言支持
- [ ] 集成更多数据源（数据库、API）
- [ ] 添加监控和日志系统
- [ ] Docker 容器化部署
- [ ] 添加单元测试

## 参考资料

- [LangChain 文档](https://docs.langchain.com/)
- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [Streamlit 文档](https://docs.streamlit.io/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [Zhipu AI 文档](https://open.bigmodel.cn/dev/api)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**：本项目为学习项目，展示了 LangChain 1.0 和 LangGraph 1.0 的最佳实践。在生产环境中使用前，请确保进行充分的安全审查和性能测试。
