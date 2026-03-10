# 项目一：LangChain 知识库及 RAG 问答系统

## 📋 项目概述

这是一个基于 LangChain 1.0 构建的**智能知识库和问答系统**，专门用于帮助开发者学习和查询 LangChain/LangGraph 的使用方法。

**核心特色**：这是一个"**用 LangChain 学习 LangChain**"的元认知项目 - 使用 LangChain 1.0 的 `create_agent` API 构建关于 LangChain 本身的知识库。

**📌 LangChain 1.0 正确用法**：
- ✅ 使用 `create_agent` 而不是链式调用（`|`）
- ✅ 检索功能作为 `@tool` 装饰的函数暴露给 agent
- ✅ Agent 自动决定何时调用工具

### 项目价值

| 面向人群 | 价值 |
|---------|------|
| **LangChain 初学者** | 通过自然语言查询快速上手，无需翻阅大量文档 |
| **面试准备者** | 了解 LangChain 1.0 工程实践的最佳方式 |
| **项目参考** | 展示如何使用 LangChain 1.0 API 构建生产级 RAG 系统 |

## 🎯 项目目标

通过本项目，你将学会：
- 构建端到端的 RAG 流水线
- 从复杂项目结构中提取和解析文档
- 使用向量数据库进行语义检索
- 实现对话历史管理和上下文感知
- 添加来源引用和置信度评估
- 优化检索质量和生成效果

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                  LangChain 知识库系统架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   知识库构建层                             │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ README   │  │ Python   │  │ 注释提取  │  │ 元数据   │ │ │
│  │  │ 解析     │  │ AST解析  │  │          │  │ 构建     │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   RAG 核心层                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ 文档分块  │  │ 向量化   │  │ 语义检索  │  │ 重排序   │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   LangGraph 工作流                         │ │
│  │  查询处理 → 文档检索 → 上下文拼接 → 回答生成 → 质量评估   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                   用户交互层                               │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ 演示模式  │  │ 交互模式  │  │ 来源引用  │              │ │
│  │  └──────────┘  └──────────┘  └──────────┘              │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
01_rag_system/
├── README.md              # 项目文档
├── main.py                # 主程序入口（RAG 系统实现）
├── knowledge_base.py      # 知识库构建模块
├── config.py              # 配置管理（可选）
├── document_loader.py     # 文档加载模块（可选）
├── text_processor.py      # 文本处理模块（可选）
├── vector_store.py        # 向量存储模块（可选）
├── retriever.py           # 检索模块（可选）
├── generator.py           # 生成模块（可选）
└── rag_chain.py           # RAG 链整合（可选）
```

## 🔧 核心组件

### 1. 知识库构建器 (knowledge_base.py)
- **功能**：解析整个 LangChain 学习项目的文档和代码
- **支持的解析**：
  - Markdown 文档（README.md）
  - Python 代码（模块 docstring、函数、类、方法文档）
  - 重要注释提取
- **元数据提取**：
  - 模块名称
  - 学习阶段（Phase 1-4）
  - 文档类型（markdown/function/class/method）
  - 源文件路径

### 2. RAG 检索增强生成系统 (main.py)
- **RAGAgent**：使用 LangChain 1.0 的 `create_agent` API
- **检索工具**：作为 `@tool` 装饰的函数（`search_knowledge_base`）
- **知识库管理器**：管理向量存储和检索
- **自动工具调用**：Agent 自动决定何时调用检索工具

### 3. 交互界面
- **演示模式**：预设示例问题展示系统能力
- **交互模式**：自由提问，支持多轮对话
- **来源追溯**：显示答案来源的模块和文件

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # 可选，用于高质量 Embeddings
```

获取 API Keys:
- Zhipu AI: https://open.bigmodel.cn/usercenter/apikeys
- OpenAI: https://platform.openai.com/ (可选，用于更好的 Embeddings)

## 🚀 快速开始

```bash
# 1. 设置环境变量
export ZHIPUAI_API_KEY="your-zhipuai-api-key"

# 2. 运行系统
cd phase4_projects/01_rag_system
python main.py

# 3. 选择模式
# - 演示模式：查看预设示例
# - 交互模式：自由提问
```

## 💡 使用示例

```python
from main import RAGConfig, KnowledgeBaseManager, RAGAgent
from pathlib import Path

# 1. 初始化配置
config = RAGConfig(
    chunk_size=800,
    chunk_overlap=150,
    top_k=5
)

# 2. 创建知识库管理器
kb_manager = KnowledgeBaseManager(config)

# 3. 构建知识库
project_root = Path("/path/to/project")
kb_manager.build_from_project(str(project_root))

# 4. 创建 RAG Agent（使用 LangChain 1.0 API）
agent = RAGAgent(kb_manager, config)

# 5. 提问
result = agent.query("什么是 LangChain？它有哪些核心组件？")
print(result['answer'])
```

**关键代码**：
```python
# 创建检索工具（作为 @tool 装饰的函数）
@tool
def search_knowledge_base(query: str) -> str:
    """在 LangChain 知识库中搜索相关信息"""
    docs = kb_manager.search(query, k=config.top_k)
    return format_results(docs)

# 使用 LangChain 1.0 的 create_agent API
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[search_knowledge_base],
    system_prompt="你是一个 LangChain 学习助手..."
)

# Agent 自动决定何时调用工具
response = agent.invoke({"messages": [{"role": "user", "content": query}]})
```

## 💡 面试话术建议

### 项目介绍（30 秒版本）

> "这是一个基于 LangChain 1.0 构建的智能知识库和问答系统。我发现 LangChain 官方文档比较分散，初学者上手困难，所以系统化整理了 25 个核心模块的知识，包括文档、代码示例和注释。然后使用 LangChain 自己的 RAG 技术构建了智能问答系统，帮助开发者通过自然语言查询快速学习 LangChain。"

### 技术亮点

1. **LangChain 1.0 API 正确使用**：
   - 使用 `create_agent` 而不是链式调用（`|`）
   - 检索功能作为 `@tool` 装饰的函数暴露给 agent
   - Agent 自动决定何时调用检索工具

2. **知识库构建**：
   - 使用 Python AST 解析代码，提取函数、类、方法的文档字符串
   - 自动解析 Markdown 文档并按标题分块
   - 构建结构化元数据（模块、阶段、类型）

3. **RAG 优化**：
   - 针对代码文档调整分块策略（800 chunk_size, 150 overlap）
   - 增加检索数量（top_k=5）获取更全面的信息
   - 优化系统提示引导 Agent 使用检索工具

### 可能的面试问题

**Q: 为什么做这个项目？**
A: 学习 LangChain 时发现官方文档分散，想做一个统一的知识入口；同时通过"用 LangChain 学 LangChain"的方式，加深对框架的理解。

**Q: 技术难点是什么？**
A: ① 代码文档的结构化解析（使用 Python AST）② 中文语境下的向量检索效果优化 ③ 代码块的分块策略调整。

**Q: 和官方文档有什么区别？**
A: 官方文档是参考手册，我的系统是问答式学习助手，支持自然语言查询，直接给出代码示例和来源追溯。

**Q: LangChain 1.0 和之前的版本有什么区别？**
A: LangChain 1.0 推荐使用 `create_agent` API 而不是链式调用。链式调用（`|`）在 1.0 中已经不推荐使用了。正确的做法是将功能作为 `@tool` 装饰的函数，然后使用 `create_agent` 创建 agent，让它自动决定何时调用工具。

## 🔧 技术栈

- **框架**: LangChain 1.0（使用 `create_agent` API）
- **LLM**: Zhipu AI (glm-4-flash)
- **向量存储**: InMemoryVectorStore (可扩展至 Chroma/Pinecone)
- **文档解析**: Python AST, Markdown 解析
- **工具调用**: `@tool` 装饰器

## 💡 最佳实践

1. **分块策略**：代码文档使用更大的 chunk_size 以保留代码完整性
2. **检索优化**：增加 top_k 值获取更全面的信息
3. **提示工程**：针对 LangChain 学习场景设计专门的 prompt 模板
4. **元数据管理**：保留详细的来源信息便于追溯

## ❓ 常见问题

### Q1: Windows 上运行时 emoji 显示乱码怎么办？

**A:** 这是 Windows 终端 GBK 编码问题。代码已包含 UTF-8 编码设置：

```python
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

### Q2: 如何选择合适的 Embeddings？

**A:**

| Embeddings | 优点 | 缺点 | 适用场景 |
|-----------|------|------|---------|
| Simple (Hash) | 无需 API Key | 语义理解弱 | 快速演示 |
| OpenAI | 语义理解强 | 需要 API Key | 生产环境 |
| Zhipu AI | 中文优化 | 需要配置 | 中文场景 |

### Q3: 如何扩展知识库？

**A:** 修改 `knowledge_base.py` 中的 `KnowledgeConfig`:

```python
config = KnowledgeConfig(
    project_root="/path/to/your/project",
    include_patterns=["*.md", "*.py", "*.txt"],
    exclude_dirs=["__pycache__", ".git"]
)
```

### Q4: 如何提高检索质量？

**A:** 几个优化策略：

```python
# 1. 使用高质量 Embeddings
OPENAI_API_KEY="your-key"  # 会自动使用 OpenAI Embeddings

# 2. 调整检索参数
config.top_k = 5  # 增加检索数量

# 3. 调整分块策略
config.chunk_size = 800  # 代码文档需要更大的块
config.chunk_overlap = 150  # 增加重叠保持上下文
```

### Q5: 这个项目的创新点是什么？

**A:**
- **元认知设计**：用 LangChain 学习 LangChain
- **系统化整理**：将 25 个模块的知识结构化
- **工程实践**：完整的 RAG 系统，不是简单 demo
- **实用价值**：真正帮助开发者学习和查阅

## 🔮 扩展方向

- 集成 Chroma 持久化向量存储
- 添加 Web 界面（Streamlit/Gradio）
- 支持多语言知识库
- 添加代码示例运行功能
- 集成 LangSmith 进行性能追踪

## 📚 参考资源

- [LangChain RAG 文档](https://python.langchain.com/docs/tutorials/rag/)
- [LangGraph 文档](https://python.langchain.com/docs/langgraph)
- [Chroma 文档](https://docs.trychroma.com/)
- [向量数据库最佳实践](https://www.pinecone.io/learn/)
