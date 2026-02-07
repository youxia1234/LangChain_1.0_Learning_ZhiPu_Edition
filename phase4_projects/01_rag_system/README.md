# 项目一：RAG 检索增强生成系统

## 📋 项目概述

本项目构建一个完整的 RAG（Retrieval-Augmented Generation）系统，支持多种文档格式、智能检索和对话式问答。这是一个生产级别的实现，整合了前面所学的所有基础知识。

## 🎯 项目目标

通过本项目，你将学会：
- 构建端到端的 RAG 流水线
- 实现多种文档加载和处理策略
- 使用向量数据库进行语义检索
- 实现对话历史管理和上下文感知
- 添加来源引用和置信度评估
- 优化检索质量和生成效果

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # 可选，用于高质量 Embeddings
```

获取 API Keys:
- Zhipu AI: https://open.bigmodel.cn/usercenter/apikeys
- OpenAI: https://platform.openai.com/ (可选，用于更好的 Embeddings)

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 系统架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 文档加载  │───▶│ 文本分块  │───▶│ 向量化   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ▼                    │
│                               ┌──────────────┐             │
│                               │  向量数据库   │             │
│                               └──────────────┘             │
│                                        │                    │
│  ┌──────────┐    ┌──────────┐         │                    │
│  │ 用户查询  │───▶│ 查询处理  │─────────┘                    │
│  └──────────┘    └──────────┘                              │
│                        │                                    │
│                        ▼                                    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 检索文档  │───▶│ 重排序   │───▶│ 生成回答  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                        │                    │
│                                        ▼                    │
│                               ┌──────────────┐             │
│                               │  带引用的回答  │             │
│                               └──────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
01_rag_system/
├── README.md              # 项目文档
├── main.py                # 主程序入口
├── config.py              # 配置管理
├── document_loader.py     # 文档加载模块
├── text_processor.py      # 文本处理模块
├── vector_store.py        # 向量存储模块
├── retriever.py           # 检索模块
├── generator.py           # 生成模块
├── rag_chain.py           # RAG 链整合
└── sample_docs/           # 示例文档
    ├── langchain_intro.txt
    └── python_basics.txt
```

## 🔧 核心组件

### 1. 文档加载器 (DocumentLoader)
- 支持多种格式：TXT、PDF、Markdown、CSV
- 自动格式检测
- 元数据提取

### 2. 文本处理器 (TextProcessor)
- 智能分块策略
- 重叠处理保持上下文
- 元数据保留

### 3. 向量存储 (VectorStore)
- 使用 Chroma 作为向量数据库
- 支持持久化存储
- 高效相似度搜索

### 4. 检索器 (Retriever)
- 语义检索
- 多种检索策略（相似度、MMR）
- 可配置的 top-k

### 5. 生成器 (Generator)
- 基于上下文的回答生成
- 来源引用
- 置信度评估

## 🚀 快速开始

```bash
# 1. 设置环境变量
export OPENAI_API_KEY="your-api-key"

# 2. 运行示例
python main.py
```

## 💡 最佳实践

1. **分块策略**: 根据文档类型选择合适的分块大小
2. **检索优化**: 使用 MMR 增加结果多样性
3. **提示工程**: 设计清晰的提示模板
4. **错误处理**: 实现完善的错误处理机制
5. **性能监控**: 使用 LangSmith 追踪系统性能

## ❓ 常见问题

### Q1: Windows 上运行时 emoji 显示乱码怎么办？

**A:** 这是 Windows 终端 GBK 编码问题。在代码开头添加：

```python
import sys
import io

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

### Q2: 为什么使用智谱 AI 而不是 Groq？

**A:**

| 特性 | Groq | 智谱 AI |
|-----|------|---------|
| 费用 | 完全免费 | 有免费额度 |
| 速度 | 极快 | 快 |
| 中文支持 | 一般 | **优秀** |
| RAG 场景 | 良好 | **更适合中文文档问答** |
| 国内网络 | 需代理 | **直接访问** |

### Q3: 如何选择合适的 chunk_size？

**A:** 根据文档类型和需求选择：

```python
# 短文档（如新闻、通知）
chunk_size=300, chunk_overlap=50

# 中等文档（如文章、报告）
chunk_size=500, chunk_overlap=100

# 长文档（如书籍、手册）
chunk_size=1000, chunk_overlap=200

# 技术文档（需要保留代码完整性）
chunk_size=1500, chunk_overlap=300
```

### Q4: 如何提高 RAG 的检索质量？

**A:** 几个优化策略：

```python
# 1. 使用更好的 Embeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. 调整检索参数
config.top_k = 5  # 增加检索数量

# 3. 使用重排序
config.search_type = "mmr"  # 最大边际相关性

# 4. 优化分块策略
config.chunk_overlap = 150  # 增加重叠保持上下文
```

### Q5: RAG 系统和普通问答有什么区别？

**A:**

| 特性 | 普通问答 | RAG 系统 |
|------|---------|----------|
| 知识来源 | 模型训练数据 | 自定义文档 |
| 信息时效性 | 截止到训练时间 | 实时更新 |
| 回答准确性 | 可能产生幻觉 | 基于真实文档 |
| 来源可追溯 | 无 | 有 |
| 知识更新 | 需要重新训练 | 更新文档即可 |

## 📚 参考资源

- [LangChain RAG 文档](https://python.langchain.com/docs/tutorials/rag/)
- [Chroma 文档](https://docs.trychroma.com/)
- [向量数据库最佳实践](https://www.pinecone.io/learn/)
