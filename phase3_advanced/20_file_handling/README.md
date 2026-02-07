# 模块 20：文件处理

## 🎯 学习目标

学习如何在 LangChain 中处理各种文件类型，包括文档加载、解析和分析。

## 🚀 环境要求

```bash
# 需要配置的环境变量
ZHIPUAI_API_KEY=your_zhipuai_api_key_here
```

获取 API Key: https://open.bigmodel.cn/usercenter/apikeys

## 📚 核心概念

### 支持的文件类型

| 类型 | 扩展名 | 加载器 |
|------|--------|--------|
| PDF | .pdf | PyPDFLoader |
| Word | .docx | Docx2txtLoader |
| 文本 | .txt | TextLoader |
| Markdown | .md | UnstructuredMarkdownLoader |
| CSV | .csv | CSVLoader |
| JSON | .json | JSONLoader |
| HTML | .html | BSHTMLLoader |

### 文档加载器基础

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 加载 PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 加载文本文件
loader = TextLoader("file.txt", encoding="utf-8")
documents = loader.load()
```

### 文档结构

```python
from langchain_core.documents import Document

# 每个文档包含
doc = Document(
    page_content="文档内容...",  # 实际文本
    metadata={                    # 元数据
        "source": "file.pdf",
        "page": 1
    }
)
```

## 🔑 关键 API

### 文本分割

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 每块最大字符数
    chunk_overlap=200,    # 块之间重叠字符数
    separators=["\n\n", "\n", "。", " "]  # 分割优先级
)

chunks = splitter.split_documents(documents)
```

### 目录加载

```python
from langchain_community.document_loaders import DirectoryLoader

# 加载目录下所有 txt 文件
loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",      # 匹配模式
    loader_cls=TextLoader
)
documents = loader.load()
```

## 📝 本模块示例

1. **单文件加载**：加载和解析单个文件
2. **批量加载**：处理目录中的多个文件
3. **智能分割**：将长文档分割成适合处理的块
4. **文档问答**：基于文档内容回答问题

## ⚠️ 注意事项

1. 大文件需要分块处理以避免超出 token 限制
2. PDF 解析质量取决于 PDF 的结构
3. 注意文件编码，中文文件建议使用 UTF-8
4. 某些加载器需要额外安装依赖

---

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
| 文件处理场景 | 良好 | **更适合中文文档处理** |
| 国内网络 | 需代理 | **直接访问** |

### Q3: 如何选择合适的 chunk_size？

**A:** 根据以下因素决定：

```python
# 短文档（< 2000 字符）
chunk_size=500, chunk_overlap=50

# 中等文档（2000-10000 字符）
chunk_size=1000, chunk_overlap=200

# 长文档（> 10000 字符）
chunk_size=2000, chunk_overlap=400

# 考虑因素：
# 1. LLM 的上下文窗口大小
# 2. 文档内容的语义完整性（尽量在段落边界分割）
# 3. 重叠部分要足够保持上下文连续性
```

### Q4: 如何处理中文文档的编码问题？

**A:** 几个最佳实践：

```python
# 1. 明确指定编码
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 2. 处理 GBK 编码（Windows 中文环境）
try:
    with open("file.txt", "r", encoding="utf-8") as f:
        content = f.read()
except UnicodeDecodeError:
    with open("file.txt", "r", encoding="gbk") as f:
        content = f.read()

# 3. 使用 TextLoader 时指定编码
from langchain_community.document_loaders import TextLoader
loader = TextLoader("file.txt", encoding="utf-8")
```

### Q5: 文档问答时如何提高准确性？

**A:** 几个技巧：

```python
# 1. 使用 RAG（检索增强生成）
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 2. 在系统提示中强调答案来源
system_prompt = """你是一个文档问答助手。
根据以下文档内容回答问题。
如果文档中没有相关信息，请明确说明"文档中未提及此信息"。
不要编造文档外的内容。
用中文简洁回答。"""

# 3. 限制上下文长度，避免信息过载
# 只传递与问题相关的文档块
```
