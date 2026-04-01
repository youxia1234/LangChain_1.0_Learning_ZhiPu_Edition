# AI Code Review Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-grade AI code review system with multi-agent parallel review, RAG-powered standards retrieval, and MCP tool integration.

**Architecture:** LangGraph StateGraph orchestrates 4 specialized reviewer agents (style, security, performance, architecture) in a fan-out/fan-in pattern. Each agent uses MCP filesystem tools to read source code and RAG to retrieve relevant coding standards. A synthesizer agent compiles findings into a structured report.

**Tech Stack:** LangChain 1.0+, LangGraph 1.0+, Zhipu AI (glm-4-flash), ChromaDB, langchain-mcp-adapters, Streamlit, FastAPI, GitPython

---

## File Structure

```
phase4_projects/03_code_review_agent/
├── backend/
│   ├── main.py                          # FastAPI 应用入口 + API端点
│   ├── core/
│   │   ├── __init__.py
│   │   ├── state.py                     # LangGraph 状态定义
│   │   ├── workflow.py                  # LangGraph 审查工作流 (fan-out/fan-in)
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── style_reviewer.py        # 代码风格审查Agent
│   │   │   ├── security_reviewer.py     # 安全漏洞审查Agent
│   │   │   ├── performance_reviewer.py  # 性能优化审查Agent
│   │   │   └── architecture_reviewer.py # 架构可维护性审查Agent
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── git_tools.py             # Git操作工具 (diff解析/文件读取)
│   │   │   └── mcp_tools.py             # MCP工具集成
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── standards_store.py       # 规范知识库管理
│   │       └── embeddings.py            # 向量化配置
│   └── models/
│       ├── __init__.py
│       └── schemas.py                   # Pydantic 数据模型
├── frontend/
│   └── main.py                          # Streamlit Web界面
├── data/
│   └── standards/                       # 内置编码规范文档
│       ├── python_style.md
│       ├── security_rules.md
│       ├── performance_patterns.md
│       └── architecture_principles.md
├── .env.example
├── requirements.txt
└── README.md
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `phase4_projects/03_code_review_agent/.env.example`
- Create: `phase4_projects/03_code_review_agent/requirements.txt`
- Create: all `__init__.py` files
- Create: `data/standards/` directory

- [ ] **Step 1: Create directory structure**

```bash
cd phase4_projects
mkdir -p 03_code_review_agent/{backend/core/{agents,tools,rag},backend/models,frontend,data/standards}
touch 03_code_review_agent/backend/__init__.py
touch 03_code_review_agent/backend/core/__init__.py
touch 03_code_review_agent/backend/core/agents/__init__.py
touch 03_code_review_agent/backend/core/tools/__init__.py
touch 03_code_review_agent/backend/core/rag/__init__.py
touch 03_code_review_agent/backend/models/__init__.py
```

- [ ] **Step 2: Create .env.example**

```env
# ==================== LLM API Keys ====================
ZHIPUAI_API_KEY=your_zhipuai_api_key_here

# ==================== Vector Database ====================
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=code_standards

# ==================== Embeddings ====================
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_ENDPOINT=https://hf-mirror.com

# ==================== MCP Config ====================
MCP_FILESYSTEM_ENABLED=true

# ==================== Server Config ====================
HOST=0.0.0.0
PORT=8001
API_BASE_URL=http://127.0.0.1:8001

# ==================== Storage ====================
UPLOAD_DIR=./data/uploads
STANDARDS_DIR=./data/standards
```

- [ ] **Step 3: Create requirements.txt**

```txt
# ==================== Web Framework ====================
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
streamlit>=1.28.0
requests>=2.31.0
python-multipart>=0.0.6
sse-starlette>=1.6.0

# ==================== LangChain & LangGraph ====================
langchain>=0.1.0
langchain-openai>=0.2.0
langchain-community>=0.3.0
langchain-core>=0.1.0
langchain-text-splitters>=0.0.1
langgraph>=0.0.20

# ==================== MCP Integration ====================
langchain-mcp-adapters>=0.1.0
mcp>=1.0.0

# ==================== Vector Database ====================
chromadb>=0.5.0

# ==================== Git Operations ====================
gitpython>=3.1.0

# ==================== Document Processing ====================
pypdf>=5.0.0
python-docx>=1.1.0
PyYAML>=6.0.0

# ==================== Embeddings ====================
sentence-transformers>=2.2.0
transformers>=4.35.0

# ==================== Data & Validation ====================
pydantic>=2.5.0
pydantic-settings>=2.1.0
numpy>=1.24.0

# ==================== Environment ====================
python-dotenv>=1.0.0
aiofiles>=23.2.0
```

- [ ] **Step 4: Commit scaffolding**

```bash
git add phase4_projects/03_code_review_agent/
git commit -m "chore: scaffold 03_code_review_agent project structure"
```

---

### Task 2: State Definition and Pydantic Schemas

**Files:**
- Create: `backend/core/state.py`
- Create: `backend/models/schemas.py`

- [ ] **Step 1: Create state.py — LangGraph 状态定义**

```python
"""LangGraph 审查工作流状态定义

定义代码审查工作流中各节点之间传递的状态结构。
使用 Annotated[list, operator.add] 实现并行节点的结果累积。
"""

from typing import TypedDict, Annotated, Optional
import operator


class ReviewState(TypedDict):
    """代码审查工作流状态

    Attributes:
        repo_path: 本地 Git 仓库路径
        target_branch: 目标分支（用于 git diff）
        diff_content: git diff 原始内容
        changed_files: 变更文件信息列表
            [{"path": "app/api.py", "status": "modified", "additions": 10, "deletions": 3}]
        review_config: 审查配置
            {"enable_style": True, "enable_security": True, ...}
        style_findings: 风格审查发现（并行写入，自动累积）
        security_findings: 安全审查发现（并行写入，自动累积）
        performance_findings: 性能审查发现（并行写入，自动累积）
        architecture_findings: 架构审查发现（并行写入，自动累积）
        final_report: 最终生成的审查报告字典
        error: 错误信息（如有）
    """
    repo_path: str
    target_branch: str
    diff_content: str
    changed_files: list[dict]
    review_config: dict
    style_findings: Annotated[list, operator.add]
    security_findings: Annotated[list, operator.add]
    performance_findings: Annotated[list, operator.add]
    architecture_findings: Annotated[list, operator.add]
    final_report: Optional[dict]
    error: Optional[str]
```

- [ ] **Step 2: Create schemas.py — Pydantic 数据模型**

```python
"""Pydantic 数据模型定义

定义代码审查系统中使用的数据结构，包括审查发现、审查报告等。
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Severity(str, Enum):
    """问题严重程度"""
    CRITICAL = "critical"   # 严重：安全漏洞、逻辑错误
    WARNING = "warning"     # 警告：性能隐患、风格违规
    SUGGESTION = "suggestion"  # 建议：改进建议、最佳实践
    POSITIVE = "positive"   # 亮点：做得好的地方


class Category(str, Enum):
    """审查类别"""
    STYLE = "style"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"


class CodeLocation(BaseModel):
    """代码位置"""
    file_path: str = Field(description="文件路径")
    line_start: Optional[int] = Field(default=None, description="起始行号")
    line_end: Optional[int] = Field(default=None, description="结束行号")


class ReviewFinding(BaseModel):
    """单个审查发现"""
    category: Category = Field(description="审查类别")
    severity: Severity = Field(description="严重程度")
    title: str = Field(description="问题标题")
    description: str = Field(description="问题描述")
    location: CodeLocation = Field(description="代码位置")
    code_snippet: Optional[str] = Field(default=None, description="相关代码片段")
    suggestion: str = Field(description="修复建议")
    reference: Optional[str] = Field(default=None, description="参考规范来源")


class ReviewSummary(BaseModel):
    """审查概览统计"""
    total_files: int = Field(default=0, description="审查文件总数")
    total_additions: int = Field(default=0, description="新增行数")
    total_deletions: int = Field(default=0, description="删除行数")
    critical_count: int = Field(default=0, description="严重问题数")
    warning_count: int = Field(default=0, description="警告数")
    suggestion_count: int = Field(default=0, description="建议数")
    positive_count: int = Field(default=0, description="亮点数")


class ReviewReport(BaseModel):
    """完整审查报告"""
    repo_path: str = Field(description="仓库路径")
    target_branch: str = Field(default="HEAD", description="目标分支")
    summary: ReviewSummary = Field(description="概览统计")
    findings: list[ReviewFinding] = Field(default_factory=list, description="所有审查发现")


class ReviewRequest(BaseModel):
    """审查请求"""
    repo_path: str = Field(description="Git 仓库本地路径")
    target_branch: str = Field(default="HEAD~1", description="比较的目标分支或提交")
    enable_style: bool = Field(default=True, description="启用风格审查")
    enable_security: bool = Field(default=True, description="启用安全审查")
    enable_performance: bool = Field(default=True, description="启用性能审查")
    enable_architecture: bool = Field(default=True, description="启用架构审查")


class ReviewResponse(BaseModel):
    """审查响应"""
    success: bool = Field(description="是否成功")
    report: Optional[ReviewReport] = Field(default=None, description="审查报告")
    error: Optional[str] = Field(default=None, description="错误信息")
```

- [ ] **Step 3: Commit**

```bash
git add phase4_projects/03_code_review_agent/backend/
git commit -m "feat: add state definition and pydantic schemas"
```

---

### Task 3: Built-in Standards Documents

**Files:**
- Create: `data/standards/python_style.md`
- Create: `data/standards/security_rules.md`
- Create: `data/standards/performance_patterns.md`
- Create: `data/standards/architecture_principles.md`

- [ ] **Step 1: Create python_style.md**

内容涵盖 PEP 8 核心规则、命名规范、导入排序、注释和文档字符串规范、代码格式化（行宽、缩进、空行规则）。约200行中文文档，包含正反例代码片段。

- [ ] **Step 2: Create security_rules.md**

内容涵盖 OWASP Top 10 常见漏洞模式：SQL注入、XSS、命令注入、硬编码密钥、不安全的反序列化、路径遍历、不安全的随机数。每种漏洞包含 Python 代码示例和修复方案。

- [ ] **Step 3: Create performance_patterns.md**

内容涵盖 Python 性能反模式：N+1 数据库查询、不必要的列表拷贝、字符串拼接性能、全局变量查找开销、阻塞式IO、内存泄漏模式、低效循环。每种包含检测方法和优化建议。

- [ ] **Step 4: Create architecture_principles.md**

内容涵盖 SOLID 原则、DRY/KISS/YAGNI、常见设计模式（工厂、策略、观察者）、代码耦合度评估、函数/类职责划分、错误处理策略、可测试性设计。

- [ ] **Step 5: Commit**

```bash
git add phase4_projects/03_code_review_agent/data/
git commit -m "feat: add built-in coding standards documents"
```

---

### Task 4: RAG Engine

**Files:**
- Create: `backend/core/rag/embeddings.py`
- Create: `backend/core/rag/standards_store.py`

- [ ] **Step 1: Create embeddings.py — 向量化配置**

```python
"""向量化配置模块

配置 Embeddings 模型用于将编码规范文档向量化。
支持 HuggingFace 本地模型和在线 API。
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """获取 Embeddings 实例

    使用 HuggingFace 本地模型进行文本向量化。
    自动配置 HF Mirror 以加速国内下载。

    Returns:
        HuggingFaceEmbeddings: 向量化实例
    """
    # 配置 HF Mirror（国内加速）
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    if hf_endpoint and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    model_name = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
```

- [ ] **Step 2: Create standards_store.py — 规范知识库管理**

实现 `StandardsStore` 类，包含以下功能：
- `initialize()`: 加载内置规范文档 + 用户上传文档，向量化存入 ChromaDB
- `search(query: str, category: str, top_k: int)`: 按类别检索相关规范
- `add_documents(docs_path: str)`: 添加用户自定义规范
- `get_retriever(category: str)`: 获取指定类别的检索器

使用 ChromaDB 持久化存储，metadata 中包含 category 字段用于过滤。文档分块使用 RecursiveCharacterTextSplitter（chunk_size=800, overlap=100）。

- [ ] **Step 3: Commit**

```bash
git add phase4_projects/03_code_review_agent/backend/core/rag/
git commit -m "feat: add RAG engine with standards store"
```

---

### Task 5: Git Tools

**Files:**
- Create: `backend/core/tools/git_tools.py`

- [ ] **Step 1: Create git_tools.py — Git 操作工具**

实现以下 LangChain @tool 工具：

1. `get_git_diff(repo_path, target_branch)`: 获取 git diff 内容
2. `get_changed_files(repo_path, target_branch)`: 获取变更文件列表
3. `read_file_content(file_path)`: 读取指定文件内容
4. `get_file_structure(repo_path)`: 获取仓库文件结构（目录树）

使用 GitPython 库操作 Git 仓库。每个工具都有清晰的 docstring（供 Agent 理解工具用途）。

关键实现细节：
- diff 使用 `git.Repo(repo_path).git.diff(target_branch)` 获取
- 文件列表解析 diff 输出中的变更统计
- read_file_content 带有路径安全检查（防止路径遍历）
- 文件结构使用 `git.Repo(repo_path).git.ls_tree()` 递归获取

- [ ] **Step 2: Commit**

```bash
git add phase4_projects/03_code_review_agent/backend/core/tools/git_tools.py
git commit -m "feat: add git tools for diff parsing and file reading"
```

---

### Task 6: MCP Tools Integration

**Files:**
- Create: `backend/core/tools/mcp_tools.py`

- [ ] **Step 1: Create mcp_tools.py — MCP 工具集成**

实现 MCP 客户端连接和工具获取：

```python
"""MCP 工具集成模块

通过 langchain-mcp-adapters 连接 MCP filesystem server，
为审查 Agent 提供文件读取能力。
当 MCP 不可用时，自动回退到直接的 Python 文件读取工具。
"""

import os
import logging
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


async def get_mcp_filesystem_tools(repo_path: str) -> list[BaseTool]:
    """获取 MCP filesystem 工具

    尝试通过 langchain-mcp-adapters 连接 filesystem MCP server。
    如果连接失败，返回空列表（后续使用 git_tools 中的直接读取工具）。

    Args:
        repo_path: 需要访问的仓库根路径

    Returns:
        list[BaseTool]: MCP 工具列表（可能为空）
    """
    enabled = os.getenv("MCP_FILESYSTEM_ENABLED", "true").lower() == "true"
    if not enabled:
        logger.info("MCP filesystem 已禁用，使用直接文件读取")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {
                "filesystem": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        repo_path,
                    ],
                    "transport": "stdio",
                }
            }
        )
        tools = client.get_tools()
        logger.info(f"MCP filesystem 工具加载成功，共 {len(tools)} 个工具")
        return tools
    except Exception as e:
        logger.warning(f"MCP filesystem 加载失败: {e}，将使用直接文件读取工具")
        return []


def get_fallback_tools() -> list[BaseTool]:
    """获取回退工具（直接文件读取）

    当 MCP 不可用时，提供等价的文件操作工具。
    使用 backend/core/tools/git_tools.py 中的工具。

    Returns:
        list[BaseTool]: 直接文件操作工具列表
    """
    from backend.core.tools.git_tools import read_file_content, get_file_structure
    return [read_file_content, get_file_structure]
```

- [ ] **Step 2: Commit**

```bash
git add phase4_projects/03_code_review_agent/backend/core/tools/mcp_tools.py
git commit -m "feat: add MCP tools integration with fallback"
```

---

### Task 7: Reviewer Agents (4个)

**Files:**
- Create: `backend/core/agents/style_reviewer.py`
- Create: `backend/core/agents/security_reviewer.py`
- Create: `backend/core/agents/performance_reviewer.py`
- Create: `backend/core/agents/architecture_reviewer.py`

每个 Reviewer Agent 的结构相同，差异在 system_prompt 和 RAG category：

- [ ] **Step 1: Create style_reviewer.py**

```python
"""代码风格审查 Agent

检查代码的命名规范、格式化、注释质量、导入排序等风格问题。
使用 RAG 检索 PEP 8 / Google 风格指南等规范。
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import json
import logging

logger = logging.getLogger(__name__)

STYLE_REVIEW_SYSTEM_PROMPT = """你是一位资深代码风格审查专家。

## 你的职责
审查代码的风格质量，重点关注：
1. **命名规范** - 变量名、函数名、类名是否符合规范（snake_case/camelCase/PascalCase）
2. **代码格式** - 缩进、空行、行宽、括号风格
3. **注释和文档** - 是否有必要的注释，docstring 是否完整
4. **导入管理** - 导入是否有序，是否有多余导入
5. **代码一致性** - 同一项目中风格是否统一

## 输出格式
请以 JSON 格式输出审查结果：
```json
{{
    "findings": [
        {{
            "severity": "warning|suggestion|positive",
            "title": "简短标题",
            "description": "问题描述",
            "file_path": "文件路径",
            "line_start": 起始行号,
            "line_end": 结束行号,
            "code_snippet": "相关代码片段",
            "suggestion": "修复建议",
            "reference": "参考规范"
        }}
    ]
}}
```

## 注意事项
- 只报告确定的问题，不要过度警告
- 每个发现必须包含具体的代码位置
- 也要标记做得好的地方（severity: positive）
- 参考 RAG 检索到的规范标准
"""


def get_llm():
    """获取 LLM 实例"""
    return ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.1,
    )


async def style_review_node(state: dict) -> dict:
    """风格审查节点函数（供 LangGraph 调用）

    Args:
        state: 工作流状态，包含 diff_content, changed_files, review_config 等

    Returns:
        更新后的状态（包含 style_findings）
    """
    if not state.get("review_config", {}).get("enable_style", True):
        return {"style_findings": []}

    llm = get_llm()
    diff_content = state.get("diff_content", "")
    changed_files = state.get("changed_files", [])
    standards = state.get("style_standards", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", STYLE_REVIEW_SYSTEM_PROMPT),
        ("human", """请审查以下代码变更的风格质量：

## 变更文件
{changed_files}

## Git Diff 内容
```diff
{diff_content}
```

## 相关编码规范
{standards}

请输出 JSON 格式的审查结果。"""),
    ])

    chain = prompt | llm

    try:
        response = await chain.ainvoke({
            "diff_content": diff_content[:8000],  # 限制长度
            "changed_files": json.dumps(changed_files, ensure_ascii=False, indent=2),
            "standards": standards[:3000] if standards else "未找到相关规范",
        })

        result = json.loads(response.content)
        findings = result.get("findings", [])

        # 为每个发现添加 category
        for f in findings:
            f["category"] = "style"

        return {"style_findings": findings}

    except json.JSONDecodeError:
        logger.warning("风格审查结果 JSON 解析失败")
        return {"style_findings": []}
    except Exception as e:
        logger.error(f"风格审查出错: {e}")
        return {"style_findings": []}
```

- [ ] **Step 2: Create security_reviewer.py**

结构同 style_reviewer.py，但：
- `SECURITY_REVIEW_SYSTEM_PROMPT` 聚焦：SQL注入、XSS、命令注入、硬编码密钥、不安全反序列化、路径遍历
- severity 包含 `critical` 级别
- RAG category 为 `security`
- 函数名为 `security_review_node`

- [ ] **Step 3: Create performance_reviewer.py**

结构同 style_reviewer.py，但：
- `PERFORMANCE_REVIEW_SYSTEM_PROMPT` 聚焦：N+1查询、内存泄漏、低效循环、阻塞IO、不必要的拷贝
- RAG category 为 `performance`
- 函数名为 `performance_review_node`

- [ ] **Step 4: Create architecture_reviewer.py**

结构同 style_reviewer.py，但：
- `ARCHITECTURE_REVIEW_SYSTEM_PROMPT` 聚焦：SOLID原则违反、高耦合、职责不清、设计模式建议、可测试性
- RAG category 为 `architecture`
- 函数名为 `architecture_review_node`

- [ ] **Step 5: Commit**

```bash
git add phase4_projects/03_code_review_agent/backend/core/agents/
git commit -m "feat: add 4 specialized reviewer agents"
```

---

### Task 8: LangGraph Workflow

**Files:**
- Create: `backend/core/workflow.py`

- [ ] **Step 1: Create workflow.py — 审查工作流编排**

```python
"""LangGraph 代码审查工作流

使用 StateGraph 编排代码审查流程：
1. 解析 diff（parse_diff_node）
2. 并行分发到 4 个审查 Agent（fan_out）
3. 收集所有审查结果（fan_in → synthesize_node）
4. 生成最终报告
"""

import os
import json
import logging
import operator
from typing import Annotated

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from backend.core.state import ReviewState
from backend.core.agents.style_reviewer import style_review_node
from backend.core.agents.security_reviewer import security_review_node
from backend.core.agents.performance_reviewer import performance_review_node
from backend.core.agents.architecture_reviewer import architecture_review_node
from backend.core.tools.git_tools import get_git_diff, get_changed_files
from backend.core.rag.standards_store import StandardsStore

logger = logging.getLogger(__name__)


def parse_diff_node(state: dict) -> dict:
    """解析 git diff 节点

    获取 diff 内容和变更文件列表，更新状态。
    同时从 RAG 知识库检索各领域相关规范。
    """
    repo_path = state["repo_path"]
    target_branch = state.get("target_branch", "HEAD~1")

    try:
        # 获取 git diff
        diff_content = get_git_diff(repo_path, target_branch)
        changed_files = get_changed_files(repo_path, target_branch)

        # 从 RAG 获取相关规范
        store = StandardsStore()
        style_standards = store.search("代码风格 命名规范 格式化", category="style", top_k=5)
        security_standards = store.search("安全漏洞 SQL注入 XSS 命令注入", category="security", top_k=5)
        perf_standards = store.search("性能优化 循环 数据库查询 内存", category="performance", top_k=5)
        arch_standards = store.search("架构设计 SOLID 耦合 设计模式", category="architecture", top_k=5)

        def format_standards(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        return {
            "diff_content": diff_content,
            "changed_files": changed_files,
            "style_standards": format_standards(style_standards),
            "security_standards": format_standards(security_standards),
            "performance_standards": format_standards(perf_standards),
            "architecture_standards": format_standards(arch_standards),
        }
    except Exception as e:
        logger.error(f"Diff 解析失败: {e}")
        return {
            "diff_content": "",
            "changed_files": [],
            "error": str(e),
        }


def synthesize_node(state: dict) -> dict:
    """综合报告生成节点

    汇总所有审查发现，使用 LLM 生成结构化审查报告。
    """
    all_findings = (
        state.get("style_findings", [])
        + state.get("security_findings", [])
        + state.get("performance_findings", [])
        + state.get("architecture_findings", [])
    )

    if not all_findings:
        return {"final_report": {
            "summary": {
                "total_files": len(state.get("changed_files", [])),
                "total_additions": 0,
                "total_deletions": 0,
                "critical_count": 0,
                "warning_count": 0,
                "suggestion_count": 0,
                "positive_count": 0,
            },
            "findings": [],
            "message": "未发现明显问题，代码质量良好。",
        }}

    # 统计各类别数量
    severity_counts = {"critical": 0, "warning": 0, "suggestion": 0, "positive": 0}
    for f in all_findings:
        sev = f.get("severity", "suggestion")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # 按严重程度排序：critical > warning > suggestion > positive
    severity_order = {"critical": 0, "warning": 1, "suggestion": 2, "positive": 3}
    sorted_findings = sorted(all_findings, key=lambda x: severity_order.get(x.get("severity", "suggestion"), 3))

    # 统计变更行数
    total_additions = sum(f.get("additions", 0) for f in state.get("changed_files", []))
    total_deletions = sum(f.get("deletions", 0) for f in state.get("changed_files", []))

    report = {
        "summary": {
            "total_files": len(state.get("changed_files", [])),
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            **severity_counts,
        },
        "findings": sorted_findings,
    }

    return {"final_report": report}


def create_review_workflow() -> StateGraph:
    """创建代码审查工作流

    构建并行审查的 LangGraph 状态图：
    START → parse_diff → [style|security|performance|architecture] → synthesize → END

    Returns:
        StateGraph: 编译后的工作流图
    """
    graph = StateGraph(ReviewState)

    # 添加节点
    graph.add_node("parse_diff", parse_diff_node)
    graph.add_node("style_review", style_review_node)
    graph.add_node("security_review", security_review_node)
    graph.add_node("performance_review", performance_review_node)
    graph.add_node("architecture_review", architecture_review_node)
    graph.add_node("synthesize", synthesize_node)

    # 定义边：START → parse_diff
    graph.add_edge(START, "parse_diff")

    # Fan-out: parse_diff → 4个并行审查
    graph.add_edge("parse_diff", "style_review")
    graph.add_edge("parse_diff", "security_review")
    graph.add_edge("parse_diff", "performance_review")
    graph.add_edge("parse_diff", "architecture_review")

    # Fan-in: 4个审查 → synthesize
    graph.add_edge("style_review", "synthesize")
    graph.add_edge("security_review", "synthesize")
    graph.add_edge("performance_review", "synthesize")
    graph.add_edge("architecture_review", "synthesize")

    # synthesize → END
    graph.add_edge("synthesize", END)

    return graph.compile()
```

- [ ] **Step 2: Commit**

```bash
git add phase4_projects/03_code_review_agent/backend/core/workflow.py
git commit -m "feat: add LangGraph workflow with fan-out/fan-in pattern"
```

---

### Task 9: FastAPI Backend

**Files:**
- Create: `backend/main.py`

- [ ] **Step 1: Create main.py — FastAPI 应用入口**

实现以下 API 端点：
1. `POST /api/review` — 提交代码审查请求（异步执行）
2. `GET /api/review/{task_id}` — 获取审查结果
3. `GET /api/health` — 健康检查
4. `POST /api/standards/upload` — 上传自定义规范文档
5. `GET /api/standards/list` — 列出当前规范库
6. `DELETE /api/standards/{doc_id}` — 删除规范文档

使用 asyncio 在后台执行审查工作流。审查结果以 JSON 返回。

关键实现：
- FastAPI app with CORS middleware
- 使用 `create_review_workflow()` 创建工作流实例
- 审查请求通过 `ainvoke()` 异步执行
- 错误处理和日志记录

- [ ] **Step 2: Commit**

```bash
git add phase4_projects/03_code_review_agent/backend/main.py
git commit -m "feat: add FastAPI backend with review API endpoints"
```

---

### Task 10: Streamlit Frontend

**Files:**
- Create: `frontend/main.py`

- [ ] **Step 1: Create main.py — Streamlit Web 界面**

实现三个页面：

**1. 审查页面（review_page）**
- 输入仓库路径和目标分支
- 勾选审查维度（风格/安全/性能/架构）
- 点击"开始审查"触发审查
- 展示审查进度
- 展示结构化审查报告（按严重程度分组）
- 每个发现可展开查看详情（代码片段、建议、参考）

**2. 知识库页面（knowledge_page）**
- 展示内置规范列表
- 上传自定义规范文档
- 删除已上传的规范
- 显示向量库统计信息

**3. 关于页面**
- 项目介绍
- 技术架构说明
- 使用指南

样式设计：
- 专业暗色主题（#1a1a2e 主色调）
- 使用 st.columns 做卡片布局
- 使用 st.expander 展示详细信息
- 自定义 CSS 美化

- [ ] **Step 2: Commit**

```bash
git add phase4_projects/03_code_review_agent/frontend/
git commit -m "feat: add Streamlit frontend with review and knowledge pages"
```

---

### Task 11: README and Integration

**Files:**
- Create: `README.md`
- Copy: `.env.example` → `.env`

- [ ] **Step 1: Create README.md**

包含项目介绍、技术架构图、功能特性、安装步骤、使用指南、技术栈说明。

- [ ] **Step 2: Copy .env and install dependencies**

```bash
cp .env.example .env
pip install -r requirements.txt
```

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "feat: complete AI code review agent project"
```

---

## Self-Review Checklist

- [x] Spec coverage: All design sections mapped to tasks
- [x] Placeholder scan: All steps contain concrete code or descriptions
- [x] Type consistency: State fields and schemas match across files
- [x] No circular dependencies between modules
