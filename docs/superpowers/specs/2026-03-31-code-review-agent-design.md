# AI Code Review Agent 设计文档

## 项目概述

**项目名**: `03_code_review_agent` — AI 智能代码审查助手

**核心功能**: 用户输入本地 Git 仓库路径，系统自动分析 git diff，调用多个专业 Agent 并行审查代码，生成结构化审查报告。

## 与现有项目差异化

| 维度 | 02_智能客服 | 03_代码审查 |
|------|-----------|-----------|
| Agent模式 | 路由式（1→1） | 并行协作式（1→N→1） |
| 输入类型 | 自然语言对话 | Git仓库/代码 |
| RAG用途 | 企业知识库问答 | 编码规范检索 |
| 输出 | 实时对话回复 | 结构化审查报告 |
| MCP | 无 | 有（filesystem等） |

## 技术栈

- LangChain 1.0+ — LLM接口、工具定义、文档处理
- LangGraph 1.0+ — 多Agent编排、状态图、并行节点
- Zhipu AI (glm-4-flash) — LLM推理
- ChromaDB — 规范知识库向量存储
- langchain-mcp-adapters — MCP工具集成
- Streamlit — Web界面
- GitPython — Git仓库操作
- FastAPI — 后端API

## 系统架构

```
Streamlit 前端: 仓库输入 | 审查配置 | 报告展示 | 知识库管理
                          │
FastAPI 后端:
  LangGraph 审查工作流:
    [协调器] → 收集diff → 分发任务
         │
    ┌────┼────┬────────┬────────┐
    风格  安全  性能    架构    RAG检索
    Agent Agent Agent  Agent    Agent
    └────┴────┴────────┴────────┘
         │
    [综合器] → 生成结构化报告

  Git工具(GitPython) | MCP工具(filesystem) | RAG引擎(ChromaDB)
```

## LangGraph 工作流

StateGraph 状态定义:
- repo_path: str — 仓库路径
- diff_content: str — git diff 内容
- changed_files: list — 变更文件列表
- review_config: dict — 审查配置
- style_findings: list — 风格审查结果
- security_findings: list — 安全审查结果
- performance_findings: list — 性能审查结果
- architecture_findings: list — 架构审查结果
- final_report: dict — 最终报告

工作流: START → parse_diff → fan_out → [style, security, performance, architecture] → fan_in → synthesize → END

## Agent 职责

| Agent | 职责 | 工具 | RAG知识源 |
|-------|------|------|----------|
| DiffParser | 解析git diff | GitPython | 无 |
| StyleReviewer | 命名、格式、注释规范 | MCP filesystem | PEP8/风格指南 |
| SecurityReviewer | 注入、密钥泄露、不安全API | MCP filesystem | OWASP规则 |
| PerformanceReviewer | N+1查询、内存泄漏、低效算法 | MCP filesystem | 性能反模式 |
| ArchitectureReviewer | 耦合度、设计模式、可测试性 | MCP filesystem | SOLID/设计模式 |
| ReportSynthesizer | 汇总结果、生成报告 | 无 | 无 |

## RAG 知识库

内置规范文档(data/standards/):
- python_style.md — PEP 8 + Google Python 风格指南
- security_rules.md — OWASP Top 10 漏洞模式
- performance_patterns.md — Python 性能反模式
- architecture_principles.md — SOLID原则 + 设计模式

支持用户上传团队规范文档。每个Agent通过元数据category过滤检索。

## 审查报告格式

结构化Markdown报告，包含:
- 概览(文件数、变更行数、问题统计)
- 严重问题(安全漏洞、逻辑错误)
- 警告(性能隐患、风格违规)
- 改进建议(架构优化、重构建议)
- 亮点(做得好的地方)

每个问题标注文件名、行号、代码片段、修复建议、参考规范。

## 文件结构

```
phase4_projects/03_code_review_agent/
├── backend/
│   ├── main.py
│   ├── core/
│   │   ├── workflow.py
│   │   ├── state.py
│   │   ├── agents/
│   │   │   ├── style_reviewer.py
│   │   │   ├── security_reviewer.py
│   │   │   ├── performance_reviewer.py
│   │   │   └── architecture_reviewer.py
│   │   ├── tools/
│   │   │   ├── git_tools.py
│   │   │   └── mcp_tools.py
│   │   └── rag/
│   │       ├── standards_store.py
│   │       └── embeddings.py
│   └── models/
│       └── schemas.py
├── frontend/
│   └── main.py
├── data/
│   └── standards/
│       ├── python_style.md
│       ├── security_rules.md
│       ├── performance_patterns.md
│       └── architecture_principles.md
├── .env.example
└── README.md
```

## MCP 集成

使用 langchain-mcp-adapters 的 MultiServerMCPClient 连接 filesystem MCP server，Agent 通过标准协议读取仓库文件。
