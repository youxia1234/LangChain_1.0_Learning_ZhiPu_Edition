"""Streamlit 前端应用

AI 代码审查助手的 Web 界面，包含：
- 审查页面：输入仓库路径，配置审查维度，展示审查报告
- 知识库页面：管理编码规范文档
- 关于页面：项目介绍和使用指南
"""

import os
import sys
import json
import requests
from pathlib import Path

import streamlit as st

# 配置页面
st.set_page_config(
    page_title="AI Code Review Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API 基础地址
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.5rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #3d3d5c;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
    }
    .severity-critical { color: #ff4757; }
    .severity-warning { color: #ffa502; }
    .severity-suggestion { color: #3498db; }
    .severity-positive { color: #2ed573; }
    .finding-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #667eea;
    }
    .finding-critical { border-left-color: #ff4757; }
    .finding-warning { border-left-color: #ffa502; }
    .finding-suggestion { border-left-color: #3498db; }
    .finding-positive { border-left-color: #2ed573; }
    .code-block {
        background: #0d1117;
        border-radius: 6px;
        padding: 0.8rem;
        font-family: 'Cascadia Code', 'Fira Code', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
        border: 1px solid #30363d;
    }
    .category-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-style { background: #6c5ce7; color: white; }
    .badge-security { background: #d63031; color: white; }
    .badge-performance { background: #00b894; color: white; }
    .badge-architecture { background: #0984e3; color: white; }
</style>
""", unsafe_allow_html=True)


def check_backend_health() -> dict:
    """检查后端健康状态"""
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "unhealthy"}


def submit_review(repo_path: str, target_branch: str, config: dict) -> dict:
    """提交审查请求"""
    try:
        payload = {
            "repo_path": repo_path,
            "target_branch": target_branch,
            **config,
        }
        resp = requests.post(f"{API_BASE}/api/review", json=payload, timeout=120)
        return resp.json()
    except requests.exceptions.Timeout:
        return {"success": False, "error": "请求超时，审查可能需要更长时间"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_standards_list() -> dict:
    """获取规范库列表"""
    try:
        resp = requests.get(f"{API_BASE}/api/standards/list", timeout=10)
        return resp.json()
    except Exception:
        return {"documents": [], "total_chunks": 0}


def upload_standard(file, category: str) -> dict:
    """上传规范文档"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/octet-stream")}
        data = {"category": category}
        resp = requests.post(
            f"{API_BASE}/api/standards/upload",
            files=files,
            data=data,
            timeout=30,
        )
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_standard(source: str) -> dict:
    """删除规范文档"""
    try:
        resp = requests.delete(f"{API_BASE}/api/standards/{source}", timeout=10)
        return resp.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_severity_icon(severity: str) -> str:
    """渲染严重程度图标"""
    icons = {
        "critical": '<span class="severity-critical">&#x1F534;</span>',
        "warning": '<span class="severity-warning">&#x1F7E1;</span>',
        "suggestion": '<span class="severity-suggestion">&#x1F535;</span>',
        "positive": '<span class="severity-positive">&#x1F7E2;</span>',
    }
    return icons.get(severity, "&#x26AA;")


def render_category_badge(category: str) -> str:
    """渲染类别标签"""
    labels = {
        "style": "风格",
        "security": "安全",
        "performance": "性能",
        "architecture": "架构",
    }
    label = labels.get(category, category)
    return f'<span class="category-badge badge-{category}">{label}</span>'


def render_report(report: dict):
    """渲染审查报告"""
    if not report:
        st.warning("未生成报告")
        return

    summary = report.get("summary", {})
    findings = report.get("findings", [])

    # 概览统计卡片
    st.markdown("## 📊 审查概览")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number">{summary.get('total_files', 0)}</div>
            <div style="color:#aaa;">审查文件</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number" style="color:#2ed573">+{summary.get('total_additions', 0)}</div>
            <div style="color:#aaa;">新增行</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number" style="color:#ff4757">-{summary.get('total_deletions', 0)}</div>
            <div style="color:#aaa;">删除行</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number severity-critical">{summary.get('critical_count', 0)}</div>
            <div style="color:#aaa;">严重</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number severity-warning">{summary.get('warning_count', 0)}</div>
            <div style="color:#aaa;">警告</div>
        </div>""", unsafe_allow_html=True)

    with col6:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-number severity-suggestion">{summary.get('suggestion_count', 0) + summary.get('positive_count', 0)}</div>
            <div style="color:#aaa;">建议/亮点</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#3d3d5c'>", unsafe_allow_html=True)

    if not findings:
        st.success("代码质量良好，未发现明显问题！")
        return

    # 按严重程度分组显示
    severity_groups = {
        "critical": {"label": "严重问题", "icon": "🔴", "findings": []},
        "warning": {"label": "警告", "icon": "🟡", "findings": []},
        "suggestion": {"label": "改进建议", "icon": "🔵", "findings": []},
        "positive": {"label": "亮点", "icon": "🟢", "findings": []},
    }

    for finding in findings:
        sev = finding.get("severity", "suggestion")
        if sev in severity_groups:
            severity_groups[sev]["findings"].append(finding)

    # 渲染每组
    for sev_key, group in severity_groups.items():
        if not group["findings"]:
            continue

        st.markdown(f"### {group['icon']} {group['label']} ({len(group['findings'])})")

        for i, finding in enumerate(group["findings"], 1):
            title = finding.get("title", "未命名问题")
            category = finding.get("category", "unknown")
            file_path = finding.get("file_path", finding.get("location", {}).get("file_path", ""))
            line_start = finding.get("line_start", finding.get("location", {}).get("line_start"))
            description = finding.get("description", "")
            suggestion = finding.get("suggestion", "")
            code_snippet = finding.get("code_snippet", "")
            reference = finding.get("reference", "")

            # 位置信息
            location_str = file_path
            if line_start:
                location_str += f":{line_start}"

            with st.expander(
                f"**{i}. {title}** — `{location_str}`",
                expanded=(sev_key == "critical"),
            ):
                # 类别和位置
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    st.markdown(render_category_badge(category), unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"**位置**: `{location_str}`")

                # 问题描述
                st.markdown(f"**问题**: {description}")

                # 代码片段
                if code_snippet:
                    st.markdown("**代码**:")
                    st.markdown(
                        f'<div class="code-block"><pre>{code_snippet}</pre></div>',
                        unsafe_allow_html=True,
                    )

                # 修复建议
                st.markdown(f"**建议**: {suggestion}")

                # 参考规范
                if reference:
                    st.markdown(f"**参考**: {reference}")

    # 下载报告
    st.markdown("<hr style='border-color:#3d3d5c'>", unsafe_allow_html=True)
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 下载 JSON 报告",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="code_review_report.json",
            mime="application/json",
        )
    with col_dl2:
        # 生成 Markdown 格式报告
        md_report = _generate_markdown_report(report)
        st.download_button(
            label="📥 下载 Markdown 报告",
            data=md_report,
            file_name="code_review_report.md",
            mime="text/markdown",
        )


def _generate_markdown_report(report: dict) -> str:
    """生成 Markdown 格式的审查报告"""
    summary = report.get("summary", {})
    findings = report.get("findings", [])

    lines = [
        "# 代码审查报告\n",
        f"**仓库**: `{report.get('repo_path', '')}`",
        f"**目标**: `{report.get('target_branch', 'HEAD~1')}`\n",
        "## 概览\n",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 审查文件 | {summary.get('total_files', 0)} |",
        f"| 新增行数 | +{summary.get('total_additions', 0)} |",
        f"| 删除行数 | -{summary.get('total_deletions', 0)} |",
        f"| 严重问题 | {summary.get('critical_count', 0)} |",
        f"| 警告 | {summary.get('warning_count', 0)} |",
        f"| 建议 | {summary.get('suggestion_count', 0)} |",
        f"| 亮点 | {summary.get('positive_count', 0)} |\n",
    ]

    severity_labels = {"critical": "严重", "warning": "警告", "suggestion": "建议", "positive": "亮点"}
    category_labels = {"style": "风格", "security": "安全", "performance": "性能", "architecture": "架构"}

    for finding in findings:
        sev = finding.get("severity", "suggestion")
        cat = finding.get("category", "unknown")
        title = finding.get("title", "")
        location = finding.get("file_path", finding.get("location", {}).get("file_path", ""))
        line = finding.get("line_start", finding.get("location", {}).get("line_start"))
        desc = finding.get("description", "")
        sug = finding.get("suggestion", "")
        ref = finding.get("reference", "")
        code = finding.get("code_snippet", "")

        loc_str = f"{location}:{line}" if line else location

        lines.append(f"### [{severity_labels.get(sev, sev)}] {title}\n")
        lines.append(f"- **类别**: {category_labels.get(cat, cat)}")
        lines.append(f"- **位置**: `{loc_str}`")
        lines.append(f"- **问题**: {desc}")
        lines.append(f"- **建议**: {sug}")
        if ref:
            lines.append(f"- **参考**: {ref}")
        if code:
            lines.append(f"\n```python\n{code}\n```\n")
        lines.append("---\n")

    return "\n".join(lines)


# ==================== 页面定义 ====================


def review_page():
    """代码审查页面"""
    st.markdown('<div class="main-header">AI Code Review Agent</div>', unsafe_allow_html=True)
    st.markdown("基于 LangChain 1.0 + LangGraph 1.0 的智能代码审查系统\n")

    # 输入区域
    col_input1, col_input2 = st.columns([3, 1])

    with col_input1:
        repo_path = st.text_input(
            "Git 仓库路径",
            placeholder="例如: E:/projects/my-project",
            help="输入本地 Git 仓库的完整路径",
        )

    with col_input2:
        target_branch = st.text_input(
            "目标分支/提交",
            value="HEAD~1",
            help="支持分支名 (main, dev) 或提交引用 (HEAD~3, abc123)",
        )

    # 审查维度配置
    st.markdown("### 审查维度配置")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    with col_s1:
        enable_style = st.checkbox("🎨 代码风格", value=True, help="命名规范、格式化、注释质量")
    with col_s2:
        enable_security = st.checkbox("🔒 安全漏洞", value=True, help="SQL注入、XSS、硬编码密钥")
    with col_s3:
        enable_performance = st.checkbox("⚡ 性能优化", value=True, help="N+1查询、内存泄漏、低效算法")
    with col_s4:
        enable_architecture = st.checkbox("🏗️ 架构设计", value=True, help="SOLID原则、耦合度、可测试性")

    config = {
        "enable_style": enable_style,
        "enable_security": enable_security,
        "enable_performance": enable_performance,
        "enable_architecture": enable_architecture,
    }

    # 提交审查
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 开始审查", type="primary", use_container_width=True, disabled=not repo_path):
        if not repo_path:
            st.error("请输入 Git 仓库路径")
            return

        with st.spinner("正在分析代码变更..."):
            # 显示进度
            progress_text = st.empty()
            progress_bar = st.progress(0)

            progress_text.text("📋 解析 Git diff...")
            progress_bar.progress(20)

            result = submit_review(repo_path, target_branch, config)

            progress_text.text("✅ 审查完成")
            progress_bar.progress(100)

        if result.get("success"):
            report = result.get("report", {})
            st.session_state["last_report"] = report
            st.rerun()
        else:
            st.error(f"审查失败: {result.get('error', '未知错误')}")

    # 显示报告（如果有）
    if "last_report" in st.session_state:
        render_report(st.session_state["last_report"])


def knowledge_page():
    """知识库管理页面"""
    st.markdown('<div class="main-header">📚 规范知识库</div>', unsafe_allow_html=True)
    st.markdown("管理编码规范文档，支持内置规范和自定义上传\n")

    # 规范库统计
    standards_data = get_standards_list()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("文档总数", len(standards_data.get("documents", [])))
    with col_s2:
        st.metric("知识分块数", standards_data.get("total_chunks", 0))

    st.markdown("<hr style='border-color:#3d3d5c'>", unsafe_allow_html=True)

    # 已有规范列表
    st.markdown("### 已加载规范")

    docs = standards_data.get("documents", [])
    if docs:
        for doc in docs:
            col_doc1, col_doc2, col_doc3, col_doc4 = st.columns([3, 2, 1, 1])

            with col_doc1:
                source_icon = "📦" if doc.get("source") == "builtin" else "📎"
                st.text(f"{source_icon} {doc.get('filename', '未知')}")

            with col_doc2:
                category_labels = {
                    "style": "🎨 风格",
                    "security": "🔒 安全",
                    "performance": "⚡ 性能",
                    "architecture": "🏗️ 架构",
                }
                cat_label = category_labels.get(doc.get("category", ""), doc.get("category", ""))
                st.text(cat_label)

            with col_doc3:
                st.text(f"{doc.get('chunk_count', 0)} 块")

            with col_doc4:
                if doc.get("source") == "user":
                    if st.button("删除", key=f"del_{doc.get('doc_id')}"):
                        result = delete_standard(doc.get("filename", ""))
                        if result.get("success"):
                            st.success("删除成功")
                            st.rerun()
                        else:
                            st.error(f"删除失败: {result.get('error', '')}")
    else:
        st.info("暂无规范文档")

    st.markdown("<hr style='border-color:#3d3d5c'>", unsafe_allow_html=True)

    # 上传规范
    st.markdown("### 上传自定义规范")

    col_up1, col_up2 = st.columns([3, 1])

    with col_up1:
        uploaded_file = st.file_uploader(
            "选择文件",
            type=["md", "txt", "pdf"],
            help="支持 Markdown (.md)、纯文本 (.txt)、PDF (.pdf)",
        )

    with col_up2:
        category = st.selectbox(
            "规范类别",
            options=["style", "security", "performance", "architecture", "custom"],
            format_func=lambda x: {
                "style": "🎨 风格",
                "security": "🔒 安全",
                "performance": "⚡ 性能",
                "architecture": "🏗️ 架构",
                "custom": "📝 自定义",
            }.get(x, x),
        )

    if uploaded_file and st.button("📤 上传", type="primary"):
        with st.spinner("正在处理文档..."):
            result = upload_standard(uploaded_file, category)
        if result.get("success"):
            st.success(f"上传成功！已生成 {result.get('chunk_count', 0)} 个知识分块")
            st.rerun()
        else:
            st.error(f"上传失败: {result.get('error', '未知错误')}")


def about_page():
    """关于页面"""
    st.markdown('<div class="main-header">关于 AI Code Review Agent</div>', unsafe_allow_html=True)

    st.markdown("""
## 项目简介

AI Code Review Agent 是一个基于 **LangChain 1.0** 和 **LangGraph 1.0** 构建的智能代码审查系统。
它能够自动分析 Git 仓库的代码变更，从多个维度并行审查代码质量，生成结构化的审查报告。

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
  ┌─────────────────────────────────┐
  │     LangGraph 并行审查工作流     │
  │  ┌──────┐ ┌──────┐ ┌──────┐    │
  │  │ 风格  │ │ 安全  │ │ 性能  │    │
  │  │Agent │ │Agent │ │Agent │    │
  │  └──┬───┘ └──┬───┘ └──┬───┘    │
  │     └────────┼────────┘        │
  │              ▼                  │
  │        ┌──────────┐            │
  │        │   综合    │            │
  │        │  报告生成  │            │
  │        └──────────┘            │
  └─────────────────────────────────┘
       │
       ▼
  结构化审查报告
```

## 核心特性

| 特性 | 说明 |
|------|------|
| **多Agent并行审查** | 4个专业Agent并行工作，覆盖风格/安全/性能/架构 |
| **RAG规范检索** | 从知识库检索编码规范，审查有据可依 |
| **MCP工具集成** | 通过MCP协议连接文件系统，标准化工具调用 |
| **LangGraph编排** | Fan-out/Fan-in并行模式，高效利用多Agent |
| **结构化报告** | 按严重程度分级，含代码位置和修复建议 |

## 使用指南

1. **启动后端**: `cd backend && python main.py`
2. **启动前端**: `cd frontend && streamlit run main.py`
3. **输入仓库路径**: 填写本地 Git 仓库的完整路径
4. **选择审查维度**: 勾选需要审查的维度
5. **开始审查**: 点击按钮，等待分析完成
6. **查看报告**: 查看结构化报告，下载 JSON/Markdown 格式

## 技术栈

- **LangChain 1.0+**: LLM 接口、工具定义、文档处理
- **LangGraph 1.0+**: 多Agent编排、并行状态图
- **Zhipu AI (glm-4-flash)**: LLM 推理引擎
- **ChromaDB**: 规范知识库向量存储
- **langchain-mcp-adapters**: MCP 工具集成
- **Streamlit**: Web 界面
- **FastAPI**: 后端 API
- **GitPython**: Git 仓库操作
""")


# ==================== 主入口 ====================

def main():
    """主函数"""
    # 侧边栏
    with st.sidebar:
        st.markdown("## 🔍 AI Code Review")
        st.markdown("---")

        page = st.radio(
            "导航",
            options=["审查代码", "知识库管理", "关于"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 后端状态
        health = check_backend_health()
        if health.get("status") == "healthy":
            st.success("后端服务: 运行中")
            st.caption(f"规范分块: {health.get('standards_loaded', 0)}")
        else:
            st.error("后端服务: 未连接")
            st.caption("请先启动后端: `python backend/main.py`")

        st.markdown("---")
        st.caption("Powered by LangChain 1.0 + LangGraph 1.0")
        st.caption("Model: Zhipu AI glm-4-flash")

    # 渲染页面
    if page == "审查代码":
        review_page()
    elif page == "知识库管理":
        knowledge_page()
    elif page == "关于":
        about_page()


if __name__ == "__main__":
    main()
