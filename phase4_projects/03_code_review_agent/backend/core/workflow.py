"""LangGraph 代码审查工作流

使用 StateGraph 编排多 Agent 并行代码审查流程：
1. parse_diff_node — 解析 git diff，检索相关规范
2. 4 个审查 Agent 并行执行（fan-out）
3. synthesize_node — 汇总结果，生成结构化报告（fan-in）

核心模式：fan-out / fan-in（并行分发 → 并行收集 → 综合）
"""

import os
import json
import logging

from langgraph.graph import StateGraph, START, END

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

    获取 diff 内容和变更文件列表，并从 RAG 知识库
    检索各领域相关的编码规范，供后续审查 Agent 使用。

    Args:
        state: 工作流状态

    Returns:
        dict: 更新的状态字段
    """
    repo_path = state["repo_path"]
    target_branch = state.get("target_branch", "HEAD~1")
    review_config = state.get("review_config", {})

    try:
        # 获取 git diff 内容
        diff_content = get_git_diff.invoke({
            "repo_path": repo_path,
            "target_branch": target_branch,
        })

        # 获取变更文件列表
        changed_files = get_changed_files.invoke({
            "repo_path": repo_path,
            "target_branch": target_branch,
        })

        logger.info(f"Diff 解析完成: {len(changed_files)} 个文件变更, {len(diff_content)} 字符")

        # 从 RAG 获取相关规范
        store = StandardsStore()
        store.initialize()

        def format_standards(docs):
            if not docs:
                return ""
            return "\n\n---\n\n".join(
                f"[{doc.metadata.get('source', '未知来源')}] {doc.page_content}"
                for doc in docs
            )

        # 根据审查配置按需检索规范
        style_standards = ""
        security_standards = ""
        performance_standards = ""
        architecture_standards = ""

        if review_config.get("enable_style", True):
            style_standards = format_standards(
                store.search("代码风格 命名规范 格式化 PEP8", category="style", top_k=5)
            )

        if review_config.get("enable_security", True):
            security_standards = format_standards(
                store.search("安全漏洞 SQL注入 XSS 命令注入 硬编码密钥", category="security", top_k=5)
            )

        if review_config.get("enable_performance", True):
            performance_standards = format_standards(
                store.search("性能优化 循环 数据库查询 内存 N+1", category="performance", top_k=5)
            )

        if review_config.get("enable_architecture", True):
            architecture_standards = format_standards(
                store.search("架构设计 SOLID 耦合 设计模式 职责", category="architecture", top_k=5)
            )

        return {
            "diff_content": diff_content,
            "changed_files": changed_files,
            "style_standards": style_standards,
            "security_standards": security_standards,
            "performance_standards": performance_standards,
            "architecture_standards": architecture_standards,
        }

    except Exception as e:
        logger.error(f"Diff 解析失败: {e}")
        return {
            "diff_content": "",
            "changed_files": [],
            "error": f"Diff 解析失败: {str(e)}",
            "style_standards": "",
            "security_standards": "",
            "performance_standards": "",
            "architecture_standards": "",
        }


def synthesize_node(state: dict) -> dict:
    """综合报告生成节点

    汇总所有审查 Agent 的发现，按严重程度排序，
    生成结构化的审查报告。

    Args:
        state: 工作流状态（包含所有 findings）

    Returns:
        dict: 包含 final_report 的状态更新
    """
    # 如果解析阶段出错，直接返回错误报告
    if state.get("error"):
        return {"final_report": {
            "summary": {
                "total_files": 0,
                "total_additions": 0,
                "total_deletions": 0,
                "critical_count": 0,
                "warning_count": 0,
                "suggestion_count": 0,
                "positive_count": 0,
            },
            "findings": [],
            "error": state["error"],
        }}

    # 收集所有发现
    all_findings = (
        state.get("style_findings", [])
        + state.get("security_findings", [])
        + state.get("performance_findings", [])
        + state.get("architecture_findings", [])
    )

    # 统计变更行数
    changed_files = state.get("changed_files", [])
    total_additions = sum(f.get("additions", 0) for f in changed_files)
    total_deletions = sum(f.get("deletions", 0) for f in changed_files)

    # 按严重程度统计
    severity_counts = {"critical": 0, "warning": 0, "suggestion": 0, "positive": 0}
    for finding in all_findings:
        sev = finding.get("severity", "suggestion")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # 按严重程度排序：critical > warning > suggestion > positive
    severity_order = {"critical": 0, "warning": 1, "suggestion": 2, "positive": 3}
    sorted_findings = sorted(
        all_findings,
        key=lambda x: severity_order.get(x.get("severity", "suggestion"), 3),
    )

    # 按类别分组
    findings_by_category = {}
    for finding in sorted_findings:
        cat = finding.get("category", "unknown")
        if cat not in findings_by_category:
            findings_by_category[cat] = []
        findings_by_category[cat].append(finding)

    report = {
        "repo_path": state.get("repo_path", ""),
        "target_branch": state.get("target_branch", "HEAD~1"),
        "summary": {
            "total_files": len(changed_files),
            "total_additions": total_additions,
            "total_deletions": total_deletions,
            **severity_counts,
        },
        "findings": sorted_findings,
        "findings_by_category": findings_by_category,
    }

    logger.info(
        f"审查报告生成完成: {len(all_findings)} 个发现 "
        f"(critical={severity_counts['critical']}, warning={severity_counts['warning']}, "
        f"suggestion={severity_counts['suggestion']}, positive={severity_counts['positive']})"
    )

    return {"final_report": report}


def create_review_workflow() -> StateGraph:
    """创建代码审查工作流图

    构建 LangGraph 状态图：
    START → parse_diff → [style|security|performance|architecture] → synthesize → END

    特点：
    - Fan-out: parse_diff 同时分发到 4 个并行审查节点
    - Fan-in: 4 个审查节点全部完成后汇总到 synthesize
    - 使用 Annotated[list, operator.add] 实现并行结果自动累积

    Returns:
        CompiledGraph: 编译后的工作流图
    """
    graph = StateGraph(ReviewState)

    # 添加所有节点
    graph.add_node("parse_diff", parse_diff_node)
    graph.add_node("style_review", style_review_node)
    graph.add_node("security_review", security_review_node)
    graph.add_node("performance_review", performance_review_node)
    graph.add_node("architecture_review", architecture_review_node)
    graph.add_node("synthesize", synthesize_node)

    # 定义边：START → parse_diff
    graph.add_edge(START, "parse_diff")

    # Fan-out: parse_diff → 4 个并行审查节点
    graph.add_edge("parse_diff", "style_review")
    graph.add_edge("parse_diff", "security_review")
    graph.add_edge("parse_diff", "performance_review")
    graph.add_edge("parse_diff", "architecture_review")

    # Fan-in: 4 个审查节点 → synthesize
    graph.add_edge("style_review", "synthesize")
    graph.add_edge("security_review", "synthesize")
    graph.add_edge("performance_review", "synthesize")
    graph.add_edge("architecture_review", "synthesize")

    # synthesize → END
    graph.add_edge("synthesize", END)

    return graph.compile()


async def run_review(repo_path: str, target_branch: str = "HEAD~1", review_config: dict = None) -> dict:
    """执行代码审查（便捷入口函数）

    Args:
        repo_path: Git 仓库路径
        target_branch: 目标分支或提交引用
        review_config: 审查配置

    Returns:
        dict: 审查报告
    """
    if review_config is None:
        review_config = {
            "enable_style": True,
            "enable_security": True,
            "enable_performance": True,
            "enable_architecture": True,
        }

    workflow = create_review_workflow()

    initial_state = {
        "repo_path": repo_path,
        "target_branch": target_branch,
        "diff_content": "",
        "changed_files": [],
        "review_config": review_config,
        "style_standards": "",
        "security_standards": "",
        "performance_standards": "",
        "architecture_standards": "",
    }

    result = await workflow.ainvoke(initial_state)
    return result.get("final_report", {})
