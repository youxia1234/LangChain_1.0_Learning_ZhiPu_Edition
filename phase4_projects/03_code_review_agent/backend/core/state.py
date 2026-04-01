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
        target_branch: 目标分支（用于 git diff，默认 HEAD~1）
        diff_content: git diff 原始内容
        changed_files: 变更文件信息列表
        review_config: 审查配置（启用哪些审查维度）
        style_standards: RAG 检索到的风格规范
        security_standards: RAG 检索到的安全规范
        performance_standards: RAG 检索到的性能规范
        architecture_standards: RAG 检索到的架构规范
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
    style_standards: str
    security_standards: str
    performance_standards: str
    architecture_standards: str
    style_findings: Annotated[list, operator.add]
    security_findings: Annotated[list, operator.add]
    performance_findings: Annotated[list, operator.add]
    architecture_findings: Annotated[list, operator.add]
    final_report: Optional[dict]
    error: Optional[str]
