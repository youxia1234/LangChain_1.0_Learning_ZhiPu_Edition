"""Git 操作工具模块

提供 Git 仓库操作相关的 LangChain 工具，包括：
- 获取 git diff 内容
- 解析变更文件列表
- 读取文件内容
- 获取仓库文件结构
"""

import os
import logging
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _validate_repo_path(repo_path: str) -> str:
    """验证仓库路径安全性

    防止路径遍历攻击，确保路径是有效的 Git 仓库。

    Args:
        repo_path: 待验证的路径

    Returns:
        str: 规范化后的绝对路径

    Raises:
        ValueError: 路径无效或不是 Git 仓库
    """
    # 解析为绝对路径
    abs_path = os.path.abspath(os.path.normpath(repo_path))

    # 检查路径是否存在
    if not os.path.isdir(abs_path):
        raise ValueError(f"路径不存在或不是目录: {abs_path}")

    # 检查是否是 Git 仓库
    git_dir = os.path.join(abs_path, ".git")
    if not os.path.exists(git_dir):
        raise ValueError(f"不是有效的 Git 仓库（缺少 .git 目录）: {abs_path}")

    return abs_path


def _validate_file_path(file_path: str, base_dir: str = None) -> str:
    """验证文件路径安全性

    防止路径遍历攻击，确保文件路径在允许的范围内。

    Args:
        file_path: 待验证的文件路径
        base_dir: 基准目录（如果指定，文件必须在此目录下）

    Returns:
        str: 规范化后的绝对路径

    Raises:
        ValueError: 路径不安全或文件不存在
    """
    abs_path = os.path.abspath(os.path.normpath(file_path))

    if base_dir:
        base = os.path.abspath(os.path.normpath(base_dir))
        if not abs_path.startswith(base):
            raise ValueError(f"文件路径不在允许的目录范围内: {file_path}")

    if not os.path.isfile(abs_path):
        raise ValueError(f"文件不存在: {abs_path}")

    return abs_path


@tool
def get_git_diff(repo_path: str, target_branch: str = "HEAD~1") -> str:
    """获取 Git 仓库的 diff 内容

    获取指定仓库相对于目标分支的代码变更差异。

    Args:
        repo_path: Git 仓库的本地路径
        target_branch: 比较目标，默认 HEAD~1（最近一次提交）。
                       支持分支名（main, dev）或提交引用（HEAD~3, abc123）。

    Returns:
        str: git diff 的文本内容
    """
    import git

    abs_path = _validate_repo_path(repo_path)

    try:
        repo = git.Repo(abs_path)
        # 获取 diff（包含变更统计和内容）
        diff_output = repo.git.diff(target_branch, "--stat", "--patch")
        return diff_output
    except git.exc.GitCommandError as e:
        error_msg = str(e)
        if "unknown revision or path not in the working tree" in error_msg:
            raise ValueError(f"目标分支或提交不存在: {target_branch}")
        raise ValueError(f"Git diff 执行失败: {error_msg}")


@tool
def get_changed_files(repo_path: str, target_branch: str = "HEAD~1") -> list[dict]:
    """获取 Git 仓库中变更的文件列表

    解析 diff 输出，返回每个变更文件的详细信息。

    Args:
        repo_path: Git 仓库的本地路径
        target_branch: 比较目标，默认 HEAD~1

    Returns:
        list[dict]: 变更文件列表，每项包含：
            - path: 文件路径
            - status: 变更状态 (added/modified/deleted/renamed)
            - additions: 新增行数
            - deletions: 删除行数
    """
    import git

    abs_path = _validate_repo_path(repo_path)

    try:
        repo = git.Repo(abs_path)
        diff_index = repo.commit(target_branch).diff("HEAD")

        changed_files = []
        for diff_item in diff_index:
            # 确定变更状态
            if diff_item.new_file:
                status = "added"
            elif diff_item.deleted_file:
                status = "deleted"
            elif diff_item.renamed_file:
                status = "renamed"
            else:
                status = "modified"

            # 获取文件路径
            file_path = diff_item.a_path if diff_item.a_path else diff_item.b_path

            # 解析变更行数
            diff_text = diff_item.diff.decode("utf-8", errors="replace") if diff_item.diff else ""
            additions = sum(1 for line in diff_text.split("\n") if line.startswith("+") and not line.startswith("+++"))
            deletions = sum(1 for line in diff_text.split("\n") if line.startswith("-") and not line.startswith("---"))

            changed_files.append({
                "path": file_path,
                "status": status,
                "additions": additions,
                "deletions": deletions,
            })

        return changed_files

    except git.exc.GitCommandError as e:
        logger.error(f"获取变更文件列表失败: {e}")
        raise ValueError(f"获取变更文件列表失败: {e}")


@tool
def read_file_content(file_path: str, start_line: int = None, end_line: int = None) -> str:
    """读取指定文件的内容

    读取文件的完整内容或指定行范围的内容。

    Args:
        file_path: 文件的完整路径
        start_line: 起始行号（从1开始），不指定则从第1行开始
        end_line: 结束行号（包含），不指定则到文件末尾

    Returns:
        str: 文件内容（可能带行号前缀）
    """
    abs_path = _validate_file_path(file_path)

    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # 截取指定行范围
        if start_line is not None:
            lines = lines[start_line - 1:]
        if end_line is not None:
            lines = lines[:end_line - (start_line or 1) + 1]

        # 添加行号
        start = start_line or 1
        numbered_lines = []
        for i, line in enumerate(lines):
            line_number = start + i
            numbered_lines.append(f"{line_number:4d} | {line.rstrip()}")

        return "\n".join(numbered_lines)

    except Exception as e:
        logger.error(f"读取文件失败 {file_path}: {e}")
        return f"读取文件失败: {e}"


@tool
def get_file_structure(repo_path: str, max_depth: int = 3) -> str:
    """获取仓库的文件目录结构

    返回仓库的目录树，用于了解项目整体结构。

    Args:
        repo_path: Git 仓库的本地路径
        max_depth: 最大遍历深度，默认3层

    Returns:
        str: 目录树文本表示
    """
    import git

    abs_path = _validate_repo_path(repo_path)

    try:
        repo = git.Repo(abs_path)
        # 使用 git ls-tree 获取跟踪的文件
        output = repo.git.ls_tree("HEAD", "-r", "--name-only")

        files = output.strip().split("\n") if output.strip() else []

        # 构建目录树
        tree_lines = []
        tree_lines.append(f"📁 {os.path.basename(abs_path)}/")

        # 按目录分组，限制深度
        dirs = {}
        for filepath in files:
            parts = filepath.split("/")
            if len(parts) > max_depth:
                # 超过深度的路径截断
                display_path = "/".join(parts[:max_depth]) + "/..."
            else:
                display_path = filepath
            dirs[display_path] = True

        # 排序输出
        sorted_paths = sorted(dirs.keys())
        for i, path in enumerate(sorted_paths):
            is_last = (i == len(sorted_paths) - 1)
            prefix = "└── " if is_last else "├── "
            indent = "│   "
            # 根据层级缩进
            depth = path.count("/")
            if depth > 0:
                tree_lines.append(f"{'│   ' * (depth)}{prefix}{os.path.basename(path)}")
            else:
                tree_lines.append(f"{prefix}{path}")

        return "\n".join(tree_lines)

    except Exception as e:
        logger.error(f"获取文件结构失败: {e}")
        return f"获取文件结构失败: {e}"
