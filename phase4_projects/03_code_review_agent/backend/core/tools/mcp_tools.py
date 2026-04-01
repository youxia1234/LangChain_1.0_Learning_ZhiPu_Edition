"""MCP 工具集成模块

通过 langchain-mcp-adapters 连接 MCP filesystem server，
为审查 Agent 提供标准化的文件读取能力。
当 MCP 不可用时，自动回退到直接的 Python 文件读取工具。
"""

import os
import logging
from typing import Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


async def get_mcp_filesystem_tools(repo_path: str) -> list[BaseTool]:
    """获取 MCP filesystem 工具

    尝试通过 langchain-mcp-adapters 连接 filesystem MCP server。
    如果连接失败（如 Node.js 未安装），返回空列表。

    Args:
        repo_path: 需要访问的仓库根路径

    Returns:
        list[BaseTool]: MCP 工具列表（可能为空）
    """
    enabled = os.getenv("MCP_FILESYSTEM_ENABLED", "true").lower() == "true"
    if not enabled:
        logger.info("MCP filesystem 已通过环境变量禁用")
        return []

    if not os.path.isdir(repo_path):
        logger.warning(f"仓库路径无效，跳过 MCP 连接: {repo_path}")
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        async with MultiServerMCPClient(
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
        ) as client:
            tools = client.get_tools()
            logger.info(f"MCP filesystem 工具加载成功，共 {len(tools)} 个工具")
            for tool in tools:
                logger.debug(f"  MCP 工具: {tool.name} - {tool.description[:80]}")
            return tools

    except ImportError:
        logger.warning("langchain-mcp-adapters 未安装，跳过 MCP 集成")
        return []
    except Exception as e:
        logger.warning(f"MCP filesystem 加载失败: {e}")
        logger.info("将使用直接文件读取工具作为替代")
        return []


def get_fallback_tools() -> list[BaseTool]:
    """获取回退工具（直接文件读取）

    当 MCP 不可用时，提供等价的文件操作工具。
    这些工具直接使用 Python 标准库读取文件。

    Returns:
        list[BaseTool]: 直接文件操作工具列表
    """
    from backend.core.tools.git_tools import read_file_content, get_file_structure

    return [read_file_content, get_file_structure]


def get_tools_for_agent(repo_path: Optional[str] = None) -> list[BaseTool]:
    """获取 Agent 可用的文件操作工具（同步版本）

    优先尝试 MCP 工具，失败时使用回退工具。
    此方法提供同步接口，适用于不需要异步的场景。

    Args:
        repo_path: 仓库路径（用于 MCP 连接）

    Returns:
        list[BaseTool]: 可用工具列表
    """
    # 对于同步场景，直接使用回退工具
    # MCP 连接是异步的，在 LangGraph 节点中通过工具调用间接使用
    return get_fallback_tools()
