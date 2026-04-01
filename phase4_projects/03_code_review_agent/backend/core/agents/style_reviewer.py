"""代码风格审查 Agent

检查代码的命名规范、格式化、注释质量、导入排序等风格问题。
使用 RAG 检索 PEP 8 / Google 风格指南等规范标准。
"""

import os
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

STYLE_REVIEW_SYSTEM_PROMPT = """你是一位资深代码风格审查专家，专注于 Python 代码质量。

## 你的职责
审查代码的风格质量，重点关注：

1. **命名规范**
   - 变量名和函数名：应使用 snake_case（如 `user_name`）
   - 类名：应使用 PascalCase（如 `UserController`）
   - 常量：应使用 UPPER_SNAKE_CASE（如 `MAX_RETRY_COUNT`）
   - 私有成员：单下划线前缀（如 `_internal_state`）

2. **代码格式**
   - 缩进：4 个空格（不用 Tab）
   - 行宽：建议不超过 120 字符
   - 空行：类之间 2 行，方法之间 1 行
   - 括号：合理使用，避免不必要的括号

3. **注释和文档**
   - 公共函数应有 docstring
   - docstring 应使用 Google 风格
   - 复杂逻辑应有行内注释
   - 注释应解释"为什么"，而非"做什么"

4. **导入管理**
   - 导入顺序：标准库 → 第三方库 → 本地模块
   - 避免使用 `from module import *`
   - 未使用的导入应清理

5. **代码一致性**
   - 同一项目中风格应统一
   - 字符串引号使用一致（单引号或双引号）

## 输出格式要求
严格以 JSON 格式输出，包含 findings 数组：
```json
{{
    "findings": [
        {{
            "severity": "warning",
            "title": "函数命名不符合规范",
            "description": "函数名 'getUserInfo' 使用了 camelCase，应使用 snake_case",
            "file_path": "app/api.py",
            "line_start": 42,
            "line_end": 45,
            "code_snippet": "def getUserInfo(user_id):",
            "suggestion": "将函数名改为 get_user_info",
            "reference": "PEP 8 - 函数和变量命名"
        }}
    ]
}}
```

severity 取值：warning（需要修改）、suggestion（建议改进）、positive（做得好）。
每个发现必须包含具体的文件路径和行号。"""


def get_llm():
    """获取 LLM 实例"""
    return ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.1,
        max_tokens=4096,
    )


async def style_review_node(state: dict) -> dict:
    """风格审查节点函数（供 LangGraph 调用）

    分析 diff 中的代码变更，检查风格质量问题。

    Args:
        state: 工作流状态

    Returns:
        更新后的状态（包含 style_findings）
    """
    # 检查是否启用风格审查
    if not state.get("review_config", {}).get("enable_style", True):
        return {"style_findings": []}

    diff_content = state.get("diff_content", "")
    changed_files = state.get("changed_files", [])
    standards = state.get("style_standards", "")

    if not diff_content:
        return {"style_findings": []}

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", STYLE_REVIEW_SYSTEM_PROMPT),
        ("human", """请审查以下代码变更的风格质量。

## 变更文件列表
{changed_files}

## Git Diff 内容
```diff
{diff_content}
```

## 相关编码规范
{standards}

请仔细分析每个变更文件的代码风格，输出 JSON 格式的审查结果。只报告确定的问题。"""),
    ])

    chain = prompt | llm

    try:
        response = await chain.ainvoke({
            "diff_content": diff_content[:8000],
            "changed_files": json.dumps(changed_files, ensure_ascii=False, indent=2),
            "standards": standards[:3000] if standards else "使用通用 PEP 8 规范",
        })

        # 解析 JSON 响应
        content = response.content
        # 尝试提取 JSON（处理可能的 markdown 代码块包裹）
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        findings = result.get("findings", [])

        # 为每个发现添加 category 标记
        for f in findings:
            f["category"] = "style"

        return {"style_findings": findings}

    except json.JSONDecodeError as e:
        logger.warning(f"风格审查结果 JSON 解析失败: {e}")
        return {"style_findings": []}
    except Exception as e:
        logger.error(f"风格审查出错: {e}")
        return {"style_findings": []}
