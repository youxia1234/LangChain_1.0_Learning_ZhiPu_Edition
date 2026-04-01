"""安全漏洞审查 Agent

检测代码中的安全漏洞，包括 SQL 注入、XSS、命令注入、硬编码密钥、
不安全的反序列化、路径遍历等安全问题。
使用 RAG 检索 OWASP 安全规则。
"""

import os
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

SECURITY_REVIEW_SYSTEM_PROMPT = """你是一位专业的应用安全工程师，专注于代码安全审计。

## 你的职责
检测代码中的安全漏洞和风险，重点关注：

1. **SQL 注入** (OWASP A03:2021)
   - 是否使用 f-string/format 拼接 SQL 语句
   - 是否使用参数化查询
   - ORM 使用是否安全

2. **命令注入** (OWASP A03:2021)
   - 是否使用 os.system() 执行外部命令
   - 是否使用 subprocess 且 shell=True
   - 用户输入是否直接拼接到命令中

3. **硬编码密钥和凭证** (OWASP A07:2021)
   - 代码中是否包含 API key、密码、token
   - 是否使用环境变量管理敏感信息
   - 配置文件中是否有明文凭证

4. **不安全的反序列化** (OWASP A08:2021)
   - 是否使用 pickle.loads() 处理不受信任的数据
   - 是否使用 yaml.load() 而非 yaml.safe_load()
   - 是否直接 eval() 用户输入

5. **路径遍历** (OWASP A01:2021)
   - 文件操作是否验证用户提供的路径
   - 是否使用 os.path.normpath 消除路径遍历
   - 文件路径是否限制在允许的目录内

6. **敏感信息泄露**
   - 错误信息是否暴露内部实现细节
   - 日志中是否记录敏感数据
   - 异常处理是否安全

7. **不安全的加密和随机数**
   - 是否使用 random 而非 secrets 生成安全令牌
   - 是否使用已弃用的加密算法（MD5, SHA1）
   - 是否使用 ECB 加密模式

## 输出格式要求
严格以 JSON 格式输出：
```json
{{
    "findings": [
        {{
            "severity": "critical",
            "title": "SQL 注入漏洞",
            "description": "使用 f-string 拼接 SQL 查询，攻击者可注入恶意 SQL",
            "file_path": "app/database.py",
            "line_start": 45,
            "line_end": 47,
            "code_snippet": "query = f\\"SELECT * FROM users WHERE id = {{user_id}}\\"",
            "suggestion": "使用参数化查询: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
            "reference": "OWASP A03:2021 - Injection"
        }}
    ]
}}
```

severity 取值：critical（必须修复）、warning（建议修复）、positive（安全实践好）。
安全问题的 severity 应偏严格。对于确定的漏洞，使用 critical。"""


def get_llm():
    """获取 LLM 实例"""
    return ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.1,
        max_tokens=4096,
    )


async def security_review_node(state: dict) -> dict:
    """安全审查节点函数（供 LangGraph 调用）

    分析 diff 中的代码变更，检测安全漏洞。

    Args:
        state: 工作流状态

    Returns:
        更新后的状态（包含 security_findings）
    """
    if not state.get("review_config", {}).get("enable_security", True):
        return {"security_findings": []}

    diff_content = state.get("diff_content", "")
    changed_files = state.get("changed_files", [])
    standards = state.get("security_standards", "")

    if not diff_content:
        return {"security_findings": []}

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SECURITY_REVIEW_SYSTEM_PROMPT),
        ("human", """请对以下代码变更进行安全审计。

## 变更文件列表
{changed_files}

## Git Diff 内容
```diff
{diff_content}
```

## 相关安全规范
{standards}

请仔细检查每个变更是否存在安全漏洞。对确定的漏洞标记为 critical。
输出 JSON 格式的审查结果。"""),
    ])

    chain = prompt | llm

    try:
        response = await chain.ainvoke({
            "diff_content": diff_content[:8000],
            "changed_files": json.dumps(changed_files, ensure_ascii=False, indent=2),
            "standards": standards[:3000] if standards else "使用 OWASP Top 10 规范",
        })

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        findings = result.get("findings", [])

        for f in findings:
            f["category"] = "security"

        return {"security_findings": findings}

    except json.JSONDecodeError as e:
        logger.warning(f"安全审查结果 JSON 解析失败: {e}")
        return {"security_findings": []}
    except Exception as e:
        logger.error(f"安全审查出错: {e}")
        return {"security_findings": []}
