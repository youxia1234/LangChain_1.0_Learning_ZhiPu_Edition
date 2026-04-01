"""性能优化审查 Agent

识别代码中的性能瓶颈和反模式，包括 N+1 查询、内存泄漏、
低效算法、阻塞 IO 等性能问题。
使用 RAG 检索性能优化最佳实践。
"""

import os
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

PERFORMANCE_REVIEW_SYSTEM_PROMPT = """你是一位 Python 性能优化专家，擅长识别代码中的性能瓶颈。

## 你的职责
审查代码的性能质量，重点关注：

1. **数据库查询效率**
   - N+1 查询问题（循环中执行数据库查询）
   - 缺少数据库索引
   - 一次性加载过多数据（应分页）
   - 未使用 select_related/prefetch_related（Django）
   - 未使用 eager loading（SQLAlchemy）

2. **循环和算法效率**
   - 嵌套循环导致 O(n²) 或更高复杂度
   - 在循环中进行不必要的重复计算
   - 可以用集合(set)查找替代列表(list)线性查找
   - 可以用列表推导替代 for 循环 append
   - 可以用生成器替代大列表（节省内存）

3. **字符串操作**
   - 循环中使用 += 拼接字符串（应使用 join 或 f-string）
   - 频繁的正则表达式编译（应预编译）
   - 不必要的字符串格式化

4. **内存使用**
   - 一次性加载大文件到内存（应流式处理）
   - 全局缓存无上限增长（应设置 LRU 缓存）
   - 循环引用导致内存泄漏
   - 不必要的数据拷贝（list.copy() 等）

5. **并发和 IO**
   - 同步阻塞 IO（应考虑异步或线程池）
   - 串行执行可并行的独立任务
   - 频繁的文件 IO（应批量处理）

6. **数据结构选择**
   - 使用 list 而非 set 进行成员检查
   - 使用 dict 而非 namedtuple/dataclass 存储简单结构
   - 频繁的排序操作（应维护有序结构）

## 输出格式要求
严格以 JSON 格式输出：
```json
{{
    "findings": [
        {{
            "severity": "warning",
            "title": "N+1 数据库查询",
            "description": "在循环中逐条查询数据库，导致大量数据库请求",
            "file_path": "app/services.py",
            "line_start": 55,
            "line_end": 60,
            "code_snippet": "for user_id in user_ids:\\n    user = db.query(User).get(user_id)",
            "suggestion": "使用批量查询: users = db.query(User).filter(User.id.in_(user_ids)).all()",
            "reference": "性能反模式 - N+1 查询"
        }}
    ]
}}
```

severity 取值：warning（明显性能问题）、suggestion（优化建议）、positive（性能实践好）。
只有可量化的性能问题才标记为 warning，一般优化建议用 suggestion。"""


def get_llm():
    """获取 LLM 实例"""
    return ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.1,
        max_tokens=4096,
    )


async def performance_review_node(state: dict) -> dict:
    """性能审查节点函数（供 LangGraph 调用）

    分析 diff 中的代码变更，识别性能瓶颈。

    Args:
        state: 工作流状态

    Returns:
        更新后的状态（包含 performance_findings）
    """
    if not state.get("review_config", {}).get("enable_performance", True):
        return {"performance_findings": []}

    diff_content = state.get("diff_content", "")
    changed_files = state.get("changed_files", [])
    standards = state.get("performance_standards", "")

    if not diff_content:
        return {"performance_findings": []}

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", PERFORMANCE_REVIEW_SYSTEM_PROMPT),
        ("human", """请对以下代码变更进行性能审查。

## 变更文件列表
{changed_files}

## Git Diff 内容
```diff
{diff_content}
```

## 相关性能优化规范
{standards}

请仔细分析每个变更的性能影响，识别潜在的性能瓶颈。
输出 JSON 格式的审查结果。"""),
    ])

    chain = prompt | llm

    try:
        response = await chain.ainvoke({
            "diff_content": diff_content[:8000],
            "changed_files": json.dumps(changed_files, ensure_ascii=False, indent=2),
            "standards": standards[:3000] if standards else "使用通用 Python 性能优化规范",
        })

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        findings = result.get("findings", [])

        for f in findings:
            f["category"] = "performance"

        return {"performance_findings": findings}

    except json.JSONDecodeError as e:
        logger.warning(f"性能审查结果 JSON 解析失败: {e}")
        return {"performance_findings": []}
    except Exception as e:
        logger.error(f"性能审查出错: {e}")
        return {"performance_findings": []}
