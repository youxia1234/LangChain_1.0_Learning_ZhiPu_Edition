"""架构与可维护性审查 Agent

评估代码的架构设计质量，包括 SOLID 原则遵循、设计模式使用、
代码耦合度、职责划分、可测试性等。
使用 RAG 检索设计模式和架构原则。
"""

import os
import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

ARCHITECTURE_REVIEW_SYSTEM_PROMPT = """你是一位资深软件架构师，专注于代码设计和可维护性审查。

## 你的职责
评估代码的架构设计质量，重点关注：

1. **SOLID 原则**
   - **单一职责 (SRP)**：一个类/函数是否只做一件事？是否需要拆分？
   - **开闭原则 (OCP)**：代码是否易于扩展而无需修改？
   - **里氏替换 (LSP)**：子类是否能正确替换父类？
   - **接口隔离 (ISP)**：接口是否精简？是否强迫实现不需要的方法？
   - **依赖倒置 (DIP)**：是否依赖抽象而非具体实现？

2. **代码耦合度**
   - 模块之间是否高度耦合？
   - 是否存在循环依赖？
   - 是否有不必要的全局状态？
   - 修改一处是否需要修改多处？

3. **设计模式**
   - 是否可以使用策略模式替代大量 if-else？
   - 是否可以使用工厂模式管理对象创建？
   - 是否可以使用观察者模式解耦事件处理？
   - 是否过度使用设计模式（anti-pattern）？

4. **错误处理**
   - 是否使用自定义异常层级？
   - 是否吞掉异常（bare except）？
   - 错误处理是否一致？
   - 是否有统一的错误处理策略？

5. **可测试性**
   - 代码是否易于单元测试？
   - 依赖是否可通过参数注入？
   - 是否有硬编码的依赖（时间、文件路径、外部API）？
   - 函数是否是纯函数或接近纯函数？

6. **代码组织和模块化**
   - 文件和目录结构是否清晰？
   - 公共接口和内部实现是否分离？
   - 配置是否外部化（环境变量、配置文件）？
   - 是否有清晰的分层（controller/service/repository）？

## 输出格式要求
严格以 JSON 格式输出：
```json
{{
    "findings": [
        {{
            "severity": "suggestion",
            "title": "函数职责过多",
            "description": "process_order() 函数同时处理验证、计算、发送通知，违反单一职责原则",
            "file_path": "app/orders.py",
            "line_start": 100,
            "line_end": 150,
            "code_snippet": "def process_order(order):\\n    validate(order)\\n    calculate_total(order)\\n    send_notification(order)",
            "suggestion": "拆分为独立函数：validate_order(), calculate_total(), notify_customer()，由上层编排调用",
            "reference": "SOLID - 单一职责原则 (SRP)"
        }}
    ]
}}
```

severity 取值：warning（明显违反原则）、suggestion（改进建议）、positive（设计优秀）。
架构问题通常用 suggestion，严重的耦合和违反 SRP 用 warning。"""


def get_llm():
    """获取 LLM 实例"""
    return ChatOpenAI(
        model="glm-4-flash",
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        temperature=0.1,
        max_tokens=4096,
    )


async def architecture_review_node(state: dict) -> dict:
    """架构审查节点函数（供 LangGraph 调用）

    分析 diff 中的代码变更，评估架构和可维护性。

    Args:
        state: 工作流状态

    Returns:
        更新后的状态（包含 architecture_findings）
    """
    if not state.get("review_config", {}).get("enable_architecture", True):
        return {"architecture_findings": []}

    diff_content = state.get("diff_content", "")
    changed_files = state.get("changed_files", [])
    standards = state.get("architecture_standards", "")

    if not diff_content:
        return {"architecture_findings": []}

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", ARCHITECTURE_REVIEW_SYSTEM_PROMPT),
        ("human", """请对以下代码变更进行架构和可维护性审查。

## 变更文件列表
{changed_files}

## Git Diff 内容
```diff
{diff_content}
```

## 相关架构设计原则
{standards}

请仔细分析代码的设计质量，评估是否遵循良好的架构原则。
输出 JSON 格式的审查结果。"""),
    ])

    chain = prompt | llm

    try:
        response = await chain.ainvoke({
            "diff_content": diff_content[:8000],
            "changed_files": json.dumps(changed_files, ensure_ascii=False, indent=2),
            "standards": standards[:3000] if standards else "使用 SOLID 原则和通用设计模式",
        })

        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        findings = result.get("findings", [])

        for f in findings:
            f["category"] = "architecture"

        return {"architecture_findings": findings}

    except json.JSONDecodeError as e:
        logger.warning(f"架构审查结果 JSON 解析失败: {e}")
        return {"architecture_findings": []}
    except Exception as e:
        logger.error(f"架构审查出错: {e}")
        return {"architecture_findings": []}
