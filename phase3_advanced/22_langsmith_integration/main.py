"""
模块 22：LangSmith 集成
学习如何追踪、监控和调试 LLM 应用
"""

import os
import sys
import time
from typing import Optional
from dotenv import load_dotenv
from functools import wraps

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取 API 密钥"
    )

# 初始化模型（使用智谱 AI）
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ============================================================
# LangSmith 配置
# ============================================================

def setup_langsmith(project_name: str = "langchain-study"):
    """配置 LangSmith 追踪"""
    
    # 检查是否有 API Key
    api_key = os.environ.get("LANGSMITH_API_KEY")
    
    if api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project_name
        print(f"✅ LangSmith 已启用 (项目: {project_name})")
        return True
    else:
        print("⚠️ 未配置 LANGSMITH_API_KEY，追踪功能未启用")
        print("   请在 .env 文件中添加：LANGSMITH_API_KEY=your_key")
        return False

# 尝试设置 LangSmith
LANGSMITH_ENABLED = setup_langsmith()

# 初始化模型

# ============================================================
# 示例 1：基本追踪
# ============================================================

def basic_tracing():
    """
    基本的 LangSmith 追踪
    启用追踪后，所有 LLM 调用自动记录
    """
    print("\n" + "=" * 60)
    print("示例 1：基本追踪")
    print("=" * 60)

    # 简单调用 - 自动追踪
    response = model.invoke("什么是 Python？用一句话回答。")
    
    print(f"响应: {response.content}")
    
    if LANGSMITH_ENABLED:
        print("\n📊 追踪数据已发送到 LangSmith")
        print("   访问 https://smith.langchain.com 查看详细信息")
    
    return response

# ============================================================
# 示例 2：带元数据的追踪
# ============================================================

def tracing_with_metadata():
    """
    添加自定义元数据到追踪
    """
    print("\n" + "=" * 60)
    print("示例 2：带元数据的追踪")
    print("=" * 60)

    # 创建带元数据的配置
    config = RunnableConfig(
        metadata={
            "user_id": "user_12345",
            "session_id": "session_67890",
            "request_type": "question",
            "app_version": "1.0.0"
        },
        tags=["study", "module_22", "demo"]
    )

    messages = [
        SystemMessage(content="你是一个友好的助手。用中文简洁回答。"),
        HumanMessage(content="LangSmith 有什么用？")
    ]

    response = model.invoke(messages, config=config)
    
    print(f"响应: {response.content}")
    print("\n添加的元数据:")
    print("  - user_id: user_12345")
    print("  - session_id: session_67890")
    print("  - tags: study, module_22, demo")
    
    return response

# ============================================================
# 示例 3：性能监控
# ============================================================

def performance_monitoring():
    """
    监控 LLM 调用的性能指标
    """
    print("\n" + "=" * 60)
    print("示例 3：性能监控")
    print("=" * 60)

    questions = [
        "1+1等于几？",
        "解释什么是机器学习，用100字以内。",
        "写一个简短的 Python 函数来计算斐波那契数列。"
    ]

    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n测试 {i}: {question[:30]}...")
        
        start_time = time.time()
        
        # 带性能元数据的调用
        config = RunnableConfig(
            metadata={
                "test_id": f"perf_test_{i}",
                "complexity": "low" if i == 1 else ("medium" if i == 2 else "high")
            },
            tags=["performance_test"]
        )
        
        response = model.invoke(question, config=config)
        
        elapsed_time = time.time() - start_time
        
        # 提取 token 使用情况（如果有）
        token_usage = getattr(response, 'usage_metadata', None)
        
        result = {
            "question": question[:30],
            "response_length": len(response.content),
            "elapsed_time": elapsed_time,
            "token_usage": token_usage
        }
        results.append(result)
        
        print(f"  响应长度: {result['response_length']} 字符")
        print(f"  耗时: {elapsed_time:.2f} 秒")
        if token_usage:
            print(f"  Token 使用: {token_usage}")

    # 汇总
    print("\n📊 性能汇总:")
    total_time = sum(r["elapsed_time"] for r in results)
    avg_time = total_time / len(results)
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  平均耗时: {avg_time:.2f} 秒")
    
    return results

# ============================================================
# 示例 4：错误追踪
# ============================================================

def error_tracking():
    """
    追踪和记录错误
    """
    print("\n" + "=" * 60)
    print("示例 4：错误追踪")
    print("=" * 60)

    def risky_operation(query: str, should_fail: bool = False):
        """模拟可能失败的操作"""
        config = RunnableConfig(
            metadata={
                "operation_type": "risky",
                "should_fail": should_fail
            },
            tags=["error_test"]
        )
        
        if should_fail:
            raise ValueError("模拟的错误：请求参数无效")
        
        return model.invoke(query, config=config)

    # 成功的调用
    print("\n测试 1: 正常调用")
    try:
        response = risky_operation("你好！")
        print(f"  成功: {response.content}")
    except Exception as e:
        print(f"  失败: {e}")

    # 失败的调用
    print("\n测试 2: 会失败的调用")
    try:
        response = risky_operation("你好！", should_fail=True)
        print(f"  成功: {response.content}")
    except Exception as e:
        print(f"  捕获错误: {e}")
        print("  💡 错误信息已记录到 LangSmith（如已启用）")

# ============================================================
# 示例 5：自定义追踪装饰器
# ============================================================

def custom_traceable(name: str = None, tags: list = None):
    """
    自定义追踪装饰器
    在没有 LangSmith 的情况下也提供本地追踪
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__
            func_tags = tags or []
            
            start_time = time.time()
            print(f"  🔍 开始追踪: {func_name}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                print(f"  ✅ 完成: {func_name} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"  ❌ 失败: {func_name} ({elapsed:.2f}s) - {e}")
                raise
        
        return wrapper
    return decorator

@custom_traceable(name="summarize_text", tags=["demo", "summarization"])
def summarize_text(text: str) -> str:
    """带追踪的文本摘要函数"""
    response = model.invoke(f"请用一句话总结：{text}")
    return response.content

def custom_decorator_demo():
    """
    演示自定义追踪装饰器
    """
    print("\n" + "=" * 60)
    print("示例 5：自定义追踪装饰器")
    print("=" * 60)

    text = "Python 是一种广泛使用的高级编程语言，以其简洁的语法和强大的功能著称。"
    
    summary = summarize_text(text)
    print(f"\n原文: {text}")
    print(f"摘要: {summary}")

# ============================================================
# 示例 6：多步骤追踪
# ============================================================

def multi_step_tracing():
    """
    追踪多步骤工作流
    """
    print("\n" + "=" * 60)
    print("示例 6：多步骤追踪")
    print("=" * 60)

    # 模拟一个多步骤的 AI 工作流
    parent_config = RunnableConfig(
        metadata={"workflow": "content_creation"},
        tags=["multi_step", "workflow"]
    )

    print("\n开始内容创作工作流...")

    # 步骤 1: 生成大纲
    print("\n  步骤 1: 生成大纲")
    step1_config = RunnableConfig(
        metadata={**parent_config.get("metadata", {}), "step": "outline"},
        tags=["step_1"]
    )
    outline = model.invoke(
        "为一篇关于'AI 的未来'的文章生成3点大纲。",
        config=step1_config
    )
    print(f"  大纲: {outline.content[:100]}...")

    # 步骤 2: 扩展第一点
    print("\n  步骤 2: 扩展内容")
    step2_config = RunnableConfig(
        metadata={**parent_config.get("metadata", {}), "step": "expand"},
        tags=["step_2"]
    )
    expanded = model.invoke(
        f"基于以下大纲，扩展第一点（50字以内）：\n{outline.content}",
        config=step2_config
    )
    print(f"  扩展内容: {expanded.content[:100]}...")

    # 步骤 3: 润色
    print("\n  步骤 3: 润色文字")
    step3_config = RunnableConfig(
        metadata={**parent_config.get("metadata", {}), "step": "polish"},
        tags=["step_3"]
    )
    polished = model.invoke(
        f"请润色以下文字，使其更专业：\n{expanded.content}",
        config=step3_config
    )
    print(f"  最终内容: {polished.content}")

    print("\n✅ 工作流完成！所有步骤已追踪记录。")
    
    return polished.content

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("LangSmith 集成教程")
    print("=" * 60)
    
    if not LANGSMITH_ENABLED:
        print("\n⚠️ 提示：配置 LANGSMITH_API_KEY 以启用完整的追踪功能")
        print("   本地演示仍可运行，但数据不会发送到 LangSmith\n")
    
    # 运行示例
    basic_tracing()
    tracing_with_metadata()
    performance_monitoring()
    error_tracking()
    custom_decorator_demo()
    multi_step_tracing()
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    if LANGSMITH_ENABLED:
        print("📊 访问 https://smith.langchain.com 查看追踪数据")
    print("=" * 60)
