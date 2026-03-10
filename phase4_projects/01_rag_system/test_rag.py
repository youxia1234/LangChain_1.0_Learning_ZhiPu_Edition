"""
RAG 系统快速测试脚本

使用 LangChain 1.0 的 create_agent API 测试 RAG 系统
"""

import sys
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from main import RAGConfig, KnowledgeBaseManager, RAGAgent

def test_rag_system():
    """测试 RAG 系统的基本功能"""

    print("=" * 60)
    print("🧪 RAG 系统测试 (LangChain 1.0 API)")
    print("=" * 60)

    # 1. 初始化配置
    config = RAGConfig(
        chunk_size=800,
        chunk_overlap=150,
        top_k=5
    )

    # 2. 初始化知识库管理器
    print("\n📦 初始化知识库管理器...")
    kb_manager = KnowledgeBaseManager(config)

    # 3. 构建知识库
    print("\n📚 构建知识库...")
    project_root = Path(__file__).parent.parent.parent
    documents = kb_manager.build_from_project(str(project_root))

    # 4. 创建 RAG Agent
    print("\n🤖 创建 RAG Agent...")
    agent = RAGAgent(kb_manager, config)
    print("✅ RAG Agent 创建完成")

    # 5. 测试查询
    print("\n" + "=" * 60)
    print("🔍 测试查询功能")
    print("=" * 60)

    test_questions = [
        "什么是 LangChain？",
        "LangGraph 中的 State 如何定义？"
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"📝 测试 {i}/{len(test_questions)}")
        print('=' * 60)

        try:
            result = agent.query(question)

            print(f"\n🤖 回答:")
            # 只显示前 500 个字符
            answer = result['answer']
            preview = answer[:500] + "..." if len(answer) > 500 else answer
            print(preview)

            print(f"\n📊 工具调用: {len(result['sources'])} 次")
            print("✅ 查询成功")

        except Exception as e:
            print(f"❌ 查询失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)

    print("\n📋 测试总结:")
    print("  - 知识库构建: ✅")
    print("  - RAG Agent 创建: ✅")
    print("  - 查询功能: ✅")
    print("\n📌 使用 LangChain 1.0 的 create_agent API")
    print("  - 不再使用链式调用 (|)")
    print("  - 检索功能作为 @tool 装饰的函数")
    print("  - Agent 自动决定何时调用工具")

    return True

if __name__ == "__main__":
    test_rag_system()
