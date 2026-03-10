"""
LangChain 知识库及 RAG 问答系统

本模块实现了一个基于 LangChain 1.0 的知识库和 RAG 问答系统，
专门用于帮助开发者学习和查询 LangChain/LangGraph 的使用方法。

核心功能：
- 自动解析项目中的所有学习模块（文档、代码、注释）
- 构建结构化的向量知识库
- 支持自然语言查询 LangChain 相关问题
- 提供代码示例、最佳实践和来源引用

💡 这是一个"用 LangChain 学习 LangChain"的元认知项目：
- 使用 LangChain 1.0 的 create_agent API 构建 RAG 系统
- 知识库内容就是 LangChain 本身的学习材料
- 帮助开发者快速查找和学习 LangChain 用法

⚠️ Embeddings 说明：
- 默认使用简单的 Hash Embeddings（用于演示）
- 如需高质量结果，请设置 OPENAI_API_KEY 使用 OpenAI Embeddings

📌 LangChain 1.0 正确用法：
- 使用 create_agent 而不是链式调用
- 检索功能作为 @tool 装饰的函数暴露给 agent
- 参考 phase2_practical/13_rag_basics 和 14_rag_advanced
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# LangChain 核心导入
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.documents import Document

# 文本处理
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 向量存储
from langchain_core.vectorstores import InMemoryVectorStore

# 导入知识库构建模块
from knowledge_base import build_knowledge_base_from_project

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取 API 密钥"
    )

# 初始化模型（使用智谱 AI）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


# ==================== 简单 Embeddings（用于演示）====================

class SimpleEmbeddings:
    """
    简单的 Embeddings 实现（用于演示）

    使用简单的 hash 生成向量，适合演示目的。
    生产环境请使用 OpenAI 或 HuggingFace Embeddings。
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        return self._embed_text(text)

    def _embed_text(self, text: str) -> List[float]:
        """简单的文本嵌入（基于字符频率）"""
        import hashlib
        # 使用文本的hash生成伪随机但确定的向量
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # 扩展到目标维度
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            # 归一化到 [-1, 1]
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            vector.append(value)

        return vector


# 选择 Embeddings
def get_embeddings():
    """根据环境选择合适的 Embeddings"""
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        try:
            from langchain_openai import OpenAIEmbeddings
            print("📊 使用 OpenAI Embeddings")
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except ImportError:
            print("⚠️ langchain_openai 未安装，使用简单 Embeddings")

    print("📊 使用简单 Embeddings（演示用）")
    return SimpleEmbeddings()


# ==================== 配置类 ====================

@dataclass
class RAGConfig:
    """RAG 系统配置 - 针对 LangChain 学习场景优化"""
    # 模型配置
    temperature: float = 0.1

    # 分块配置 - 针对代码和技术文档优化
    chunk_size: int = 800  # 增大以保留更多代码上下文
    chunk_overlap: int = 150  # 增加重叠以保持代码完整性

    # 检索配置
    top_k: int = 5  # 检索数量

    # 知识库配置
    project_root: str = "."  # 项目根目录


# ==================== 知识库管理器 ====================

class KnowledgeBaseManager:
    """知识库管理器：构建和管理向量知识库"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        self.embeddings = get_embeddings()
        self.vector_store = None
        self.documents = []

    def build_from_project(self, project_root: str) -> List[Document]:
        """从项目构建知识库"""
        print("\n" + "=" * 60)
        print("📚 构建知识库")
        print("=" * 60)

        # 使用 knowledge_base 模块解析项目
        self.documents = build_knowledge_base_from_project(project_root)

        print(f"\n📊 共提取 {len(self.documents)} 个文档片段")

        # 分割文档
        print("\n✂️ 分割文档...")
        chunks = self.text_splitter.split_documents(self.documents)
        print(f"   生成了 {len(chunks)} 个文本块")

        # 创建向量存储
        print("\n🔢 创建向量存储...")
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        print("   向量存储创建完成")

        return self.documents

    def search(self, query: str, k: int = None) -> List[Document]:
        """搜索知识库"""
        if not self.vector_store:
            raise ValueError("知识库未构建，请先调用 build_from_project()")

        k = k or self.config.top_k
        return self.vector_store.similarity_search(query, k=k)

    def search_with_scores(self, query: str, k: int = None) -> List[tuple]:
        """搜索知识库并返回相似度分数"""
        if not self.vector_store:
            raise ValueError("知识库未构建，请先调用 build_from_project()")

        k = k or self.config.top_k
        return self.vector_store.similarity_search_with_score(query, k=k)


# ==================== RAG Agent（使用 LangChain 1.0 API）====================

class RAGAgent:
    """
    RAG 问答 Agent

    使用 LangChain 1.0 的 create_agent API 构建，
    检索功能作为 tool 暴露给 agent。
    """

    def __init__(self, kb_manager: KnowledgeBaseManager, config: RAGConfig):
        self.kb_manager = kb_manager
        self.config = config
        self.agent = None
        self._create_agent()

    def _create_agent(self):
        """创建 RAG Agent"""
        # 创建检索工具
        @tool
        def search_knowledge_base(query: str) -> str:
            """在 LangChain 知识库中搜索相关信息

            Args:
                query: 搜索查询，可以是关于 LangChain/LangGraph 的任何问题

            Returns:
                检索到的相关文档内容
            """
            docs = self.kb_manager.search(query, k=self.config.top_k)

            # 格式化检索结果
            result_parts = []
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                module = metadata.get("module", "unknown")
                source = metadata.get("source", "unknown")
                doc_type = metadata.get("type", "unknown")

                result_parts.append(
                    f"[来源 {i}]\n"
                    f"模块: {module}\n"
                    f"文件: {source}\n"
                    f"类型: {doc_type}\n"
                    f"内容: {doc.page_content}"
                )

            return "\n\n".join(result_parts)

        @tool
        def get_module_overview(module_name: str) -> str:
            """获取特定模块的概览信息

            Args:
                module_name: 模块名称，如 "01_hello_langchain"

            Returns:
                该模块的概览信息
            """
            # 搜索特定模块的文档
            query = f"{module_name} 模块 概述 介绍"
            docs = self.kb_manager.search(query, k=10)

            # 筛选属于该模块的文档
            module_docs = [
                doc for doc in docs
                if doc.metadata.get("module") == module_name
            ]

            if not module_docs:
                return f"未找到模块 '{module_name}' 的信息"

            # 提取 markdown 类型的文档（通常是概述）
            overview_docs = [
                doc for doc in module_docs
                if doc.metadata.get("type") == "markdown"
            ]

            if overview_docs:
                return overview_docs[0].page_content[:1000]

            return module_docs[0].page_content[:1000]

        # 使用 LangChain 1.0 的 create_agent API
        self.agent = create_agent(
            model=model,
            tools=[search_knowledge_base, get_module_overview],
            system_prompt="""你是一个专业的 LangChain 和 LangGraph 学习助手。

重要：你必须始终使用 search_knowledge_base 工具来搜索知识库，即使你认为自己知道答案。

你的职责：
1. 基于知识库中的信息，准确回答关于 LangChain/LangGraph 的问题
2. 提供清晰的代码示例和最佳实践
3. 解释核心概念和 API 用法
4. 指出常见的使用陷阱和注意事项

工作流程：
1. 当用户提问时，首先使用 search_knowledge_base 工具搜索相关信息（必须执行）
2. 如果用户询问特定模块，使用 get_module_overview 获取模块概览
3. 基于检索到的信息组织回答

回答格式要求：
- 使用 Markdown 格式（代码块用 ``` 包裹）
- 重要概念用粗体标记
- 代码示例要完整可运行
- 在回答末尾标注信息来源（模块名和文件名）

重要规则：
1. 必须使用 search_knowledge_base 工具搜索相关信息
2. 只使用知识库中的信息来回答问题
3. 如果知识库中没有相关信息，请诚实地说"根据知识库中的信息，我没有找到相关内容"
4. 对于代码示例，确保包含必要的导入语句
"""
        )

    def query(self, question: str) -> Dict[str, Any]:
        """执行查询"""
        if not self.agent:
            raise ValueError("Agent 未初始化")

        print(f"\n{'='*60}")
        print(f"🔍 问题：{question}")
        print('='*60)

        # 调用 agent
        response = self.agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })

        # 获取回答
        answer = response["messages"][-1].content
        print("💬 生成回答完成")

        # 尝试获取来源信息（从工具调用中）
        sources = []
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    if "name" in call and call["name"] == "search_knowledge_base":
                        # 从工具输出中提取来源
                        sources.append({
                            "tool": "search_knowledge_base",
                            "query": call["args"].get("query", "")
                        })

        return {
            "answer": answer,
            "sources": sources,
            "all_messages": response["messages"]
        }


# ==================== 主程序 ====================

def build_knowledge_base(config: RAGConfig, kb_manager: KnowledgeBaseManager):
    """构建知识库"""
    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    print(f"\n📂 项目根目录: {project_root}")

    # 构建知识库
    return kb_manager.build_from_project(str(project_root))


def interactive_mode(agent: RAGAgent):
    """交互式问答模式"""
    print("\n" + "=" * 60)
    print("💬 进入交互式问答模式")
    print("=" * 60)
    print("\n提示：")
    print("  - 输入你的问题，系统将基于知识库回答")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'module:模块名' 查看特定模块概览")
    print("")

    while True:
        try:
            question = input("\n👤 你的问题: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break

            if question.lower() == 'clear':
                # 重新创建 agent 以清空历史
                agent._create_agent()
                print("✅ 对话历史已清空")
                continue

            # 查询
            result = agent.query(question)

            # 显示回答
            print("\n🤖 回答:")
            print("-" * 60)
            print(result['answer'])
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


def demo_mode(agent: RAGAgent):
    """演示模式 - 预设示例问题"""
    print("\n" + "=" * 60)
    print("📝 演示模式 - 示例问题")
    print("=" * 60)

    demo_questions = [
        "什么是 LangChain？它有哪些核心组件？",
        "LangGraph 中的 State 是如何定义的？",
        "如何创建一个简单的 Agent？",
        "RAG 系统的工作流程是什么？",
        "如何使用 MemorySaver 保存对话历史？"
    ]

    for i, q in enumerate(demo_questions, 1):
        print(f"\n{'=' * 60}")
        print(f"📝 示例 {i}/{len(demo_questions)}")
        print(f"👤 问题: {q}")
        print('=' * 60)

        result = agent.query(q)

        print(f"\n🤖 回答:")
        print(result['answer'][:1000] if len(result['answer']) > 1000 else result['answer'])

        if len(result['answer']) > 1000:
            print("\n... (回答已截断)")

        print(f"\n📊 工具调用: {len(result['sources'])} 次")


def main():
    """主程序"""
    print("=" * 60)
    print("🚀 LangChain 知识库及 RAG 问答系统")
    print("=" * 60)
    print("\n💡 这是一个 '用 LangChain 学习 LangChain' 的系统")
    print("   知识库来源于完整的 LangChain 1.0 学习项目")
    print("\n📌 使用 LangChain 1.0 的 create_agent API 构建")

    # 1. 初始化配置
    config = RAGConfig(
        chunk_size=800,
        chunk_overlap=150,
        top_k=5,
        project_root=str(Path(__file__).parent.parent.parent)
    )

    # 2. 初始化知识库管理器
    print("\n📦 初始化知识库管理器...")
    kb_manager = KnowledgeBaseManager(config)

    # 3. 构建知识库
    build_knowledge_base(config, kb_manager)

    # 4. 创建 RAG Agent
    print("\n🤖 创建 RAG Agent...")
    agent = RAGAgent(kb_manager, config)
    print("✅ RAG Agent 创建完成")

    # 5. 选择模式
    print("\n" + "=" * 60)
    print("请选择运行模式:")
    print("  1. 演示模式 - 运行预设示例问题")
    print("  2. 交互模式 - 自由提问")
    print("=" * 60)

    choice = input("\n请输入选择 (1/2，默认 1): ").strip()

    if choice == '2':
        interactive_mode(agent)
    else:
        demo_mode(agent)
        # 演示结束后提供交互选项
        print("\n" + "=" * 60)
        enter_interactive = input("是否进入交互模式继续提问？(y/n): ").strip().lower()
        if enter_interactive == 'y':
            interactive_mode(agent)

    print("\n" + "=" * 60)
    print("✅ 系统运行结束！")
    print("=" * 60)


if __name__ == "__main__":
    main()
