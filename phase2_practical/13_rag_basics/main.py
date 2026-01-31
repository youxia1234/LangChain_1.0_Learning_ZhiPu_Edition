"""
LangChain 1.0 - RAG Basics (RAG 基础)
=====================================

本模块重点讲解：
1. 文档加载 (Document Loaders)
2. 文本分割 (Text Splitters)
3. 向量嵌入 (Embeddings)
4. 向量存储 (Vector Stores) - Pinecone 免费版
5. 检索 (Retrieval)
6. RAG 问答链

注意：
- 本模块使用智谱 AI (glm-4-flash)
- 示例 1-3 不需要 Pinecone API
- 示例 4-6 需要 Pinecone API（可选）
- 支持 HF Mirror 解决国内连接问题
"""

# ==================== 配置 HF Mirror（必须在所有导入之前！） ====================
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from pinecone import Pinecone, ServerlessSpec
import time

# 获取脚本所在目录
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

# 确保 data 目录存在
DATA_DIR.mkdir(exist_ok=True)

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取密钥"
    )

# 初始化模型（使用智谱 AI）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# ==================== Embeddings 初始化函数 ====================
def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    获取 Embeddings 模型（支持国内环境）

    Args:
        model_name: 模型名称，默认使用 sentence-transformers/all-MiniLM-L6-v2

    Returns:
        HuggingFaceEmbeddings 实例
    """
    try:
        print(f"\n正在加载 Embeddings 模型: {model_name}")
        print("使用 HF Mirror 加速 (https://hf-mirror.com)")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True},  # 归一化提升相似度计算
        )
        print("[OK] Embeddings 模型加载成功")
        return embeddings

    except Exception as e:
        print(f"\n[ERROR] HuggingFace 加载失败: {e}")
        print("\n尝试以下解决方案：")
        print("方案1: 设置环境变量")
        print("  set HF_ENDPOINT=https://hf-mirror.com")
        print("\n方案2: 手动下载模型")
        print("  访问 https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2")
        print("  下载到本地，然后使用 model_name='本地路径'")
        print("\n方案3: 使用其他 Embeddings 服务")
        print("  - Jina AI (免费额度): https://jina.ai/embeddings")
        print("  - 智谱 AI Embeddings API")
        raise


if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
    print("\n[警告] 未设置 PINECONE_API_KEY")
    print("如需运行 Pinecone 相关示例，请：")
    print("1. 访问 https://www.pinecone.io/ 注册免费账号")
    print("2. 获取 API Key")
    print("3. 在 .env 文件中设置 PINECONE_API_KEY=你的key")
    print("\n当前将跳过需要 Pinecone 的示例（示例 4-6）")


# ============================================================================
# 示例 1：文档加载 - Document Loaders
# ============================================================================
def example_1_document_loaders():
    """
    示例1：文档加载

    Document Loaders 将各种数据源转换为 LangChain Document 对象
    """
    print("\n" + "="*70)
    print("示例 1：文档加载 - Document Loaders")
    print("="*70)

    # 创建示例文本文件
    sample_text = """LangChain 是一个用于构建 LLM 应用的框架。

它提供了以下核心组件：
1. Models - 语言模型接口
2. Prompts - 提示词模板
3. Chains - 链式调用
4. Agents - 智能代理
5. Memory - 记忆管理

LangChain 1.0 引入了重大改进，包括：
- 更简洁的 API
- 更好的性能
- 内置的 LangGraph 支持
- 强大的中间件系统

RAG (Retrieval-Augmented Generation) 是 LangChain 的核心应用场景之一。
它结合了检索和生成，让 LLM 能够访问外部知识库。"""

    # 保存到文件
    doc_path = DATA_DIR / "langchain_intro.txt"
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    print(f"\n[OK] 创建示例文档: {doc_path}")

    # 使用 TextLoader 加载
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()

    print(f"\n加载结果:")
    print(f"  文档数量: {len(documents)}")
    print(f"  第一个文档:")
    print(f"    内容长度: {len(documents[0].page_content)} 字符")
    print(f"    元数据: {documents[0].metadata}")
    print(f"    内容预览: {documents[0].page_content[:100]}...")

    print("\n关键点:")
    print("  - TextLoader 加载文本文件")
    print("  - 返回 Document 对象列表")
    print("  - Document 包含 page_content 和 metadata")
    print("\n其他常用 Loaders:")
    print("  - PyPDFLoader - 加载 PDF")
    print("  - WebBaseLoader - 爬取网页")
    print("  - CSVLoader - 加载 CSV")

    return documents

# ============================================================================
# 示例 2：文本分割 - Text Splitters
# ============================================================================
def example_2_text_splitters(documents):
    """
    示例2：文本分割

    将长文档分割成小块，便于嵌入和检索
    """
    print("\n" + "="*70)
    print("示例 2：文本分割 - Text Splitters")
    print("="*70)

    # 创建分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,        # 每块最大字符数
        chunk_overlap=50,      # 块之间的重叠字符数
        length_function=len,   # 计算长度的函数
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]  # 分割优先级
    )

    print("\n配置:")
    print(f"  chunk_size: 200 字符")
    print(f"  chunk_overlap: 50 字符（防止信息被截断）")
    print(f"  分割优先级: 段落 -> 行 -> 句子 -> 空格 -> 字符")

    # 分割文档
    chunks = splitter.split_documents(documents)
    print(f"\n分割结果:")
    print(f"  原文档数: {len(documents)}")
    print(f"  分割后: {len(chunks)} 块")

    print(f"\n前 3 块内容:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n  块 {i}:")
        print(f"    长度: {len(chunk.page_content)} 字符")
        print(f"    内容: {chunk.page_content[:80]}...")

    print("\n关键点:")
    print("  - chunk_size 控制块大小")
    print("  - chunk_overlap 防止信息被截断")
    print("  - separators 定义分割优先级")
    print("  - RecursiveCharacterTextSplitter 智能分割")

    return chunks

# ============================================================================
# 示例 3：向量嵌入 - Embeddings
# ============================================================================
def example_3_embeddings():
    """
    示例3：向量嵌入

    将文本转换为向量，用于相似度计算
    """
    print("\n" + "="*70)
    print("示例 3：向量嵌入 - Embeddings")
    print("="*70)

    print("\n使用 HuggingFace 免费模型:")
    print("  模型: sentence-transformers/all-MiniLM-L6-v2")
    print("  维度: 384")
    print("  特点: 小巧、快速、免费")

    # 创建嵌入模型（使用封装函数）
    embeddings = get_embeddings()

    # 嵌入单个文本
    text = "LangChain 是一个 LLM 应用框架"
    vector = embeddings.embed_query(text)

    print(f"\n嵌入示例:")
    print(f"  文本: {text}")
    print(f"  向量维度: {len(vector)}")
    print(f"  向量类型: {type(vector)}")
    print(f"  向量前 5 个值: {vector[:5]}")

    # 嵌入多个文本
    texts = [
        "LangChain 是一个框架",
        "Python 是一种编程语言",
        "LangChain 用于构建 LLM 应用"
    ]
    vectors = embeddings.embed_documents(texts)

    print(f"\n批量嵌入:")
    print(f"  文本数: {len(texts)}")
    print(f"  向量数: {len(vectors)}")
    print(f"  每个向量维度: {len(vectors[0])}")

    # 计算相似度（简单示例）
    import numpy as np

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    sim_01 = cosine_similarity(vectors[0], vectors[1])
    sim_02 = cosine_similarity(vectors[0], vectors[2])

    print(f"\n相似度计算:")
    print(f"  '{texts[0]}' vs '{texts[1]}': {sim_01:.4f}")
    print(f"  '{texts[0]}' vs '{texts[2]}': {sim_02:.4f}")
    print(f"  -> 相同主题的文本相似度更高")

    print("\n关键点:")
    print("  - embed_query() - 嵌入单个查询")
    print("  - embed_documents() - 批量嵌入文档")
    print("  - 使用免费的 HuggingFace 模型（无需 API key）")
    print("  - 向量可用于相似度搜索")

    return embeddings

# ============================================================================
# 示例 4：Pinecone 向量存储 - 创建索引
# ============================================================================
def example_4_pinecone_setup():
    """
    示例4：Pinecone 设置

    创建 Pinecone serverless 索引（免费层级）
    """
    print("\n" + "="*70)
    print("示例 4：Pinecone 向量存储 - 创建索引")
    print("="*70)

    if not PINECONE_API_KEY or PINECONE_API_KEY == "your_pinecone_api_key_here":
        print("\n[警告] 跳过：需要设置 PINECONE_API_KEY")
        return None, None

    # 初始化 Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 索引配置
    index_name = "langchain-rag-demo"
    dimension = 384  # 与 all-MiniLM-L6-v2 模型维度匹配

    print(f"\n索引配置:")
    print(f"  名称: {index_name}")
    print(f"  维度: {dimension}")
    print(f"  类型: Serverless (免费层级)")
    print(f"  区域: us-east-1 (AWS)")

    # 检查索引是否存在
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name in existing_indexes:
        print(f"\n[OK] 索引已存在，直接使用")
        index = pc.Index(index_name)#获取 Pinecone 索引的句柄，之后通过句柄进行操作
    else:
        print(f"\n创建新索引...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",  # 相似度度量
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # 免费层级可用区域
            )
        )

        # 等待索引就绪
        print("等待索引初始化...")
        time.sleep(10)
        index = pc.Index(index_name)
        print("[OK] 索引创建完成")

    # 获取索引统计
    stats = index.describe_index_stats()
    print(f"\n索引统计:")
    print(f"  向量数: {stats.get('total_vector_count', 0)}")
    print(f"  维度: {stats.get('dimension', 'N/A')}")

    print("\n关键点:")
    print("  - Pinecone 提供免费 serverless 层级")
    print("  - dimension 必须与 embedding 模型匹配")
    print("  - metric='cosine' 用于相似度计算")
    print("  - ServerlessSpec 配置云和区域")

    # 创建 embeddings（使用封装函数）
    embeddings = get_embeddings()

    return index_name, embeddings

#示例3创建返回的embeddings被示例4创建返回的覆盖了，传入示例5的是示例4所创建的embeddings

# ============================================================================
# 示例 5：文档索引 - 存入向量数据库
# ============================================================================
def example_5_index_documents(index_name, embeddings, chunks):
    """
    示例5：文档索引

    将分割后的文档存入 Pinecone
    """
    print("\n" + "="*70)
    print("示例 5：文档索引 - 存入向量数据库")
    print("="*70)

    if not index_name or not embeddings:
        print("\n[警告] 跳过：需要 Pinecone 配置")
        return None

    print(f"\n准备索引 {len(chunks)} 个文档块...")

    # 使用 PineconeVectorStore.from_documents()
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    print(f"[OK] 文档已索引到 Pinecone")

    # 测试检索
    query = "LangChain 的核心组件是什么？"
    print(f"\n测试检索:")
    print(f"  查询: {query}")

    results = vectorstore.similarity_search(query, k=2)

    print(f"  返回 {len(results)} 个最相关的文档块:\n")
    for i, doc in enumerate(results, 1):
        print(f"  结果 {i}:")
        print(f"    内容: {doc.page_content[:100]}...")
        print()

    print("关键点:")
    print("  - from_documents() 自动嵌入并存储")
    print("  - similarity_search() 检索相似文档")
    print("  - k=2 返回最相关的 2 个结果")

    return vectorstore

# ============================================================================
# 示例 6：RAG 问答 - 使用 Agent
# ============================================================================
def example_6_rag_qa(vectorstore):
    """
    示例6：RAG 问答

    使用 LangChain 1.0 的 create_agent 创建 RAG 问答系统
    """
    print("\n" + "="*70)
    print("示例 6：RAG 问答 - 使用 Agent")
    print("="*70)

    if not vectorstore:
        print("\n[警告] 跳过：需要 Pinecone vectorstore")
        return

    # 创建检索工具
    @tool
    def search_knowledge_base(query: str) -> str:
        """在知识库中搜索相关信息，返回相关文档内容"""
        docs = vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

    # 使用 LangChain 1.0 的 create_agent API
    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt="""你是一个知识助手，可以访问知识库。

当用户提问时：
1. 使用 search_knowledge_base 工具搜索相关信息
2. 基于搜索到的内容回答问题
3. 如果知识库中没有相关信息，诚实告知用户"""
    )

    # 测试问答
    questions = [
        "LangChain 有哪些核心组件？",
        "RAG 是什么？",
        "LangChain 1.0 有什么改进？"
    ]

    for question in questions:
        print(f"\n问题: {question}")
        try:
            response = agent.invoke({
                "messages": [{"role": "user", "content": question}]
            })
            # 获取最后一条消息（AI 的回答）
            answer = response["messages"][-1].content
            print(f"回答: {answer[:200]}...")
        except Exception as e:
            print(f"[错误] 查询失败: {str(e)[:100]}...")
        print("-" * 70)

    print("\n关键点:")
    print("  - create_agent 是 LangChain 1.0 的核心 API")
    print("  - from langchain.agents import create_agent")
    print("  - Agent 自动决定是否调用工具")
    print("  - 基于检索结果生成答案")
    print("  - 这就是 RAG (检索增强生成)")

# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - RAG Basics (RAG 基础)")
    print("="*70)

    try:
        # 1. 文档加载
        documents = example_1_document_loaders()

        # 2. 文本分割
        chunks = example_2_text_splitters(documents)

        # 3. 向量嵌入
        embeddings = example_3_embeddings()

        # 4. Pinecone 设置（需要 API）
        index_name, embeddings = example_4_pinecone_setup()

        # 5. 文档索引（需要 API）
        vectorstore = example_5_index_documents(index_name, embeddings, chunks)

        # 6. RAG 问答（需要 API）
        example_6_rag_qa(vectorstore)

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  1. Document Loaders - 加载文档")
        print("  2. Text Splitters - 分割文本")
        print("  3. Embeddings - 向量嵌入")
        print("  4. Vector Stores - 向量存储（Pinecone）")
        print("  5. Similarity Search - 相似度检索")
        print("  6. RAG - 检索增强生成")
        print("\nRAG 工作流:")
        print("  文档 -> 分割 -> 嵌入 -> 存储")
        print("  查询 -> 检索 -> 提供给 LLM -> 生成答案")
        print("\n注意：")
        print("  - 示例 1-3 可以正常运行（不需要 Pinecone）")
        print("  - 示例 4-6 需要 PINECONE_API_KEY")
        print("\n下一步：")
        print("  14_rag_advanced - RAG 进阶（混合搜索、重排序）")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
