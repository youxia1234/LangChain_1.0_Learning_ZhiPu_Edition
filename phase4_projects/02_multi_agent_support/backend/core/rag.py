"""
RAG 引擎模块

本模块实现：
- 文档加载和处理
- 向量化（优先使用智谱 AI）
- ChromaDB 向量存储集成（本地部署，无网络限制）
- 语义检索
- 知识库管理
"""

import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import uuid

# LangChain 核心导入
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)

# ChromaDB 向量存储
from langchain_chroma import Chroma

# 加载环境变量
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# ==================== 配置 HF Mirror ====================
# 必须在导入 HuggingFace 之前设置
os.environ['HF_ENDPOINT'] = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')

# 禁用遥测和离线模式（避免网络连接问题）
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


# ==================== Embeddings 模型 ====================

def get_embeddings():
    """
    获取 Embeddings 模型（优先使用智谱 AI）

    Returns:
        Embeddings 实例
    """
    # 优先尝试使用智谱 AI embeddings（国内无限制）
    zhipuai_key = os.getenv("ZHIPUAI_API_KEY")
    if zhipuai_key and zhipuai_key != "your_zhipuai_api_key_here":
        try:
            from langchain_openai import OpenAIEmbeddings

            print("[INFO] 使用智谱 AI Embeddings API")
            embeddings = OpenAIEmbeddings(
                model="embedding-2",
                openai_api_key=zhipuai_key,
                openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
            )
            print("   [OK] 智谱 AI Embeddings 初始化成功")
            return embeddings
        except Exception as e:
            print(f"[WARN] 智谱 AI Embeddings 初始化失败: {e}")
            print("       尝试使用本地模型...")

    # 回退到本地 HuggingFace 模型
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = os.getenv(
            "EMBEDDINGS_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print(f"[INFO] 使用本地 HuggingFace 模型: {model_name}")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

        print("   [OK] 本地 Embeddings 模型加载成功")
        return embeddings

    except Exception as e:
        print(f"[ERROR] Embeddings 加载失败: {e}")
        raise


# ==================== ChromaDB 配置 ====================

# 持久化目录
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")


# ==================== 文档处理器 ====================

class DocumentProcessor:
    """文档处理器"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        self.embeddings = get_embeddings()

    def load_pdf(self, file_path: str) -> List[Document]:
        """加载 PDF 文档"""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def load_text(self, file_path: str) -> List[Document]:
        """加载文本文档"""
        loader = TextLoader(file_path, encoding='utf-8')
        return loader.load()

    def load_directory(self, directory: str, glob: str = "**/*.*") -> List[Document]:
        """加载目录中的所有文档"""
        loader = DirectoryLoader(
            directory,
            glob=glob,
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        return loader.load()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        chunks = self.text_splitter.split_documents(documents)
        print(f"   [FILE] 分割: {len(documents)} 个文档 -> {len(chunks)} 个块")
        return chunks

    def add_metadata(self, documents: List[Document], category: str, source: str) -> List[Document]:
        """添加元数据"""
        for doc in documents:
            doc.metadata.update({
                "category": category,
                "source": source,
                "upload_date": datetime.now().isoformat()
            })
        return documents


# ==================== RAG 引擎 ====================

class RAGEngine:
    """RAG 引擎（使用 ChromaDB）"""

    def __init__(self):
        self.embeddings = get_embeddings()
        self.collection_name = os.getenv("CHROMA_COLLECTION", "customer_service_kb")
        self.vector_stores = {}  # 按类别存储向量存储
        self.processor = DocumentProcessor()

        # 确保持久化目录存在
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)

        print(f"[OK] ChromaDB 初始化成功")
        print(f"[INFO] 持久化目录: {CHROMA_PERSIST_DIR}")

    def _get_vector_store(self, category: str) -> Chroma:
        """获取或创建指定类别的向量存储"""
        if category not in self.vector_stores:
            persist_directory = os.path.join(CHROMA_PERSIST_DIR, category)

            self.vector_stores[category] = Chroma(
                collection_name=f"{self.collection_name}_{category}",
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )

        return self.vector_stores[category]

    def index_document(self, file_path: str, category: str) -> Dict[str, Any]:
        """
        索引单个文档到 ChromaDB

        Args:
            file_path: 文件路径
            category: 知识库类别（products/technical/faq）

        Returns:
            索引结果
        """
        print(f"\n[KB] 索引文档: {file_path}")
        print(f"   类别: {category}")

        # 1. 加载文档
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.pdf':
            documents = self.processor.load_pdf(file_path)
        else:
            documents = self.processor.load_text(file_path)

        print(f"   [OK] 加载: {len(documents)} 页")

        # 2. 添加元数据
        source = Path(file_path).name
        documents = self.processor.add_metadata(documents, category, source)

        # 3. 分割文档
        chunks = self.processor.split_documents(documents)

        # 4. 向量化并存储到 ChromaDB
        print(f"   [VECTOR] 开始向量化 {len(chunks)} 个文档块并存储到 ChromaDB...")
        print(f"   [INFO] 这可能需要一些时间，请耐心等待...")

        batch_size = 50  # 每批处理 50 个块
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        try:
            vector_store = self._get_vector_store(category)

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1

                print(f"   [PROGRESS] 处理批次 {batch_num}/{total_batches} ({len(batch)} 个块)...")

                # 添加文档到 ChromaDB
                vector_store.add_documents(batch)

                print(f"   [OK] 批次 {batch_num}/{total_batches} 完成")

            print(f"   [OK] 向量化完成，共处理 {len(chunks)} 个文档块")

            return {
                "status": "success",
                "document_id": str(uuid.uuid4()),
                "chunks": len(chunks),
                "category": category,
                "source": source
            }

        except Exception as e:
            print(f"[ERROR] ChromaDB 索引失败: {e}")
            raise

    def index_directory(self, directory: str, category: str) -> Dict[str, Any]:
        """
        索引整个目录

        Args:
            directory: 目录路径
            category: 知识库类别

        Returns:
            索引结果
        """
        print(f"\n[KB] 索引目录: {directory}")
        print(f"   类别: {category}")

        # 1. 加载目录中的所有文档
        documents = self.processor.load_directory(directory)

        print(f"   加载: {len(documents)} 个文档")

        # 2. 添加元数据
        documents = self.processor.add_metadata(documents, category, directory)

        # 3. 分割文档
        chunks = self.processor.split_documents(documents)

        # 4. 向量化并存储
        print("   [VECTOR] 向量化并存储到 ChromaDB...")

        try:
            vector_store = self._get_vector_store(category)
            vector_store.add_documents(chunks)

            return {
                "status": "success",
                "chunks": len(chunks),
                "category": category,
                "source": directory
            }

        except Exception as e:
            print(f"[ERROR] ChromaDB 索引失败: {e}")
            raise

    def search(self, query: str, category: str = None, k: int = 3) -> List[Document]:
        """
        在知识库中搜索

        Args:
            query: 搜索查询
            category: 知识库类别（None 表示搜索所有）
            k: 返回结果数量

        Returns:
            相关文档列表
        """
        try:
            if category:
                # 搜索特定类别
                if category not in self.vector_stores:
                    # 尝试加载已存在的向量存储
                    persist_directory = os.path.join(CHROMA_PERSIST_DIR, category)
                    if Path(persist_directory).exists():
                        self.vector_stores[category] = Chroma(
                            collection_name=f"{self.collection_name}_{category}",
                            embedding_function=self.embeddings,
                            persist_directory=persist_directory
                        )
                    else:
                        return []

                vector_store = self.vector_stores[category]
                results = vector_store.similarity_search(query, k=k)
            else:
                # 搜索所有类别
                results = []
                for cat in self.vector_stores.keys():
                    cat_results = self.vector_stores[cat].similarity_search(query, k=k)
                    results.extend(cat_results)

                # 按相似度排序（取前 k 个）
                # ChromaDB 已经按相似度排序，所以直接去重并限制数量
                unique_results = []
                seen_content = set()
                for doc in results:
                    if doc.page_content not in seen_content:
                        unique_results.append(doc)
                        seen_content.add(doc.page_content)
                        if len(unique_results) >= k:
                            break

                results = unique_results

            return results

        except Exception as e:
            print(f"[WARN] ChromaDB 搜索失败: {e}")
            return []

    def delete_documents(self, category: str, filter_dict: Dict = None) -> bool:
        """
        删除文档（通过类别）

        Args:
            category: 知识库类别
            filter_dict: 额外的过滤条件

        Returns:
            是否成功
        """
        try:
            if category in self.vector_stores:
                # 删除整个集合
                vector_store = self.vector_stores[category]

                # 获取所有 ID 并删除
                collection = vector_store._collection
                collection.delete(where={"category": category})

                # 从内存中移除
                del self.vector_stores[category]

                print(f"[OK] 已删除类别 '{category}' 的所有文档")
                return True
            else:
                print(f"[WARN] 类别 '{category}' 不存在")
                return False

        except Exception as e:
            print(f"[ERROR] 删除失败: {e}")
            return False

    def list_documents(self, category: str = None) -> List[Dict[str, Any]]:
        """
        列出知识库中的文档

        Args:
            category: 知识库类别

        Returns:
            文档列表
        """
        # ChromaDB 不直接支持列出所有文档
        # 这里返回模拟数据，实际使用中可以维护一个文档索引数据库
        return [
            {
                "id": "doc-1",
                "filename": "product_manual.pdf",
                "category": "products",
                "chunks": 15,
                "upload_date": "2024-12-20T10:00:00Z"
            },
            {
                "id": "doc-2",
                "filename": "troubleshooting_guide.pdf",
                "category": "technical",
                "chunks": 25,
                "upload_date": "2024-12-20T11:00:00Z"
            }
        ]


# ==================== 便捷函数 ====================

def create_rag_engine():
    """创建 RAG 引擎实例"""
    return RAGEngine()


if __name__ == "__main__":
    # 测试 RAG 引擎
    print("=" * 60)
    print("RAG 引擎测试 (ChromaDB)")
    print("=" * 60)

    try:
        # 1. 初始化
        print("\n1. 初始化 RAG 引擎...")
        rag = create_rag_engine()
        print("   [OK] 初始化完成")

        # 2. 测试搜索（如果有数据）
        print("\n2. 测试搜索功能...")
        results = rag.search("蓝牙耳机连接", category="technical", k=2)
        print(f"   找到 {len(results)} 个相关文档")

        for i, doc in enumerate(results, 1):
            print(f"\n   [{i}] {doc.metadata.get('source', 'unknown')}")
            print(f"   {doc.page_content[:100]}...")

        print("\n[OK] 测试完成")

    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
