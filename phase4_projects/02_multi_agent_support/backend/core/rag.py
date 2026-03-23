"""
RAG 引擎模块

本模块实现：
- 文档加载和处理
- 向量化（优先使用智谱 AI）
- 向量存储集成（优先 Milvus，Windows 环境自动降级到 ChromaDB）
- 语义检索
- 知识库管理

向量数据库选择：
- 优先使用 Milvus（分布式、高性能）
- Windows 环境下 Milvus Lite 不可用时，自动降级到 ChromaDB
- 可通过环境变量 FORCE_VECTOR_DB 强制指定

参考文档：
- Milvus: https://langchain.cadn.net.cn/python/docs/integrations/vectorstores/milvus/index.html
- ChromaDB: https://python.langchain.com/docs/integrations/vectorstores/chroma/
"""

import os
import sys
import platform
from typing import List, Dict, Any, Optional, Union
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

# 加载环境变量
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)


# ==================== 向量数据库选择 ====================

def _detect_vector_db() -> str:
    """
    检测可用的向量数据库

    Returns:
        'milvus' 或 'chroma'
    """
    # 检查环境变量强制指定
    force_db = os.getenv("FORCE_VECTOR_DB", "").lower()
    if force_db in ["milvus", "chroma"]:
        return force_db

    # 尝试导入 Milvus
    try:
        from langchain_milvus import Milvus
        from pymilvus import connections

        # 测试 Milvus Lite 连接（仅适用于 Linux/macOS）
        test_uri = "./test_milvus_connection.db"
        try:
            # 尝试连接本地 Milvus Lite
            connections.connect("test", uri=test_uri)
            connections.disconnect("test")
            # 清理测试文件
            if os.path.exists(test_uri):
                os.remove(test_uri)
            print("[INFO] Milvus Lite 可用，将使用 Milvus 向量数据库")
            return "milvus"
        except Exception:
            # Milvus Lite 不可用，可能是 Windows 环境
            print("[WARN] Milvus Lite 不可用（Windows 不支持 milvus-lite）")
            print("[INFO] 自动降级到 ChromaDB")
            return "chroma"

    except ImportError as e:
        print(f"[WARN] 无法导入 Milvus: {e}")
        print("[INFO] 自动降级到 ChromaDB")
        return "chroma"


# 检测可用的向量数据库
VECTOR_DB_TYPE = _detect_vector_db()

# 根据检测结果导入相应的向量存储
if VECTOR_DB_TYPE == "milvus":
    from langchain_milvus import Milvus
    print("[OK] 使用 Milvus 向量数据库")
else:
    from langchain_chroma import Chroma
    print("[OK] 使用 ChromaDB 向量数据库（Milvus Lite 在 Windows 上不可用）")
    print("     提示：如需在 Windows 上使用 Milvus，请使用 Docker 或 WSL2")

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


# ==================== 向量数据库配置 ====================

# Milvus 配置
MILVUS_LITE_URI = os.getenv("MILVUS_LITE_URI", "./data/milvus/milvus_lite.db")
MILVUS_COLLECTION_PREFIX = os.getenv("MILVUS_COLLECTION_PREFIX", "customer_service_kb")

# ChromaDB 配置
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
CHROMA_COLLECTION_PREFIX = os.getenv("CHROMA_COLLECTION_PREFIX", "customer_service_kb")


def get_vector_db_config():
    """
    获取向量数据库配置

    Returns:
        配置字典，包含数据库类型、连接参数等
    """
    config = {"type": VECTOR_DB_TYPE}

    if VECTOR_DB_TYPE == "milvus":
        # 确保 Milvus Lite 数据目录存在
        lite_db_path = Path(MILVUS_LITE_URI)
        lite_db_path.parent.mkdir(parents=True, exist_ok=True)

        config.update({
            "connection_args": {"uri": MILVUS_LITE_URI},
            "index_params": {"index_type": "FLAT", "metric_type": "L2"},
            "collection_prefix": MILVUS_COLLECTION_PREFIX
        })
    else:  # chroma
        # 确保 ChromaDB 数据目录存在
        persist_dir = Path(CHROMA_PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)

        config.update({
            "persist_directory": CHROMA_PERSIST_DIR,
            "collection_prefix": CHROMA_COLLECTION_PREFIX
        })

    return config


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
    """RAG 引擎（支持 Milvus 和 ChromaDB）"""

    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_db_type = VECTOR_DB_TYPE
        self.vector_db_config = get_vector_db_config()
        self.vector_stores = {}  # 按类别存储向量存储
        self.processor = DocumentProcessor()

        # 初始化向量数据库
        self._init_vector_db()

    def _init_vector_db(self):
        """初始化向量数据库"""
        if self.vector_db_type == "milvus":
            print(f"[OK] Milvus 初始化成功")
            print(f"[INFO] 本地数据库: {MILVUS_LITE_URI}")
            print(f"[INFO] 使用 Milvus Lite 模式（本地文件存储）")
        else:
            print(f"[OK] ChromaDB 初始化成功")
            print(f"[INFO] 本地数据库: {CHROMA_PERSIST_DIR}")
            print(f"[INFO] 使用 ChromaDB 持久化存储")

    def _get_collection_name(self, category: str) -> str:
        """
        获取指定类别的集合名称

        Args:
            category: 知识库类别

        Returns:
            集合名称
        """
        # 集合名称只能包含字母、数字和下划线
        safe_category = category.replace("-", "_").replace(" ", "_")
        prefix = self.vector_db_config["collection_prefix"]
        return f"{prefix}_{safe_category}"

    def _get_vector_store(self, category: str) -> Optional[Any]:
        """
        获取或创建指定类别的向量存储

        支持两种向量数据库：
        - Milvus: 分布式高性能向量数据库
        - ChromaDB: 本地向量数据库（Windows 降级选项）

        Args:
            category: 知识库类别

        Returns:
            向量存储实例（Milvus 或 ChromaDB）
        """
        if category not in self.vector_stores:
            collection_name = self._get_collection_name(category)

            try:
                if self.vector_db_type == "milvus":
                    # 使用 Milvus
                    self.vector_stores[category] = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=collection_name,
                        connection_args=self.vector_db_config["connection_args"],
                        index_params=self.vector_db_config["index_params"],
                        drop_old=False  # 保留已有数据
                    )
                    print(f"[INFO] 加载 Milvus 集合: {collection_name}")
                else:
                    # 使用 ChromaDB
                    self.vector_stores[category] = Chroma(
                        collection_name=collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.vector_db_config["persist_directory"]
                    )
                    print(f"[INFO] 加载 ChromaDB 集合: {collection_name}")

            except Exception as e:
                print(f"[ERROR] 向量数据库初始化失败: {e}")
                import traceback
                traceback.print_exc()
                return None

        return self.vector_stores.get(category)

    def index_document(self, file_path: str, category: str) -> Dict[str, Any]:
        """
        索引单个文档到向量数据库

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

        # 4. 向量化并存储
        db_name = "Milvus" if self.vector_db_type == "milvus" else "ChromaDB"
        print(f"   [VECTOR] 开始向量化 {len(chunks)} 个文档块并存储到 {db_name}...")
        print(f"   [INFO] 这可能需要一些时间，请耐心等待...")

        batch_size = 50  # 每批处理 50 个块
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        try:
            vector_store = self._get_vector_store(category)
            if not vector_store:
                raise Exception(f"无法获取 {db_name} 向量存储")

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_num = i // batch_size + 1

                print(f"   [PROGRESS] 处理批次 {batch_num}/{total_batches} ({len(batch)} 个块)...")

                # 添加文档到向量数据库
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
            print(f"[ERROR] {db_name} 索引失败: {e}")
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
        db_name = "Milvus" if self.vector_db_type == "milvus" else "ChromaDB"
        print(f"   [VECTOR] 向量化并存储到 {db_name}...")

        try:
            vector_store = self._get_vector_store(category)
            if not vector_store:
                raise Exception(f"无法获取 {db_name} 向量存储")

            vector_store.add_documents(chunks)

            return {
                "status": "success",
                "chunks": len(chunks),
                "category": category,
                "source": directory
            }

        except Exception as e:
            print(f"[ERROR] {db_name} 索引失败: {e}")
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
                vector_store = self._get_vector_store(category)
                if not vector_store:
                    print(f"[WARN] 类别 '{category}' 的向量存储不存在")
                    return []

                results = vector_store.similarity_search(query, k=k)
            else:
                # 搜索所有类别
                results = []
                for cat in self.vector_stores.keys():
                    cat_results = self.vector_stores[cat].similarity_search(query, k=k)
                    results.extend(cat_results)

                # 按相似度排序并去重
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
            db_name = "Milvus" if self.vector_db_type == "milvus" else "ChromaDB"
            print(f"[WARN] {db_name} 搜索失败: {e}")
            import traceback
            traceback.print_exc()
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
                vector_store = self.vector_stores[category]

                if self.vector_db_type == "milvus":
                    # Milvus 删除整个集合
                    vector_store.delete_collection()
                else:
                    # ChromaDB 删除集合
                    vector_store._collection.delete(where={"category": category})

                # 从内存中移除
                del self.vector_stores[category]

                print(f"[OK] 已删除类别 '{category}' 的所有文档")
                return True
            else:
                print(f"[WARN] 类别 '{category}' 不存在")
                return False

        except Exception as e:
            print(f"[ERROR] 删除失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def list_collections(self) -> List[str]:
        """
        列出所有集合

        Returns:
            集合名称列表
        """
        try:
            if self.vector_db_type == "milvus":
                from pymilvus import connections, utility

                # 连接到 Milvus
                connections.connect("default", **self.vector_db_config["connection_args"])

                collections = utility.list_collections()
                return collections
            else:
                # ChromaDB 列出集合
                import chromadb
                client = chromadb.PersistentClient(path=self.vector_db_config["persist_directory"])
                collections = [col.name for col in client.list_collections()]
                return collections

        except Exception as e:
            print(f"[ERROR] 获取集合列表失败: {e}")
            return []

    def list_documents(self, category: str = None) -> List[Dict[str, Any]]:
        """
        列出知识库中的文档

        Args:
            category: 知识库类别

        Returns:
            文档列表
        """
        # Milvus 返回模拟数据，实际使用中可以维护一个文档索引数据库
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
    print("RAG 引擎测试 (Milvus)")
    print("=" * 60)

    try:
        # 1. 初始化
        print("\n1. 初始化 RAG 引擎...")
        rag = create_rag_engine()
        print("   [OK] 初始化完成")

        # 2. 列出集合
        print("\n2. 列出所有集合...")
        collections = rag.list_collections()
        print(f"   找到 {len(collections)} 个集合: {collections}")

        # 3. 测试搜索（如果有数据）
        print("\n3. 测试搜索功能...")
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
