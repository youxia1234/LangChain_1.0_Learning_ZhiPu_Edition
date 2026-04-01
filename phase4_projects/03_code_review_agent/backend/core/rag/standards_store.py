"""规范知识库管理模块

管理编码规范文档的向量存储，支持：
- 内置规范文档的自动加载
- 用户上传自定义规范
- 按类别检索相关规范
- ChromaDB 持久化存储
"""

import os
import logging
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

from backend.core.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# 类别到文件名的映射
CATEGORY_FILE_MAP = {
    "style": "python_style.md",
    "security": "security_rules.md",
    "performance": "performance_patterns.md",
    "architecture": "architecture_principles.md",
}


class StandardsStore:
    """编码规范知识库

    使用 ChromaDB 管理编码规范的向量存储和检索。
    支持按类别过滤检索，内置规范自动加载。

    Attributes:
        persist_dir: ChromaDB 持久化目录
        collection_name: 集合名称
        embeddings: 向量化实例
        client: ChromaDB 客户端
        collection: ChromaDB 集合
        vectorstore: LangChain Chroma 向量存储
        initialized: 是否已初始化
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """初始化规范知识库

        Args:
            persist_dir: ChromaDB 持久化目录，默认从环境变量读取
            collection_name: 集合名称，默认从环境变量读取
        """
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma"
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "code_standards"
        )
        self.embeddings = get_embeddings()
        self.vectorstore = None
        self.initialized = False

        # 文本分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "],
            length_function=len,
        )

    def initialize(self) -> None:
        """初始化知识库

        加载 ChromaDB（如已有持久化数据则直接加载），
        否则自动加载内置规范文档并构建向量索引。
        """
        if self.initialized:
            return

        os.makedirs(self.persist_dir, exist_ok=True)

        # 尝试加载已有的向量存储
        try:
            from langchain_community.vectorstores import Chroma

            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )

            # 检查是否已有数据
            count = self.vectorstore._collection.count()
            if count > 0:
                logger.info(f"加载已有规范知识库，共 {count} 条记录")
                self.initialized = True
                return

        except Exception as e:
            logger.warning(f"加载已有向量存储失败: {e}，将重新构建")

        # 加载内置规范文档
        self._load_builtin_standards()
        self.initialized = True

    def _load_builtin_standards(self) -> None:
        """加载内置规范文档

        读取 data/standards/ 目录下的 Markdown 文件，
        解析并添加元数据后存入 ChromaDB。
        """
        standards_dir = os.getenv("STANDARDS_DIR", "./data/standards")

        # 自动检测脚本所在目录的相对路径
        if not os.path.isabs(standards_dir):
            base_dir = Path(__file__).parent.parent.parent.parent  # 回到项目根目录
            standards_dir = str(base_dir / standards_dir)

        if not os.path.isdir(standards_dir):
            logger.warning(f"规范目录不存在: {standards_dir}，跳过内置规范加载")
            return

        all_documents = []

        for category, filename in CATEGORY_FILE_MAP.items():
            filepath = os.path.join(standards_dir, filename)
            if not os.path.isfile(filepath):
                logger.warning(f"规范文件不存在: {filepath}")
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()

                # 创建文档对象，添加元数据
                doc = Document(
                    page_content=content,
                    metadata={
                        "category": category,
                        "source": filename,
                        "type": "builtin",
                    },
                )

                # 分块
                chunks = self.text_splitter.split_documents([doc])

                # 为每个分块添加更丰富的元数据
                for i, chunk in enumerate(chunks):
                    chunk.metadata["chunk_index"] = i
                    chunk.metadata["total_chunks"] = len(chunks)

                all_documents.extend(chunks)
                logger.info(f"加载规范文档: {filename} ({category}) - {len(chunks)} 个分块")

            except Exception as e:
                logger.error(f"加载规范文件失败 {filename}: {e}")

        if all_documents:
            self._add_documents_to_store(all_documents)
            logger.info(f"内置规范加载完成，共 {len(all_documents)} 个分块")

    def _add_documents_to_store(self, documents: list[Document]) -> None:
        """将文档添加到向量存储

        Args:
            documents: 要添加的文档列表
        """
        from langchain_community.vectorstores import Chroma

        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
        else:
            self.vectorstore.add_documents(documents)

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
    ) -> list[Document]:
        """检索相关规范文档

        Args:
            query: 查询文本
            category: 过滤类别（style/security/performance/architecture）
            top_k: 返回最大文档数

        Returns:
            list[Document]: 相关文档列表
        """
        if not self.initialized:
            self.initialize()

        if self.vectorstore is None:
            return []

        try:
            # 构建过滤条件
            filter_dict = {}
            if category:
                filter_dict["category"] = category

            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query, k=top_k, filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(query, k=top_k)

            return results

        except Exception as e:
            logger.error(f"规范检索失败: {e}")
            return []

    def get_retriever(self, category: Optional[str] = None, top_k: int = 5):
        """获取指定类别的检索器

        Args:
            category: 过滤类别
            top_k: 返回最大文档数

        Returns:
            VectorStoreRetriever: 检索器实例
        """
        if not self.initialized:
            self.initialize()

        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")

        search_kwargs = {"k": top_k}
        if category:
            search_kwargs["filter"] = {"category": category}

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def add_user_documents(self, filepath: str, category: str) -> int:
        """添加用户自定义规范文档

        Args:
            filepath: 文档文件路径
            category: 文档类别

        Returns:
            int: 添加的分块数量
        """
        if not self.initialized:
            self.initialize()

        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        try:
            # 根据文件类型加载内容
            if ext == ".md":
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            elif ext == ".txt":
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            elif ext == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(filepath)
                content = "\n\n".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )
            else:
                raise ValueError(f"不支持的文件类型: {ext}")

            doc = Document(
                page_content=content,
                metadata={
                    "category": category,
                    "source": filename,
                    "type": "user",
                },
            )

            chunks = self.text_splitter.split_documents([doc])
            self._add_documents_to_store(chunks)

            logger.info(f"用户文档已添加: {filename} - {len(chunks)} 个分块")
            return len(chunks)

        except Exception as e:
            logger.error(f"添加用户文档失败: {e}")
            raise

    def list_documents(self) -> list[dict]:
        """列出知识库中的所有文档

        Returns:
            list[dict]: 文档信息列表
        """
        if not self.initialized:
            self.initialize()

        if self.vectorstore is None:
            return []

        try:
            # 从 ChromaDB 获取所有元数据
            collection = self.vectorstore._collection
            results = collection.get(include=["metadatas"])

            # 按来源分组统计
            doc_info = {}
            for metadata in results.get("metadatas", []):
                source = metadata.get("source", "unknown")
                if source not in doc_info:
                    doc_info[source] = {
                        "doc_id": source,
                        "filename": source,
                        "category": metadata.get("category", "unknown"),
                        "chunk_count": 0,
                        "source": metadata.get("type", "unknown"),
                    }
                doc_info[source]["chunk_count"] += 1

            return list(doc_info.values())

        except Exception as e:
            logger.error(f"获取文档列表失败: {e}")
            return []

    def delete_document(self, source: str) -> bool:
        """删除指定来源的所有文档

        Args:
            source: 文档来源标识（文件名）

        Returns:
            bool: 是否删除成功
        """
        if not self.initialized:
            self.initialize()

        if self.vectorstore is None:
            return False

        try:
            collection = self.vectorstore._collection
            # 获取该来源的所有文档ID
            results = collection.get(
                where={"source": source},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info(f"已删除文档: {source} ({len(ids)} 个分块)")
                return True
            return False

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False

    def get_stats(self) -> dict:
        """获取知识库统计信息

        Returns:
            dict: 统计信息
        """
        if not self.initialized:
            self.initialize()

        if self.vectorstore is None:
            return {"total_chunks": 0, "documents": []}

        collection = self.vectorstore._collection
        total = collection.count()
        docs = self.list_documents()

        return {
            "total_chunks": total,
            "documents": docs,
            "categories": list(set(d["category"] for d in docs)),
        }
