"""
混合 RAG 检索引擎

结合 BM25 关键词检索和向量检索（Milvus/ChromaDB），
使用 EnsembleRetriever 实现 RRF (Reciprocal Rank Fusion) 算法融合结果。

优势：
- BM25：精确匹配专有名词、版本号、代码片段
- 向量检索：理解语义和同义词
- 混合检索：结合两者优势，全面覆盖
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# LangChain 核心导入
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

# 本地模块导入
from .rag import RAGEngine, get_embeddings, VECTOR_DB_TYPE


class HybridRAGEngine(RAGEngine):
    """
    混合 RAG 引擎

    继承自 RAGEngine，添加 BM25 检索能力，
    实现向量 + BM25 的混合检索。
    """

    def __init__(self, bm25_weight: float = 0.4, vector_weight: float = 0.6):
        """
        初始化混合 RAG 引擎

        Args:
            bm25_weight: BM25 检索权重 (0-1)
            vector_weight: 向量检索权重 (0-1)
                           两者之和应该为 1.0
        """
        super().__init__()
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # BM25 检索器缓存（按类别存储）
        self.bm25_retrievers: Dict[str, BM25Retriever] = {}

        # Ensemble 检索器缓存（按类别存储）
        self.ensemble_retrievers: Dict[str, EnsembleRetriever] = {}

        print(f"[HybridRAG] 初始化混合检索引擎")
        print(f"[HybridRAG] BM25 权重: {bm25_weight:.1%}")
        print(f"[HybridRAG] 向量权重: {vector_weight:.1%}")

    def _get_bm25_retriever(self, category: str, documents: List[Document] = None) -> BM25Retriever:
        """
        获取或创建指定类别的 BM25 检索器

        Args:
            category: 知识库类别
            documents: 文档列表（用于创建新检索器）

        Returns:
            BM25Retriever 实例
        """
        if category not in self.bm25_retrievers and documents:
            # 创建新的 BM25 检索器
            self.bm25_retrievers[category] = BM25Retriever.from_documents(
                documents=documents,
                k=5  # 检索 top-5
            )
            print(f"[HybridRAG] 创建 {category} 类别的 BM25 检索器")

        return self.bm25_retrievers.get(category)

    def _get_ensemble_retriever(
        self,
        category: str,
        documents: List[Document] = None
    ) -> Optional[EnsembleRetriever]:
        """
        获取或创建指定类别的混合检索器

        Args:
            category: 知识库类别
            documents: 文档列表（用于创建新检索器）

        Returns:
            EnsembleRetriever 实例
        """
        if category not in self.ensemble_retrievers:
            # 获取向量检索器（_get_vector_store 只需要 category 参数）
            vector_store = self._get_vector_store(category)
            if not vector_store:
                return None

            vector_retriever = vector_store.as_retriever(
                search_kwargs={"k": 5}
            )

            # 获取 BM25 检索器
            bm25_retriever = self._get_bm25_retriever(category, documents)
            if not bm25_retriever:
                return None

            # 创建混合检索器
            self.ensemble_retrievers[category] = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[self.bm25_weight, self.vector_weight]
            )
            print(f"[HybridRAG] 创建 {category} 类别的混合检索器")

        return self.ensemble_retrievers.get(category)

    def search_hybrid(
        self,
        query: str,
        category: str = None,
        k: int = 3
    ) -> List[Document]:
        """
        混合检索（推荐使用）

        结合 BM25 关键词检索和向量语义检索，
        使用 RRF 算法融合结果。

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
                ensemble_retriever = self._get_ensemble_retriever(category)
                if not ensemble_retriever:
                    print(f"[WARN] 类别 {category} 的混合检索器未初始化")
                    # 回退到纯向量检索
                    return self.search(query, category=category, k=k)

                results = ensemble_retriever.invoke(query)
                return results[:k]
            else:
                # 搜索所有类别：分别搜索后合并
                all_results = []

                for cat in self.vector_stores.keys():
                    ensemble_retriever = self._get_ensemble_retriever(cat)
                    if ensemble_retriever:
                        results = ensemble_retriever.invoke(query)
                        all_results.extend(results)

                # 按相关性排序（EnsembleRetriever 已排序）
                return all_results[:k]

        except Exception as e:
            print(f"[ERROR] 混合检索失败: {e}")
            print(f"[WARN] 回退到纯向量检索")
            # 回退到纯向量检索
            return self.search(query, category=category, k=k)

    def index_document_hybrid(
        self,
        file_path: str,
        category: str
    ) -> Dict[str, Any]:
        """
        索引文档到向量存储和 BM25（混合索引）

        Args:
            file_path: 文档文件路径
            category: 知识库类别

        Returns:
            索引结果
        """
        # 先使用父类方法进行向量索引
        result = self.index_document(file_path, category)

        # 更新 BM25 检索器
        if result["status"] == "success":
            # 重新加载该类别的所有文档以更新 BM25
            self._update_bm25_for_category(category)

            result["hybrid_index"] = True
            result["message"] += " | 已添加到 BM25 索引"

        return result

    def _update_bm25_for_category(self, category: str):
        """
        为指定类别更新 BM25 索引

        支持两种向量数据库：
        - Milvus: 使用 Milvus API 获取所有文档
        - ChromaDB: 使用 ChromaDB API 获取所有文档

        Args:
            category: 知识库类别
        """
        try:
            # 从向量存储中获取所有文档
            vector_store = self._get_vector_store(category)
            if not vector_store:
                return

            documents = []

            if self.vector_db_type == "milvus":
                # 使用 Milvus API 获取文档
                from pymilvus import connections, Collection

                collection_name = self._get_collection_name(category)

                # 连接到 Milvus
                connections.connect("default", **self.vector_db_config["connection_args"])

                # 获取集合
                collection = Collection(collection_name)
                collection.load()

                # 获取所有文档（使用 expr 查询所有数据）
                results = collection.query(
                    expr="",
                    output_fields=["*"]
                )

                # 构建文档列表
                for result in results:
                    # Milvus 存储文档内容和元数据
                    text = result.get("text", "")
                    metadata = {k: v for k, v in result.items() if k != "text"}
                    documents.append(Document(page_content=text, metadata=metadata))

            else:
                # 使用 ChromaDB API 获取文档
                import chromadb

                collection_name = self._get_collection_name(category)
                persist_dir = self.vector_db_config["persist_directory"]

                # 连接到 ChromaDB
                client = chromadb.PersistentClient(path=persist_dir)
                collection = client.get_collection(name=collection_name)

                # 获取所有文档
                results = collection.get()

                # 构建文档列表
                for i, doc_id in enumerate(results.get('ids', [])):
                    text = results.get('documents', [])[i] if i < len(results.get('documents', [])) else ""
                    metadata = results.get('metadatas', [])[i] if i < len(results.get('metadatas', [])) else {}
                    documents.append(Document(page_content=text, metadata=metadata))

            # 重新创建 BM25 检索器
            if documents:
                # 清除旧的 BM25 检索器和混合检索器
                if category in self.bm25_retrievers:
                    del self.bm25_retrievers[category]
                if category in self.ensemble_retrievers:
                    del self.ensemble_retrievers[category]

                # 创建新的 BM25 检索器
                self.bm25_retrievers[category] = BM25Retriever.from_documents(
                    documents=documents,
                    k=5
                )

                db_name = "Milvus" if self.vector_db_type == "milvus" else "ChromaDB"
                print(f"[HybridRAG] 更新 {category} 类别的 BM25 索引")
                print(f"[HybridRAG] 文档数量: {len(documents)}")
                print(f"[HybridRAG] 向量数据库: {db_name}")

        except Exception as e:
            print(f"[ERROR] 更新 BM25 索引失败: {e}")
            import traceback
            traceback.print_exc()

    def compare_retrieval_methods(
        self,
        query: str,
        category: str = None,
        k: int = 3
    ) -> Dict[str, Any]:
        """
        对比不同检索方法的结果

        Args:
            query: 搜索查询
            category: 知识库类别
            k: 返回结果数量

        Returns:
            对比结果
        """
        results = {
            "query": query,
            "methods": {}
        }

        # 1. 纯 BM25 检索
        if category and category in self.bm25_retrievers:
            bm25_results = self.bm25_retrievers[category].invoke(query)[:k]
            results["methods"]["bm25"] = {
                "count": len(bm25_results),
                "docs": [doc.page_content[:100] + "..." for doc in bm25_results]
            }

        # 2. 纯向量检索
        vector_results = self.search(query, category=category, k=k)
        results["methods"]["vector"] = {
            "count": len(vector_results),
            "docs": [doc.page_content[:100] + "..." for doc in vector_results]
        }

        # 3. 混合检索
        hybrid_results = self.search_hybrid(query, category=category, k=k)
        results["methods"]["hybrid"] = {
            "count": len(hybrid_results),
            "docs": [doc.page_content[:100] + "..." for doc in hybrid_results]
        }

        return results


    def _get_collection_name(self, category: str) -> str:
        """
        获取指定类别的集合名称（使用父类方法）

        Args:
            category: 知识库类别

        Returns:
            集合名称
        """
        # 调用父类方法以保持一致性
        return super()._get_collection_name(category)


# ==================== 便捷函数 ====================

def create_hybrid_rag_engine(
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6
) -> HybridRAGEngine:
    """
    创建混合 RAG 引擎

    Args:
        bm25_weight: BM25 权重 (推荐 0.3-0.5)
        vector_weight: 向量权重 (推荐 0.5-0.7)

    Returns:
        HybridRAGEngine 实例

    权重建议：
    - 技术文档/代码：bm25_weight=0.5, vector_weight=0.5 (平衡)
    - 产品介绍/对话：bm25_weight=0.3, vector_weight=0.7 (偏语义)
    - 配置/手册：bm25_weight=0.6, vector_weight=0.4 (偏精确)
    """
    return HybridRAGEngine(
        bm25_weight=bm25_weight,
        vector_weight=vector_weight
    )


def get_hybrid_rag_manager(enable_rag: bool = True) -> HybridRAGEngine:
    """
    获取混合 RAG 管理器（兼容原有接口）

    Args:
        enable_rag: 是否启用 RAG

    Returns:
        RAG 引擎实例
    """
    if not enable_rag:
        return None

    return create_hybrid_rag_engine()
