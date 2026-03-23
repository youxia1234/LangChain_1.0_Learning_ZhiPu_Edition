"""
知识库管理模块

本模块负责：
- 知识库初始化
- 文档上传和管理
- 知识库统计
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from .rag import RAGEngine


class KnowledgeBaseManager:
    """知识库管理器"""

    def __init__(self, enable_rag: bool = False, rag_engine=None):
        # RAG 引擎（可以外部传入）
        self.rag_engine = rag_engine
        self.enable_rag = enable_rag

        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
        self.knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", "./data/knowledge"))

        # 确保目录存在
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def _get_rag_engine(self):
        """获取 RAG 引擎"""
        return self.rag_engine

    def set_rag_engine(self, rag_engine):
        """设置 RAG 引擎"""
        self.rag_engine = rag_engine
        self.enable_rag = (rag_engine is not None)

    def save_uploaded_file(self, file_content: bytes, filename: str, category: str) -> str:
        """
        保存上传的文件

        Args:
            file_content: 文件内容
            filename: 文件名
            category: 知识库类别

        Returns:
            保存的文件路径
        """
        # 创建类别目录
        category_dir = self.upload_dir / category
        category_dir.mkdir(exist_ok=True)

        # 保存文件
        file_path = category_dir / filename
        with open(file_path, "wb") as f:
            f.write(file_content)

        return str(file_path)

    def process_and_index_document(self, file_path: str, category: str) -> Dict[str, Any]:
        """
        处理并索引文档

        Args:
            file_path: 文件路径
            category: 知识库类别

        Returns:
            处理结果
        """
        rag_engine = self._get_rag_engine()

        if rag_engine is None:
            # RAG 未启用，返回模拟结果
            print(f"[WARN] RAG engine is None, document not indexed")
            return {
                "status": "success",
                "document_id": "mock-id",
                "chunks": 0,
                "category": category,
                "source": file_path,
                "message": "RAG is disabled - document saved but not indexed"
            }

        try:
            print(f"[INFO] Indexing document: {file_path} to category: {category}")
            result = rag_engine.index_document(file_path, category)
            print(f"[OK] Document indexed successfully: {result['chunks']} chunks")
            return result
        except Exception as e:
            # RAG 索引失败，但文档已保存
            print(f"[WARN] RAG 索引失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "partial_success",
                "document_id": "local-id",
                "chunks": 0,
                "category": category,
                "source": file_path,
                "message": f"文档已保存，但索引失败: {str(e)}"
            }

    def search_knowledge(self, query: str, category: str = None, k: int = 3) -> List[Dict[str, Any]]:
        """
        搜索知识库

        Args:
            query: 搜索查询
            category: 知识库类别
            k: 返回结果数量

        Returns:
            搜索结果
        """
        rag_engine = self._get_rag_engine()
        if rag_engine is None:
            # RAG 未启用，返回空结果
            return []
        docs = rag_engine.search(query, category, k)

        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "preview": doc.page_content[:200] + "..."
            })

        return results

    def list_documents(self, category: str = None) -> List[Dict[str, Any]]:
        """
        列出知识库中的文档

        Args:
            category: 知识库类别

        Returns:
            文档列表
        """
        # 扫描上传目录
        documents = []

        for category_dir in self.upload_dir.iterdir():
            if category_dir.is_dir():
                cat = category_dir.name

                # 如果指定了类别，只返回该类别的文档
                if category and category != cat:
                    continue

                for file_path in category_dir.glob("*.*"):
                    if file_path.is_file():
                        stat = file_path.stat()
                        # 使用 | 作为分隔符，避免与文件名中的连字符冲突
                        documents.append({
                            "id": f"{cat}|{file_path.name}",
                            "filename": file_path.name,
                            "category": cat,
                            "size": stat.st_size,
                            "upload_date": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })

        return documents

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息
        """
        documents = self.list_documents()

        # 按类别统计
        category_stats = {}
        for doc in documents:
            cat = doc["category"]
            category_stats[cat] = category_stats.get(cat, 0) + 1

        return {
            "total_documents": len(documents),
            "category_stats": category_stats,
            "categories": list(category_stats.keys())
        }

    def delete_document(self, document_id: str) -> bool:
        """
        删除文档

        Args:
            document_id: 文档ID（格式：category|filename）

        Returns:
            是否成功
        """
        # 解析 document_id (format: category|filename)
        parts = document_id.split("|", 1)
        if len(parts) != 2:
            print(f"[ERROR] Invalid document_id format: {document_id}")
            return False

        category, filename = parts

        # 删除文件
        file_path = self.upload_dir / category / filename
        if file_path.exists():
            file_path.unlink()
            print(f"[OK] Deleted file: {file_path}")
        else:
            print(f"[WARN] File not found: {file_path}")

        # 从 Milvus 删除（如果 RAG 启用）
        rag_engine = self._get_rag_engine()
        if rag_engine is None:
            # RAG 未启用，只删除文件
            return True
        return rag_engine.delete_documents(category)

    def get_category_path(self, category: str) -> Path:
        """获取类别目录路径"""
        return self.upload_dir / category


# 全局知识库管理器实例
_kb_manager = None


def get_kb_manager(enable_rag: bool = False, rag_engine=None) -> KnowledgeBaseManager:
    """
    获取知识库管理器实例（单例）

    Args:
        enable_rag: 是否启用 RAG
        rag_engine: 外部创建的 RAG 引擎（可选）
    """
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager(enable_rag=enable_rag, rag_engine=rag_engine)
    elif rag_engine is not None and _kb_manager.rag_engine is None:
        # 如果外部传入了 rag_engine，而管理器还没有引擎，则设置
        _kb_manager.set_rag_engine(rag_engine)
    return _kb_manager
