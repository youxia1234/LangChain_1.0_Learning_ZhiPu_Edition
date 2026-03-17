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

    def __init__(self, enable_rag: bool = False):
        # RAG 引擎默认不初始化（避免网络连接问题）
        self.rag_engine = None
        self.enable_rag = enable_rag

        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
        self.knowledge_dir = Path(os.getenv("KNOWLEDGE_DIR", "./data/knowledge"))

        # 确保目录存在
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def _get_rag_engine(self):
        """懒加载 RAG 引擎"""
        if self.rag_engine is None and self.enable_rag:
            from .rag import RAGEngine
            self.rag_engine = RAGEngine()
        return self.rag_engine

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
            return {
                "status": "success",
                "document_id": "mock-id",
                "chunks": 0,
                "category": category,
                "source": file_path,
                "message": "RAG is disabled - document saved but not indexed"
            }

        try:
            return rag_engine.index_document(file_path, category)
        except Exception as e:
            # RAG 索引失败，但文档已保存
            print(f"[WARN] RAG 索引失败: {e}")
            print(f"[INFO] 文档已保存到本地，但未索引到 Pinecone")
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
                        documents.append({
                            "id": f"{cat}-{file_path.name}",
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
            document_id: 文档ID

        Returns:
            是否成功
        """
        # 解析 document_id (format: category-filename)
        parts = document_id.split("-", 1)
        if len(parts) != 2:
            return False

        category, filename = parts

        # 删除文件
        file_path = self.upload_dir / category / filename
        if file_path.exists():
            file_path.unlink()

        # 从 Pinecone 删除（如果 RAG 启用）
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


def get_kb_manager(enable_rag: bool = False) -> KnowledgeBaseManager:
    """获取知识库管理器实例（单例）"""
    global _kb_manager
    if _kb_manager is None:
        _kb_manager = KnowledgeBaseManager(enable_rag=enable_rag)
    return _kb_manager
