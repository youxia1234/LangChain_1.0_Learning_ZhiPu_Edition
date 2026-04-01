"""
检索结果重排序模块

实现 RAG 检索后处理：
- LLM Re-ranking：使用 LLM 对检索结果二次评分排序
- 相似度阈值过滤：过滤低质量结果
- 上下文压缩：去重和压缩减少 token 消耗
- MMR 多样性：确保结果多样性
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from dotenv import load_dotenv

# 加载环境变量
env_path = Path(__file__).parent.parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# 禁用 LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

# 初始化 LLM
_llm = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/",
    temperature=0.1,
)


# ==================== LLM 重排序器 ====================

class LLMReranker:
    """
    使用 LLM 对检索结果进行二次评分排序

    原理：将查询和候选文档一起输入 LLM，让 LLM 判断每个文档与查询的相关性，
    输出排序后的文档索引。无需额外训练模型，适用于任意领域。
    """

    RERANK_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一个文档相关性评估专家。你需要根据用户查询，对以下文档片段进行相关性评分和排序。

评分标准（0-10 分）：
- 10 分：完全匹配，直接回答了查询
- 7-9 分：高度相关，包含大部分所需信息
- 4-6 分：部分相关，包含一些有用信息
- 1-3 分：几乎无关
- 0 分：完全不相关

返回 JSON 格式（按相关性从高到低排序）：
[{{"index": 文档编号, "score": 分数, "reason": "评分原因"}}]

只返回 JSON 数组。"""),
        ("human", """用户查询：{query}

文档列表：
{documents}

请评分并排序：""")
    ])

    def __init__(self, llm=None, top_k: int = 5, min_score: float = 4.0):
        """
        初始化重排序器

        Args:
            llm: LLM 实例
            top_k: 返回前 K 个结果
            min_score: 最低相关分数阈值
        """
        self.llm = llm or _llm
        self.top_k = top_k
        self.min_score = min_score
        self.chain = self.RERANK_PROMPT | self.llm | StrOutputParser()

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None,
        min_score: float = None
    ) -> List[Document]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回前 K 个（覆盖默认值）
            min_score: 最低分数阈值（覆盖默认值）

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        top_k = top_k or self.top_k
        min_score = min_score or self.min_score

        # 文档数量少于 2，无需重排序
        if len(documents) <= 2:
            return documents[:top_k]

        try:
            # 构建文档列表文本
            doc_texts = []
            for i, doc in enumerate(documents):
                # 截取前 300 字符避免过长
                content = doc.page_content[:300]
                doc_texts.append(f"[{i}] {content}")

            docs_text = "\n\n".join(doc_texts)

            # 调用 LLM 评分
            result = self.chain.invoke({
                "query": query,
                "documents": docs_text
            })

            # 解析结果
            result = result.strip()
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            scores = json.loads(result)

            # 按分数过滤和排序
            scored_docs = []
            for item in scores:
                idx = item.get("index", -1)
                score = item.get("score", 0)
                if 0 <= idx < len(documents) and score >= min_score:
                    # 将分数添加到元数据
                    doc = documents[idx]
                    doc.metadata["rerank_score"] = score
                    doc.metadata["rerank_reason"] = item.get("reason", "")
                    scored_docs.append((score, doc))

            # 按分数降序排序
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            reranked = [doc for _, doc in scored_docs[:top_k]]

            print(f"[Reranker] {len(documents)} → {len(reranked)} 个文档（阈值: {min_score}）")
            return reranked

        except Exception as e:
            print(f"[Reranker] 重排序失败: {e}，返回原始结果")
            return documents[:top_k]


# ==================== 上下文压缩器 ====================

class ContextCompressor:
    """
    上下文压缩器

    去除检索结果中的重复内容，压缩上下文减少 token 消耗。
    """

    @staticmethod
    def deduplicate(documents: List[Document], similarity_threshold: float = 0.85) -> List[Document]:
        """
        去除内容高度相似的文档

        Args:
            documents: 文档列表
            similarity_threshold: 相似度阈值（基于 Jaccard 相似度）

        Returns:
            去重后的文档列表
        """
        if not documents:
            return []

        def jaccard_similarity(text1: str, text2: str) -> float:
            """计算 Jaccard 相似度（基于字符级 n-gram）"""
            set1 = set(text1[i:i+3] for i in range(len(text1) - 2))
            set2 = set(text2[i:i+3] for i in range(len(text2) - 2))
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0

        unique_docs = [documents[0]]

        for doc in documents[1:]:
            is_duplicate = False
            for existing in unique_docs:
                similarity = jaccard_similarity(doc.page_content, existing.page_content)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)

        removed = len(documents) - len(unique_docs)
        if removed > 0:
            print(f"[Compressor] 去重: {len(documents)} → {len(unique_docs)} (去除 {removed} 个重复)")

        return unique_docs

    @staticmethod
    def truncate(documents: List[Document], max_chars: int = 4000) -> List[Document]:
        """
        截断文档内容，确保总字符数不超过限制

        Args:
            documents: 文档列表
            max_chars: 最大总字符数

        Returns:
            截断后的文档列表
        """
        total_chars = 0
        truncated = []

        for doc in documents:
            content_len = len(doc.page_content)
            if total_chars + content_len <= max_chars:
                truncated.append(doc)
                total_chars += content_len
            else:
                # 部分截取
                remaining = max_chars - total_chars
                if remaining > 100:  # 至少保留 100 字符
                    truncated_doc = Document(
                        page_content=doc.page_content[:remaining] + "...",
                        metadata=doc.metadata
                    )
                    truncated.append(truncated_doc)
                break

        return truncated


# ==================== 统一重排序器 ====================

class Reranker:
    """
    统一重排序器

    集成 LLM 重排序、去重、截断等后处理功能。
    """

    def __init__(
        self,
        enable_llm_rerank: bool = True,
        enable_dedup: bool = True,
        enable_truncate: bool = True,
        top_k: int = 5,
        min_score: float = 4.0,
        max_context_chars: int = 4000
    ):
        """
        初始化重排序器

        Args:
            enable_llm_rerank: 是否启用 LLM 重排序
            enable_dedup: 是否启用去重
            enable_truncate: 是否启用截断
            top_k: 返回前 K 个结果
            min_score: LLM 重排序最低分数
            max_context_chars: 最大上下文字符数
        """
        self.enable_llm_rerank = enable_llm_rerank
        self.enable_dedup = enable_dedup
        self.enable_truncate = enable_truncate
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chars = max_context_chars

        # 初始化子组件
        self.llm_reranker = LLMReranker(top_k=top_k, min_score=min_score) if enable_llm_rerank else None
        self.compressor = ContextCompressor()

        print(f"[Reranker] 初始化完成 (llm_rerank={enable_llm_rerank}, dedup={enable_dedup}, truncate={enable_truncate})")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Document]:
        """
        对检索结果进行后处理

        流程：去重 → LLM 重排序 → 截断

        Args:
            query: 查询文本
            documents: 检索到的文档列表
            top_k: 返回前 K 个

        Returns:
            处理后的文档列表
        """
        if not documents:
            return []

        top_k = top_k or self.top_k

        # Step 1: 去重
        if self.enable_dedup:
            documents = self.compressor.deduplicate(documents)

        # Step 2: LLM 重排序
        if self.enable_llm_rerank and self.llm_reranker:
            documents = self.llm_reranker.rerank(query, documents, top_k=top_k)

        # Step 3: 截断
        if self.enable_truncate:
            documents = self.compressor.truncate(documents, max_chars=self.max_context_chars)

        return documents[:top_k]


# ==================== 便捷函数 ====================

def create_reranker(
    enable_llm_rerank: bool = True,
    top_k: int = 5,
    min_score: float = 4.0,
    max_context_chars: int = 4000
) -> Reranker:
    """创建重排序器"""
    return Reranker(
        enable_llm_rerank=enable_llm_rerank,
        top_k=top_k,
        min_score=min_score,
        max_context_chars=max_context_chars
    )
