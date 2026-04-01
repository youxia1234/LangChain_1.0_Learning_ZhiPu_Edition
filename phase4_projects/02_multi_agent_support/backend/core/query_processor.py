"""
查询处理器模块

实现 RAG 查询优化：
- Query Rewriting：将用户口语化提问改写为精确检索词
- Multi-Query：生成多个查询变体，扩大召回率
- Query Classification：判断查询类型，动态选择检索策略
"""

import os
import json
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
    temperature=0.3,
)


# ==================== 查询类型枚举 ====================

QueryType = Literal["exact_match", "semantic", "hybrid"]


# ==================== 查询重写器 ====================

class QueryRewriter:
    """
    查询重写器

    将用户的口语化提问改写为适合检索的精确查询词。
    """

    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一个查询优化专家。你的任务是将用户的口语化提问改写为适合知识库检索的精确查询词。

改写规则：
1. 提取核心实体和关键词（产品名、型号、技术术语等）
2. 去除口语化表达（"你们"、"怎么"、"什么"等）
3. 补充隐含的专业术语
4. 保持简洁，用空格分隔关键词

示例：
- "你们公司是做什么的" → "公司主营业务 产品线 行业领域"
- "有没有防水的手表" → "防水手表 IP等级 产品型号"
- "质量怎么样" → "质量管理体系 ISO认证 产品检测"
- "能不能定制" → "OEM ODM 定制化生产 贴牌代工"
- "蓝牙耳机多少钱" → "蓝牙耳机 价格 报价 型号对比"

只输出改写后的查询词，不要其他内容。"""),
        ("human", "{query}")
    ])

    def __init__(self, llm=None):
        self.llm = llm or _llm
        self.chain = self.REWRITE_PROMPT | self.llm | StrOutputParser()

    def rewrite(self, query: str) -> str:
        """
        改写查询

        Args:
            query: 原始查询

        Returns:
            改写后的查询词
        """
        try:
            result = self.chain.invoke({"query": query})
            rewritten = result.strip()
            print(f"[QueryRewriter] '{query}' → '{rewritten}'")
            return rewritten
        except Exception as e:
            print(f"[QueryRewriter] 重写失败: {e}，使用原始查询")
            return query


# ==================== 多查询生成器 ====================

class MultiQueryGenerator:
    """
    多查询生成器

    为原始查询生成多个变体，扩大检索召回率。
    """

    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """你是一个查询扩展专家。为用户的原始查询生成 3 个不同角度的查询变体，用于从知识库中检索更全面的信息。

变体要求：
- 从不同角度表达相同的信息需求
- 使用同义词和相关术语
- 一个精确匹配，一个语义扩展，一个宽泛搜索

返回格式（JSON 数组）：
["变体1", "变体2", "变体3"]

只返回 JSON 数组，不要其他内容。"""),
        ("human", "{query}")
    ])

    def __init__(self, llm=None):
        self.llm = llm or _llm
        self.chain = self.MULTI_QUERY_PROMPT | self.llm | StrOutputParser()

    def generate(self, query: str, num_variants: int = 3) -> List[str]:
        """
        生成查询变体

        Args:
            query: 原始查询
            num_variants: 变体数量

        Returns:
            查询变体列表（包含原始查询）
        """
        try:
            result = self.chain.invoke({"query": query})
            result = result.strip()

            # 去除可能的 markdown 代码块
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            variants = json.loads(result)

            if isinstance(variants, list):
                # 确保不超过指定数量
                variants = variants[:num_variants]
                print(f"[MultiQuery] '{query}' → {len(variants)} 个变体")
                return variants

        except (json.JSONDecodeError, Exception) as e:
            print(f"[MultiQuery] 生成失败: {e}，使用原始查询")

        return [query]


# ==================== 查询分类器 ====================

class QueryClassifier:
    """
    查询分类器

    判断查询类型，动态选择检索策略：
    - exact_match：包含具体型号、编号、参数 → 侧重 BM25
    - semantic：模糊描述、概念性提问 → 侧重向量检索
    - hybrid：混合型 → 平衡检索
    """

    CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """分析用户查询，判断检索策略类型。

分类标准：
- exact_match：包含具体型号、产品编号、认证标准号、精确参数（如 "WX-200"、"ISO9001"、"IP67"、"CE认证"）
- semantic：模糊描述、概念性提问（如 "质量怎么样"、"有什么优势"、"技术实力"）
- hybrid：同时包含具体和模糊信息（如 "WX-200和ZX-100哪个防水好"）

返回 JSON：
{"type": "exact_match 或 semantic 或 hybrid", "reason": "分类原因"}

只返回 JSON。"""),
        ("human", "{query}")
    ])

    def __init__(self, llm=None):
        self.llm = llm or _llm
        self.chain = self.CLASSIFY_PROMPT | self.llm | StrOutputParser()

    def classify(self, query: str) -> Dict[str, Any]:
        """
        分类查询

        Args:
            query: 用户查询

        Returns:
            {"type": "exact_match"|"semantic"|"hybrid", "reason": "..."}
        """
        try:
            result = self.chain.invoke({"query": query})
            result = result.strip()

            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()

            parsed = json.loads(result)

            if "type" in parsed and parsed["type"] in ["exact_match", "semantic", "hybrid"]:
                print(f"[QueryClassifier] '{query}' → {parsed['type']}")
                return parsed

        except (json.JSONDecodeError, Exception) as e:
            print(f"[QueryClassifier] 分类失败: {e}")

        # 默认使用混合检索
        return {"type": "hybrid", "reason": "分类失败，默认混合"}


# ==================== 查询处理器（统一接口） ====================

class QueryProcessor:
    """
    查询处理器

    统一封装查询重写、多查询生成、查询分类。
    提供一站式查询预处理。
    """

    def __init__(self, llm=None, enable_rewrite: bool = True, enable_multi_query: bool = False):
        """
        初始化查询处理器

        Args:
            llm: LLM 实例
            enable_rewrite: 是否启用查询重写
            enable_multi_query: 是否启用多查询生成
        """
        self.llm = llm or _llm
        self.enable_rewrite = enable_rewrite
        self.enable_multi_query = enable_multi_query

        self.rewriter = QueryRewriter(self.llm)
        self.multi_query = MultiQueryGenerator(self.llm)
        self.classifier = QueryClassifier(self.llm)

        print(f"[QueryProcessor] 初始化完成 (rewrite={enable_rewrite}, multi_query={enable_multi_query})")

    def process(self, query: str) -> Dict[str, Any]:
        """
        处理查询（一站式）

        Args:
            query: 用户原始查询

        Returns:
            {
                "original_query": 原始查询,
                "rewritten_query": 改写后的查询,
                "queries": [所有检索查询],
                "query_type": "exact_match"|"semantic"|"hybrid",
                "retrieval_weights": {"bm25": float, "vector": float}
            }
        """
        # 1. 查询分类
        classification = self.classifier.classify(query)
        query_type = classification["type"]

        # 2. 根据查询类型确定检索权重
        retrieval_weights = self._get_retrieval_weights(query_type)

        # 3. 查询重写
        rewritten_query = query
        if self.enable_rewrite:
            rewritten_query = self.rewriter.rewrite(query)

        # 4. 多查询生成
        queries = [rewritten_query]
        if self.enable_multi_query:
            variants = self.multi_query.generate(rewritten_query)
            queries.extend(variants)

        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "queries": queries,
            "query_type": query_type,
            "retrieval_weights": retrieval_weights,
        }

    def _get_retrieval_weights(self, query_type: str) -> Dict[str, float]:
        """
        根据查询类型确定 BM25 和向量检索的权重

        Args:
            query_type: 查询类型

        Returns:
            {"bm25": 权重, "vector": 权重}
        """
        if query_type == "exact_match":
            return {"bm25": 0.65, "vector": 0.35}  # 精确匹配 → 侧重 BM25
        elif query_type == "semantic":
            return {"bm25": 0.25, "vector": 0.75}  # 语义理解 → 侧重向量
        else:  # hybrid
            return {"bm25": 0.4, "vector": 0.6}    # 混合 → 平衡


# ==================== 便捷函数 ====================

def create_query_processor(
    enable_rewrite: bool = True,
    enable_multi_query: bool = False
) -> QueryProcessor:
    """创建查询处理器"""
    return QueryProcessor(
        enable_rewrite=enable_rewrite,
        enable_multi_query=enable_multi_query
    )
