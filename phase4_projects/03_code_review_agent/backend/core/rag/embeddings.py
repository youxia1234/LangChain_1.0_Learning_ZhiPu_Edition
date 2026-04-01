"""向量化配置模块

配置 Embeddings 模型用于将编码规范文档向量化。
支持 HuggingFace 本地模型，自动配置 HF Mirror 加速国内下载。
"""

import os
import logging

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# 全局缓存，避免重复加载模型
_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """获取 Embeddings 实例（单例模式）

    使用 HuggingFace 本地模型进行文本向量化。
    首次调用时加载模型并缓存，后续调用直接返回缓存实例。

    Returns:
        HuggingFaceEmbeddings: 向量化实例
    """
    global _embeddings_instance

    if _embeddings_instance is not None:
        return _embeddings_instance

    # 配置 HF Mirror（国内加速）
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    if hf_endpoint and "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    model_name = os.getenv(
        "EMBEDDINGS_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    logger.info(f"加载 Embeddings 模型: {model_name}")

    _embeddings_instance = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embeddings 模型加载完成")
    return _embeddings_instance
