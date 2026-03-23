"""
测试 Milvus 导入
"""
import os
import sys
from pathlib import Path

print("=" * 60)
print("Milvus 导入测试")
print("=" * 60)

# 检查 pymilvus
print("\n1. 检查 pymilvus:")
try:
    import pymilvus
    print(f"   [OK] pymilvus 版本: {pymilvus.__version__}")
except ImportError as e:
    print(f"   [FAIL] pymilvus: {e}")
    sys.exit(1)

# 检查 langchain-milvus
print("\n2. 检查 langchain-milvus:")
try:
    from langchain_milvus import Milvus
    print(f"   [OK] langchain-milvus.Milvus")
    print(f"        模块: {Milvus.__module__}")
except ImportError as e:
    print(f"   [FAIL] langchain-milvus: {e}")

# 检查 langchain-community
print("\n3. 检查 langchain-community:")
try:
    from langchain_community.vectorstores import Milvus as OldMilvus
    print(f"   [OK] langchain_community.vectorstores.Milvus (旧版)")
    print(f"        模块: {OldMilvus.__module__}")
except ImportError as e:
    print(f"   [FAIL] langchain-community: {e}")

# 测试 Milvus 初始化
print("\n4. 测试 Milvus 初始化:")
try:
    from langchain_openai import OpenAIEmbeddings
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    embeddings = OpenAIEmbeddings(
        model="embedding-2",
        openai_api_key=os.getenv("ZHIPUAI_API_KEY", "test"),
        openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
    )

    from langchain_milvus import Milvus

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": "./test_milvus.db"},
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        drop_old=True
    )
    print(f"   [OK] Milvus 初始化成功")
except Exception as e:
    print(f"   [FAIL] Milvus 初始化失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
