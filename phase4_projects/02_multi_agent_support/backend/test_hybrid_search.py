"""
混合检索测试脚本

测试 BM25 + 向量的混合检索功能，并与单一检索方法对比。
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from core.hybrid_rag import HybridRAGEngine


def test_hybrid_search():
    """测试混合检索功能"""
    print("=" * 70)
    print("混合检索测试")
    print("=" * 70)

    # 初始化混合检索引擎
    print("\n[1] 初始化混合检索引擎...")
    engine = HybridRAGEngine(bm25_weight=0.4, vector_weight=0.6)

    # 测试查询
    test_queries = [
        ("蓝牙耳机连接问题", "语义查询"),
        ("ORD001 订单状态", "精确匹配"),
        ("BM25 算法原理", "专有名词"),
        ("如何优化性能", "概念查询")
    ]

    print("\n[2] 执行检索测试...")
    print("-" * 70)

    for query, query_type in test_queries:
        print(f"\n查询: {query}")
        print(f"类型: {query_type}")
        print("-" * 40)

        # 使用混合检索
        results = engine.search_hybrid(query, k=3)

        if results:
            print(f"找到 {len(results)} 个结果:")
            for i, doc in enumerate(results, 1):
                content = doc.page_content
                # 截断过长内容
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  {i}. {content}")
        else:
            print("  未找到相关结果")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


def test_comparison():
    """对比不同检索方法"""
    print("\n" + "=" * 70)
    print("检索方法对比测试")
    print("=" * 70)

    engine = HybridRAGEngine()

    # 对比测试
    query = "蓝牙耳机无法连接"
    print(f"\n查询: {query}")
    print("-" * 70)

    try:
        comparison = engine.compare_retrieval_methods(query, k=3)

        for method, data in comparison["methods"].items():
            print(f"\n{method.upper()} 检索:")
            print(f"  结果数量: {data['count']}")
            if data['docs']:
                for i, doc in enumerate(data['docs'], 1):
                    print(f"  {i}. {doc}")
    except Exception as e:
        print(f"对比测试失败: {e}")


def test_weight_adjustment():
    """测试不同权重配置"""
    print("\n" + "=" * 70)
    print("权重配置测试")
    print("=" * 70)

    query = "如何优化系统性能"

    # 测试不同权重
    weights = [
        (0.3, 0.7, "偏向语义"),
        (0.5, 0.5, "平衡"),
        (0.7, 0.3, "偏向精确")
    ]

    for bm25_w, vector_w, desc in weights:
        print(f"\n{desc} (BM25={bm25_w}, Vector={vector_w}):")
        print("-" * 40)

        engine = HybridRAGEngine(bm25_weight=bm25_w, vector_weight=vector_w)
        results = engine.search_hybrid(query, k=2)

        if results:
            for i, doc in enumerate(results, 1):
                content = doc.page_content[:80] + "..."
                print(f"  {i}. {content}")


if __name__ == "__main__":
    # 基础测试
    test_hybrid_search()

    # 对比测试
    # test_comparison()

    # 权重测试
    # test_weight_adjustment()

    print("\n提示：取消注释以运行更多测试")
