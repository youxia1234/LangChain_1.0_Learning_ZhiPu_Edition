"""
流式输出测试脚本

测试 /api/chat/stream 接口的流式输出功能
"""

import requests
import json

# 配置
API_BASE_URL = "http://localhost:8000"

def test_stream_chat():
    """测试流式聊天接口"""
    print("=" * 60)
    print("流式输出测试")
    print("=" * 60)

    # 测试消息
    test_message = "我的蓝牙耳机无法连接怎么办？"

    print(f"\n用户消息: {test_message}")
    print("\n" + "=" * 60)
    print("流式响应:")
    print("=" * 60 + "\n")

    try:
        # 发送 POST 请求到流式接口
        response = requests.post(
            f"{API_BASE_URL}/api/chat/stream",
            json={"message": test_message},
            stream=True,  # 启用流式接收
            timeout=60
        )

        response.raise_for_status()

        # 逐行读取 SSE 数据
        for line in response.iter_lines():
            if line:
                # 解码并解析 SSE 数据
                line_str = line.decode('utf-8')

                # SSE 格式：data: <json>\n\n
                if line_str.startswith('data: '):
                    json_str = line_str[6:]  # 移除 "data: " 前缀

                    try:
                        chunk = json.loads(json_str)

                        # 根据类型处理不同的 chunk
                        chunk_type = chunk.get("type", "unknown")

                        if chunk_type == "intent":
                            print(f"[意图] {chunk['intent']} (置信度: {chunk['confidence']:.2f})")

                        elif chunk_type == "content":
                            print(f"[内容] {chunk['content'][:100]}...")

                        elif chunk_type == "quality":
                            print(f"[质量] 评分: {chunk['quality_score']:.2f}")

                        elif chunk_type == "escalate":
                            print(f"[升级] {chunk['content'][:100]}...")

                        elif chunk_type == "final":
                            print("\n" + "=" * 60)
                            print("[最终响应]")
                            print("=" * 60)
                            print(f"\n完整回复:\n{chunk['response']}")
                            print(f"\n意图: {chunk['intent']}")
                            print(f"置信度: {chunk['confidence']:.2f}")
                            print(f"质量评分: {chunk['quality_score']:.2f}")
                            print(f"是否升级: {chunk['escalated']}")
                            if chunk.get('sources'):
                                print(f"\n来源信息: {len(chunk['sources'])} 个")

                            print("\n" + "=" * 60)
                            print("[测试完成]")
                            print("=" * 60)
                            break

                        elif chunk_type == "error":
                            print(f"[错误] {chunk['error']}")
                            break

                    except json.JSONDecodeError as e:
                        print(f"[解析错误] {e}")
                        print(f"原始数据: {json_str}")

    except requests.exceptions.RequestException as e:
        print(f"[请求错误] {e}")
    except Exception as e:
        print(f"[未知错误] {e}")


def test_normal_chat():
    """测试普通聊天接口（对比）"""
    print("\n\n" + "=" * 60)
    print("普通接口对比测试")
    print("=" * 60)

    test_message = "我的蓝牙耳机无法连接怎么办？"

    print(f"\n用户消息: {test_message}")
    print("\n" + "=" * 60)
    print("同步响应:")
    print("=" * 60 + "\n")

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"message": test_message},
            timeout=60
        )

        response.raise_for_status()
        result = response.json()

        print(f"\n完整回复:\n{result['response']}")
        print(f"\n意图: {result['intent']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"质量评分: {result['quality_score']:.2f}")

    except Exception as e:
        print(f"[错误] {e}")


if __name__ == "__main__":
    # 测试流式输出
    test_stream_chat()

    # 对比测试普通接口
    test_normal_chat()

    print("\n\n提示：对比两次测试可以看到流式输出的实时反馈效果")
