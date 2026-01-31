"""
简单测试：验证内存功能
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY"
    )

# 初始化模型（使用智谱 AI）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


print("=" * 70)
print("测试：InMemorySaver 内存功能")
print("=" * 70)

# 创建带内存的 Agent
agent = create_agent(
    model=model,
    tools=[],
    system_prompt="你是一个有帮助的助手。",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "test_session"}}

print("\n第一轮对话：")
print("用户: 我叫张三")
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫张三"}]},
    config=config
)
print(f"Agent: {response1['messages'][-1].content}")

print("\n第二轮对话：")
print("用户: 我叫什么？")
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么？"}]},
    config=config
)
print(f"Agent: {response2['messages'][-1].content}")

print("\n" + "=" * 70)
print("内存状态：")
print(f"  总消息数: {len(response2['messages'])}")
print(f"  thread_id: {config['configurable']['thread_id']}")
print("=" * 70)

if "张三" in response2['messages'][-1].content:
    print("\n测试成功！Agent 记住了名字。")
else:
    print("\n警告：Agent 可能没有正确记住")

print("\n测试完成！")
