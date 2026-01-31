"""
invoke 方法深入实践 - 配合 README.md 学习
==========================================

本文件提供 invoke 方法的实战练习代码
建议先阅读 README.md 中的 "invoke 方法 - 调用模型（深入详解）" 部分
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()
GROQ_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 GROQ_API_KEY\n"
        "访问 https://console.groq.com/keys 获取免费密钥"
    )

# 初始化模型
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=GROQ_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here_replace_this":
    print("请先在 .env 文件中设置有效的 GROQ_API_KEY")
    exit(1)

# 初始化模型
# model 已在文件开头通过 get_model() 初始化

print("="*70)
print(" invoke 方法深入实践")
print("="*70)

# ============================================================================
# 练习 1：理解三种输入格式
# ============================================================================
def exercise_1_input_formats():
    """
    练习目标：理解 invoke 的三种输入格式
    - 格式 1：纯字符串
    - 格式 2：字典列表（推荐）
    - 格式 3：消息对象
    """
    print("\n" + "="*70)
    print("练习 1：三种输入格式对比")
    print("="*70)

    # 格式 1：纯字符串
    print("\n【格式 1：纯字符串】")
    print("代码：model.invoke('什么是Python？')")
    # # response1 = model.invoke("什么是 Python？用一句话回答")
    # print(f"回复：{response1.content}\n")
    response1 = model.invoke("什么是python？用一句话回答")
    print(f"{response1.content}\n")
    # 格式 2：字典列表（推荐）
    print("【格式 2：字典列表（推荐）】")
    print("代码：model.invoke([{'role': 'system', ...}, {'role': 'user', ...}])")
    messages2 = [
        {"role": "system", "content": "你是一个简洁的助手，回答限制在30字以内"},
        {"role": "user", "content": "什么是 Python？"}
    ]
    response2 = model.invoke(messages2)
    
    print(f"回复：{response2.content}\n")

    # 格式 3：消息对象
    print("【格式 3：消息对象】")
    print("代码：model.invoke([SystemMessage(...), HumanMessage(...)])")
    from langchain_core.messages import SystemMessage, HumanMessage
    messages3 = [
        SystemMessage(content="你是一个幽默的助手，喜欢用比喻"),
        HumanMessage(content="什么是 Python？")
    ]
    response3 = model.invoke(messages3)
    print(f"回复：{response3.content}\n")

    print("💡 观察：三种格式的回复有何不同？")
    print("   - 格式1：无系统提示，回复较长")
    print("   - 格式2：有系统提示（简洁），回复较短")
    print("   - 格式3：有系统提示（幽默），回复风格不同")

# ============================================================================
# 练习 2：系统提示的威力
# ============================================================================
def exercise_2_system_prompt():
    """
    练习目标：理解 system 角色的作用
    通过不同的系统提示，让 AI 扮演不同角色
    """
    print("\n" + "="*70)
    print("练习 2：系统提示的威力")
    print("="*70)

    question = "什么是递归？"

    # 角色 1：专业教师
    print(f"\n问题：{question}\n")
    print("【角色 1：专业教师】")
    messages = [
        {"role": "system", "content": "你是一个严肃的计算机科学教授，回答要学术化、专业化"},
        {"role": "user", "content": question}
    ]
    response1 = model.invoke(messages)
    print(f"{response1.content}\n")

    # 角色 2：5岁小孩的老师
    print("【角色 2：儿童教育者】")
    messages = [
        {"role": "system", "content": "你在给5岁小孩解释概念，要用简单的语言和生动的比喻"},
        {"role": "user", "content": question}
    ]
    response2 = model.invoke(messages)
    print(f"{response2.content}\n")

    # 角色 3：诗人
    print("【角色 3：诗人】")
    messages = [
        {"role": "system", "content": "你是一个诗人，喜欢用诗歌的形式回答问题"},
        {"role": "user", "content": question}
    ]
    response3 = model.invoke(messages)
    print(f"{response3.content}\n")

    print("💡 体会：同一个问题，不同的系统提示，得到完全不同的回答！")

# ============================================================================
# 练习 3：多轮对话 - 理解对话历史
# ============================================================================
def exercise_3_conversation():
    """
    练习目标：理解如何构建多轮对话
    关键：每次都要传递完整的对话历史
    """
    print("\n" + "="*70)
    print("练习 3：多轮对话实践")
    print("="*70)

    # 初始化对话
    conversation = [
        {"role": "system", "content": "你是一个友好的 Python 助手"}
    ]

    # 第一轮对话
    print("\n【第 1 轮】")
    conversation.append({"role": "user", "content": "我想学习 Python，从哪里开始？"})
    print(f"用户：{conversation[-1]['content']}")

    response1 = model.invoke(conversation)
    print(f"AI：{response1.content}")

    # 保存 AI 的回复到历史
    conversation.append({"role": "assistant", "content": response1.content})

    # 第二轮对话
    print("\n【第 2 轮】")
    conversation.append({"role": "user", "content": "那数据类型有哪些？"})
    print(f"用户：{conversation[-1]['content']}")

    response2 = model.invoke(conversation)
    print(f"AI：{response2.content}")

    conversation.append({"role": "assistant", "content": response2.content})

    # 第三轮对话 - 测试上下文记忆
    print("\n【第 3 轮 - 测试记忆】")
    conversation.append({"role": "user", "content": "我刚才第一个问题问的是什么？"})
    print(f"用户：{conversation[-1]['content']}")

    response3 = model.invoke(conversation)
    print(f"AI：{response3.content}")

    # 打印完整对话历史
    print("\n" + "-"*70)
    print("完整对话历史：")
    for i, msg in enumerate(conversation, 1):
        role = msg['role']
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  {i}. [{role}] {content}")

    print(f"\n💡 观察：对话列表包含 {len(conversation)} 条消息")
    print("   AI 能记住之前的对话，因为我们每次都传递了完整历史！")

# ============================================================================
# 练习 4：错误的多轮对话示例
# ============================================================================
def exercise_4_wrong_conversation():
    """
    练习目标：理解为什么必须传递对话历史
    演示：如果不传递历史，AI 会"失忆"
    """
    print("\n" + "="*70)
    print("练习 4：错误示例 - AI 失忆")
    print("="*70)

    print("\n【错误做法：不保存对话历史】\n")

    # 第一次对话
    print("第 1 轮：")
    response1 = model.invoke("我叫张三")
    print(f"用户：我叫张三")
    print(f"AI：{response1.content}\n")

    # 第二次对话 - 没有传递历史
    print("第 2 轮：")
    response2 = model.invoke("我叫什么名字？")
    print(f"用户：我叫什么名字？")
    print(f"AI：{response2.content}\n")

    print("❌ 问题：AI 不记得你叫张三！因为没有传递对话历史\n")

    print("【正确做法：保存并传递对话历史】\n")

    conversation = []

    # 第一次对话
    print("第 1 轮：")
    conversation.append({"role": "user", "content": "我叫李四"})
    print(f"用户：我叫李四")

    response1 = model.invoke(conversation)
    print(f"AI：{response1.content}")

    conversation.append({"role": "assistant", "content": response1.content})

    # 第二次对话 - 传递了历史
    print("\n第 2 轮：")
    conversation.append({"role": "user", "content": "我叫什么名字？"})
    print(f"用户：我叫什么名字？")

    response2 = model.invoke(conversation)
    print(f"AI：{response2.content}\n")

    print("✅ 成功：AI 记住了你叫李四！")
    print("\n💡 关键：必须传递完整的对话历史列表")

# ============================================================================
# 练习 5：理解返回值
# ============================================================================
def exercise_5_response_structure():
    """
    练习目标：理解 invoke 返回的 AIMessage 对象
    学会访问各种有用的信息
    """
    print("\n" + "="*70)
    print("练习 5：深入理解返回值")
    print("="*70)

    response = model.invoke("用20个字解释什么是人工智能")

    print("\n返回值是一个 AIMessage 对象，包含丰富的信息：\n")

    # 1. 主要内容
    print("【1. 主要内容】")
    print(f"response.content = '{response.content}'")
    print(f"类型：{type(response.content)}\n")

    # 2. 消息 ID
    print("【2. 消息 ID】")
    print(f"response.id = '{response.id}'")
    print("用途：追踪特定的对话消息\n")

    # 3. 响应元数据
    print("【3. 响应元数据】")
    metadata = response.response_metadata
    print(f"模型名称：{metadata.get('model_name')}")
    print(f"结束原因：{metadata.get('finish_reason')}")
    print(f"模型提供商：{metadata.get('model_provider')}\n")

    # 4. Token 使用情况
    print("【4. Token 使用情况】")
    usage = metadata.get('token_usage', {})
    print(f"输入 tokens：{usage.get('prompt_tokens')}")
    print(f"输出 tokens：{usage.get('completion_tokens')}")
    print(f"总计 tokens：{usage.get('total_tokens')}")
    print(f"输入处理时间：{usage.get('prompt_time'):.4f} 秒")
    print(f"输出生成时间：{usage.get('completion_time'):.4f} 秒\n")

    # 5. 计算成本（示例）
    print("【5. 成本估算（假设每千tokens $0.1）】")
    total_tokens = usage.get('total_tokens', 0)
    cost = total_tokens / 1000 * 0.1
    print(f"本次调用成本：${cost:.6f}")

    print("\n💡 提示：Token 统计对成本控制很重要！")

# ============================================================================
# 练习 6：实战 - 构建一个简单的聊天机器人
# ============================================================================
def exercise_6_chatbot():
    """
    练习目标：综合运用所学知识，构建一个简单的聊天机器人
    """
    print("\n" + "="*70)
    print("练习 6：实战 - 简单聊天机器人")
    print("="*70)

    print("\n这是一个简单的聊天机器人示例")
    print("它会记住对话历史，并统计 token 使用情况")
    print("输入 'quit' 退出\n")

    # 初始化对话
    conversation = [
        {"role": "system", "content": "你是一个友好、幽默的助手，喜欢帮助用户"}
    ]

    total_tokens_used = 0
    turn = 0

    # 模拟几轮对话（非交互式）
    demo_questions = [
        "你好！",
        "你能做什么？",
        "告诉我一个编程笑话",
        "我想学 Python，有什么建议吗？"
    ]

    for question in demo_questions:
        turn += 1
        print(f"\n--- 第 {turn} 轮 ---")
        print(f"用户：{question}")

        # 添加用户消息
        conversation.append({"role": "user", "content": question})

        # 调用模型
        response = model.invoke(conversation)

        # 显示 AI 回复
        print(f"AI：{response.content}")

        # 统计 token
        usage = response.response_metadata.get('token_usage', {})
        tokens = usage.get('total_tokens', 0)
        total_tokens_used += tokens
        print(f"[本轮使用 {tokens} tokens，累计 {total_tokens_used} tokens]")

        # 保存 AI 回复到历史
        conversation.append({"role": "assistant", "content": response.content})

    print("\n" + "="*70)
    print(f"对话结束！共进行 {turn} 轮对话")
    print(f"总计使用 {total_tokens_used} tokens")
    print(f"对话历史包含 {len(conversation)} 条消息")
    print("="*70)

# ============================================================================
# 运行所有练习
# ============================================================================
def main():
    """运行所有练习"""
    try:
        exercise_1_input_formats()

        input("\n按 Enter 继续下一个练习...")
        exercise_2_system_prompt()

        input("\n按 Enter 继续下一个练习...")
        exercise_3_conversation()

        input("\n按 Enter 继续下一个练习...")
        exercise_4_wrong_conversation()

        input("\n按 Enter 继续下一个练习...")
        exercise_5_response_structure()

        input("\n按 Enter 继续下一个练习...")
        exercise_6_chatbot()

        print("\n" + "="*70)
        print(" 🎉 所有练习完成！")
        print("="*70)
        print("\n你已经掌握了 invoke 方法的核心用法：")
        print("  ✅ 三种输入格式")
        print("  ✅ 系统提示的作用")
        print("  ✅ 多轮对话的实现")
        print("  ✅ 对话历史的管理")
        print("  ✅ 返回值的解析")
        print("  ✅ Token 使用统计")
        print("\n建议：")
        print("  1. 重新运行这个文件，仔细观察每个输出")
        print("  2. 修改代码，尝试不同的系统提示")
        print("  3. 阅读 README.md 的详细文档")
        print("  4. 继续学习下一个模块：02_prompt_templates")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n运行出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
