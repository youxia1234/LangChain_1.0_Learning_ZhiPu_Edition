"""
LangChain 1.0 基础教程 - 提示词模板 (Prompt Templates)
========================================================

本文件演示如何使用 LangChain 的提示词模板系统
涵盖以下核心概念：
1. PromptTemplate - 简单文本模板
2. ChatPromptTemplate - 聊天消息模板
3. 模板变量和格式化
4. 消息模板的组合
5. 实际应用场景

作者：LangChain 学习者
日期：2025-11
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "你的智谱API密钥填在这里":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取密钥"
    )

# 初始化模型 - 使用智谱清言 GLM-4-Flash
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ============================================================================
# 示例 1：为什么需要提示词模板？
# ============================================================================
def example_1_why_templates():
    """
    示例1：对比字符串拼接 vs 模板

    问题：字符串拼接容易出错、难维护、不可复用
    解决：使用提示词模板
    """
    print("\n" + "="*70)
    print("示例 1：为什么需要提示词模板？")
    print("="*70)

    # [X] 不推荐：使用字符串拼接
    print("\n【方式 1：字符串拼接（不推荐）】")
    print("-"*70)

    topic = "Python"
    difficulty = "初学者"

    # 难以维护，容易出错
    prompt_str = f"你是一个{difficulty}级别的编程导师。请用简单易懂的语言解释{topic}。"
    print(f"提示词：{prompt_str}")

    response = model.invoke(prompt_str)
    print(f"AI 回复：{response.content[:100]}...\n")

    # [OK] 推荐：使用 PromptTemplate
    print("【方式 2：使用 PromptTemplate（推荐）】")
    print("-"*70)

    # 创建可复用的模板
    template = PromptTemplate.from_template(
        "你是一个{difficulty}级别的编程导师。请用简单易懂的语言解释{topic}。"
    )
    
    print(f"模板：{template.template}")
    print(f"变量：{template.input_variables}")

    # 使用模板生成提示词
    prompt = template.format(difficulty=difficulty, topic=topic)
    print(f"生成的提示词：{prompt}")

    response = model.invoke(prompt)
    print(f"AI 回复：{response.content[:100]}...\n")

    print("[TIPS] 优势：")
    print("  1. 可复用 - 同一个模板可以用于不同的输入")
    print("  2. 可维护 - 模板和数据分离，易于修改")
    print("  3. 类型安全 - 自动验证变量")
    print("  4. 可测试 - 更容易编写测试用例")

# ============================================================================
# 示例 2：PromptTemplate 基础用法
# ============================================================================
def example_2_prompt_template_basics():
    """
    示例2：PromptTemplate 的基本用法

    PromptTemplate 用于简单的文本模板
    适合单一提示词的场景
    """
    print("\n" + "="*70)
    print("示例 2：PromptTemplate 基础用法")
    print("="*70)

    # 方法 1：使用 from_template（最简单）
    print("\n【方法 1：from_template（推荐）】")
    template1 = PromptTemplate.from_template(
        "将以下文本翻译成{language}：\n{text}"
    )

    prompt1 = template1.format(language="法语", text="Hello, how are you?")
    print(f"生成的提示词：\n{prompt1}\n")

    response1 = model.invoke(prompt1)
    print(f"AI 回复：{response1.content}\n")

    # 方法 2：显式指定变量（更严格）
    print("【方法 2：显式指定变量】")
    template2 = PromptTemplate(
        input_variables=["product", "feature"],
        template="为{product}写一句广告语，重点突出{feature}特点。"
    )

    prompt2 = template2.format(product="智能手表", feature="超长续航")
    print(f"生成的提示词：\n{prompt2}\n")

    response2 = model.invoke(prompt2)
    print(f"AI 回复：{response2.content}\n")

    # 方法 3：使用 invoke（直接生成消息）
    print("【方法 3：使用 invoke（更方便）】")
    template3 = PromptTemplate.from_template(
        "写一首关于{theme}的{style}风格的诗，不超过4行。"
    )

    # invoke 直接返回格式化后的值
    prompt_value = template3.invoke({"theme": "春天", "style": "现代"})
    print(f"生成的提示词：\n{prompt_value.text}\n")

# ============================================================================
# 示例 3：ChatPromptTemplate - 聊天消息模板
# ============================================================================
def example_3_chat_prompt_template():
    """
    示例3：ChatPromptTemplate 的基本用法

    ChatPromptTemplate 用于构建聊天消息
    支持 system、user、assistant 多种角色
    """
    print("\n" + "="*70)
    print("示例 3：ChatPromptTemplate - 聊天消息模板")
    print("="*70)

    # 方法 1：使用元组格式（最简单，推荐）
    print("\n【方法 1：元组格式（推荐）】")

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，擅长{expertise}。"),
        ("user", "请帮我{task}")
    ])

    print(f"模板变量：{chat_template.input_variables}")

    # 格式化模板
    messages = chat_template.format_messages(
        role="Python 导师",
        expertise="用简单的方式解释复杂概念",
        task="解释什么是列表推导式"
    )

    print("\n生成的消息：")
    for msg in messages:
        print(f"  {msg.type}: {msg.content}")

    response = model.invoke(messages)
    print(f"\nAI 回复：{response.content[:150]}...\n")

    # 方法 2：使用字符串简写（最简洁）
    print("【方法 2：字符串简写】")

    simple_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手"),
        ("user", "{question}")
    ])

    messages = simple_template.format_messages(question="什么是机器学习？")
    response = model.invoke(messages)
    print(f"AI 回复：{response.content[:100]}...\n")



# ============================================================================
# 示例 4：多轮对话模板
# ============================================================================
def example_4_conversation_template():
    """
    示例4：构建多轮对话的模板
    包含系统提示、对话历史和当前问题
    """
    print("\n" + "="*70)
    print("示例 4：多轮对话模板")
    print("="*70)

    # 创建包含对话历史的模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}。{instruction}"),
        ("user", "{question1}"),
        ("assistant", "{answer1}"),
        ("user", "{question2}")
    ])

    print("模板结构：")
    print("  1. System: 设定角色和指令")
    print("  2. User: 第一个问题")
    print("  3. Assistant: 第一个回答")
    print("  4. User: 第二个问题（基于上下文）\n")

    # 填充模板
    messages = template.format_messages(
        role="Python 专家",
        instruction="回答要简洁、准确",
        question1="什么是列表？",
        answer1="列表是 Python 中的有序可变集合，用方括号 [] 表示。",
        question2="它和元组有什么区别？"  # 基于上下文的问题
    )

    print("生成的完整对话：")
    for i, msg in enumerate(messages, 1):
        content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        print(f"  {i}. [{msg.type}] {content_preview}")

    response = model.invoke(messages)
    print(f"\nAI 回复：{response.content}\n")

# ============================================================================
# 示例 5：使用 MessagePromptTemplate（高级）
# ============================================================================
def example_5_message_templates():
    """
    示例5：使用 MessagePromptTemplate 类

    提供更细粒度的控制
    """
    print("\n" + "="*70)
    print("示例 5：MessagePromptTemplate 类（高级用法）")
    print("="*70)

    # 分别创建不同类型的消息模板
    system_template = SystemMessagePromptTemplate.from_template(
        "你是一个{profession}，你的特长是{specialty}。"
    )

    human_template = HumanMessagePromptTemplate.from_template(
        "关于{topic}，我想知道{question}"
    )

    # 组合成 ChatPromptTemplate
    chat_template = ChatPromptTemplate.from_messages([
        system_template,
        human_template
    ])

    print("模板组件：")
    print(f"  1. SystemMessagePromptTemplate")
    print(f"  2. HumanMessagePromptTemplate")
    print(f"\n总变量：{chat_template.input_variables}\n")

    # 使用模板
    messages = chat_template.format_messages(
        profession="数据科学家",
        specialty="用数据讲故事",
        topic="数据可视化",
        question="如何选择合适的图表类型？"
    )

    response = model.invoke(messages)
    print(f"AI 回复：{response.content[:200]}...\n")

# ============================================================================
# 示例 6：部分变量（Partial Variables）
# ============================================================================
def example_6_partial_variables():
    """
    示例6：部分变量 - 预填充某些变量

    适用场景：
    - 某些变量固定不变
    - 需要创建模板变体
    """
    print("\n" + "="*70)
    print("示例 6：部分变量（Partial Variables）")
    print("="*70)

    # 创建原始模板
    original_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，你的目标用户是{audience}。"),
        ("user", "请{task}")
    ])

    print(f"原始模板变量：{original_template.input_variables}\n")

    # 部分填充：固定 role 和 audience
    partially_filled = original_template.partial(
        role="科技博客作者",
        audience="程序员"
    )

    print(f"部分填充后的变量：{partially_filled.input_variables}\n")

    # 现在只需要提供 task
    messages1 = partially_filled.format_messages(
        task="写一篇关于 Python 装饰器的文章开头"
    )

    response1 = model.invoke(messages1)
    print(f"文章 1：{response1.content[:150]}...\n")

    # 复用模板，不同的 task
    messages2 = partially_filled.format_messages(
        task="写一篇关于异步编程的文章开头"
    )

    response2 = model.invoke(messages2)
    print(f"文章 2：{response2.content[:150]}...\n")

# ============================================================================
# 示例 9：与 LCEL 链式调用（预览）
# ============================================================================
def example_9_lcel_chains():
    """
    示例9：模板 + 模型的链式调用

    LangChain Expression Language (LCEL)
    """
    print("\n" + "="*70)
    print("示例 9：LCEL 链式调用（预览）")
    print("="*70)

    # 创建模板
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}"),
        ("user", "{input}")
    ])

    # 使用 | 运算符创建链
    chain = template | model

    print("链的组成：")
    print("  模板 | 模型")
    print("  (Template) | (LLM)\n")

    # 直接调用链
    response = chain.invoke({
        "role": "幽默的程序员",
        "input": "解释什么是bug"
    })

    print(f"AI 回复：{response.content}\n")

    print("[TIPS] 链式调用的优势：")
    print("  1. 代码更简洁")
    print("  2. 组件可复用")
    print("  3. 易于调试和监控")
    print("  （详细内容将在后续模块学习）")

# ============================================================================
# 主程序
# ============================================================================
def main():
    """运行所有示例"""
    print("\n" + "="*70)
    print(" LangChain 1.0 基础教程 - 提示词模板")
    print("="*70)

    try:
        example_1_why_templates()
        print("\n---\n")

        example_2_prompt_template_basics()
        print("\n---\n")

        example_3_chat_prompt_template()
        print("\n---\n")

        example_4_conversation_template()
        print("\n---\n")

        example_5_message_templates()
        print("\n---\n")

        example_6_partial_variables()
        print("\n---\n")

        example_9_lcel_chains()

        print("\n" + "="*70)
        print(" 所有示例运行完成！")
        print("="*70)
        print("\n你已经学会了：")
        print("  [OK] PromptTemplate 基础用法")
        print("  [OK] ChatPromptTemplate 聊天模板")
        print("  [OK] 多轮对话模板")
        print("  [OK] 部分变量填充")
        print("  [OK] 模板组合")
        print("  [OK] 可复用模板库")
        print("  [OK] LCEL 链式调用预览")
        print("\n下一步学习：")
        print("  - 03_messages: 深入理解消息类型")
        print("  - 04_custom_tools: 创建自定义工具")

    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n运行出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
