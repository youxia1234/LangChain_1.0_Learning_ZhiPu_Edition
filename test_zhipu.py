"""
测试智谱清言 API 配置
====================
运行此脚本来验证你的智谱 API Key 是否配置正确
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "你的智谱API密钥填在这里":
    print("请先在 .env 文件中填入你的智谱 API Key")
    print("\n步骤：")
    print("1. 打开 .env 文件")
    print("2. 将 ZHIPUAI_API_KEY=你的智谱API密钥填在这里")
    print("3. 改成 ZHIPUAI_API_KEY=你实际获取到的密钥")
    print("\n获取密钥地址：https://open.bigmodel.cn/usercenter/apikeys")
    exit(1)

print("="*60)
print("测试智谱清言 API 连接")
print("="*60)

# 智谱 AI 支持的模型：
# glm-4-flash (免费，速度快)
# glm-4-plus (强大)
# glm-4-0520
# glm-4-air (经济实惠)
# glm-4 (通用)

print("\n正在初始化 GLM 模型...")
print("使用模型: glm-4-flash (免费版)")

try:
    # 使用 ChatOpenAI 兼容智谱 API
    model = ChatOpenAI(
        model="glm-4-flash",
        api_key=ZHIPUAI_API_KEY,
        base_url="https://open.bigmodel.cn/api/paas/v4/"
    )

    print("\n[OK] 模型初始化成功!\n")

    # 测试调用
    print("发送测试消息...")
    response = model.invoke("你好！请用一句话介绍你自己。")

    print("="*60)
    print("GLM 模型的回复：")
    print("="*60)
    print(response.content)
    print("\n[OK] API 配置成功！可以开始学习了！")

    print("\n" + "="*60)
    print("回复的详细信息：")
    print("="*60)
    print(f"回复类型: {type(response).__name__}")
    print(f"回复内容: {response.content}")

    if hasattr(response, 'response_metadata'):
        print(f"元数据: {response.response_metadata}")

except Exception as e:
    print(f"\n[ERROR] 调用失败: {e}")
    print("\n可能的原因：")
    print("1. API Key 不正确")
    print("2. 网络连接问题")
    print("3. API 配额用完")
    print("\n请检查后重试")
    import traceback
    traceback.print_exc()
