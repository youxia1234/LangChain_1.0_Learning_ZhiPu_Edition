"""
模块 15：工具与 Agent 进阶
演示高级工具定义、验证、组合和生产级实践

运行: python main.py
"""

import os
import sys
import json
import time
from typing import Optional, List
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent  # LangChain 1.0 统一 API

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取 API 密钥"
    )

# 初始化模型（使用智谱 AI）
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ============================================================
# 示例 1: 高级工具定义与参数验证
# ============================================================

def example_1_advanced_tools():
    """
    演示使用 Pydantic 进行高级工具定义和参数验证
    """
    print("\n" + "=" * 60)
    print("示例 1: 高级工具定义与参数验证")
    print("=" * 60)
    
    from langchain_core.tools import tool, StructuredTool
    from pydantic import BaseModel, Field
    
    # 方式1: 简单 @tool 装饰器
    @tool
    def simple_calculator(expression: str) -> str:
        """执行简单的数学计算。
        
        Args:
            expression: 数学表达式，如 "2 + 3 * 4"
        """
        try:
            # 安全地计算表达式
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含不允许的字符"
            result = eval(expression)
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    # 方式2: Pydantic 参数模型
    class WeatherInput(BaseModel):
        """天气查询参数"""
        city: str = Field(description="城市名称")
        unit: str = Field(default="celsius", description="温度单位: celsius 或 fahrenheit")

    @tool(args_schema=WeatherInput)
    def get_weather(city: str, unit: str = "celsius") -> str:
        """获取指定城市的天气信息。"""
        # 在工具内部验证参数（Pydantic V2 兼容方式）
        if unit not in ["celsius", "fahrenheit"]:
            return "错误: unit 必须是 'celsius' 或 'fahrenheit'"

        # 模拟天气数据
        weather_data = {
            "北京": {"temp_c": 15, "condition": "晴"},
            "上海": {"temp_c": 20, "condition": "多云"},
            "深圳": {"temp_c": 25, "condition": "阴"},
        }

        data = weather_data.get(city, {"temp_c": 18, "condition": "未知"})
        temp = data["temp_c"] if unit == "celsius" else data["temp_c"] * 9/5 + 32
        unit_symbol = "°C" if unit == "celsius" else "°F"

        return f"{city}天气: {data['condition']}, 温度: {temp:.1f}{unit_symbol}"
    
    # 方式3: StructuredTool
    def translate_text(text: str, target_lang: str) -> str:
        """翻译文本（模拟）"""
        translations = {
            "ch": f"[中文] {text}",
            "en": f"[English] {text}",
            "ja": f"[日本語] {text}",
        }
        return translations.get(target_lang, f"[{target_lang}] {text}")
    
    translate_tool = StructuredTool.from_function(
        func=translate_text,
        name="translate",
        description="将文本翻译成指定语言。支持: ch，en, ja ",
    )
    
    # 测试工具
    print("\n📌 测试简单计算器:")
    print(simple_calculator.invoke({"expression": "2 + 3 * 4"}))
    print(simple_calculator.invoke({"expression": "(10 - 5) / 2"}))
    
    print("\n📌 测试天气查询:")
    print(get_weather.invoke({"city": "北京", "unit": "celsius"}))
    print(get_weather.invoke({"city": "上海", "unit": "fahrenheit"}))
    
    print("\n📌 测试翻译工具:")
    print(translate_tool.invoke({"text": "你好世界", "target_lang": "ch"}),"（仅作模拟，未进行功能实现）")
    print(translate_tool.invoke({"text": "Hello World", "target_lang": "en"}),"（仅作模拟，未进行功能实现）")
    
    print("\n✅ 工具信息:")
    for t in [simple_calculator, get_weather, translate_tool]:
        print(f"  - {t.name}: {t.description[:50]}...")

# ============================================================
# 示例 2: 工具错误处理
# ============================================================

def example_2_error_handling():
    """
    演示工具的错误处理机制
    """
    print("\n" + "=" * 60)
    print("示例 2: 工具错误处理")
    print("=" * 60)

    from langchain_core.tools import tool

    # 方式1: 基本错误处理（在工具内部处理）
    @tool
    def safe_divide(a: float, b: float) -> str:
        """安全的除法运算。

        Args:
            a: 被除数
            b: 除数
        """
        try:
            if b == 0:
                return "⚠️ 错误: 除数不能为零！请提供一个非零的除数。"
            return f"{a} ÷ {b} = {a / b:.4f}"
        except Exception as e:
            return f"⚠️ 计算错误: {str(e)}"

    # 方式2: 自定义错误处理（返回友好错误信息）
    @tool
    def fetch_data(url: str) -> str:
        """从 URL 获取数据（模拟）。

        Args:
            url: 要获取数据的 URL
        """
        # 验证 URL
        if not url.startswith("http"):
            return "⚠️ 操作失败: URL 必须以 http:// 或 https:// 开头\n💡 建议: 请检查 URL 格式是否正确"
        if "error" in url:
            return "⚠️ 操作失败: 无法连接到服务器\n💡 建议: 请检查网络连接或稍后重试"
        return f"✅ 成功获取数据: {url}"

    # 方式3: 带重试的工具
    @tool
    def unreliable_api(query: str) -> str:
        """模拟不稳定的 API 调用。"""
        import random
        if random.random() < 0.3:  # 30% 失败率
            return f"⚠️ API 临时不可用"
        return f"✅ API 返回: {query}"

    def with_retry(tool_func, max_retries: int = 3):
        """添加重试逻辑的包装器"""
        def wrapper(*args, **kwargs):
            last_result = None
            for attempt in range(max_retries):
                result = tool_func.invoke(*args, **kwargs)
                # 如果结果包含错误信息，重试
                if "⚠️" in result and attempt < max_retries - 1:
                    print(f"  重试 {attempt + 1}/{max_retries}...")
                    time.sleep(0.1)
                    last_result = result
                else:
                    return result
            return last_result or f"失败（{max_retries}次重试后）"
        return wrapper

    # 测试
    print("\n📌 测试安全除法:")
    print(safe_divide.invoke({"a": 10.0, "b": 3.0}))
    print(safe_divide.invoke({"a": 10.0, "b": 0.0}))  # 触发错误

    print("\n📌 测试自定义错误处理:")
    print(fetch_data.invoke({"url": "https://api.example.com/data"}))
    print(fetch_data.invoke({"url": "invalid-url"}))  # 触发错误

    print("\n📌 测试带重试的 API:")
    retry_api = with_retry(unreliable_api)
    print(retry_api({"query": "测试查询"}))

# ============================================================
# 示例 3: 监控回调
# ============================================================

def example_3_monitoring():
    """
    演示工具执行监控和日志记录
    """
    print("\n" + "=" * 60)
    print("示例 3: 监控回调")
    print("=" * 60)
    
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tools import tool
    from typing import Any, Dict
    
    class ToolMonitor(BaseCallbackHandler):
        """工具执行监控器"""
        
        def __init__(self):
            self.call_count = 0
            self.total_duration = 0.0
            self.tool_stats = {}
            self._start_time = None
        
        def on_tool_start(
            self, 
            serialized: Dict[str, Any], 
            input_str: str, 
            **kwargs
        ) -> None:
            self._start_time = time.time()
            tool_name = serialized.get("name", "unknown")
            print(f"  🔧 工具开始: {tool_name}")
            print(f"     输入: {input_str[:100]}...")
            
            if tool_name not in self.tool_stats:
                self.tool_stats[tool_name] = {"calls": 0, "duration": 0.0}
        
        def on_tool_end(self, output: str, **kwargs) -> None:
            duration = time.time() - self._start_time
            self.call_count += 1
            self.total_duration += duration
            
            print(f"  ✅ 工具完成 (耗时: {duration:.3f}s)")
            print(f"     输出: {str(output)[:100]}...")
        
        def on_tool_error(self, error: Exception, **kwargs) -> None:
            print(f"  ❌ 工具错误: {error}")
        
        def get_stats(self) -> dict:
            """获取统计信息"""
            return {
                "total_calls": self.call_count,
                "total_duration": f"{self.total_duration:.3f}s",
                "avg_duration": f"{self.total_duration/max(1, self.call_count):.3f}s"
            }
    
    # 创建监控器
    monitor = ToolMonitor()
    
    @tool
    def slow_operation(data: str) -> str:
        """模拟耗时操作。"""
        time.sleep(0.1)  # 模拟处理时间
        return f"处理完成: {data}"
    
    @tool
    def fast_operation(data: str) -> str:
        """快速操作。"""
        return f"快速结果: {data}"
    
    # 测试（注意：独立调用工具时回调不会自动触发）
    print("\n📌 模拟工具执行监控:")
    
    # 手动模拟监控过程
    for i, (tool_func, data) in enumerate([
        (slow_operation, "慢数据"),
        (fast_operation, "快数据1"),
        (fast_operation, "快数据2"),
    ]):
        monitor.on_tool_start({"name": tool_func.name}, data)
        try:
            result = tool_func.invoke({"data": data})
            monitor.on_tool_end(result)
        except Exception as e:
            monitor.on_tool_error(e)
    
    print("\n📊 监控统计:")
    stats = monitor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

# ============================================================
# 示例 4: 工具组合与复用
# ============================================================

def example_4_tool_composition():
    """
    演示工具的组合和复用模式
    """
    print("\n" + "=" * 60)
    print("示例 4: 工具组合与复用")
    print("=" * 60)
    
    from langchain_core.tools import tool
    from typing import List
    
    # 基础工具
    @tool
    def search_products(query: str) -> List[dict]:
        """搜索商品。"""
        # 模拟数据库搜索
        products = [
            {"id": "P001", "name": "iPhone 15", "price": 7999, "stock": 50},
            {"id": "P002", "name": "MacBook Pro", "price": 14999, "stock": 20},
            {"id": "P003", "name": "AirPods Pro", "price": 1899, "stock": 100},
        ]
        results = [p for p in products if query.lower() in p["name"].lower()]
        return results if results else [{"message": f"未找到 '{query}' 相关商品"}]
    
    @tool
    def check_inventory(product_id: str) -> dict:
        """检查库存。"""
        inventory = {
            "P001": {"available": True, "quantity": 50},
            "P002": {"available": True, "quantity": 20},
            "P003": {"available": True, "quantity": 100},
        }
        return inventory.get(product_id, {"available": False, "quantity": 0})
    
    @tool
    def calculate_discount(price: float, discount_percent: float) -> dict:
        """计算折扣价格。"""
        discount_amount = price * (discount_percent / 100)
        final_price = price - discount_amount
        return {
            "original_price": price,
            "discount": f"{discount_percent}%",
            "savings": discount_amount,
            "final_price": final_price
        }
    
    # 组合工具：搜索并检查库存
    @tool
    def search_and_check(query: str) -> str:
        """搜索商品并检查库存状态。"""
        # 调用搜索工具
        products = search_products.invoke({"query": query})
        
        if isinstance(products, list) and products and "message" not in products[0]:
            results = []
            for p in products:
                # 检查库存
                inventory = check_inventory.invoke({"product_id": p["id"]})
                status = "有货" if inventory["available"] else "缺货"
                results.append(
                    f"- {p['name']}: ¥{p['price']} ({status}, 库存{inventory['quantity']})"
                )
            return "\n".join(results)
        return f"未找到 '{query}' 相关商品"
    
    # 工厂函数：创建特定类型的搜索工具
    def create_category_search(category: str):
        """创建特定分类的搜索工具"""
        @tool
        def category_search(query: str) -> str:
            """在指定分类中搜索商品。"""
            return f"[{category}] 搜索 '{query}' 的结果..."

        # 修改工具名称和描述
        category_search.name = f"search_{category.lower()}"
        # 使用 __doc__ 修改 docstring（更可靠的方式）
        category_search.description = f"在{category}分类中搜索相关商品"
        return category_search
    
    # 创建多个分类搜索工具
    electronics_search = create_category_search("电子产品")
    clothing_search = create_category_search("服装")
    
    # 测试
    print("\n📌 测试基础商品搜索:")
    result = search_products.invoke({"query": "iPhone"})
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    print("\n📌 测试组合工具（搜索+库存检查）:")
    print(search_and_check.invoke({"query": "Pro"}))
    
    print("\n📌 测试折扣计算:")
    discount_result = calculate_discount.invoke({"price": 7999.0, "discount_percent": 15.0})
    print(json.dumps(discount_result, ensure_ascii=False, indent=2))
    
    print("\n📌 工厂创建的工具:")
    print(f"  - {electronics_search.name}: {electronics_search.description}")
    print(f"  - {clothing_search.name}: {clothing_search.description}")

# ============================================================
# 示例 5: 完整的智能体示例（需要 API Key）
# ============================================================

def example_5_complete_agent():
    """
    演示完整的生产级智能体
    使用智谱 AI (glm-4-flash)
    """
    print("\n" + "=" * 60)
    print("示例 5: 完整的智能体示例")
    print("=" * 60)

    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.tools import tool
    from langchain_core.callbacks import BaseCallbackHandler
    from pydantic import BaseModel, Field

    # ====== 定义工具 ======

    class OrderInput(BaseModel):
        order_id: str = Field(description="订单号，格式: ORD+数字")

    # 模拟数据库
    ORDERS = {
        "ORD001": {"status": "已发货", "product": "iPhone 15", "amount": 7999.0},
        "ORD002": {"status": "处理中", "product": "MacBook Pro", "amount": 14999.0},
        "ORD003": {"status": "已完成", "product": "AirPods Pro", "amount": 1899.0},
    }

    @tool(args_schema=OrderInput)
    def check_order(order_id: str) -> str:
        """查询订单状态。"""
        order = ORDERS.get(order_id.upper())
        if not order:
            return f"未找到订单 {order_id}。可用订单: {', '.join(ORDERS.keys())}"
        return json.dumps({
            "订单号": order_id.upper(),
            "商品": order["product"],
            "状态": order["status"],
            "金额": f"¥{order['amount']}"
        }, ensure_ascii=False, indent=2)

    @tool
    def get_faq(topic: str) -> str:
        """获取常见问题解答。

        Args:
            topic: 问题主题，如"配送"、"退货"、"支付"
        """
        faqs = {
            "配送": "普通快递3-5天送达，顺丰1-2天送达，偏远地区可能延迟1-2天",
            "退货": "支持7天无理由退货，商品需保持原包装完好",
            "支付": "支持微信、支付宝、银行卡等多种支付方式",
            "发票": "订单完成后可在'我的订单'中申请电子发票"
        }

        for key, value in faqs.items():
            if key in topic:
                return f"📖 关于【{key}】:\n{value}"

        return f"暂无关于'{topic}'的FAQ，常见主题: {', '.join(faqs.keys())}"

    @tool
    def calculate_shipping(address: str) -> str:
        """计算配送费用。

        Args:
            address: 收货地址
        """
        # 简单的运费计算逻辑
        if any(city in address for city in ["北京", "上海", "广州", "深圳"]):
            return f"配送到 {address}: 免运费（一线城市包邮）"
        elif any(region in address for region in ["新疆", "西藏", "内蒙古"]):
            return f"配送到 {address}: ¥25（偏远地区）"
        else:
            return f"配送到 {address}: ¥10（标准运费）"

    # ====== 监控回调 ======

    class AgentMonitor(BaseCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):
            print(f"  🔧 调用: {serialized.get('name', '?')}")

        def on_tool_end(self, output, **kwargs):
            print(f"  ✅ 返回: {str(output)[:80]}...")

    # ====== 创建智能体（使用智谱 AI） ======

    # 使用全局定义的 model（智谱 AI）
    tools = [check_order, get_faq, calculate_shipping]
    memory = MemorySaver()

    system_prompt = """你是一个专业的电商客服助手。

你可以：
1. 查询订单状态（可用订单号: ORD001, ORD002, ORD003）
2. 回答常见问题（配送、退货、支付、发票）
3. 计算配送费用

请保持友好、专业的态度，回答要简洁明了。"""

    agent = create_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        system_prompt=system_prompt
    )

    # ====== 测试对话 ======

    config = {
        "configurable": {"thread_id": "demo_session"},
        "callbacks": [AgentMonitor()]
    }

    test_queries = [
        "帮我查一下订单 ORD001",
        "配送一般要多久？",
        "我在杭州，运费是多少？"
    ]

    print("\n📞 开始客服对话模拟:")
    print("-" * 40)

    for query in test_queries:
        print(f"\n👤 用户: {query}")

        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config
            )
            ai_message = response["messages"][-1].content
            print(f"🤖 客服: {ai_message}")
        except Exception as e:
            print(f"❌ 错误: {e}")

        print("-" * 40)

# ============================================================
# 主函数
# ============================================================

def main():
    """运行所有示例"""
    print("╔" + "═" * 58 + "╗")
    print("║" + " 模块 15: 工具与 Agent 进阶 ".center(56) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # 运行示例
    example_1_advanced_tools()
    example_2_error_handling()
    example_3_monitoring()
    example_4_tool_composition()
    example_5_complete_agent()
    
    print("\n" + "=" * 60)
    print("🎉 所有示例运行完成！")
    print("=" * 60)
    
    print("""
📚 本模块要点总结:

1. 工具定义方式:
   - @tool 装饰器（简单场景）
   - Pydantic args_schema（生产级，推荐）
   - StructuredTool（完全控制）

2. 错误处理:
   - handle_tool_error=True 启用错误处理
   - 自定义错误处理函数
   - 重试机制

3. 监控与日志:
   - BaseCallbackHandler 实现监控
   - on_tool_start/on_tool_end 回调

4. 工具组合:
   - 工具间调用
   - 工厂函数创建工具
   - 管道模式

5. 生产级实践:
   - 参数验证
   - 完善的错误消息
   - 监控和日志

6. Agent API:
   - 使用 create_agent (langchain.agents)
   - langgraph.prebuilt.create_react_agent 已弃用
""")

if __name__ == "__main__":
    main()
