"""
LangChain 1.0 - Validation & Retry (验证和重试)
===============================================

本模块重点讲解：
1. with_retry() - 自动重试机制
2. with_fallbacks() - 降级/备用方案
3. Pydantic 验证错误处理
4. 自定义验证逻辑
5. 重试循环实现

注意：
- 智谱 AI 不支持原生 with_structured_output()
- 本模块使用 JSON 解析方式实现结构化输出
- 示例 3、5、6 已启用（不需要 LLM 调用）
- 示例 1、2、4、7 已注释（需要网络调用）
"""

import os
import json
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional, List, TypeVar, Type
from enum import Enum
import time

# 加载环境变量
load_dotenv()
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

if not ZHIPUAI_API_KEY or ZHIPUAI_API_KEY == "your_zhipuai_api_key_here":
    raise ValueError(
        "\n请先在 .env 文件中设置有效的 ZHIPUAI_API_KEY\n"
        "访问 https://open.bigmodel.cn/usercenter/apikeys 获取密钥"
    )

# 初始化模型（使用智谱 AI）
model = ChatOpenAI(
    model="glm-4-flash",
    api_key=ZHIPUAI_API_KEY,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


# ==================== 辅助函数 ====================

# Windows GBK 编码安全打印
def safe_print(content):
    """安全打印，处理特殊字符"""
    if isinstance(content, str):
        try:
            print(content.encode('gbk', errors='ignore').decode('gbk'))
        except:
            print(content.encode('utf-8', errors='ignore').decode('utf-8'))
    else:
        print(content)

T = TypeVar('T', bound=BaseModel)

def structured_output(prompt: str, output_class: Type[T], llm=None) -> T:
    """
    结构化输出函数（使用 JSON 解析）

    智谱 AI 不支持 with_structured_output()，使用 JSON 解析方式
    """
    if llm is None:
        llm = model

    # 获取 Pydantic 模型的完整 JSON schema（包含所有定义）
    full_schema = output_class.model_json_schema()

    # 提取所有定义的模型
    definitions = full_schema.get('$defs', {}) or full_schema.get('definitions', {})

    def get_field_type_info(field_info, field_name):
        """递归获取字段类型信息"""
        desc = field_info.get('description', '')
        field_type = field_info.get('type', 'string')

        # 检查是否有枚举值
        if 'enum' in field_info:
            enum_values = field_info['enum']
            if isinstance(enum_values, list) and len(enum_values) > 0:
                enum_str = " / ".join(enum_values)
                return f"{desc}，类型: 枚举 [{enum_str}]", enum_values[0]

        # 检查是否是数组
        if field_type == 'array' and 'items' in field_info:
            items = field_info['items']
            # 检查 items 是否有 enum（枚举数组）
            if 'enum' in items:
                enum_values = items['enum']
                enum_str = " / ".join(enum_values)
                return f"{desc}，类型: 枚举数组 [{enum_str}]", [enum_values[0]]
            elif '$ref' in items:
                ref = items['$ref']
                ref_name = ref.split('/')[-1] if '/' in ref else ref
                return f"{desc}，类型: {ref_name}对象数组", [{"示例": "值"}]
            elif 'properties' in items:
                nested = []
                for k, v in items['properties'].items():
                    nested.append(f"{k}:{v.get('type','string')}")
                nested_example = {k: "示例" for k in items['properties'].keys()}
                return f"{desc}，类型: 对象数组 [{', '.join(nested)}]", [nested_example]
            else:
                return f"{desc}，类型: 数组", ["值1", "值2"]

        # 检查是否是引用类型（单个对象或枚举）
        if '$ref' in field_info:
            ref = field_info['$ref']
            ref_name = ref.split('/')[-1] if '/' in ref else ref
            # 从 definitions 中获取引用的模型结构
            if ref_name in definitions:
                ref_def = definitions[ref_name]
                # 检查是否是枚举定义
                if 'enum' in ref_def:
                    enum_values = ref_def['enum']
                    if isinstance(enum_values, list) and len(enum_values) > 0:
                        enum_str = " / ".join(enum_values)
                        return f"{desc}，类型: 枚举 [{enum_str}]", enum_values[0]
                # 普通对象定义
                ref_props = ref_def.get('properties', {})
                nested = []
                nested_example = {}
                for k, v in ref_props.items():
                    v_type = v.get('type', 'string')
                    nested.append(f"{k}:{v_type}")
                    if v_type == 'string':
                        nested_example[k] = "示例"
                    elif v_type == 'integer':
                        nested_example[k] = 123
                    elif v_type == 'number':
                        nested_example[k] = 123.45
                    elif v_type == 'boolean':
                        nested_example[k] = True
                return f"{desc}，类型: {ref_name}对象 [{', '.join(nested)}]", nested_example
            return f"{desc}，类型: 对象", {"示例": "值"}

        # 检查是否有内联嵌套属性
        if 'properties' in field_info:
            nested = []
            nested_example = {}
            for k, v in field_info['properties'].items():
                v_type = v.get('type', 'string')
                nested.append(f"{k}:{v_type}")
                if v_type == 'string':
                    nested_example[k] = "示例"
                elif v_type == 'integer':
                    nested_example[k] = 123
            return f"{desc}，类型: 对象 [{', '.join(nested)}]", nested_example

        # 基本类型
        if field_type == 'string':
            return f"{desc}，类型: 字符串", "示例值"
        elif field_type == 'integer':
            return f"{desc}，类型: 整数", 123
        elif field_type == 'number':
            return f"{desc}，类型: 数字", 123.45
        elif field_type == 'boolean':
            return f"{desc}，类型: 布尔值", True
        else:
            return f"{desc}，类型: {field_type}", "示例值"

    properties = full_schema.get('properties', {})
    required = full_schema.get('required', [])

    # 构建字段说明
    field_descriptions = []
    for field_name, field_info in properties.items():
        required_mark = "（必填）" if field_name in required else "（可选）"
        type_info, _ = get_field_type_info(field_info, field_name)
        field_descriptions.append(f"- {field_name}: {type_info}{required_mark}")

    fields_str = "\n".join(field_descriptions)
    required_str = ", ".join(required) if required else "无"

    # 构建示例
    example_values = {}
    for field_name, field_info in properties.items():
        field_type = field_info.get('type', 'string')
        _, example = get_field_type_info(field_info, field_name)

        if example is not None:
            example_values[field_name] = example
        elif field_type == 'string':
            example_values[field_name] = "示例值"
        elif field_type == 'integer':
            example_values[field_name] = 123
        elif field_type == 'number':
            example_values[field_name] = 123.45
        elif field_type == 'boolean':
            example_values[field_name] = True
        else:
            example_values[field_name] = "示例值"

    example_json = json.dumps(example_values, ensure_ascii=False, indent=2)

    json_prompt = f"""{prompt}

请从上述文本中提取信息，并按照以下 JSON 格式返回：

字段说明：
{fields_str}

必填字段：{required_str}

返回格式示例：
```json
{example_json}
```

重要要求：
1. 只返回JSON对象，不要添加任何解释性文字
2. 所有必填字段必须填写实际提取的值
3. 字符串值用双引号，数字值不要用引号
4. 注意区分单个对象和对象数组
5. 使用 ```json ``` 代码块包裹结果"""

    response = llm.invoke([HumanMessage(content=json_prompt)])
    content = response.content.strip()

    # 清理 Markdown 格式
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    # 解析 JSON
    try:
        data = json.loads(content.strip())
        return output_class.model_validate(data)
    except Exception as e:
        print(f"  [错误] JSON 解析失败: {e}")
        print(f"  [调试] 原始响应: {content[:500]}...")
        raise ValueError(f"无法解析结构化输出: {e}")



# ============================================================================
# 示例 1：with_retry() - 自动重试
# ============================================================================
def example_1_with_retry():
    """
    示例1：使用 with_retry() 处理网络错误

    当遇到临时性错误（网络超时、API限流等）时自动重试
    """
    print("\n" + "="*70)
    print("示例 1：with_retry() - 自动重试机制")
    print("="*70)

    # 创建带重试的 LLM
    llm_with_retry = model.with_retry(
        retry_if_exception_type=(ConnectionError, TimeoutError),  # 重试的异常类型
        wait_exponential_jitter=True,  # 指数退避 + 随机抖动
        stop_after_attempt=3  # 最多重试 3 次
    )

    print("\n配置:")
    print("  - 重试异常: ConnectionError, TimeoutError")
    print("  - 最大重试次数: 3")
    print("  - 退避策略: 指数退避 + 随机抖动")

    try:
        print("\n调用 LLM (如果失败会自动重试)...")
        response = llm_with_retry.invoke("你好")
        print(f"响应: {response.content[:50]}...")
        print("\n[OK] 调用成功")
    except Exception as e:
        print(f"\n[X] 重试 3 次后仍然失败: {e}")

    print("\n关键点:")
    print("  - with_retry() 是 Runnable 接口的方法")
    print("  - 适用于临时性错误（网络波动、API限流）")
    print("  - 不适用于逻辑错误（提示词错误、参数错误）")

# ============================================================================
# 示例 2：with_fallbacks() - 降级方案
# ============================================================================
def example_2_with_fallbacks():
    """
    示例2：使用 with_fallbacks() 实现降级

    主模型失败时，自动切换到备用模型

    注意：由于本教程使用单一 API 密钥，此示例仅展示用法
    在实际生产环境中，可以配置不同提供商的模型作为降级选项
    """
    print("\n" + "="*70)
    print("示例 2：with_fallbacks() - 降级/备用方案")
    print("="*70)

    # 主模型
    primary_model = model

    # 备用模型（在实际场景中，可以使用不同提供商的模型）
    # 例如：主模型用 GPT-4，备用用 Claude，最后用 Llama
    # 这里使用同一个模型作为示例
    fallback_model = model

    # 配置降级
    llm_with_fallbacks = primary_model.with_fallbacks([fallback_model])

    print("\n配置:")
    print("  - 主模型: glm-4-flash (智谱 AI)")
    print("  - 备用模型: 相同模型（示例）")

    print("\n说明:")
    print("  - 在生产环境中，应该配置不同提供商的模型")
    print("  - 例如: openai:gpt-4 → anthropic:claude → groq:llama")
    print("  - 这样即使某个提供商宕机，系统仍能继续运行")

    try:
        response = llm_with_fallbacks.invoke("用一句话介绍 Python")
        print(f"\n响应: {response.content[:100]}...")
        print("\n关键点:")
        print("  - 主模型成功 -> 使用主模型响应")
        print("  - 主模型失败 -> 自动切换到备用模型")
        print("  - 适用于高可用性场景")
    except Exception as e:
        print(f"\n所有模型都失败: {e}")

# ============================================================================
# 示例 3：Pydantic 字段验证
# ============================================================================
class User(BaseModel):
    """用户信息（带验证）"""
    name: str = Field(description="姓名", min_length=2, max_length=20)
    age: int = Field(description="年龄", ge=0, le=150)  # 0-150 岁
    email: str = Field(description="邮箱")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """自定义邮箱验证"""
        if '@' not in v:
            raise ValueError('邮箱必须包含 @')
        return v

def example_3_pydantic_validation():
    """
    示例3：Pydantic 内置验证

    使用 Field 约束和自定义验证器
    """
    print("\n" + "="*70)
    print("示例 3：Pydantic 字段验证")
    print("="*70)

    print("\n测试 1: 有效数据")
    try:
        user = User(name="张三", age=30, email="zhang@example.com")
        print(f"[OK] 验证通过: {user.name}, {user.age}, {user.email}")
    except ValidationError as e:
        print(f"[X] 验证失败: {e}")

    print("\n测试 2: 年龄超出范围")
    try:
        user = User(name="李四", age=200, email="li@example.com")
        print(f"[OK] 验证通过: {user}")
    except ValidationError as e:
        print(f"[X] 验证失败: 年龄必须在 0-150 之间")
        print(f"   错误详情: {e.errors()[0]['msg']}")

    print("\n测试 3: 邮箱格式错误")
    try:
        user = User(name="王五", age=25, email="invalid-email")
        print(f"[OK] 验证通过: {user}")
    except ValidationError as e:
        print(f"[X] 验证失败: 邮箱格式错误")
        print(f"   错误详情: {e.errors()[0]['msg']}")

    print("\n关键点:")
    print("  - Field(ge=, le=) - 数值范围约束")
    print("  - Field(min_length=, max_length=) - 字符串长度约束")
    print("  - @field_validator - 自定义验证逻辑")
    print("  - ValidationError - 验证失败时抛出")

# ============================================================================
# 示例 4：LLM 输出验证 + 重试
# ============================================================================
class Product(BaseModel):
    """产品信息（严格验证）"""
    name: str = Field(description="产品名称（字符串类型）", min_length=2)
    price: float = Field(description="价格，数字类型（必须 > 0）", gt=0)
    stock: int = Field(description="库存，整数类型（必须 >= 0）", ge=0)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v.lower() == "unknown":
            raise ValueError('产品名称不能是 unknown')
        return v

def example_4_llm_validation_retry():
    """
    示例4：LLM 输出验证 + 重试循环

    如果 LLM 输出不符合验证规则，重新提示并重试
    """
    print("\n" + "="*70)
    print("示例 4：LLM 输出验证 + 重试循环")
    print("="*70)

    max_retries = 3

    # 使用一个简单的测试案例（改为正常价格，避免触发验证错误）
    text = "iPhone 15 售价 5999 元，库存 50 件"

    print(f"\n提取文本: {text}")
    print(f"验证规则: price > 0, stock >= 0, name 不能是 'unknown'\n")

    for attempt in range(1, max_retries + 1):
        print(f"尝试 {attempt}/{max_retries}...")

        try:
            # 调用 LLM
            prompt = f"""从以下文本提取产品信息。
重要：price 必须是数字类型（不是字符串），stock 必须是整数类型。

文本: {text}"""
            result = structured_output(prompt, Product)

            # 如果到这里，说明验证通过
            print(f"[OK] 提取成功!")
            print(f"  产品: {result.name}")
            print(f"  价格: {result.price} 元")
            print(f"  库存: {result.stock} 件")
            break

        except ValidationError as e:
            print(f"[X] Pydantic 验证失败: {e.errors()[0]['msg']}")

            if attempt < max_retries:
                error_msg = e.errors()[0]['msg']
                text = f"{text}\n注意: {error_msg}"
                print(f"  -> 修正提示后重试...\n")
            else:
                print(f"  -> 已达到最大重试次数")

        except Exception as e:
            # 捕获其他错误（如 JSON 解析错误）
            print(f"[X] 错误: {e}")

            if attempt < max_retries:
                print(f"  -> 重试...\n")
                # 强化提示
                text = f"{text}\n重要: price 和 stock 必须是数字类型，不能是字符串"
            else:
                print(f"  -> 已达到最大重试次数")

    print("\n关键点:")
    print("  - ValidationError 捕获 Pydantic 验证失败")
    print("  - Exception 捕获 JSON 解析失败")
    print("  - 在提示中强调类型要求")
    print("  - 限制最大重试次数防止无限循环")

# ============================================================================
# 示例 5：自定义验证函数
# ============================================================================
class Article(BaseModel):
    """文章信息"""
    title: str = Field(description="标题")
    content: str = Field(description="内容")
    word_count: int = Field(description="字数")

def validate_article(article: Article) -> bool:
    """
    自定义验证逻辑

    检查 word_count 是否与 content 实际字数接近
    """
    actual_count = len(article.content)
    claimed_count = article.word_count

    # 允许 10% 误差
    tolerance = 0.1
    lower_bound = actual_count * (1 - tolerance)
    upper_bound = actual_count * (1 + tolerance)

    if not (lower_bound <= claimed_count <= upper_bound):
        return False

    return True

def example_5_custom_validation():
    """
    示例5：自定义验证函数

    Pydantic 验证之外的业务逻辑验证
    """
    print("\n" + "="*70)
    print("示例 5：自定义验证函数")
    print("="*70)

    print("\n测试 1: 字数匹配")
    article1 = Article(
        title="测试文章",
        content="这是一篇测试文章的内容",
        word_count=12
    )

    if validate_article(article1):
        print(f"[OK] 验证通过: 声称字数 {article1.word_count}，实际 {len(article1.content)}")
    else:
        print(f"[X] 验证失败: 字数不匹配")

    print("\n测试 2: 字数不匹配（相差太大）")
    article2 = Article(
        title="测试文章",
        content="短内容",
        word_count=1000  # 明显错误
    )

    if validate_article(article2):
        print(f"[OK] 验证通过")
    else:
        print(f"[X] 验证失败: 声称 {article2.word_count} 字，实际只有 {len(article2.content)} 字")

    print("\n关键点:")
    print("  - Pydantic 验证类型和格式")
    print("  - 自定义函数验证业务逻辑")
    print("  - 可以结合使用实现完整验证")

# ============================================================================
# 示例 6：完整的验证 + 重试工作流
# ============================================================================
class ExtractedData(BaseModel):
    """提取的数据（完整验证）"""
    name: str = Field(description="名称（字符串类型）", min_length=1)
    value: float = Field(description="数值（数字类型，必须 > 0）", gt=0)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if v.strip() == "":
            raise ValueError('名称不能为空')
        return v.strip()

def extract_with_validation(text: str, max_retries: int = 3) -> Optional[ExtractedData]:
    """
    带验证的提取函数

    Args:
        text: 待提取的文本
        max_retries: 最大重试次数

    Returns:
        提取的数据（验证通过）或 None（失败）
    """
    current_text = text

    for attempt in range(1, max_retries + 1):
        try:
            # 调用 LLM（强调类型）
            prompt = f"""提取以下文本中的信息。
重要：value 必须是数字类型（float），不能是字符串。

{current_text}"""
            result = structured_output(prompt, ExtractedData)

            # 额外的业务验证（Pydantic 已经检查了 gt=0）
            # 所有验证通过
            return result

        except ValidationError as e:
            error_msg = e.errors()[0]['msg']
            if attempt < max_retries:
                current_text = f"{text}\n\n注意：{error_msg}。请重新提取。"
            else:
                return None

        except Exception as e:
            # 捕获 JSON 解析错误
            if attempt < max_retries:
                current_text = f"{text}\n\n重要：value 必须是数字类型，不能是字符串。"
            else:
                return None

    return None

def example_6_complete_workflow():
    """
    示例6：完整的验证 + 重试工作流

    展示生产环境中的最佳实践
    """
    print("\n" + "="*70)
    print("示例 6：完整的验证 + 重试工作流")
    print("="*70)

    test_cases = [
        "产品 A 的价值是 999.99 元",
        "产品 B 的价值是 1299 元",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print(f"文本: {text}")

        result = extract_with_validation(text, max_retries=2)

        if result:
            print(f"[OK] 提取成功:")
            print(f"  名称: {result.name}")
            print(f"  数值: {result.value}")
        else:
            print(f"[X] 提取失败（重试 2 次后仍无法通过验证）")

    print("\n关键点:")
    print("  - 封装验证逻辑到函数中")
    print("  - 清晰的错误处理")
    print("  - 返回 Optional 表示可能失败")
    print("  - 适合集成到生产系统")

# ============================================================================
# 示例 7：组合使用 retry + fallbacks + validation
# ============================================================================
def example_7_combined():
    """
    示例7：组合使用多种策略

    网络重试 + 输出验证

    注意：由于本教程使用单一 API，这里演示 with_retry 的用法
    在生产环境中，还可以添加 with_fallbacks 实现模型降级
    """
    print("\n" + "="*70)
    print("示例 7：组合策略 - retry + validation")
    print("="*70)

    # 创建带重试的 LLM
    llm_with_retry = model.with_retry(
        retry_if_exception_type=(ConnectionError, TimeoutError),
        stop_after_attempt=2
    )

    print("\n配置:")
    print("  - 网络重试: 最多 2 次")
    print("  - 输出验证: Pydantic 模型验证")

    print("\n生产环境建议:")
    print("  可以组合使用:")
    print("  1. with_retry() - 网络错误自动重试")
    print("  2. with_fallbacks() - 模型降级（主模型失败时切换备用模型）")
    print("  3. Pydantic 验证 - 数据类型和格式验证")

    try:
        prompt = """提取以下文本中的信息。
重要：value 必须是数字类型（float）。

产品 C 的价值是 1299 元"""
        result = structured_output(prompt, ExtractedData, llm=llm_with_retry)
        print(f"\n[OK] 成功提取:")
        print(f"  名称: {result.name}")
        print(f"  数值: {result.value}")
    except Exception as e:
        print(f"\n[X] 提取失败: {e}")

    print("\n关键点:")
    print("  - 调用顺序很重要！")
    print("  - 原生 with_structured_output() 需要在最前面")
    print("  - 然后 retry、fallbacks")
    print("  - 本模块使用 JSON 解析方式，故不受此限制")
    print("  - 多层防护: 验证 → 重试 → 降级")

# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" LangChain 1.0 - Validation & Retry (验证和重试)")
    print("="*70)

    try:
        example_1_with_retry()

        example_2_with_fallbacks()

        example_3_pydantic_validation()

        example_4_llm_validation_retry()

        example_5_custom_validation()

        example_6_complete_workflow()

        example_7_combined()

        print("\n" + "="*70)
        print(" 完成！")
        print("="*70)
        print("\n核心要点：")
        print("  1. with_retry() - 网络错误自动重试")
        print("  2. with_fallbacks() - 模型降级/备用方案")
        print("  3. Pydantic Field 约束 - 类型和格式验证")
        print("  4. @field_validator - 自定义字段验证")
        print("  5. ValidationError - 捕获验证失败")
        print("  6. 重试循环 - LLM 输出验证失败时重试")
        print("  7. 组合策略 - retry + fallbacks + validation")
        print("\n生产环境建议：")
        print("  - 网络调用 → with_retry()")
        print("  - 高可用性 → with_fallbacks()")
        print("  - 数据质量 → Pydantic 验证 + 重试循环")
        print("\n下一步：")
        print("  13_rag_basics - RAG 基础（文档加载、向量存储、检索）")

    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
