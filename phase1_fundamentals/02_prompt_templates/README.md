# 02 - Prompt Templates: 提示词模板

## 学习目标

通过本模块，你将学习：

1. **为什么需要提示词模板**
   - 字符串拼接的问题
   - 模板的优势

2. **PromptTemplate**
   - 基本用法
   - 变量替换
   - 格式化方法

3. **ChatPromptTemplate**
   - 聊天消息模板
   - 多角色支持
   - 对话历史管理

4. **高级特性**
   - 部分变量
   - 模板组合
   - 可复用模板库

5. **LCEL 链式调用**
   - 模板与模型的组合
   - 管道运算符

---

## 核心概念详解

### 1. 为什么需要提示词模板？

#### 🔴 问题：字符串拼接的缺点

```python
# ❌ 不推荐的做法
user_name = "张三"
topic = "Python"

prompt = f"你好 {user_name}，我来帮你学习 {topic}"
```

**问题：**
- ❌ 难以维护和修改
- ❌ 容易出现格式错误
- ❌ 不能复用
- ❌ 难以测试
- ❌ 混合了逻辑和数据

#### ✅ 解决方案：使用模板

```python
# ✅ 推荐的做法
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "你好 {user_name}，我来帮你学习 {topic}"
)

prompt = template.format(user_name="张三", topic="Python")
```

**优势：**
- ✅ **可复用** - 一个模板，多次使用
- ✅ **可维护** - 模板和数据分离
- ✅ **类型安全** - 自动验证变量
- ✅ **可测试** - 更容易编写测试
- ✅ **可组合** - 可以组合多个模板

---

### 2. PromptTemplate - 简单文本模板

`PromptTemplate` 用于创建**简单的文本提示词**，适合单一提示的场景。

#### 基本语法

```python
from langchain_core.prompts import PromptTemplate

# 方法 1：from_template（最简单，推荐）
template = PromptTemplate.from_template("你的模板文本 {变量名}")

# 方法 2：完整定义
template = PromptTemplate(
    input_variables=["变量1", "变量2"],
    template="你的模板文本 {变量1} 和 {变量2}"
)
```

#### 创建模板的三种方法

**方法 1：from_template（推荐）**

```python
template = PromptTemplate.from_template(
    "将以下文本翻译成{language}：\n{text}"
)

# 自动识别变量
print(template.input_variables)  # ['language', 'text']
```

**方法 2：显式指定变量**

```python
template = PromptTemplate(
    input_variables=["product", "feature"],
    template="为{product}写一句广告语，重点突出{feature}特点。"
)
```

**方法 3：部分变量预填充**

```python
template = PromptTemplate.from_template(
    "你是一个{role}，请{task}"
)

# 预填充 role
partial_template = template.partial(role="Python 导师")

# 现在只需要提供 task
prompt = partial_template.format(task="解释装饰器")
```

#### 使用模板

**方式 1：format() - 返回字符串**

```python
template = PromptTemplate.from_template("你好 {name}")

# 返回格式化后的字符串
prompt_str = template.format(name="张三")
print(prompt_str)  # "你好 张三"

# 直接传递给模型
response = model.invoke(prompt_str)
```

**方式 2：invoke() - 返回 PromptValue**

```python
template = PromptTemplate.from_template("你好 {name}")

# 返回 PromptValue 对象
prompt_value = template.invoke({"name": "张三"})

# 获取文本
print(prompt_value.text)  # "你好 张三"
```

#### 实用示例

**示例 1：翻译模板**

```python
translator = PromptTemplate.from_template(
    "将以下{source_lang}文本翻译成{target_lang}：\n{text}"
)

prompt = translator.format(
    source_lang="英语",
    target_lang="中文",
    text="Hello, how are you?"
)
```

**示例 2：代码生成模板**

```python
code_generator = PromptTemplate.from_template(
    "用{language}编写一个{functionality}的函数。\n"
    "要求：\n"
    "1. {requirement1}\n"
    "2. {requirement2}"
)

prompt = code_generator.format(
    language="Python",
    functionality="计算斐波那契数列",
    requirement1="使用递归实现",
    requirement2="添加类型注解"
)
```

---

### 3. ChatPromptTemplate - 聊天消息模板

`ChatPromptTemplate` 用于创建**聊天格式的消息**，支持多种角色（system、user、assistant）。

#### 为什么需要 ChatPromptTemplate？

**PromptTemplate vs ChatPromptTemplate：**

| 特性 | PromptTemplate | ChatPromptTemplate |
|------|----------------|-------------------|
| 输出格式 | 纯文本字符串 | 消息列表 |
| 角色支持 | ❌ 无 | ✅ system/user/assistant |
| 对话历史 | ❌ 不支持 | ✅ 支持 |
| 适用场景 | 简单提示 | 聊天、对话、多轮交互 |

#### 基本语法

```python
from langchain_core.prompts import ChatPromptTemplate

# 使用元组格式（推荐）
template = ChatPromptTemplate.from_messages([
    ("system", "系统提示"),
    ("user", "用户消息 {variable}"),
    ("assistant", "AI 回复"),
    ("user", "下一个用户消息")
])
```

#### 消息类型

| 角色字符串 | 含义 | 用途 |
|-----------|------|------|
| `"system"` | 系统消息 | 设定 AI 的行为、角色、规则 |
| `"user"` / `"human"` | 用户消息 | 用户的输入/问题 |
| `"assistant"` / `"ai"` | AI 消息 | AI 的回复（用于对话历史） |

#### 创建方法

**方法 1：元组格式（最简单，推荐）**

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("user", "{question}")
])

messages = template.format_messages(
    role="Python 导师",
    
    question="什么是装饰器？"
)
```

**方法 2：字符串简写**

```python
# 单独的字符串会被解释为 user 消息
template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    "{user_input}"  # 相当于 ("user", "{user_input}")
])
```

**方法 3：使用 MessagePromptTemplate（高级）**

```python
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

system_template = SystemMessagePromptTemplate.from_template(
    "你是一个{role}"
)
human_template = HumanMessagePromptTemplate.from_template(
    "{question}"
)

template = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])
```

#### 使用模板

**方式 1：format_messages() - 返回消息列表**

```python
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("user", "{input}")
])

# 返回消息列表
messages = template.format_messages(
    role="助手",
    input="你好"
)

# 直接传递给模型
response = model.invoke(messages)
```

**方式 2：invoke() - 返回 ChatPromptValue**

```python
# 返回 ChatPromptValue 对象
prompt_value = template.invoke({
    "role": "助手",
    "input": "你好"
})

# 获取消息列表
messages = prompt_value.to_messages()
```

#### 实用示例

**示例 1：简单聊天**

```python
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的{role}，擅长{skill}"),
    ("user", "{question}")
])

messages = chat_template.format_messages(
    role="编程导师",
    skill="用简单语言解释复杂概念",
    question="什么是递归？"
)

response = model.invoke(messages)
```

**示例 2：多轮对话**

```python
conversation_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("user", "{question1}"),
    ("assistant", "{answer1}"),
    ("user", "{question2}")
])

messages = conversation_template.format_messages(
    role="Python 专家",
    question1="什么是列表？",
    answer1="列表是 Python 的有序可变集合。",
    question2="它和元组有什么区别？"  # 基于上下文
)
```

**示例 3：结构化指令**

```python
structured_template = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个{domain}专家。\n"
     "回答风格：{style}\n"
     "回答长度：{length}字以内"),
    ("user", "{question}")
])

messages = structured_template.format_messages(
    domain="机器学习",
    style="技术性强、简洁",
    length="100",
    question="什么是梯度下降？"
)
```

---

### 4. 高级特性

#### 4.1 部分变量（Partial Variables）

预填充某些固定不变的变量，创建模板的变体。

**使用场景：**
- 某些变量在所有调用中都相同
- 需要为不同用户/场景创建定制模板

**语法：**

```python
# 原始模板
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}，目标用户是{audience}"),
    ("user", "{task}")
])

# 部分填充
customer_support_template = template.partial(
    role="客服专员",
    audience="普通用户"
)

# 现在只需要提供 task
messages = customer_support_template.format_messages(
    task="解释退款政策"
)
```

**实用示例：**

```python
# 基础翻译模板
translator = ChatPromptTemplate.from_messages([
    ("system", "你是专业翻译，精通{source}和{target}"),
    ("user", "翻译：{text}")
])

# 创建英译中的专用模板
en_to_zh = translator.partial(source="英语", target="中文")

# 创建中译英的专用模板
zh_to_en = translator.partial(source="中文", target="英语")

# 使用
messages1 = en_to_zh.format_messages(text="Hello")
messages2 = zh_to_en.format_messages(text="你好")
```

#### 4.2 模板组合

将多个模板片段组合成复杂的提示词。

**方法 1：字符串组合**

```python
# 定义可复用的部分
role_part = "你是一个{domain}专家。"
style_part = "回答风格：{style}。"
constraint_part = "限制：{constraint}。"

# 组合
full_system = role_part + style_part + constraint_part

template = ChatPromptTemplate.from_messages([
    ("system", full_system),
    ("user", "{question}")
])
```

**方法 2：使用 + 运算符**

```python
template1 = ChatPromptTemplate.from_messages([
    ("system", "你是助手")
])

template2 = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

# 组合（LangChain 1.0 支持）
combined = template1 + template2
```

#### 4.3 可复用模板库

在实际项目中，建议创建模板库。

**示例：模板库**

```python
# templates.py
from langchain_core.prompts import ChatPromptTemplate

class PromptLibrary:
    """可复用的提示词模板库"""

    TRANSLATOR = ChatPromptTemplate.from_messages([
        ("system", "你是专业翻译，精通{source_lang}和{target_lang}"),
        ("user", "翻译以下文本：\n{text}")
    ])

    CODE_REVIEWER = ChatPromptTemplate.from_messages([
        ("system", "你是{language}代码审查专家，重点关注{focus}"),
        ("user", "审查代码：\n```{language}\n{code}\n```")
    ])

    SUMMARIZER = ChatPromptTemplate.from_messages([
        ("system", "你是内容摘要专家"),
        ("user", "将以下内容总结为{num}个要点：\n{content}")
    ])

    TUTOR = ChatPromptTemplate.from_messages([
        ("system", "你是{subject}导师，学生水平：{level}"),
        ("user", "{question}")
    ])

# 使用
from templates import PromptLibrary

messages = PromptLibrary.TRANSLATOR.format_messages(
    source_lang="英语",
    target_lang="中文",
    text="Hello World"
)
```

---

### 5. LCEL 链式调用（预览）

**LCEL** = LangChain Expression Language，LangChain 的表达式语言。

#### 什么是链（Chain）？

链是将多个组件连接在一起的方式，形成处理流程。

```
输入 → 模板 → 模型 → 输出
```

#### 使用管道运算符 `|`

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# 创建组件
template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("user", "{input}")
])

model = init_chat_model("groq:llama-3.3-70b-versatile")

# 使用 | 创建链
chain = template | model

# 直接调用链
response = chain.invoke({
    "role": "Python 导师",
    "input": "什么是装饰器？"
})

print(response.content)
```

#### 链的优势

| 优势 | 说明 |
|------|------|
| **简洁** | 一行代码完成多步操作 |
| **可读** | 清晰展示数据流向 |
| **可组合** | 可以轻松添加/删除组件 |
| **可复用** | 链本身可以作为组件 |

**详细内容将在后续模块学习。**

---

## 完整示例代码说明

`main.py` 包含 9 个渐进式示例：

1. **示例 1** - 为什么需要模板？对比字符串拼接
2. **示例 2** - PromptTemplate 基础（3种创建方法）
3. **示例 3** - ChatPromptTemplate 基础（3种创建方法）
4. **示例 4** - 多轮对话模板
5. **示例 5** - MessagePromptTemplate 类（高级）
6. **示例 6** - 部分变量预填充
7. **示例 7** - 模板组合
8. **示例 8** - 可复用模板库
9. **示例 9** - LCEL 链式调用

---

## 运行示例

```bash
cd phase1_fundamentals/02_prompt_templates
python main.py
```

---

## 常见问题 (FAQ)

### Q1: PromptTemplate 和 ChatPromptTemplate 有什么区别?

**A:**

| 特性 | PromptTemplate | ChatPromptTemplate |
|------|----------------|-------------------|
| 输出 | 字符串 | 消息列表 |
| 角色 | 无 | system/user/assistant |
| 适用场景 | 简单提示 | 聊天、对话 |

**建议：**
- 简单场景 → `PromptTemplate`
- 聊天场景 → `ChatPromptTemplate`（推荐）

### Q2: 什么时候使用部分变量？

**A:** 当某些变量在多次调用中保持不变时：

```python
# 场景：为不同部门创建专用模板
base_template = ChatPromptTemplate.from_messages([
    ("system", "你是{department}的{role}"),
    ("user", "{task}")
])

# IT 部门
it_template = base_template.partial(
    department="IT 部门",
    role="技术支持"
)

# 销售部门
sales_template = base_template.partial(
    department="销售部门",
    role="销售顾问"
)
```

### Q3: 如何在模板中使用换行和特殊字符？

**A:** 使用三引号字符串：

```python
template = PromptTemplate.from_template("""
你是一个{role}。

请完成以下任务：
1. {task1}
2. {task2}

注意事项：
- {note1}
- {note2}
""")
```

### Q4: 模板变量可以是什么类型？

**A:** 通常是字符串，但也可以是其他可转换为字符串的类型：

```python
template = PromptTemplate.from_template(
    "生成{count}个关于{topic}的想法"
)

# count 是整数
prompt = template.format(count=5, topic="创新")
```

### Q5: 如何处理可选变量？

**A:** 使用部分变量或默认值：

```python
# 方法 1：部分变量
template = PromptTemplate.from_template(
    "{greeting} {name}，{message}"
)
template_with_default = template.partial(greeting="你好")

# 方法 2：在应用层处理
def create_prompt(name, message, greeting="你好"):
    return template.format(
        greeting=greeting,
        name=name,
        message=message
    )
```

---

## 最佳实践

### 1. 模板命名规范

```python
# ✅ 好的命名
translator_template = ...
code_review_template = ...
customer_support_template = ...

# ❌ 不好的命名
template1 = ...
t = ...
my_template = ...
```

### 2. 组织模板

```python
# templates/
# ├── __init__.py
# ├── common.py        # 通用模板
# ├── translation.py   # 翻译相关
# └── coding.py        # 编程相关

# common.py
from langchain_core.prompts import ChatPromptTemplate

FRIENDLY_ASSISTANT = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的助手"),
    ("user", "{input}")
])
```

### 3. 文档化模板

```python
class Templates:
    """项目模板库"""

    TRANSLATOR = ChatPromptTemplate.from_messages([
        ("system", "你是专业翻译"),
        ("user", "翻译：{text}")
    ])
    """
    翻译模板

    变量:
        - text: 要翻译的文本

    示例:
        messages = TRANSLATOR.format_messages(text="Hello")
    """
```

### 4. 测试模板

```python
def test_translator_template():
    """测试翻译模板"""
    template = PromptLibrary.TRANSLATOR

    # 测试变量识别
    assert "text" in template.input_variables

    # 测试格式化
    messages = template.format_messages(text="Hello")
    assert len(messages) == 2
    assert messages[0].type == "system"
    assert messages[1].type == "user"
```

---

## 下一步学习

完成本模块后，继续学习：

1. **03_messages** - 深入理解消息类型和对话管理
2. **04_custom_tools** - 创建自定义工具
3. **05_simple_agent** - 使用 `create_agent` 构建第一个 Agent

---

## 参考资源

- [LangChain Prompts 文档](https://docs.langchain.com/oss/python/docs/how_to/prompts)
- [ChatPromptTemplate API](https://docs.langchain.com/oss/python/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
- [LCEL 文档](https://docs.langchain.com/oss/python/docs/concepts/lcel)

---

## 小结

通过本模块，你已经学习了：

- ✅ 为什么需要提示词模板
- ✅ PromptTemplate 的基本用法
- ✅ ChatPromptTemplate 的强大功能
- ✅ 部分变量的应用
- ✅ 模板组合技巧
- ✅ 构建可复用模板库
- ✅ LCEL 链式调用预览

**恭喜！你已经掌握了 LangChain 提示词模板的核心知识！** 🎉
