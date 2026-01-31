# 免费 LLM API 使用指南（2025）

本指南帮助您获取和使用**完全免费**或**低成本**的 LLM API，用于 LangChain 学习。

## 🌟 推荐排行（按免费程度）

### ⭐⭐⭐ 1. Groq API（强烈推荐！）

**为什么选择 Groq？**
- ✅ **完全免费**，无需信用卡
- ✅ **速度极快**（使用专用 LPU 硬件，比 GPU 快 10倍+）
- ✅ **LangChain 原生支持**（官方 `langchain-groq` 包）
- ✅ 支持最新开源模型（Llama 3.3 70B, Mixtral 8x7B 等）

**获取步骤：**
1. 访问：https://console.groq.com/keys
2. 使用 Google/GitHub 账号登录（无需信用卡）
3. 点击 "Create API Key"
4. 复制 API Key 到 `.env` 文件

**在 LangChain 中使用：**
```bash
pip install langchain-groq
```

```python
from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.3-70b-versatile",  # 或 mixtral-8x7b-32768
    temperature=0.7,
    groq_api_key="your_groq_api_key"
)

response = model.invoke("Hello, how are you?")
print(response.content)
```

**限制：**
- 每分钟约 30 次请求（对学习足够）
- 每天约 14,400 次请求

---

### ⭐⭐⭐ 2. Google Gemini API（免费额度慷慨）

**为什么选择 Gemini？**
- ✅ 免费额度非常慷慨
- ✅ 性能优秀（Gemini 1.5 Flash 很快）
- ✅ 支持多模态（文本、图像、视频）
- ✅ Google 官方支持

**免费额度：**
- 每分钟 15 次请求
- 每天 1,500 次请求
- Gemini 1.5 Flash 和 Pro 都免费

**获取步骤：**
1. 访问：https://aistudio.google.com/apikey
2. 使用 Google 账号登录
3. 点击 "Get API Key" → "Create API Key"
4. 复制 API Key

**在 LangChain 中使用：**
```bash
pip install langchain-google-genai
```

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # 或 gemini-1.5-pro
    google_api_key="your_google_api_key"
)

response = model.invoke("Explain quantum computing")
print(response.content)
```

---

### ⭐⭐ 3. DeepSeek API（成本极低）

**为什么选择 DeepSeek？**
- ✅ 成本仅为 OpenAI 的 **2%**
- ✅ 性能接近 GPT-4
- ✅ 128K 上下文窗口
- ✅ 中国团队开发，支持中文

**价格：**
- 输入：$0.28 / 1M tokens
- 输出：$0.42 / 1M tokens
- （100万 tokens 总成本约 $0.70，而 GPT-4 需要 $30+）

**获取步骤：**
1. 访问：https://platform.deepseek.com/
2. 注册账号（需要手机号）
3. 新用户通常有免费额度
4. 获取 API Key

**在 LangChain 中使用：**
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="your_deepseek_api_key",
    openai_api_base="https://api.deepseek.com"
)
```

---

### ⭐⭐ 4. Claude API（学生/研究者免费）

**免费途径：**

#### 方法 1：学生计划（推荐）
- 网址：https://anthropic.com/students
- 条件：在校学生（需要 .edu 邮箱）
- 额度：**$500 免费额度**
- 申请：填写简单表格即可

#### 方法 2：研究者计划
- 网址：https://anthropic.com/research
- 条件：从事 AI 安全/对齐研究
- 额度：根据研究需求提供

#### 方法 3：云平台赠金
- **AWS Activate**：通过 AWS Bedrock 使用 Claude，可获得 $300-$300,000 额度
- **Google Cloud**：通过 Vertex AI 使用 Claude，新用户 $300 额度

**在 LangChain 中使用：**
```bash
pip install langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    anthropic_api_key="your_anthropic_api_key"
)
```

---

### ⭐ 5. Together AI（有免费层）

**特点：**
- 免费层可用于测试
- 支持多种开源模型
- 价格比 OpenAI 便宜

**获取步骤：**
1. 访问：https://api.together.xyz/
2. 注册账号
3. 获取免费额度

---

### ⭐ 6. Hugging Face Inference API（免费但有限制）

**特点：**
- 完全免费
- 限制：速度较慢，有请求限制
- 适合学习和实验

**获取步骤：**
1. 访问：https://huggingface.co/settings/tokens
2. 创建 Access Token
3. 使用 Inference API

```bash
pip install langchain-huggingface
```

---

## 💡 推荐组合策略

### 学习阶段建议

**阶段一（基础学习）：**
```
主要：Groq API（免费 + 快速）
备用：Google Gemini（免费额度大）
```

**阶段二-三（进阶学习）：**
```
主要：Groq API / Gemini
多模态：Google Gemini（支持图像）
备用：DeepSeek（低成本）
```

**阶段四（项目实战）：**
```
主要：DeepSeek（低成本高性能）
高质量任务：Claude（如果有学生额度）
快速任务：Groq
```

## 📊 成本对比表

| API | 1M Input Tokens | 1M Output Tokens | 特点 |
|-----|-----------------|------------------|------|
| **Groq** | **免费** | **免费** | ⚡ 速度极快 |
| **Gemini Flash** | **免费** | **免费** | 🎯 额度大 |
| DeepSeek | $0.28 | $0.42 | 💰 极低成本 |
| Claude Haiku | $0.25 | $1.25 | 🚀 快速便宜 |
| Claude Sonnet | $3.00 | $15.00 | 🧠 高质量 |
| GPT-4o | $2.50 | $10.00 | 🏆 OpenAI |
| GPT-4o mini | $0.15 | $0.60 | 📦 小模型 |

## ⚠️ 注意事项

### 1. API Key 安全
- ❌ 永远不要提交 `.env` 文件到 Git
- ✅ 使用 `.gitignore` 忽略敏感文件
- ✅ 定期轮换 API Keys
- ✅ 设置使用限额避免超支

### 2. 免费额度管理
- 📊 定期检查使用情况
- 🔄 轮换使用不同的免费 API
- 💾 本地缓存响应减少重复请求
- 🎯 开发时使用免费 API，生产时考虑付费

### 3. 速率限制
大多数免费 API 都有速率限制，注意：
- 添加重试逻辑
- 使用指数退避策略
- 避免并发请求过多

### 4. 学生身份验证
对于 Claude 学生计划：
- 需要有效的 .edu 邮箱
- 通常需要学生证明
- 额度可能有使用期限

## 🚀 快速开始

### 1. 最简单方案（Groq）

```bash
# 1. 获取 Groq API Key
# 访问：https://console.groq.com/keys

# 2. 安装依赖
pip install langchain langchain-groq

# 3. 创建 .env 文件
echo "GROQ_API_KEY=your_key_here" > .env

# 4. 测试
python phase1_fundamentals/01_hello_langchain/main.py
```

### 2. 多模型切换

在 `config.py` 中配置：
```python
import os
from dotenv import load_dotenv

load_dotenv()

# 根据环境变量自动选择可用的模型
def get_default_model():
    if os.getenv("GROQ_API_KEY"):
        return "groq:llama-3.3-70b-versatile"
    elif os.getenv("GOOGLE_API_KEY"):
        return "google:gemini-1.5-flash"
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic:claude-sonnet-4-5"
    else:
        raise ValueError("No API key found!")
```

## 📚 相关资源

- Groq 文档：https://console.groq.com/docs
- Gemini API 文档：https://ai.google.dev/docs
- DeepSeek 文档：https://platform.deepseek.com/docs
- Claude API 文档：https://docs.anthropic.com/

## ❓ 常见问题

**Q: 我是学生，最推荐哪个？**
A: 1) Groq（完全免费） 2) 申请 Claude 学生计划（$500 额度） 3) Google Gemini

**Q: Groq 和 Gemini 哪个更好？**
A: Groq 速度极快但模型选择少；Gemini 支持多模态且额度大。建议两个都用。

**Q: DeepSeek 需要付费吗？**
A: 需要，但成本极低。$5 可以用很久（相当于 OpenAI 的 $250）

**Q: 免费 API 有哪些限制？**
A: 主要是速率限制（每分钟请求数）和每日配额。对学习来说完全够用。

**Q: 可以同时使用多个 API 吗？**
A: 可以！建议配置多个 API Key，轮换使用。

---

💡 **开始使用建议：先从 Groq 开始（完全免费），然后逐步尝试其他 API！**
