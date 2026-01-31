# 项目修改记录 (Project Modifications)

本文档记录了对原 [LangChain 1.0 & LangGraph 1.0 学习项目](https://github.com/Mason-zy/Langchain1.0-Study) 的所有修改。

---

## 修改概述

| 修改项 | 原版本 | 修改后版本 |
|-------|-------|-----------|
| **API 提供商** | Groq (llama-3.3-70b-versatile) | 智谱 AI (glm-4-flash) |
| **环境变量** | `GROQ_API_KEY` | `ZHIPUAI_API_KEY` |
| **HuggingFace 镜像** | 无 | 添加 HF Mirror 支持 |
| **交互方式** | 部分模块使用 `input()` | 移除交互式输入 |

---

## 模块级别修改详情

### 模块 11: Structured Output (结构化输出)

**文件**: `phase2_practical/11_structured_output/main.py`

#### 修改内容：

1. **模型配置** (第 45-50 行)
   ```python
   # 原版本
   model = init_chat_model("groq:llama-3.3-70b-versatile")

   # 修改后
   from langchain_openai import ChatOpenAI
   model = ChatOpenAI(
       model="glm-4-flash",
       api_key=os.getenv("ZHIPUAI_API_KEY"),
       base_url="https://open.bigmodel.cn/api/paas/v4/"
   )
   ```

2. **创建 Agent** (第 116-120 行)
   - 保持使用 `create_agent`，符合 LangChain 1.0 API 规范

---

### 模块 12: Validation Retry (验证与重试)

**文件**: `phase2_practical/12_validation_retry/main.py`

#### 修改内容：

1. **模型配置** (第 47-52 行)
   ```python
   # 修改后
   from langchain_openai import ChatOpenAI
   model = ChatOpenAI(
       model="glm-4-flash",
       api_key=os.getenv("ZHIPUAI_API_KEY"),
       base_url="https://open.bigmodel.cn/api/paas/v4/"
   )
   ```

2. **创建 Agent** (第 112-116 行)
   - 使用 `create_agent` 并设置 `system_prompt`

---

### 模块 13: RAG Basics (RAG 基础)

**文件**: `phase2_practical/13_rag_basics/main.py`

#### 修改内容：

1. **HF Mirror 配置** (第 16-18 行) - 新增
   ```python
   import os
   os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
   ```
   > **重要**: 此配置必须放在所有导入之前，否则 HuggingFace 连接超时

2. **模型配置** (第 72-77 行)
   ```python
   model = ChatOpenAI(
       model="glm-4-flash",
       api_key=os.getenv("ZHIPUAI_API_KEY"),
       base_url="https://open.bigmodel.cn/api/paas/v4/"
   )
   ```

3. **Embeddings 获取函数** (第 80-105 行) - 新增
   ```python
   def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
       """获取 HuggingFace Embeddings 模型的封装函数"""
       try:
           print(f"\n正在加载 Embeddings 模型: {model_name}")
           print("使用 HF Mirror 加速 (https://hf-mirror.com)")

           embeddings = HuggingFaceEmbeddings(
               model_name=model_name,
               encode_kwargs={'normalize_embeddings': True},
           )
           print("[OK] Embeddings 模型加载成功")
           return embeddings
       except Exception as e:
           print(f"\n[ERROR] HuggingFace 加载失败: {e}")
           print("\n替代方案:")
           print("1. 设置环境变量: set HF_ENDPOINT=https://hf-mirror.com")
           print("2. 手动下载模型到本地")
           print("3. 使用其他 Embeddings 服务 (Jina AI, 智谱 AI 等)")
           raise
   ```

4. **移除 input() 交互** (main 函数)

---

### 模块 14: RAG Advanced (RAG 进阶)

**文件**: `phase2_practical/14_rag_advanced/main.py`

#### 修改内容：

1. **HF Mirror 配置** (第 16-18 行) - 新增
   ```python
   import os
   os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
   ```

2. **模型配置** (第 51-56 行)
   ```python
   model = ChatOpenAI(
       model="glm-4-flash",
       api_key=os.getenv("ZHIPUAI_API_KEY"),
       base_url="https://open.bigmodel.cn/api/paas/v4/"
   )
   ```

3. **Embeddings 获取函数** (第 60-92 行) - 新增

4. **移除 input() 交互** (main 函数)

---

## README 文档修改

### 模块 11 README

**文件**: `phase2_practical/11_structured_output/README.md`

- 更新了在线阶段代码示例，使用智谱 AI 替代 Groq
- 添加了 HF Mirror 配置说明

### 模块 12 README

**文件**: `phase2_practical/12_validation_retry/README.md`

- 同步更新了代码示例和配置说明

### 模块 13 README

**文件**: `phase2_practical/13_rag_basics/README.md`

- 更新环境变量要求（添加 `ZHIPUAI_API_KEY`）
- 添加 HF Mirror 配置说明
- 更新代码示例

### 模块 14 README

**文件**: `phase2_practical/14_rag_advanced/README.md`

- 添加 Q7: 为什么使用智谱 AI 而不是 Groq？
- 添加 Q8: 国内用户 HuggingFace 连接问题解决方案

---

## 技术要点说明

### 1. HF Mirror 配置原理

```python
# 必须在导入 HuggingFaceEmbeddings 之前设置
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

**原因**: `HuggingFaceEmbeddings` 在导入时会读取环境变量来配置下载地址。

### 2. 智谱 AI 集成方式

使用 `ChatOpenAI` 类兼容智谱 API：
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPUAI_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)
```

### 3. create_agent vs LangGraph

本项目坚持使用 LangChain 1.0 核心方法：
```python
from langchain.agents import create_agent  # ✅ 正确

# 而非
from langgraph.prebuilt import create_react_agent  # ❌ 已弃用
```

---

## 修改原因

### 为什么选择智谱 AI？

| 特性 | Groq | 智谱 AI |
|-----|------|---------|
| 费用 | 完全免费 | 有免费额度 |
| 速度 | 极快 | 快 |
| 中文支持 | 一般 | **优秀** |
| 工具调用稳定性 | 良好 | **更好** |
| 国内网络 | 需代理 | **直接访问** |

### 为什么添加 HF Mirror？

- 国内访问 HuggingFace 官方源速度慢或超时
- HF Mirror (https://hf-mirror.com) 提供国内加速镜像
- 首次下载模型后会缓存到本地，无需重复下载

---

## 测试结果

所有修改后的模块均已测试通过：

| 模块 | 测试状态 | 备注 |
|------|---------|-----|
| 模块 11 | ✅ 通过 | 结构化输出正常 |
| 模块 12 | ✅ 通过 | 验证重试机制正常 |
| 模块 13 | ✅ 通过 | RAG 基础功能正常 |
| 模块 14 | ✅ 通过 | 混合检索功能正常 |

---

## 未来计划

- [ ] 继续学习并修改模块 15 及后续模块
- [ ] 每完成一个模块，提交一次 Git commit
- [ ] 最终完成全部 22 个模块 + 3 个项目的修改

---

## 联系方式

如有问题或建议，欢迎提 Issue 或 Pull Request。
