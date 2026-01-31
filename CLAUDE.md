# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **comprehensive learning repository for LangChain 1.0 and LangGraph 1.0**, designed as a systematic tutorial for developers to learn how to build LLM-driven applications. The codebase covers everything from basic concepts to advanced implementations and production-ready projects.

## Setup and Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv (faster):
uv pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Required Environment Variables

Minimum required in `.env`:
- `GROQ_API_KEY` - Groq API (free, recommended) - Get from https://console.groq.com/keys

Optional:
- `OPENAI_API_KEY` - Required for image processing modules (19, 21)
- `LANGCHAIN_API_KEY` - For LangSmith monitoring/observability
- `PINECONE_API_KEY` - For advanced RAG examples

### Verify Installation

```bash
python phase1_fundamentals/01_hello_langchain/main.py
```

## Running Code

Each module is self-contained with its own `main.py`:

```bash
# Run a specific module
cd phase1_fundamentals/01_hello_langchain
python main.py

# Run tests (if available)
python test.py
```

## Architecture

### Project Structure

The repository is organized into **4 progressive phases**:

1. **Phase 1: Fundamentals** (`phase1_fundamentals/`) - Basic LLM interactions, prompts, messages, tools, simple agents
2. **Phase 2: Practical** (`phase2_practical/`) - Memory, context management, checkpointing, middleware, structured outputs, RAG basics
3. **Phase 3: Advanced** (`phase3_advanced/`) - LangGraph, multi-agent systems, routing, multi-modal processing, LangSmith, error handling
4. **Phase 4: Projects** (`phase4_projects/`) - Production-ready systems (RAG, multi-agent support, research assistant)

### Key Technologies

- **LangChain 1.0+** - Core framework with new `create_agent` API
- **LangGraph 1.0+** - Agent runtime with persistence and streaming
- **Groq** - Primary model provider (free and fast)
- **ChromaDB** - Default vector store for RAG
- **SQLite** - Default checkpoint storage via `langgraph-checkpoint-sqlite`

### LangChain 1.0 Critical API Changes

**IMPORTANT**: This codebase uses LangChain 1.0 APIs, which have significant changes from earlier versions:

1. **Agent Creation** - Use `create_agent` from `langchain.agents`:
   ```python
   from langchain.agents import create_agent  # ✅ Correct
   # NOT: from langgraph.prebuilt import create_react_agent  # ❌ Deprecated
   ```

2. **Model Initialization**:
   ```python
   from langchain.chat_models import init_chat_model
   model = init_chat_model("groq:llama-3.3-70b-versatile")
   ```

3. **Agent Configuration**:
   ```python
   agent = create_agent(
       model=model,
       tools=[tool1, tool2],
       system_prompt="Agent behavior instructions"  # LangChain 1.0 uses system_prompt
   )
   ```

4. **For Advanced Customization** - When `create_agent` is insufficient, use LangGraph directly:
   ```python
   from langgraph.graph import StateGraph, START, END
   from typing import TypedDict, Annotated
   from langgraph.graph.message import add_messages

   class State(TypedDict):
       messages: Annotated[list, add_messages]

   graph = StateGraph(State)
   graph.add_node("chat", chat_node)
   graph.add_edge(START, "chat")
   graph.add_edge("chat", END)
   app = graph.compile()
   ```

### Module Structure

Each module follows a consistent pattern:
- `main.py` - Main tutorial code with comprehensive examples and Chinese docstrings
- `test.py` - Unit tests (where applicable)
- Subdirectories for tools, data, or resources as needed

### Code Style

- All docstrings and comments are in **Chinese**
- Comprehensive examples with detailed explanations
- Progressive learning curve - each module builds on previous concepts
- Production-ready code with error handling and validation

## Development Notes

### Testing

```bash
# Run tests for a module
cd phase1_fundamentals/01_hello_langchain
python test.py

# Run with pytest (if configured)
pytest
```

### Code Quality Tools

The repository includes:
- `black` - Code formatting
- `isort` - Import sorting
- `flake8` - Linting
- `mypy` - Type checking
- `pytest` - Testing framework

### Common Issues

1. **Agent not calling tools**: Check tool docstrings are clear and descriptive - the AI reads these to understand when to use tools
2. **No conversation memory**: Agents require full message history, not just the latest message
3. **Import errors**: Ensure you're using LangChain 1.0 import paths (see Critical API Changes above)

## Learning Path

Follow the phases sequentially for best results:
1. Start with `phase1_fundamentals/01_hello_langchain`
2. Progress through each module in order
3. Each phase assumes knowledge from previous phases
4. Phase 4 projects combine all learned concepts

## Reference Resources

- LangChain Documentation: https://docs.langchain.com/oss/python/langchain/
- LangGraph Documentation: https://docs.langchain.com/oss/python/langgraph
- Migration Guide: https://docs.langchain.com/oss/python/migrate/langchain-v1
