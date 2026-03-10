"""
LangChain 知识库构建模块

本模块负责从 LangChain 学习项目中提取和构建知识库：
- 解析所有模块的 README.md 和 main.py 文件
- 提取代码注释和文档字符串
- 构建结构化的元数据
- 支持增量更新
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ast
import re

# 设置 UTF-8 编码输出（解决 Windows emoji 显示问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from langchain_core.documents import Document


@dataclass
class KnowledgeConfig:
    """知识库配置"""
    # 项目根目录
    project_root: str = "."
    # 排除的目录
    exclude_dirs: List[str] = None
    # 包含的文件模式
    include_patterns: List[str] = None
    # 最大代码块大小
    max_code_chunk_size: int = 2000
    # 代码块重叠
    code_chunk_overlap: int = 200

    def __post_init__(self):
        if self.exclude_dirs is None:
            self.exclude_dirs = [
                "__pycache__",
                ".git",
                ".venv",
                "venv",
                "node_modules",
                ".pytest_cache",
                "sample_docs",
                "templates"
            ]
        if self.include_patterns is None:
            self.include_patterns = ["*.md", "*.py", "*.txt"]


class CodeExtractor:
    """从 Python 代码中提取文档和注释"""

    @staticmethod
    def extract_docstring(code: str) -> str:
        """提取模块级 docstring"""
        try:
            tree = ast.parse(code)
            if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                node = tree.body[0]
                if ast.get_docstring(node):
                    return ast.get_docstring(node)
            # 检查模块级 docstring
            if ast.get_docstring(tree):
                return ast.get_docstring(tree)
        except:
            pass
        return ""

    @staticmethod
    def extract_functions(code: str) -> List[Dict[str, str]]:
        """提取所有函数及其文档字符串"""
        functions = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "args": [arg.arg for arg in node.args.args],
                        "lineno": node.lineno
                    }
                    functions.append(func_info)
        except:
            pass
        return functions

    @staticmethod
    def extract_classes(code: str) -> List[Dict[str, str]]:
        """提取所有类及其文档字符串"""
        classes = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "docstring": ast.get_docstring(node) or "",
                        "methods": [],
                        "lineno": node.lineno
                    }
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_info = {
                                "name": item.name,
                                "docstring": ast.get_docstring(item) or ""
                            }
                            class_info["methods"].append(method_info)
                    classes.append(class_info)
        except:
            pass
        return classes

    @staticmethod
    def extract_comments(code: str) -> List[str]:
        """提取注释"""
        comments = []
        # 提取单行注释
        for line in code.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') and not stripped.startswith('#!'):
                comments.append(stripped[1:].strip())
        return comments


class LangChainKnowledgeBuilder:
    """LangChain 知识库构建器"""

    def __init__(self, config: KnowledgeConfig = None):
        self.config = config or KnowledgeConfig()
        self.code_extractor = CodeExtractor()
        self.project_root = Path(self.config.project_root)

    def find_modules(self) -> List[Path]:
        """查找所有模块目录"""
        modules = []
        phases = ["phase1_fundamentals", "phase2_practical",
                  "phase3_advanced", "phase4_projects"]

        for phase in phases:
            phase_path = self.project_root / phase
            if phase_path.exists():
                for item in phase_path.iterdir():
                    if item.is_dir() and item.name not in self.config.exclude_dirs:
                        modules.append(item)

        return sorted(modules)

    def parse_module(self, module_path: Path) -> List[Document]:
        """解析单个模块，返回文档列表"""
        documents = []
        module_name = module_path.name
        relative_path = module_path.relative_to(self.project_root)

        # 解析 README.md
        readme_path = module_path / "README.md"
        if readme_path.exists():
            docs = self._parse_markdown(readme_path, module_name, relative_path)
            documents.extend(docs)

        # 解析 main.py
        main_path = module_path / "main.py"
        if main_path.exists():
            docs = self._parse_python_file(main_path, module_name, relative_path)
            documents.extend(docs)

        # 解析其他 Python 文件（如子模块）
        for py_file in module_path.glob("*.py"):
            if py_file.name != "main.py" and not py_file.name.startswith("_"):
                docs = self._parse_python_file(py_file, module_name, relative_path)
                documents.extend(docs)

        return documents

    def _parse_markdown(self, file_path: Path, module_name: str,
                       relative_path: Path) -> List[Document]:
        """解析 Markdown 文件"""
        documents = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 分块处理 Markdown
            sections = self._split_markdown_sections(content)

            for section_name, section_content in sections:
                metadata = {
                    "source": str(file_path.relative_to(self.project_root)),
                    "module": module_name,
                    "type": "markdown",
                    "section": section_name,
                    "phase": str(relative_path.parts[0]) if len(relative_path.parts) > 0 else ""
                }

                doc = Document(
                    page_content=section_content,
                    metadata=metadata
                )
                documents.append(doc)

        except Exception as e:
            print(f"⚠️ 解析 Markdown 文件失败 {file_path}: {e}")

        return documents

    def _split_markdown_sections(self, content: str) -> List[tuple]:
        """将 Markdown 按标题分块"""
        sections = []
        current_section = "概述"
        current_content = []

        for line in content.split('\n'):
            # 检测标题（一级到三级）
            if line.startswith('### '):
                # 保存当前section
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = line[4:].strip()
                current_content = [line]
            elif line.startswith('## '):
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = line[3:].strip()
                current_content = [line]
            elif line.startswith('# '):
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = line[2:].strip()
                current_content = [line]
            else:
                current_content.append(line)

        # 添加最后一个section
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))

        return sections

    def _parse_python_file(self, file_path: Path, module_name: str,
                          relative_path: Path) -> List[Document]:
        """解析 Python 文件，提取代码和文档"""
        documents = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # 提取模块级文档
            module_doc = self.code_extractor.extract_docstring(code)
            if module_doc:
                metadata = {
                    "source": str(file_path.relative_to(self.project_root)),
                    "module": module_name,
                    "type": "module_docstring",
                    "phase": str(relative_path.parts[0]) if len(relative_path.parts) > 0 else ""
                }
                documents.append(Document(page_content=module_doc, metadata=metadata))

            # 提取函数文档
            functions = self.code_extractor.extract_functions(code)
            for func in functions:
                if func["docstring"]:
                    content = f"函数 {func['name']}:\n参数: {', '.join(func['args'])}\n{func['docstring']}"
                    metadata = {
                        "source": str(file_path.relative_to(self.project_root)),
                        "module": module_name,
                        "type": "function",
                        "function_name": func["name"],
                        "lineno": func["lineno"],
                        "phase": str(relative_path.parts[0]) if len(relative_path.parts) > 0 else ""
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

            # 提取类文档
            classes = self.code_extractor.extract_classes(code)
            for cls in classes:
                if cls["docstring"]:
                    # 类文档
                    content = f"类 {cls['name']}:\n{cls['docstring']}"
                    metadata = {
                        "source": str(file_path.relative_to(self.project_root)),
                        "module": module_name,
                        "type": "class",
                        "class_name": cls["name"],
                        "lineno": cls["lineno"],
                        "phase": str(relative_path.parts[0]) if len(relative_path.parts) > 0 else ""
                    }
                    documents.append(Document(page_content=content, metadata=metadata))

                # 方法文档
                for method in cls["methods"]:
                    if method["docstring"]:
                        content = f"类 {cls['name']} 的方法 {method['name']}:\n{method['docstring']}"
                        metadata = {
                            "source": str(file_path.relative_to(self.project_root)),
                            "module": module_name,
                            "type": "method",
                            "class_name": cls["name"],
                            "method_name": method["name"],
                            "phase": str(relative_path.parts[0]) if len(relative_path.parts) > 0 else ""
                        }
                        documents.append(Document(page_content=content, metadata=metadata))

            # 提取重要注释（以特定标记开头的）
            important_comments = []
            for line in code.split('\n'):
                stripped = line.strip()
                if stripped.startswith('# ') and any(keyword in line for keyword in
                    ['核心', '重要', '说明', '配置', '实现', '原理', '示例']):
                    important_comments.append(stripped[1:].strip())

            if important_comments:
                content = "重要注释:\n" + "\n".join(important_comments)
                metadata = {
                    "source": str(file_path.relative_to(self.project_root)),
                    "module": module_name,
                    "type": "comments",
                    "phase": str(relative_path.parts[0]) if len(relative_path.parts) > 0 else ""
                }
                documents.append(Document(page_content=content, metadata=metadata))

        except Exception as e:
            print(f"⚠️ 解析 Python 文件失败 {file_path}: {e}")

        return documents

    def build_knowledge_base(self) -> List[Document]:
        """构建完整的知识库"""
        all_documents = []
        modules = self.find_modules()

        print(f"📚 发现 {len(modules)} 个学习模块")
        print("=" * 60)

        for i, module_path in enumerate(modules, 1):
            module_name = module_path.name
            print(f"📖 [{i}/{len(modules)}] 解析模块: {module_name}")

            docs = self.parse_module(module_path)
            print(f"   提取了 {len(docs)} 个文档片段")
            all_documents.extend(docs)

        print("=" * 60)
        print(f"✅ 知识库构建完成！共 {len(all_documents)} 个文档片段")

        # 打印统计信息
        self._print_statistics(all_documents)

        return all_documents

    def _print_statistics(self, documents: List[Document]):
        """打印知识库统计信息"""
        print("\n📊 知识库统计:")
        print("-" * 60)

        # 按类型统计
        type_counts = {}
        phase_counts = {}
        module_counts = {}

        for doc in documents:
            doc_type = doc.metadata.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            phase = doc.metadata.get("phase", "unknown")
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

            module = doc.metadata.get("module", "unknown")
            module_counts[module] = module_counts.get(module, 0) + 1

        print(f"按文档类型:")
        for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {doc_type}: {count}")

        print(f"\n按学习阶段:")
        for phase, count in sorted(phase_counts.items()):
            print(f"  - {phase}: {count}")

        print(f"\n文档最多的模块 (Top 5):")
        for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {module}: {count}")

        print("-" * 60)


# ==================== 便捷函数 ====================

def build_knowledge_base_from_project(project_root: str = ".") -> List[Document]:
    """
    从项目根目录构建知识库

    Args:
        project_root: 项目根目录路径

    Returns:
        文档列表
    """
    config = KnowledgeConfig(project_root=project_root)
    builder = LangChainKnowledgeBuilder(config)
    return builder.build_knowledge_base()


if __name__ == "__main__":
    # 测试知识库构建
    import sys
    project_root = Path(__file__).parent.parent.parent
    print(f"项目根目录: {project_root}")
    print()

    documents = build_knowledge_base_from_project(str(project_root))

    # 示例：打印前3个文档
    print("\n📄 示例文档（前3个）:")
    print("=" * 60)
    for i, doc in enumerate(documents[:3], 1):
        print(f"\n[文档 {i}]")
        print(f"来源: {doc.metadata.get('source')}")
        print(f"类型: {doc.metadata.get('type')}")
        print(f"内容预览: {doc.page_content[:200]}...")
