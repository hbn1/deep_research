"""
RAG 入库模块：CLI 入口 + 运行时动态入库支持。
支持格式：TXT / MD / PDF / DOCX / 代码文件
"""
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from app.mult_agents.rag.core import RAGSystem, RAGConfig, RAGManager
from app.mult_agents.config import AppConfig

INPUT_PATH = Path(__file__).resolve().parents[3] / "front" / "agent_front" / "README.md"
COLLECTION_NAME = ""
MILVUS_HOST = ""
MILVUS_PORT = 0
EMBEDDING_MODEL = "text-embedding-v1"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def _collect_paths(input_path):
    if input_path.is_file():
        return [input_path]
    patterns = ("*.txt", "*.md", "*.markdown", "*.py", "*.js", "*.ts", "*.java", "*.go", "*.rs")
    paths = []
    for pat in patterns:
        paths.extend(sorted(input_path.rglob(pat)))
    return paths


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = AppConfig.from_file()
    collection_name = COLLECTION_NAME or config.milvus_collection
    milvus_host = MILVUS_HOST or config.milvus_host
    milvus_port = MILVUS_PORT or config.milvus_port

    rag_cfg = RAGConfig(
        milvus_host=milvus_host,
        milvus_port=milvus_port,
        collection_name=collection_name,
        embedding_model=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    rag = RAGSystem(api_key=config.api_key, config=rag_cfg)

    input_path = INPUT_PATH.expanduser().resolve()
    if not input_path.exists():
        print(f"Input path not found: {input_path}")
        return

    paths = _collect_paths(input_path)
    if not paths:
        print(f"No ingestable files found: {input_path}")
        return

    total_chunks = rag.ingest_paths(paths)
    print(f"Ingestion complete | files={len(paths)} | chunks={total_chunks} | collection={collection_name}")


if __name__ == "__main__":
    main()