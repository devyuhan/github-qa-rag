"""GitHub repo fetching, chunking, and Pinecone upsert."""

import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from llama_index.core import (
    Settings as LISettings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import Settings

# File extensions considered documentation (use SentenceSplitter).
_DOC_EXTENSIONS = {".md", ".rst", ".txt"}

# Map file extension to tree-sitter language for CodeSplitter.
_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".yaml": "yaml",
    ".json": "json",
}


def _clone_repo(owner: str, repo: str, branch: str, token: str) -> Path:
    """Shallow-clone a GitHub repo into a temp directory. Returns the path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"ghqa_{repo}_"))
    url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"
    print(f"  Cloning {owner}/{repo} (branch={branch}) ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, url, str(tmp_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return tmp_dir


def _classify(file_path: str) -> tuple[str, bool]:
    """Return (file_type extension, is_doc bool) for a path."""
    ext = PurePosixPath(file_path).suffix.lower()
    return ext, ext in _DOC_EXTENSIONS


def _split_documents(documents: list[Document]) -> list:
    """Split documents using CodeSplitter for code and SentenceSplitter for docs."""
    sentence_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    code_nodes = []
    doc_nodes = []

    for doc in documents:
        file_path = doc.metadata.get("file_path", "")
        ext = PurePosixPath(file_path).suffix.lower()

        if ext in _DOC_EXTENSIONS:
            doc_nodes.append(doc)
        elif ext in _LANG_MAP:
            lang = _LANG_MAP[ext]
            splitter = CodeSplitter(
                language=lang, chunk_lines=40, chunk_lines_overlap=15, max_chars=1500
            )
            code_nodes.extend(splitter.get_nodes_from_documents([doc]))
        else:
            doc_nodes.append(doc)

    text_nodes = sentence_splitter.get_nodes_from_documents(doc_nodes)
    return code_nodes + text_nodes


def _enrich_metadata(
    documents: list[Document], owner: str, repo: str, repo_root: Path
) -> None:
    """Add contextual chunk headers and metadata to each document in-place."""
    for doc in documents:
        # Convert absolute path to repo-relative path
        abs_path = doc.metadata.get("file_path", "unknown")
        try:
            rel_path = str(Path(abs_path).relative_to(repo_root))
        except ValueError:
            rel_path = abs_path
        doc.metadata["file_path"] = rel_path

        ext, is_doc = _classify(rel_path)
        directory = str(PurePosixPath(rel_path).parent)

        doc.metadata.update(
            {"file_type": ext, "directory": directory, "is_doc": is_doc}
        )
        # Prepend provenance header so the LLM knows file context.
        header = f"[{owner}/{repo}] {rel_path}\n\n"
        doc.set_content(header + doc.text)


def _get_or_create_pinecone_index(settings: Settings):
    """Return a Pinecone index object, creating it if needed."""
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [i.name for i in pc.list_indexes()]
    if settings.pinecone_index_name not in existing:
        print(f"Creating Pinecone index '{settings.pinecone_index_name}' ...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embed_dimensions,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(settings.pinecone_index_name)


def ingest(
    owner: str,
    repo: str,
    branch: str = "main",
    extensions: list[str] | None = None,
) -> None:
    """Clone a GitHub repo, chunk it, and upsert into Pinecone."""
    settings = Settings()

    # Configure global LlamaIndex settings
    LISettings.embed_model = OpenAIEmbedding(
        model=settings.embed_model_name,
        api_key=settings.openai_api_key,
    )

    exts = extensions or list(settings.default_extensions)

    # --- 1. Shallow-clone the repo ---
    print(f"Loading {owner}/{repo} (branch={branch}) ...")
    repo_dir = _clone_repo(owner, repo, branch, settings.github_token)

    try:
        # Build glob patterns for SimpleDirectoryReader
        required_exts = [e if e.startswith(".") else f".{e}" for e in exts]

        documents = SimpleDirectoryReader(
            input_dir=str(repo_dir),
            required_exts=required_exts,
            recursive=True,
            exclude=[
                "node_modules",
                ".git",
                "__pycache__",
                "*.lock",
                "*.min.js",
            ],
        ).load_data()
        print(f"  Loaded {len(documents)} file(s).")

        # --- 2. Enrich metadata + contextual headers ---
        _enrich_metadata(documents, owner, repo, repo_dir)

        # --- 3. Split ---
        nodes = _split_documents(documents)
        print(f"  Created {len(nodes)} chunk(s).")

        # --- 4. Upsert into Pinecone ---
        pinecone_index = _get_or_create_pinecone_index(settings)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("  Upserting into Pinecone ...")
        VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
        )
        print("Done.")
    finally:
        # Clean up the temp clone
        shutil.rmtree(repo_dir, ignore_errors=True)
