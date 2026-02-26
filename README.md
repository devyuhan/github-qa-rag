# GitHub QA RAG

RAG-based Q&A over any GitHub repository. Ingest a repo's code and docs into a vector database, then ask natural-language questions grounded in the actual source.

## Stack

- **LlamaIndex** — orchestration, chunking, query engine
- **Pinecone** — vector storage (cosine similarity, 1536 dims)
- **OpenAI** — embeddings (`text-embedding-3-small`)
- **Claude** — LLM for answer generation
- **tree-sitter** — code-aware splitting that respects function/class boundaries

## Setup

```bash
# 1. Create a virtualenv (Python 3.10+)
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Fill in: GITHUB_TOKEN, OPENAI_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY
```

## Usage

### Ingest a repository

```bash
python -m src ingest --owner <owner> --repo <repo> [--branch main] [--extensions .py .md]
```

This shallow-clones the repo, splits code with `CodeSplitter` and docs with `SentenceSplitter`, enriches metadata (file path, type, directory), and upserts into Pinecone.

### Ask questions

```bash
python -m src ask
```

Starts an interactive Q&A loop. Each answer includes source file paths and similarity scores.

### Ask with evaluation

```bash
python -m src ask --eval
```

Runs faithfulness and relevancy evaluation on each answer using LlamaIndex evaluators.

## Project Structure

```
src/
├── config.py      # Settings & env var loading
├── ingest.py      # Git clone → chunk → Pinecone upsert
├── query.py       # Retrieval + Claude generation
├── evaluate.py    # Faithfulness & relevancy scoring
└── cli.py         # CLI entry point (ingest / ask subcommands)
```
