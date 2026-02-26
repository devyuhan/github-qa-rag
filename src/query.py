"""Retrieval + Claude generation."""

from llama_index.core import Settings as LISettings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import Settings


def _build_query_engine(settings: Settings):
    """Connect to existing Pinecone index and return a query engine."""
    # Configure global LlamaIndex settings
    LISettings.embed_model = OpenAIEmbedding(
        model=settings.embed_model_name,
        api_key=settings.openai_api_key,
    )
    LISettings.llm = Anthropic(
        model=settings.llm_model_name,
        api_key=settings.anthropic_api_key,
    )

    pc = Pinecone(api_key=settings.pinecone_api_key)
    pinecone_index = pc.Index(settings.pinecone_index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index.as_query_engine(similarity_top_k=settings.similarity_top_k)


_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _build_query_engine(Settings())
    return _engine


def ask(question: str):
    """Ask a question and return the LlamaIndex Response object."""
    engine = _get_engine()
    return engine.query(question)
