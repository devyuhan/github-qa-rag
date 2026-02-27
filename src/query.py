"""Retrieval + Claude generation with conversation memory."""

from llama_index.core import Settings as LISettings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.config import Settings


def _build_chat_engine(settings: Settings):
    """Connect to existing Pinecone index and return a chat engine with memory."""
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

    # CohereRerank: fetch more candidates then rerank down to top_n
    node_postprocessors = []
    top_k = settings.similarity_top_k

    if settings.cohere_api_key:
        from llama_index.postprocessor.cohere_rerank import CohereRerank

        reranker = CohereRerank(
            api_key=settings.cohere_api_key,
            top_n=settings.rerank_top_n,
            model="rerank-english-v3.0",
        )
        node_postprocessors.append(reranker)
        top_k = 20  # fetch more candidates for the reranker
        print(f"  CohereRerank enabled (top_n={settings.rerank_top_n})")

    return index.as_chat_engine(
        chat_mode="condense_plus_context",
        similarity_top_k=top_k,
        node_postprocessors=node_postprocessors or None,
    )


_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _build_chat_engine(Settings())
    return _engine


def ask(question: str):
    """Ask a question and return the chat response (supports conversation memory)."""
    engine = _get_engine()
    return engine.chat(question)


def reset():
    """Clear conversation memory so the next question starts fresh."""
    engine = _get_engine()
    engine.reset()
