import pandas as pd
import os
import argparse
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, "data")


def get_vectordb_dir(article):
    d = os.path.join(DATA, "vectordb", f"article{article}")
    os.makedirs(d, exist_ok=True)
    return d


def load_text(article):
    text = pd.read_pickle(os.path.join(DATA, "processed", f"article{article}", "outcome_cases.pkl"))
    text['Facts'] = text['Facts'].str.replace('\n', ' ')
    loader = DataFrameLoader(text, page_content_column='Facts')
    return loader.load()


def load_openai_embeddings(chunk_size=2048, chunk_overlap=100):
    """Original paper embedding: text-embedding-3-large via OpenAI API."""
    from langchain_openai import OpenAIEmbeddings
    try:
        import sys
        from echr.retrieval.openai_vector_db import openai_key
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)
    except ImportError:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base', chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return embeddings, text_splitter


def load_qwen_embeddings(chunk_size=2048, chunk_overlap=100, model_name="Qwen/Qwen3-Embedding-8B"):
    """Thesis embedding: Qwen3-Embedding-8B, local HuggingFace inference."""
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return embeddings, text_splitter


def build_and_save(documents, text_splitter, embeddings, filename, article):
    """Chunk documents, embed, build FAISS index, save via save_local."""
    print(f"  splitting {len(documents)} documents...")
    chunks = text_splitter.split_documents(documents)
    print(f"  {len(chunks)} chunks → embedding and indexing...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    out_path = os.path.join(get_vectordb_dir(article), filename)
    os.makedirs(out_path, exist_ok=True)
    vector_store.save_local(out_path)
    print(f"  saved -> {out_path}")
    return vector_store


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="Article number: 3, 6, or 8")
    parser.add_argument("--embedding", default="qwen", choices=["openai", "qwen"],
                        help="'openai' = text-embedding-3-large (original paper); "
                             "'qwen' = Qwen/Qwen3-Embedding-8B (thesis)")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--model-name", default=None,
                        help="Override model path/name (optional)")
    args = parser.parse_args()

    print(f"[art {args.article}] loading documents...")
    documents = load_text(args.article)

    if args.embedding == "openai":
        short_name = "openai"
        embeddings, text_splitter = load_openai_embeddings(args.chunk_size, args.chunk_overlap)
    else:
        model_name = args.model_name or "Qwen/Qwen3-Embedding-8B"
        short_name = "qwen3-8b"
        embeddings, text_splitter = load_qwen_embeddings(args.chunk_size, args.chunk_overlap, model_name)

    filename = f"chunk_{args.chunk_size}_embedding_{short_name}_raw"
    print(f"[art {args.article}] building FAISS index: {filename}")

    build_and_save(documents, text_splitter, embeddings, filename, args.article)
    print(f"[art {args.article}] done.")
