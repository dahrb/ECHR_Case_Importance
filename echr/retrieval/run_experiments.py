import os
import argparse
import pickle
import pandas as pd
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, models
from echr.retrieval.initialise_vector_dbs import get_vectordb_dir, DATA

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REAL_K = 100
K_to_retrieve = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

results_dict = {}
vector_store = None  # set in __main__ before apply


def date_filter(doc, start_date):
    return doc['date'] < start_date


def _dedup_ordered(docs):
    """Deduplicate by appno while preserving rank order (keep first occurrence)."""
    seen = set()
    result = []
    for doc in docs:
        appno = doc.metadata['appno']
        if appno not in seen:
            seen.add(appno)
            result.append(appno)
    return result


def filter_new(comm_case, docs, similarity='cosine'):
    start_date = pd.Timestamp(comm_case['doc_date'])
    filtered_results = [r for r in docs if date_filter(r.metadata, start_date)]
    results = _dedup_ordered(filtered_results)[:REAL_K]

    K_results = K_to_retrieve
    while len(results) < REAL_K:
        K_results *= 2
        if similarity == 'cosine':
            docs = vector_store.similarity_search(comm_case['Subject Matter'], K_results)
        elif similarity == 'mmr':
            docs = vector_store.max_marginal_relevance_search(comm_case['Subject Matter'], K_results)
        else:  # bm25 — BM25Retriever has no similarity_search; widen k and re-invoke
            vector_store.k = K_results
            docs = vector_store.invoke(comm_case['Subject Matter'])
        filtered_results = [r for r in docs if date_filter(r.metadata, start_date)]
        results = _dedup_ordered(filtered_results)[:REAL_K]
        if K_results > 5000:
            raise ValueError('K_results > 5000: not enough pre-date cases to fill REAL_K')

    results_dict[comm_case['Filename']] = results


def select_embedding(embedding_name, chunk_size, short_name, article):
    vdb_dir = get_vectordb_dir(article)
    index_path = os.path.join(vdb_dir, f"chunk_{chunk_size}_embedding_{short_name}")

    if embedding_name == 'openai':
        from langchain_openai import OpenAIEmbeddings
        try:
            import sys; sys.path.insert(0, os.path.join(REPO, "old", "VectorDB"))
            from API_key import openai_key
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)
        except ImportError:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    elif 'qwen' in embedding_name.lower():
        model_name = embedding_name if '/' in embedding_name else "Qwen/Qwen3-Embedding-8B"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    else:
        # bert / legal-bert / longformer — original paper models
        word_embedding_model = models.Transformer(embedding_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        model_save_path = f"{short_name}_sentence_transformer_model"
        st_model.save(model_save_path)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_save_path,
            multi_process=True,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="Article number: 3, 6, or 8")
    parser.add_argument("-c", "--chunk-size", type=int, default=2048)
    parser.add_argument("-o", "--chunk-overlap", type=int, default=100)
    parser.add_argument("-e", "--embedding-name", default="Qwen/Qwen3-Embedding-8B",
                        help="'openai', 'Qwen/Qwen3-Embedding-8B', 'BM25', or HF model ID")
    parser.add_argument("-n", "--short-name", default="qwen3-8b_raw",
                        help="Short name used in saved filenames")
    parser.add_argument("-s", "--similarity", default="cosine", choices=["cosine", "mmr"])
    args = parser.parse_args()

    comm_cases = pd.read_pickle(
        os.path.join(DATA, "processed", f"article{args.article}", "splits", "comm_test.pkl")
    )

    print(f"[art {args.article}] {len(comm_cases)} comm cases | "
          f"embedding={args.embedding_name} chunk={args.chunk_size} sim={args.similarity}", flush=True)

    if args.embedding_name == 'BM25':
        out_path = os.path.join(get_vectordb_dir(args.article), f"{args.short_name}_results.pkl")
        bm25_path = os.path.join(get_vectordb_dir(args.article), "bm25_retriever.pkl")
        print(f"[art {args.article}] loading BM25 retriever...", flush=True)
        with open(bm25_path, "rb") as f:
            vector_store = pickle.load(f)
        print(f"[art {args.article}] BM25 loaded; {len(comm_cases)} cases to process", flush=True)

        # resume: skip already-done cases
        if os.path.exists(out_path):
            existing = pd.read_pickle(out_path)
            results_dict.update(existing)
            comm_cases = comm_cases[~comm_cases['Filename'].isin(results_dict)]
            print(f"[art {args.article}] resuming — {len(results_dict)} done, {len(comm_cases)} remaining", flush=True)

        vector_store.k = K_to_retrieve
        checkpoint_interval = 200
        for idx, (_, row) in enumerate(comm_cases.iterrows()):
            filter_new(row, vector_store.invoke(row['Subject Matter']), 'bm25')
            if (idx + 1) % checkpoint_interval == 0:
                pd.to_pickle(results_dict, out_path)
                print(f"[art {args.article}] checkpoint {idx+1}/{len(comm_cases)}", flush=True)
    else:
        vector_store = select_embedding(
            args.embedding_name, args.chunk_size, args.short_name, args.article
        )
        if args.similarity == 'cosine':
            comm_cases.apply(
                lambda x: filter_new(
                    x, vector_store.similarity_search(x['Subject Matter'], K_to_retrieve), 'cosine'
                ), axis=1
            )
        else:
            comm_cases.apply(
                lambda x: filter_new(
                    x, vector_store.max_marginal_relevance_search(x['Subject Matter'], K_to_retrieve), 'mmr'
                ), axis=1
            )
        out_path = os.path.join(
            get_vectordb_dir(args.article),
            f"{args.similarity}_{args.short_name}_chunk_{args.chunk_size}_results.pkl"
        )

    pd.to_pickle(results_dict, out_path)
    print(f"[art {args.article}] results saved -> {out_path}")
