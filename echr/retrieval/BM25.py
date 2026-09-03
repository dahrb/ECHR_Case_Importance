import os
import argparse
import pickle
from langchain_community.retrievers import BM25Retriever
from echr.retrieval.initialise_vector_dbs import load_text, get_vectordb_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True, help="Article number: 3, 6, or 8")
    parser.add_argument("-k", type=int, default=100, help="Number of documents to retrieve")
    args = parser.parse_args()

    print(f"[art {args.article}] loading documents...")
    documents = load_text(args.article)

    retriever = BM25Retriever.from_documents(documents, k=args.k)

    out_path = os.path.join(get_vectordb_dir(args.article), "bm25_retriever.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(retriever, f)
    print(f"[art {args.article}] BM25 retriever saved -> {out_path}")
