
import pandas as pd
import os
os.chdir('/users/sgdbareh/volatile/ECHR_Importance')
from API_key import openai_key
from openai import OpenAI
client = OpenAI(api_key=openai_key)
import torch 
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_openai import OpenAIEmbeddings
from sklearn.model_selection import ParameterGrid
from initialise_vector_dbs import save_vectors, load_text, load_embeddings
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever




if __name__ == '__main__':
   
    documents = load_text()
    
    retriever = BM25Retriever.from_documents(documents,k=100)

    with open('bm25_retriever.pkl', 'wb') as f:
        pickle.dump(retriever, f)
    
    print('BM25 retriever saved')






