
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


if __name__ == '__main__':
    
    for i in [512,1024,2048]:
        
        print('start')
        documents = load_text()
        chunk_size = i
        if i == 512:
            chunk_overlap = 50
        else:
            chunk_overlap = 100
        setup = 'raw'
        embedding_name = "text-embedding-3-large"
        short_name = 'openai'

        embeddings = OpenAIEmbeddings(
        model=embedding_name,
        openai_api_key=openai_key)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name='cl100k_base',chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        print('embedding_name:', embedding_name, 'chunk_size:', chunk_size, 'chunk_overlap:', chunk_overlap)    
        
        print('vector store created')

        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy=DistanceStrategy.COSINE,
        )

        print('initialised vector store')

        store = InMemoryStore()

        retriever_new = ParentDocumentRetriever(vectorstore=vector_store,docstore=store,child_splitter=text_splitter)
        retriever_new.add_documents(documents)

        print('docs added')

        # Save the vector store
        save_vectors(vector_store, f"chunk_{chunk_size}_embedding_{short_name}_{setup}")

        print('vector store saved')






