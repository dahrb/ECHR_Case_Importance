
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
    
    for i in [1024,2048]:
        
        print('start')
        documents = load_text()
        chunk_size = i
        chunk_overlap = 100
        embedding_name = "lexlms/legal-longformer-base"
        setup = 'raw'
        short_name = 'longformer'

        # Load the LegalBert model and tokenizer
        model = AutoModel.from_pretrained(embedding_name)
        tokenizer = AutoTokenizer.from_pretrained(embedding_name)
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer,chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        # Create a SentenceTransformers model
        word_embedding_model = models.Transformer(embedding_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        sentence_transformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # Save the model
        sentence_transformer_model.save('bert-sentence-transformer')

        print('embedding_name:', embedding_name, 'chunk_size:', chunk_size, 'chunk_overlap:', chunk_overlap)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="bert-sentence-transformer",
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}  # Set `True` for cosine similarity
            )
        
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






