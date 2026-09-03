
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

# def embed_text(text,mean=True, tokenizer=None, model=None):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     if mean:
#         return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#     else:
#         return outputs.last_hidden_state[:,0].squeeze().numpy()

def save_vectors(vector_store, filename):
    #save the retriever to disk
    faiss.write_index(vector_store.index, f"/users/sgdbareh/volatile/ECHR_Importance/VectorDB/faiss_index_{filename}.bin")

    # Save document store and metadata
    with open(f"/users/sgdbareh/volatile/ECHR_Importance/VectorDB/docstore_{filename}.pkl", "wb") as f:
        pickle.dump(vector_store.docstore, f)
    with open(f"/users/sgdbareh/volatile/ECHR_Importance/VectorDB/index_to_docstore_id_{filename}.pkl", "wb") as f:
        pickle.dump(vector_store.index_to_docstore_id, f)

def load_text():
    text = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/Art_3_Data_Process/outcome_cases.pkl')
    text['Facts'] = text['Facts'].str.replace('\n', ' ')
    # load documents
    loader = DataFrameLoader(text,page_content_column='Facts')
    # docs loaded
    documents = loader.load()
    return documents
#
def load_embeddings(chunk_size=512, chunk_overlap=50,embedding_name="nlpaueb/legal-bert-base-uncased"):
    #load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(embedding_name)
    model = AutoModel.from_pretrained(embedding_name)
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return tokenizer, model, text_splitter
#
def load_openai_embeddings(chunk_size=512, chunk_overlap=50):
    #load embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name='gpt-4o',chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return embeddings, text_splitter


if __name__ == '__main__':
    print('start')
    documents = load_text()

    hyperparameters_BERT = {
        'chunk_size': [512],
        'chunk_overlap': [50],
        'embedding_name': ["nlpaueb/legal-bert-base-uncased",'google-bert/bert-base-uncased','openai']
    }

    #hyperparameters_OpenAI = {
    #    'chunk_size': (1024,2048),
    #    'chunk_overlap': 100,   
    #    'embedding_name': ("lexlms/legal-longformer-base","openai") 
    #}

    # Generate all possible combinations of hyperparameters
    grid = ParameterGrid(hyperparameters_BERT)

    # Iterate over each combination of hyperparameters
    for params in grid:
        chunk_size = params['chunk_size']
        chunk_overlap = params['chunk_overlap']
        embedding_name = params['embedding_name']

        print('embedding_name:', embedding_name, 'chunk_size:', chunk_size, 'chunk_overlap:', chunk_overlap)
        
        if embedding_name == "openai":
            embeddings, text_splitter = load_openai_embeddings(chunk_size, chunk_overlap)
            vector_store = create_vector_store(documents, text_splitter, embeddings)
        else:
            tokenizer, model, text_splitter = load_embeddings(chunk_size, chunk_overlap, embedding_name)
            vector_store = create_vector_store(documents, text_splitter, lambda x: embed_text(x,tokenizer,model))
        
        print('vector store created')

        # Save the vector store
        save_vectors(vector_store, f"chunk_{chunk_size}_embedding_{embedding_name}")

        print('vector store saved')

    




