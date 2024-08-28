#To Do

from datetime import datetime
import pandas as pd
import os
os.chdir('/users/sgdbareh/volatile/ECHR_Importance')
from API_key import openai_key
from openai import OpenAI
client = OpenAI(api_key=openai_key)
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, models
from optparse import OptionParser
import torch

REAL_K = 100
K_to_retrieve = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

results_dict = {}

# Define the filter function
def date_filter(doc, start_date):
    #print(doc)
    doc_date = doc['date']
    #print(doc_date)
    #print(start_date)
    return doc_date < start_date

def filter_new(comm_case,docs,similarity='cosine'):
    #print(docs)
    start_date = pd.Timestamp(comm_case['doc_date'])
    #[print(doc.metadata) for doc in docs]
    filtered_results = [result for result in docs if date_filter(result.metadata, start_date)]
    #print(filtered_results)
    appno_list = [doc.metadata['appno'] for doc in filtered_results]
    appno_list = list(set(appno_list))
    #print(appno_list)
    results = appno_list[:REAL_K]

    K_results = K_to_retrieve

    while len(results) < REAL_K:

        #print(len(results))

        K_results = K_results*2

        if similarity == 'cosine':
            docs = vector_store.similarity_search(comm_case['Subject Matter'],K_results )
        else:
            docs = vector_store.max_marginal_relevance_search(comm_case['Subject Matter'],K_results)
        filtered_results = [result for result in docs if date_filter(result.metadata, start_date)]
        appno_list = [doc.metadata['appno'] for doc in filtered_results]
        appno_list = list(set(appno_list))
        results = appno_list[:REAL_K]

        if K_results > 5000:
            raise ValueError('K_results > 1000')

    results_dict[comm_case['Filename']] = results

def select_embedding(embedding_name, chunk_size, chunk_overlap, short_name=None):
    # Load document store and metadata
    with open(f"/users/sgdbareh/volatile/ECHR_Importance/VectorDB/docstore_chunk_{chunk_size}_embedding_{short_name}.pkl", "rb") as f:
        docstore = pickle.load(f)
    with open(f"/users/sgdbareh/volatile/ECHR_Importance/VectorDB/index_to_docstore_id_chunk_{chunk_size}_embedding_{short_name}.pkl", "rb") as f:
        index_to_docstore_id = pickle.load(f)

    if 'bert' in embedding_name or 'longformer' in embedding_name:

        # Load FAISS index
        index = faiss.read_index(f"/users/sgdbareh/volatile/ECHR_Importance/VectorDB/faiss_index_chunk_{chunk_size}_embedding_{short_name}.bin")

        # Load the LegalBert model and tokenizer
        #model = AutoModel.from_pretrained(embedding_name)
        #tokenizer = AutoTokenizer.from_pretrained(embedding_name)
        #text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer,chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        # Create a SentenceTransformers model
        word_embedding_model = models.Transformer(embedding_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        sentence_transformer_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        model_name = f'{short_name}_sentence_transformer_model'
        # Save the model
        sentence_transformer_model.save(f'{short_name}_sentence_transformer_model')

        if DEVICE == "cuda":
            embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}  # Set `True` for cosine similarity
            )
        else:
            embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            multi_process=True,
            encode_kwargs={"normalize_embeddings": True}  # Set `True` for cosine similarity
            )

        #print('embedding_name:', embedding_name, 'chunk_size:', chunk_size, 'chunk_overlap:', chunk_overlap)

    else:
        embeddings = OpenAIEmbeddings(
        openai_api_key=openai_key,
        model=embedding_name,
        )

        #text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name='cl100k_base',chunk_size=chunk_size,chunk_overlap=chunk_overlap)

        index = faiss.read_index(f"/users/sgdbareh/volatile/ECHR_Importance/VectorDB/faiss_index_chunk_{chunk_size}_embedding_{short_name}.bin")

    
    vector_store_NEW = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    distance_strategy=DistanceStrategy.COSINE,
    )

    return vector_store_NEW



if __name__ == '__main__':
    comm_cases = pd.read_pickle('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/train.pkl')
    #comm_cases = comm_cases[:1]
    #read in from arguments
    parser = OptionParser(usage='usage: -c chunk_size -o chunk_overlap -e embedding_name -n short_name -s similarity')   
    parser.add_option("-c", "--chunk_size", action = "store", type = "int", dest = "chunk_size", default = 512)
    parser.add_option("-o", "--chunk_overlap", action = "store", type = "int", dest = "chunk_overlap", default = 50)
    parser.add_option("-e", "--embedding_name", action = "store", type = "string", dest = "embedding_name",default = 'nlpaueb/legal-bert-base-uncased')
    parser.add_option("-n", "--short_name", action = "store", type = "string", dest = "short_name", default = 'legal-bert_raw')
    parser.add_option("-s", "--similarity", action = "store", type = "string", dest = "similarity", default = 'cosine')
    (options, _) = parser.parse_args()

    chunk_size = options.chunk_size
    chunk_overlap = options.chunk_overlap
    embedding_name = options.embedding_name
    short_name = options.short_name
    similarity = options.similarity

    print('embedding_name:', embedding_name, 'chunk_size:', chunk_size, 'chunk_overlap:', chunk_overlap)

    if embedding_name == 'BM25':
        with open('/users/sgdbareh/volatile/ECHR_Importance/VectorDB/bm25_retriever.pkl', 'rb') as f:
            vector_store = pickle.load(f)
        
        comm_cases.apply(lambda x: filter_new(x,vector_store.invoke(x['Subject Matter'])),axis=1)

        pd.to_pickle(results_dict,f'/users/sgdbareh/volatile/ECHR_Importance/VectorDB/{short_name}_results.pkl')

    else:
        vector_store = select_embedding(embedding_name, chunk_size, chunk_overlap,short_name)

        if similarity == 'cosine':
            comm_cases.apply(lambda x: filter_new(x,vector_store.similarity_search(x['Subject Matter'], K_to_retrieve,similarity=similarity)),axis=1)
        else:
            comm_cases.apply(lambda x: filter_new(x,vector_store.max_marginal_relevance_search(x['Subject Matter'], K_to_retrieve,similarity=similarity)),axis=1)

        pd.to_pickle(results_dict,f'/users/sgdbareh/volatile/ECHR_Importance/VectorDB/{similarity}_{short_name}_chunk_{chunk_size}_results.pkl')