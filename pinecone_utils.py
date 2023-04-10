# pinecone_utils.py
import pinecone
from typing import Dict, List

def init_pinecone(api_key: str, environment: str):
    pinecone.init(api_key=api_key, environment=environment)

def create_pinecone_index(index_name: str, dimension: int, metric: str, pod_type: str):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension, metric=metric, pod_type=pod_type)

def get_pinecone_index(index_name: str):
    return pinecone.Index(index_name)

def query_index(index, query_embedding, top_k, include_metadata=True, namespace):
    return index.query(query_embedding, top_k=top_k, include_metadata=include_metadata, namespace=namespace)

def upsert_to_index(index, data, namespace):
    index.upsert(data, namespace=namespace)

def delete_pinecone_index(index_name: str):
    pinecone.deinit(index_name)

