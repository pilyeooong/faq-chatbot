from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from itertools import islice
import os
import pickle
import tiktoken

load_dotenv()

COLLECTION_NAME = "faq_collection"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_TOKENS = 4096
OVERLAP = MAX_TOKENS // 4
WINDOW_SIZE = MAX_TOKENS // 2

def get_chroma_db_client():
    client = PersistentClient(path=f"./chroma_db_{MAX_TOKENS}")

    return client

def get_or_create_collection(name=COLLECTION_NAME):
    client = get_chroma_db_client()
    embedding_function = get_embedding_function(EMBEDDING_MODEL)
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function
    )
    
    return collection

def get_tokenizer(model_name):
    tokenizer = tiktoken.encoding_for_model(model_name)
    return tokenizer

def sliding_window_text(text, window_size=WINDOW_SIZE, overlap=OVERLAP, max_tokens=MAX_TOKENS):
    tokenizer = get_tokenizer(EMBEDDING_MODEL)
    tokens = tokenizer.encode(text)

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        if end == len(tokens):
            break
        start += window_size - overlap

    return chunks

def get_embedding_function(model_name):
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=model_name
    )
    return embedding_function

def load_pkl_file(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data

def get_retriever(text):
    collection = get_or_create_collection(COLLECTION_NAME)
    retrieved = collection.query(
        query_texts=[text], 
        n_results=5 
    )
    results = retrieved.get("documents", [])
    return results[0]

def batch_iterator(data_dict, batch_size=10):
    iterator = iter(data_dict.items())
    for first in iterator:
        yield [first] + list(islice(iterator, batch_size - 1))

def embed_data():
    file_path = "final_result.pkl"
    data = load_pkl_file(file_path)
    collection = get_or_create_collection(COLLECTION_NAME)

    existing_ids = set()
    if collection.count() > 0:
        existing_data = collection.get()
        existing_ids = set(existing_data['ids'])

    batch_size = 20
    for batch_idx, batch in enumerate(batch_iterator(data, batch_size)):
        documents = []
        metadatas = []
        ids = []

        for idx, (question, answer) in enumerate(batch):
            data_id = f"faq_{batch_idx}_{idx}"
            
            if data_id in existing_ids:
                continue

            if isinstance(question, str) and isinstance(answer, str):
                answer_chunks = sliding_window_text(answer, window_size=WINDOW_SIZE, overlap=OVERLAP, max_tokens=MAX_TOKENS)
                for chunk_idx, chunk in enumerate(answer_chunks):
                    chunk_id = f"{data_id}_chunk{chunk_idx}"
                    if chunk_id in existing_ids:
                        continue

                    documents.append(chunk)
                    metadatas.append({"question": question, "chunk_index": chunk_idx})
                    ids.append(chunk_id)

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Batch {batch_idx + 1} processed.")
    
def main():
    embed_data()

if __name__ == "__main__":
    main()