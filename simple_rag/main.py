from models import LocalEmbeddingModel, VLLMGenerationModel
from utils import load_data
import json
import os
import tqdm
import faiss
import numpy as np
import pickle
class simpleRAG:
    def __init__(self, task_name,
                corpus,
                queries,
                answers,
                embedding_model_path,
                generation_model_name,
                base_url = "http://localhost:8000/v1"):
        self.corpus = corpus
        self.queries = queries
        self.answers = answers
        self.task_name = task_name
        self.embedding_model = LocalEmbeddingModel(embedding_model_path)
        self.generation_model = VLLMGenerationModel(generation_model_name, base_url)

    def save_vector_db(self, index, doc_map):
        if not os.path.exists(self.task_name):
            os.makedirs(self.task_name)
        faiss.write_index(index, f"{self.task_name}/{self.task_name}.index")
        with open(f"{self.task_name}/{self.task_name}_metadata.pkl", "wb") as f:
            pickle.dump(doc_map, f)
        print(f"Vector database saved to {self.task_name}/{self.task_name}.index and {self.task_name}/{self.task_name}_metadata.pkl")

    def embed_corpus(self, corpus):

        if os.path.exists(f"{self.task_name}/{self.task_name}.index") and os.path.exists(f"{self.task_name}/{self.task_name}_metadata.pkl"):
            print("Loading existing FAISS index and metadata...")
            self.faiss_index = faiss.read_index(f"{self.task_name}/{self.task_name}.index")
            with open(f"{self.task_name}/{self.task_name}_metadata.pkl", "rb") as f:
                self.doc_map = pickle.load(f)
            print("FAISS index and metadata loaded successfully.")
            return
        
        print("Embedding corpus documents and building index...")
        
        all_embeddings_list = []
        self.doc_map = []

        for doc_idx, chunks in enumerate(tqdm.tqdm(corpus)):
            embedded_chunks = self.embedding_model.embed_documents(chunks, show_progress_bar=False)
            
            for i, vector in enumerate(embedded_chunks):
                all_embeddings_list.append(vector)
                self.doc_map.append((doc_idx, i))

        all_embeddings = np.array(all_embeddings_list).astype('float32')
        num_vectors, dimension = all_embeddings.shape
        print(f"Total vectors: {num_vectors}, Dimension: {dimension}")

        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(all_embeddings)
        
        print("FAISS index built with {} vectors.".format(self.faiss_index.ntotal))

        self.save_vector_db(self.faiss_index, self.doc_map)

    def embed_queries(self, queries, batch_size=64):

        processed_queries = [i['query'] for i in queries.values()]
        query_ids = list(queries.keys())

        print("Embedding queries...")
        query_embeddings = self.embedding_model.embed_documents(processed_queries) 
        
        query_embeddings = np.array(query_embeddings).astype('float32')
        print(f"Embedded {len(query_embeddings)} queries with dimension {query_embeddings.shape[1]}.")
        return query_embeddings, query_ids


    def retrieve(self, query_embeddings, query_ids, top_k=5):
        print("Retrieving relevant contexts for queries...")
        distances, indices = self.faiss_index.search(query_embeddings, top_k)
        batch_results = {}

        print("Processing retrieval results...")

        for i, file_name in enumerate(query_ids):
            query_text = self.queries[file_name]
            top_k_indices = indices[i]
            
            retrieved_chunks = []
            for idx in top_k_indices:
                if idx == -1: continue
                doc_idx, chunk_idx = self.doc_map[idx]
                content = self.corpus[doc_idx][chunk_idx]
                retrieved_chunks.append(content)
            
            
            batch_results[file_name] = {
                "query": query_text,
                "contexts": retrieved_chunks
            }
            
        return batch_results
        

    def generate(self, batch_results):
        print("Generating answers for queries...")
        generated_answers = {}
        for file_name, data in tqdm.tqdm(batch_results.items(), desc="Generating Answers"):
            query = data["query"]
            contexts = data["contexts"]
            generated_answer = self.generation_model.generate_answer(query, contexts)
            generated_answers[file_name] = generated_answer

        return generated_answers

    def answer_query(self, query):
        retrieved_info = self.retrieve(query)
        response = self.generate(retrieved_info)
        return response
    
if __name__ == "__main__":

    corpus, queries, answers = load_data(
        corpus_path = "/data2/huirutao/open-rag-bench/open_ragbench/corpus",
        query_path = "/data2/huirutao/open-rag-bench/open_ragbench/queries.json",
        answer_path = "/data2/huirutao/open-rag-bench/open_ragbench/answers.json",
    )

    rag_system = simpleRAG(
        task_name = "simple_rag_1",
        corpus = corpus,
        queries = queries,
        answers = answers,
        embedding_model_path = "/data2/huirutao/open-rag-bench/open-rag-bench/models/stella_en_1.5B_v5",
        generation_model_name = "qwen2_5-14b-awq",
        base_url = "http://localhost:8000/v1"
    )

    rag_system.embed_corpus(corpus)
    query_embeddings, query_ids = rag_system.embed_queries(queries)
    retrieved_info = rag_system.retrieve(query_embeddings, query_ids, top_k=5)
    generated_answers = rag_system.generate(retrieved_info)
    with open(os.path.join("/data2/huirutao/simple-RAG/simple_rag_1", "generated_answers.json"), "w", encoding="utf-8") as f:
        json.dump(generated_answers, f, ensure_ascii=False, indent=4)