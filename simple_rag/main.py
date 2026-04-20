from models import LocalEmbeddingModel, VLLMGenerationModel
from utils import load_data
import json
import os
import tqdm
import faiss
import numpy as np
import pickle
import math
import concurrent.futures
import json
import os
import tqdm

def process_single_corpus_file(args):
    """
    为了方便多进程传参，建议接收一个元组 args: (file_name, corpus_path)
    """
    file_name, corpus_path = args
    if not file_name.endswith(".json"):
        return None
    
    doc_id = file_name.replace(".json", "")
    file_path = os.path.join(corpus_path, file_name)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            line = f.readline()
            if not line: return None
            data = json.loads(line)
        
        title = data.get('title', "")
        texts = []
        metas = []
        
        for section in data.get('sections', []):
            clean_text = section['text'].replace('#', '').strip()
            combined = f"Title: {title} | Section {section['section_id']}: {clean_text}"
            texts.append(combined)
            metas.append({
                "doc_id": doc_id, 
                "section_id": section['section_id']
            })
        return (texts, metas)
    except Exception:
        return None


class simpleRAG:
    def __init__(self, task_name,
                embedding_model_path,
                generation_model_name,
                base_url = "http://localhost:8000/v1"):
        self.task_name = task_name
        self.embedding_model = LocalEmbeddingModel(embedding_model_path)
        # self.generation_model = VLLMGenerationModel(generation_model_name, base_url)

    def load_data(self, corpus_path, query_path, answer_path):
        # 获取文件列表
        corpus_files = [f for f in os.listdir(corpus_path) if f.endswith(".json")]
        
        # 准备传给多进程的参数列表：[(文件名1, 路径), (文件名2, 路径), ...]
        process_args = [(f, corpus_path) for f in corpus_files]
        
        max_io_workers = 4 
        print(f"🚀 正在以并行模式加载语料 (Workers: {max_io_workers})...")
        
        self.corpus = []
        self.corpus_metadata = []

        # 使用顶级函数进行 map
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_io_workers) as executor:
            # 使用 list() 强制迭代并显示进度条
            results = list(tqdm.tqdm(
                executor.map(process_single_corpus_file, process_args), 
                total=len(corpus_files)
            ))

        # 整理结果
        for res in results:
            if res:
                texts, metas = res
                self.corpus.append(texts)
                self.corpus_metadata.append(metas)

        # 加载 Queries 和 Answers
        print("🔍 正在加载查询和标注数据...")
        with open(query_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)
        with open(answer_path, "r", encoding="utf-8") as f:
            self.answers = json.load(f)

        print(f"✅ 加载完成！")


    def save_vector_db(self, index, doc_map):
        self._ensure_task_dir()
        faiss.write_index(index, f"{self.task_name}/{self.task_name}.index")
        with open(f"{self.task_name}/{self.task_name}_metadata.pkl", "wb") as f:
            pickle.dump(doc_map, f)
        print(f"Vector database saved to {self.task_name}/{self.task_name}.index and {self.task_name}/{self.task_name}_metadata.pkl")

    def _ensure_task_dir(self):
        os.makedirs(self.task_name, exist_ok=True)

    def _default_generated_answers_path(self):
        return os.path.join(self.task_name, "generated_answers.json")

    def _default_evaluation_path(self):
        return os.path.join(self.task_name, "evaluation_details.json")

    def _default_query_embeddings_path(self):
        return os.path.join(self.task_name, "query_embeddings.npy")

    def _default_query_ids_path(self):
        return os.path.join(self.task_name, "query_ids.pkl")

    def embed_corpus(self):
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

        for doc_idx, chunks in enumerate(tqdm.tqdm(self.corpus)):
            embedded_chunks = self.embedding_model.embed_documents(chunks, show_progress_bar=False)
            
            for i, vector in enumerate(embedded_chunks):
                all_embeddings_list.append(vector)
                # 修改：doc_map 现在存储 (文档索引, 分段索引, 物理元数据)
                # 物理元数据包含了 doc_id 和 section_id
                self.doc_map.append({
                    "doc_idx": doc_idx,
                    "chunk_idx": i,
                    "metadata": self.corpus_metadata[doc_idx][i]
                })

        all_embeddings = np.array(all_embeddings_list).astype('float32')
        num_vectors, dimension = all_embeddings.shape
        print(f"Total vectors: {num_vectors}, Dimension: {dimension}")

        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(all_embeddings)
        
        print("FAISS index built with {} vectors.".format(self.faiss_index.ntotal))

        self.save_vector_db(self.faiss_index, self.doc_map)

    def embed_queries(self, batch_size=64):
        self._ensure_task_dir()
        query_embeddings_path = self._default_query_embeddings_path()
        query_ids_path = self._default_query_ids_path()

        if os.path.exists(query_embeddings_path) and os.path.exists(query_ids_path):
            print("Loading existing query embeddings and query ids...")
            query_embeddings = np.load(query_embeddings_path)
            with open(query_ids_path, "rb") as f:
                query_ids = pickle.load(f)
            print(f"Loaded {len(query_embeddings)} query embeddings from {query_embeddings_path}.")
            return query_embeddings, query_ids

        processed_queries = [i['query'] for i in self.queries.values()]
        query_ids = list(self.queries.keys())

        print("Embedding queries...")
        query_embeddings = self.embedding_model.embed_documents(processed_queries)
        query_embeddings = np.array(query_embeddings).astype('float32')

        np.save(query_embeddings_path, query_embeddings)
        with open(query_ids_path, "wb") as f:
            pickle.dump(query_ids, f)

        print(f"Embedded {len(query_embeddings)} queries with dimension {query_embeddings.shape[1]}.")
        print(f"Query embeddings saved to: {query_embeddings_path}")
        return query_embeddings, query_ids
        

    

    def retrieve(self, query_embeddings, query_ids, top_k=5):
        print("Retrieving relevant contexts for queries...")
        distances, indices = self.faiss_index.search(query_embeddings, top_k)
        batch_results = {}

        for i, q_id in enumerate(query_ids):
            retrieved_chunks = []
            retrieved_metadatas = [] # 新增：记录物理 ID
            
            for idx in indices[i]:
                if idx == -1: continue
                entry = self.doc_map[idx]
                doc_idx = entry["doc_idx"]
                chunk_idx = entry["chunk_idx"]
                
                retrieved_chunks.append(self.corpus[doc_idx][chunk_idx])
                retrieved_metadatas.append(entry["metadata"]) # 拿到 doc_id 和 section_id
            
            batch_results[q_id] = {
                "query": self.queries[q_id]['query'],
                "contexts": retrieved_chunks,
                "metadatas": retrieved_metadatas # 传递给 generate
            }
        return batch_results

    def generate(self, batch_results, save_path=None):
        if save_path is None:
            save_path = self._default_generated_answers_path()
            self._ensure_task_dir()

        if os.path.exists(save_path):
            print(f"Found existing generated answers at {save_path}, loading it.")
            with open(save_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print("Generating answers for queries...")
        generated_results = {}
        
        for q_id, data in tqdm.tqdm(batch_results.items(), desc="Generating Answers"):
            query = data["query"]
            retrieved_contexts = data["contexts"]
            retrieved_metadatas = data["metadatas"] # 拿到检索阶段存下的元数据
            
            response_dict = self.generation_model.generate_answer(query, retrieved_contexts)
            
            generated_results[q_id] = {
                "answer": response_dict.get('answer', ""),
                "sources": retrieved_contexts,
                # 核心修改：将检索到的物理 ID 直接存入结果，供评估脚本读取
                "retrieved_ids": retrieved_metadatas, 
                "source_ids": response_dict.get('sources', []),
                "found": response_dict.get('found', False),
                "confidence": response_dict.get('confidence', 0.0)
            }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(generated_results, f, ensure_ascii=False, indent=4)

        print(f"Generated answers saved to: {save_path}")
        return generated_results
    
    def _calculate_ndcg(self, rank, k=5):
        """内部辅助函数：计算 NDCG"""
        if rank > 0 and rank <= k:
            return 1.0 / math.log2(rank + 1)
        return 0.0

    def evaluate_retrieval(self, results, qrels_path, output_path=None):
        """
        评估检索性能并记录每个查询的得分。
        :param results: retrieve() 的输出，或 generate() 的输出，必须包含检索元数据字段
        :param qrels_path: qrels.json 的路径
        :param output_path: 详细结果保存路径，默认保存到 task_name 目录下
        """
        if output_path is None:
            self._ensure_task_dir()
            output_path = self._default_evaluation_path()

        if os.path.exists(output_path):
            print(f"Found existing evaluation file at {output_path}, loading it.")
            with open(output_path, "r", encoding="utf-8") as f:
                final_output = json.load(f)
            return final_output.get("summary", final_output)

        print(f"Loading qrels from {qrels_path}...")
        with open(qrels_path, 'r', encoding='utf-8') as f:
            qrels = json.load(f)

        detailed_queries = {}
        hits = 0
        mrr_sum = 0
        precision_sum = 0
        ndcg_sum = 0
        valid_count = 0

        print("Evaluating retrieval performance...")
        for q_id, res_data in results.items():
            if q_id not in qrels:
                continue
            
            valid_count += 1
            gt = qrels[q_id]
            # 获取检索到的物理 ID 列表：支持 retrieve() 输出中的 metadatas 或 generate() 输出中的 retrieved_ids
            retrieved_ids = res_data.get('retrieved_ids', res_data.get('metadatas', []))[:5]
            
            found_at_rank = -1
            for rank, item in enumerate(retrieved_ids):
                if str(item.get('doc_id')) == str(gt['doc_id']) and \
                   int(item.get('section_id', -1)) == int(gt['section_id']):
                    found_at_rank = rank + 1
                    break
            
            # 计算指标
            cur_hit = 1 if found_at_rank != -1 else 0
            cur_mrr = 1.0 / found_at_rank if found_at_rank != -1 else 0.0
            cur_ndcg = self._calculate_ndcg(found_at_rank, k=5)
            cur_precision = 0.2 if found_at_rank != -1 else 0.0

            # 累计
            hits += cur_hit
            mrr_sum += cur_mrr
            ndcg_sum += cur_ndcg
            precision_sum += cur_precision

            # 记录该 Query 的详细得分
            detailed_queries[q_id] = {
                "hit": cur_hit,
                "rank": found_at_rank,
                "mrr": round(cur_mrr, 4),
                "ndcg": round(cur_ndcg, 4),
                "is_success": cur_hit > 0,
                "ground_truth": gt
            }

        # 计算平均值
        summary = {
            "avg_hit_rate": hits / valid_count if valid_count > 0 else 0,
            "avg_mrr": mrr_sum / valid_count if valid_count > 0 else 0,
            "avg_ndcg": ndcg_sum / valid_count if valid_count > 0 else 0,
            "avg_precision": precision_sum / valid_count if valid_count > 0 else 0,
            "total_evaluated": valid_count
        }

        # 保存到本地
        final_output = {
            "summary": summary,
            "details": detailed_queries
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print("-" * 30)
        print(f"Recall@5 (Hit Rate): {summary['avg_hit_rate']:.4%}")
        print(f"MRR@5:                {summary['avg_mrr']:.4f}")
        print(f"NDCG@5:               {summary['avg_ndcg']:.4f}")
        print(f"Results saved to: {output_path}")
        print("-" * 30)
        
        return summary

    
if __name__ == "__main__":
    rag_system = simpleRAG(
        task_name = "simple_rag_with_evaluation",
        embedding_model_path = "/data2/huirutao/open-rag-bench/open-rag-bench/models/stella_en_1.5B_v5",
        generation_model_name = "qwen2_5-14b-awq",
        base_url = "http://localhost:8000/v1"
    )

    rag_system.load_data(
        corpus_path = "/data2/huirutao/open-rag-bench/open_ragbench/corpus",
        query_path = "/data2/huirutao/open-rag-bench/open_ragbench/queries.json",
        answer_path = "/data2/huirutao/open-rag-bench/open_ragbench/answers.json",
    )

    rag_system.embed_corpus()
    query_embeddings, query_ids = rag_system.embed_queries()

    retrieved_info = rag_system.retrieve(query_embeddings, query_ids, top_k=5)

    rag_system.evaluate_retrieval(
        results = retrieved_info,
        qrels_path = "/data2/huirutao/open-rag-bench/open_ragbench/qrels.json"
    )

    # generated_answers = rag_system.generate(retrieved_info)