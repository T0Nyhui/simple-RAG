import os
import json
import tqdm

def preprocess_corpus(json_line):
    data = json.loads(json_line)
    title = data['title']
    full_text_list = []
    
    for section in data['sections']:
        
        clean_text = section['text'].replace('#', '').strip()
        
        combined = f"Title: {title} | Section {section['section_id']}: {clean_text}"
        full_text_list.append(combined)
    
    return full_text_list


def load_data(corpus_path, query_path, answer_path):

    corpus = []
    corpus_paths = os.listdir(corpus_path)
    print("Loading corpus data...")
    for file in tqdm.tqdm(corpus_paths):
        if file.endswith(".json"):
            with open(os.path.join(corpus_path, file), "r", encoding="utf-8") as f:
                corpus.append(preprocess_corpus(f.readline()))
    print(f"Loaded {len(corpus)} corpus documents.")


    print("Loading queries data...")
    with open(query_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries.")

    print("Loading answers data...")
    with open(answer_path, "r", encoding="utf-8") as f:
        answers = json.load(f)
    print(f"Loaded {len(answers)} answers.")


    return corpus, queries, answers


if __name__ == "__main__":
    corpus, queries, answers = load_data(
        corpus_path = "/data2/huirutao/open-rag-bench/open_ragbench/corpus",
        query_path = "/data2/huirutao/open-rag-bench/open_ragbench/queries.json",
        answer_path = "/data2/huirutao/open-rag-bench/open_ragbench/answers.json"
    )
    print(corpus[0])
    print(queries)
    print(answers)