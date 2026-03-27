from typing import List
from openai import OpenAI


class VLLMModel:
    def __init__(self, model_name, base_url = "http://localhost:8000/v1"):
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key="vllm-embedding")
    
    def embed_text(self, text : str) -> List[float]:
        responce = self.client.embeddings.create(
            input = text,
            model = self.model_name
        )
        return responce.data[0].embedding

    def embed_documents(self, texts : List[str]) -> List[List[float]]:
        responce = self.client.embeddings.create(
            input = texts,
            model = self.model_name
        )
        return [it.embedding for it in responce.data]
    
if __name__ == "__main__":
    print("开始")
    model = VLLMModel("stella-v5")
    print(model.embed_text("你好")[0:2])

    print([it[0:2] for it in model.embed_documents([
        "你好",
        "谢谢",
        "再见"
    ])])