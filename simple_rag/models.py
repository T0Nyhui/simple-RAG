from typing import List
from openai import OpenAI


class VLLMEmbeddingModel:
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
    


class VLLMGenerationModel:
    def __init__(self, model_name, base_url = "http://localhost:8000/v1"):
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key="vllm-gen")
    
    def generate_answer(self,  query: str, contexts: List[str]):
        system_prompt = (
            "You are a helpful AI assistant that provides answers in JSON format. "
            "You must base your answer strictly on the provided context. "
            "Your response must be a valid JSON object with the following keys:\n"
            "1. 'answer': (string) A concise and accurate response to the user's question.\n"
            "2. 'sources': (list of integers) The IDs of the sources used to generate the answer.\n"
            "3. 'found': (boolean) True if the information is present in the context, False otherwise.\n"
            "4. 'confidence': (float) A score between 0 and 1 representing your certainty."
        )

        formatted_context = "\n\n".join([f"Source [{i+1}]: {c}" for i, c in enumerate(contexts)])
        
        user_input = (
            f"Context Information:\n"
            f"---------------------\n"
            f"{formatted_context}\n"
            f"---------------------\n\n"
            f"Question: {query}\n\n"
            f"Return the response as a JSON object. Example:\n"
            "{\n"
            "  \"answer\": \"The capital of France is Paris.\",\n"
            "  \"sources\": [1],\n"
            "  \"found\": true,\n"
            "  \"confidence\": 0.95\n"
            "}"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0
        )

        response_content = response.choices[0].message.content
        try:
            import json
            return json.loads(response_content)
        except json.JSONDecodeError:
            clean_content = response_content.replace("```json", "").replace("```", "").strip()
            try:
                return json.loads(clean_content)
            except:
                return {"error": "Failed to parse JSON", "raw": response_content}
        
    
if __name__ == "__main__":

    '''
    print("开始")
    model = VLLMEmbeddingModel("stella-v5")
    print(model.embed_text("你好")[0:2])

    print([it[0:2] for it in model.embed_documents([
        "你好",
        "谢谢",
        "再见"
    ])])
    '''

    model = VLLMGenerationModel("qwen2_5-14b-awq")
    print(model.generate_answer(
        query = "什么是人工智能？",
        contexts = [
            "人工智能（Artificial Intelligence，简称AI）是指通过计算机系统模拟人类智能的能力，使其能够执行通常需要人类智能才能完成的任务，如学习、推理、问题解决、语言理解和感知等。人工智能技术包括机器学习、深度学习、自然语言处理、计算机视觉等多个领域，广泛应用于自动驾驶、医疗诊断、金融分析、智能客服等多个行业。"
        ]
    ))