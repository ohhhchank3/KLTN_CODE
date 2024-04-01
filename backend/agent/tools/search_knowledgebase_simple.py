import asyncio
import json

from backend.agent import model_contain
from backend.chat.knowledge_base_chat import knowledge_base_chat
from configs import MAX_TOKENS, SCORE_THRESHOLD, VECTOR_SEARCH_TOP_K


async def search_knowledge_base_iter(database: str, query: str) -> str:
    response = await knowledge_base_chat(query=query,
                                         knowledge_base_name=database,
                                         model_name=model_contain.MODEL.model_name,
                                         temperature=0.01,
                                         history=[],
                                         top_k=VECTOR_SEARCH_TOP_K,
                                         max_tokens=MAX_TOKENS,
                                         prompt_name="knowledge_base_chat",
                                         score_threshold=SCORE_THRESHOLD,
                                         stream=False)

    contents = ""
    async for data in response.body_iterator: # Dữ liệu ở đây là một chuỗi JSON
        data = json.loads(data)
        contents = data["answer"]
        docs = data["docs"]
    return contents

def search_knowledgebase_simple(query: str):
    return asyncio.run(search_knowledge_base_iter(query))


if __name__ == "__main__":
    result = search_knowledgebase_simple("Tỉ lệ nam nữ trong lĩnh vực dữ liệu lớn")
    print("Answer:", result)
