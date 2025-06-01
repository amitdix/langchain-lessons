from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


llm = HuggingFaceEndpoint(repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                          task="text-generation")
model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of Delhi?")
print(result.content)
