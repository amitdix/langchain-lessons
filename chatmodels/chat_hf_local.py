from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

load_dotenv()

llm =HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                          task="text-generation")

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of Delhi?")
print(result.content)