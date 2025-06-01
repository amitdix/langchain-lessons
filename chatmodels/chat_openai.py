from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.8, max_completion_tokens=15)

print(llm.invoke("Suggest me a good hill station in karnataka?").content) 