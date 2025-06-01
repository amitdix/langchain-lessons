from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=1.8, max_completion_tokens=15)

print(llm.invoke("Suggest me a good hill station in karnataka?").content)  