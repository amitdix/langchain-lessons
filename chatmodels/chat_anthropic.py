from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=1.8)

print(llm.invoke("Suggest me a good hill station in karnataka?").content)