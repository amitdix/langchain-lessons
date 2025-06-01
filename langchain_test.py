from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


# 1. Initialize the LLM
# You can specify the model using the 'model_name' parameter.
# For example, to use 'gpt-3.5-turbo', set model_name="gpt-3.5-turbo"
# The OpenAI LLM class accepts several parameters. Here are some of the main ones:
# - temperature: Controls randomness of output (float, default 0.7)
# - model_name: The name of the OpenAI model to use (str, e.g., "gpt-3.5-turbo")
# - openai_api_key: Your OpenAI API key (str, optional if set in environment)
# - max_tokens: Maximum number of tokens to generate (int, optional)
# - streaming: Whether to stream responses (bool, default False)
# - n: Number of completions to generate (int, default 1)
# - stop: Sequences where the API will stop generating further tokens (str or list, optional)
# - best_of: Generates best_of completions server-side and returns the "best" (int, optional)
# - logprobs: Include the log probabilities on the logprobs most likely tokens (int, optional)
# - presence_penalty: Penalizes new tokens based on whether they appear in the text so far (float, optional)
# - frequency_penalty: Penalizes new tokens based on their frequency in the text so far (float, optional)
# - user: A unique identifier representing your end-user (str, optional)
# - organization: OpenAI organization ID (str, optional)
# - model_kwargs: Additional model-specific parameters (dict, optional)
llm = OpenAI(
    temperature=0.7,
    model_name="gpt-4.1-nano",
    # openai_api_key="YOUR_API_KEY",
    # max_tokens=256,
    # streaming=False,
    # n=1,
    # stop=None,
    # best_of=1,
    # logprobs=None,
    # presence_penalty=0.0,
    # frequency_penalty=0.0,
    # user=None,
    # organization=None,
    # model_kwargs=None
)
# 2. Create a prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Tell me a fun fact about {topic}."
)

# 3. Create the chain
# chain = LLMChain(llm=llm, prompt=prompt)

# 4. Run it
# response = chain.run("black holes")
# print(response) 


# Load a PDF file
loader = PyPDFLoader("/Users/amitdixit/langchain-env/Dakshina-ComputerProject-9B.pdf")
documents = loader.load()
print(documents[0].page_content[:300])  # Preview first chunk


text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

retriever = db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)

response = qa_chain.run("What is this document about?")
print(response)