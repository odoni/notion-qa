"""Ask a question to the notion database."""
import faiss
from langchain.chains import RetrievalQAWithSourcesChain
import pickle
import argparse
# from local_api import LocalApi
from langchain import HuggingFaceHub

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs_local.index")

with open("faiss_store_local.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})
# llm=LocalApi()


chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=store.vectorstore.as_retriever())
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
