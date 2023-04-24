"""Ask a question to the notion database."""
import faiss
from langchain import HuggingFaceHub
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse

parser = argparse.ArgumentParser(description='Ask a question to the notion DB.')
parser.add_argument('question', type=str, help='The question to ask the notion DB')
args = parser.parse_args()

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

store.index = index

llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0.7, "max_length":512})

chain = VectorDBQAWithSourcesChain.from_llm(llm=llm, vectorstore=store)
result = chain({"question": args.question})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
