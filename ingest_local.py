"""This is the logic for ingesting Notion data into LangChain."""
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
# import signal

# signal.signal(signal.SIGSEGV, signal.SIG_IGN)

print("Loading files")
# Here we load in the data in the format that Notion exports it in.
ps = list(Path("/Users/felipe.odoni/Documents/glo_dataset").glob("**/*.txt"))

data = []
sources = []
for p in ps:
    with open(p) as f:
        data.append(f.read())
    sources.append(p)

print("Spliting files")
# Here we split the documents, as needed, into smaller chunks.
# We do this due to the context limits of the LLMs.
text_splitter = CharacterTextSplitter(chunk_size=3000, separator="\n")
docs = []
metadatas = []
for i, d in enumerate(data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadatas.extend([{"source": sources[i]}] * len(splits))

print("Creating vector store")
# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, HuggingFaceEmbeddings(), metadatas=metadatas)
faiss.write_index(store.index, "docs_globose.index")
store.index = None
with open("faiss_store_globose.pkl", "wb") as f:
    pickle.dump(store, f)
