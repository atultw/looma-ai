from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

faiss_db = FAISS.load_local("faiss_index", hf)

while True:
    query = input("Search: ")
    docs = faiss_db.similarity_search(query)
    for doc in docs:
        print(doc)
    print("------------------ \n\n\n")