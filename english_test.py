from langchain.chains import LLMChain, MapReduceDocumentsChain, StuffDocumentsChain, ReduceDocumentsChain

from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# llm = LlamaCpp(
#     model_path="llama-2-7b.Q4_K_M.gguf",
#     temperature=0.4,
#     max_tokens=64,
#     top_p=1,
#     n_gpu_layers=24,
#     callback_manager=callback_manager,
#     verbose=True,  # Verbose is required to pass to the callback manager
#     n_ctx=4096,
# )

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from langchain_community.document_loaders import PyPDFLoader

chapters = [(5 + 1, 5 + 12), (5 + 13, 5 + 26), (5 + 27, 5 + 39), (5 + 40, 5 + 51), (5 + 52, 5 + 61), (5 + 62, 5 + 74),
            (5 + 75, 5 + 86), (5 + 87, 5 + 96), (5 + 97, 5 + 109), (5 + 110, 5 + 120), (5 + 121, 5 + 132),
            (5 + 133, 5 + 144), (5 + 145, 5 + 159), (5 + 160, 5 + 168), (5 + 169, 5 + 179), (5 + 180, 5 + 190),
            (5 + 191, 5 + 5 + 200)]

loader = PyPDFLoader("English_5_0001.pdf")
pages = loader.load_and_split()
db = FAISS(hf)
for chapter in chapters:
    FAISS.add_documents()
    docs = pages[chapter[0]:chapter[1]]
    db = FAISS.from_documents(docs, hf)
    print(db.index.ntotal)
    db.save_local("faiss_index")


