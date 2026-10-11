from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings,OllamaLLM
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage , SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    collection_name="rag_documents",
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}

)

query = "When did SpaceX receive funding from Google?"

retreiver = db.as_retriever(search_kwargs = {"k":5})

relevant_docs = retreiver.invoke(query)

print(f"User Query:{query}")

print("---Context---")

for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")


# using LLM from here 

combined_input = f"""Based on the following documents, please answer the question:{query}

Documents:{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear , helpful answer using only the information from these documents. 

"""

model = OllamaLLM(model= "llama3.2:3b")

messages = [
    SystemMessage(content = "You are helpful assistant"),
    HumanMessage(content = combined_input)
]

result = model.invoke(messages)

print("\n---Generated response---")

print("Content only")

print(result)


