import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
llm = OllamaLLM(model="llama3.2:3b")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

#to load the documents 
def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory{docs_path} does not exist.Please check again")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        
    )
    documents = loader.load()

    if len(documents) ==0:
        raise FileNotFoundError(f"No txt files exit in {docs_path}")

    for i , doc in enumerate(documents[:2]):
        print(f"\n Documents {i+1}:")
        print(f"Source:{doc.metadata['source']}")
        print(f"Content length:{len(doc.page_content)} characters")
        print(f"Content preview:{doc.page_content[:100]}")
        print(f"metadata:{doc.metadata}")
    
    return documents

#to  split the documents in chunks

def split_documents(documents, chunk_size=800,chunk_overlap=0):
    print("splitting the documents in chunks") 
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,

    )
    
    chunks = text_splitter.split_documents(documents)
    if chunks:
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n---Chunk {i+1}---")
            print(f"Source:{chunk.metadata['source']}")
            print(f" length:{len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-"* 50)
        if len(chunks) > 5:
            print(f"\n... and {len(chunks)-5} more chunks")

    return chunks

def create_vector_store(chunks,persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDb")
  
    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    print("---Creating Vector Store ")
    vectorstore = Chroma(
        # documents=chunks, 
        collection_name="rag_documents",
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    batch_size = 20

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        vectorstore.add_documents(batch)

        completed = min(start + batch_size, len(chunks))
        print(f"Embedded {completed}/{len(chunks)} chunks")

    print("---Finished creating vector store---")
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():

    #first we load the documents 
    documents = load_documents(docs_path="docs")

    # SPLITTING DOCUMENT INTO CHUNK
    chunks = split_documents(documents)

    #creating a vector db

    vectorstore = create_vector_store(chunks)
if __name__ =="__main__":
    main()
