from unittest import result
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import OllamaEmbeddings,OllamaLLM

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(collection_name="rag_documents",
    persist_directory=persistent_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space":"cosine"}
)

model = OllamaLLM(model = "llama3.2:3b")

chat_history = []

def ask_question(user_question):
    print(f"\n-- You asked a question{user_question}"),

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be a standalone and searchable. Just return the rewritten questions"),
        ] + chat_history +[
            HumanMessage(content=f"New Question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question = result.strip()
        print(f"Searching for:{search_question}")
    else:
        search_question= user_question

    #initialising the retriever then invoking it 
    retriever = db.as_retriever(search_kwargs ={"k":3})
    docs= retriever.invoke(search_question)


    #find relevant documents
    print(f"Found {len(docs)} relevant documents")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"Doc {i}: {preview}")

    combined_input = f"""Based on the following documents, please answer the question:{user_question}

    Documents:{chr(10).join([f"-{doc.page_content}" for doc in docs])}

    Please provide a clear , helpful answer using only the information from these documents. 

    """

    messages = [ SystemMessage(content = "You are helpful assistant")]+ chat_history +[HumanMessage(content = combined_input)]

    result = model.invoke(messages)
    answer = result

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(SystemMessage(content=answer))

 
    print(f"Answer:{answer}")
    return answer 


def start_chat():
    print(f"Ask me questions. Type quit to exit the chat")

    while True:
        question= input("\n Your Question :")

        if question.lower() == 'quit':
             print("Goodbye")
             break
        ask_question(question)

if __name__ == '__main__':
    start_chat()

