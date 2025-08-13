import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("API_Key")
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL = "qwen/qwen3-32b"

custom_prompt_template = """
You are a helpful, concise, and accurate assistant.
Use ONLY the information provided in the context to answer the user's question.
If the context does not contain enough information, clearly say: 
"I am sorry, I do not have enough information to answer that."

Context:
{context}

User Question:
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

def load_llm():
    return ChatGroq(model_name=GROQ_MODEL, temperature=0.5, max_tokens=512)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain():
    db = load_vectorstore()
    llm = load_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt()}
    )

# Query Interface

if __name__ == "__main__":
    qa_chain = build_qa_chain()
    user_query = input("Write Query here: ")
    response = qa_chain.invoke({'query': user_query})
    print("Result:", response["result"])
    print("SOURCE_DOCUMENTS:", response["source_documents"])
