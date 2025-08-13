def main():
    print("Hello from chatbot!")


if __name__ == "__main__":
    main()
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# DATA Path

DATA_PATH =r"C:\Desktop\Chatbot\Datasets"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
print("length od pdf pages:",len(documents))

# Create Chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                  chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Length of Text chunks", len(text_chunks))

# Vector Embedding

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()

# Store Embeddings

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)