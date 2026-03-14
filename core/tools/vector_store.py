import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def build_vector_store():
    """Reads the 10-K, chunks it, and saves it to a local Chroma Vector Database."""
    file_path = "core/data/tsla_10k.txt"
    db_dir = "core/data/chroma_db"
    
    print("Initializing embedding model (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"Reading document from {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        print("Chunking text file...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=200
        )
        chunks = splitter.create_documents([text])
        print(f"Document successfully split into {len(chunks)} chunks.")
        
        print(f"Building ChromaDB Vector Store at {db_dir}...")
        print("embedding chunks into vectors...")
        

        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=db_dir
        )
        
        print("\nSuccess! Tesla 10-K has been embedded and saved to your local Vector Database.")

    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {file_path}")

    

if __name__ == "__main__":
    build_vector_store()