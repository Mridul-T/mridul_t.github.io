import os
from PyPDF2 import PdfReader
import chromadb
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm  # Import tqdm for progress tracking
import torch
import time

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Initialize SentenceTransformer model for embeddings

embedding_function = SentenceTransformerEmbeddings(model_name="./all-MiniLM-L6-v2")
# Initialize Chroma vector store for document embeddings
client = chromadb.PersistentClient(path="/data/chroma_db")

collection = client.get_or_create_collection("my_content_engine")

# Function to extract text from PDFs
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to generate embeddings for document content and store in vector store
def process_and_store_document(file_path, doc_id):
    # Extract text from the PDF
    text = extract_text_from_pdf(file_path)
    print(text[:300])
    # Split text into smaller chunks (you can customize chunking strategy)
    chunker = SemanticChunker(embedding_function)
    chunks = chunker.split_text(text)
    if not chunks:
        print(f"No chunks created from document {doc_id} at {file_path}")
        return
    embeddings=[]
    for text_chunk in chunks:
        if not text_chunk:
            print(f"Empty text chunk in document {doc_id} at {file_path}")
            continue
        try:
            embedding = embedding_function.embed_documents(text_chunk)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error embedding chunk: {e}")

    # Store embeddings in the vector store with progress tracking
    for idx, embedding in tqdm(enumerate(embeddings), total=len(embeddings), desc=f"Processing {doc_id}"):
        collection.add(
            documents=[doc_id],
            ids=[f"{doc_id}_{idx}"],
            embeddings=[embedding],
            metadatas=[{"chunk": chunks[idx]}]
        )

# Process each document and store embeddings
pdf_files = {
    "google_10k": "data/goog-10-k-2023.pdf",
    "tesla_10k": "data/tsla-20231231-gen.pdf",
    "uber_10k": "data/uber-10-k-2023.pdf"
}

for doc_id, file_path in tqdm(pdf_files.items(), desc="Processing documents"):
    process_and_store_document(file_path, doc_id)
