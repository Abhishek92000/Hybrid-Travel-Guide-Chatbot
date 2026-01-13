import os
from langchain_community.document_loaders import DirectoryLoader,UnstructuredFileLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import nltk

nltk.download("punkt")
nltk.download("stopwords")

print("Startin RAG ingestion pipeline")

Doc_path = r"D:\travel\rag_dataset"
Vector_db_path = r"D:\travel\rag_code\vector_data_set"
Collections_name = "place_info_v1"

embedding = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)

loader = DirectoryLoader(
    path=Doc_path,
    glob="*.pdf",
    loader_cls=UnstructuredFileLoader
)

documents = loader.load()
print(f"load documents: {len(documents)}")

for doc in documents:
    doc.metadata["source"] = doc.metadata.get("source", "unknown")
    doc.metadata["doc_type"] = "pdf"

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"raw chunks: {len(chunks)}")

clean_chunks = []
for i, chunk in enumerate(chunks):
    if len(chunk.page_content.strip())>50:
        chunk.metadata["chunk_id"] = f"{chunk.metadata['source']}_chunk_{i}"
        clean_chunks.append(chunk)


print(f"✅ Valid chunks: {len(clean_chunks)}")

# for i in clean_chunks:
#     print(i)
Chroma.from_documents(
    documents=clean_chunks,
    embedding=embedding,
    persist_directory=Vector_db_path,
    collection_name=Collections_name
)

print("🧠 Vector DB built successfully")
