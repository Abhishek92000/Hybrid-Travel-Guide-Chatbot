import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Configuration
vector_db_path = r"D:\travel\rag_code\vector_data_set"
collection_name = "place_info_v1"

# 1. Setup Embeddings (Must match the one used in ingestion)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Setup LLM
llm = ChatOllama(
    model="gemma3:1b",  # Fixed: removed trailing space
    temperature=0.2
)

# 3. Load Vector Store
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding,
    persist_directory=vector_db_path
)

# 4. Define Retriever
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={"k": 4}
)

# 5. Execute Query
query = "see give me all places name"
docs = retriever.invoke(query)

# 6. Format Context and Prompt
# Fixed: changed 'page_contant' to 'page_content'
context = "\n\n".join([doc.page_content for doc in docs])

# Added a "Safe Guard" to the prompt
prompt = f"""
You are a helpful travel assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I'm sorry, I don't have historical information on that place yet."

Context: 
{context}

Question: {query}
"""

# 7. Get Response
response = llm.invoke(prompt)
print(response.content) # Added .content to print only the text message