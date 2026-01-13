🏝️ Mumbai Hybrid Travel Guide
An intelligent, context-aware travel assistant for Mumbai. This project uniquely combines Generative AI (RAG) for historical knowledge with Deterministic Machine Learning for precise logistics (Hotels, Restaurants, and Trips).

✨ Key Features
🧠 Intelligent RAG Brain: Leverages ChromaDB and Gemma-3 to answer complex questions about Mumbai's history and landmarks using PDF-sourced data.

🏨 Smart Hotel Matcher: Uses a Logistic Regression pipeline to recommend accommodations based on budget, area, and room type.

🍴 Restaurant Discovery: Specialized logic to find dining spots tailored to group size and cuisine preferences.

🗺️ Trip Planner & Fare Engine: A custom mathematical engine that calculates real-time Mumbai transport costs (Auto/Car/Bus) based on vehicle capacity and distance.


🧠 Technical Architecture & Stack
This project implements a Hybrid AI Architecture (RAG + Deterministic ML). This design pattern solves the "Hallucination Problem" by routing mathematical and structured data tasks to specialized ML models, while using the LLM only for natural language understanding and historical retrieval.

Orchestration Framework: LangChain

Used to manage the conversation flow, maintain chat_history, and build the retrieval chain.

Provides the interface for seamless communication between the LLM and the Vector Database.

LLM Engine: Ollama running Gemma-3:1b

A lightweight, state-of-the-art local model used for intent detection and response generation.

Vector Store: ChromaDB

A high-performance local vector database used to store and retrieve unstructured knowledge from PDF travel guides.

Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

A specialized NLP model used to convert text into 384-dimensional vectors for accurate semantic search.

ML Framework: Scikit-Learn

Powers the recommendation engines using Logistic Regression Pipelines, including StandardScaler for numerical normalization and OneHotEncoder for categorical features.

🔄 The Hybrid Workflow
Intent Classification: LangChain analyzes the user's input for specific keywords (Hotel, Food, Trip).

Contextual Retrieval: For general queries, LangChain uses the Chroma retriever to find the most relevant "chunks" of data from the PDF.

Deterministic Execution: For logistics, the system triggers custom Python functions where Scikit-Learn predicts the best matches based on a calculated Match Score.

Final Response: The bot assembles the data into a clean, Markdown-formatted response for the user.
