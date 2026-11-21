#!/usr/bin/env python3
"""
Test script for RAG system - runs tests programmatically without interactive input.
"""

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


def setup_rag():
    """Configure LLM, embeddings, and load documents to build index."""
    # Configure local Ollama LLM (Llama 3) with 360s timeout
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)

    # Configure local Ollama embedding model
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    # Load PDF documents from data directory
    documents = SimpleDirectoryReader("./data").load_data()

    if not documents:
        print("No PDF documents found in ./data directory.")
        return None

    print(f"Loaded {len(documents)} document(s). Building index...")

    # Create vector index from documents
    index = VectorStoreIndex.from_documents(documents)

    print("Index built successfully.")
    return index


def query_rag(index, question: str) -> str:
    """Query the RAG system with a question."""
    query_engine = index.as_query_engine(response_mode="compact")
    response = query_engine.query(question)
    response_text = str(response).strip()

    # Detect when information is not available in the PDF
    unavailable_phrases = [
        "cannot answer",
        "don't know",
        "not mentioned",
        "not found",
        "no information",
        "unable to answer",
        "insufficient information",
        "does not contain",
        "could not find",
    ]
    if not response_text or any(phrase in response_text.lower() for phrase in unavailable_phrases):
        return "Information not available in the provided PDF documents."

    return response_text


def run_tests():
    """Run all test cases."""
    print("=" * 60)
    print("RAG SYSTEM TESTS")
    print("=" * 60)

    # Build index
    index = setup_rag()
    if index is None:
        print("FAIL: Could not build index")
        return

    print("\n--- TEST 1: PDF Ingestion ---")
    print("PASS: PDF loaded and index built successfully")

    print("\n--- TEST 2: Relevant Question (Explicit in PDF) ---")
    q1 = "What is a vector database?"
    print(f"Question: {q1}")
    ans1 = query_rag(index, q1)
    print(f"Answer: {ans1}")
    if "vector database" in ans1.lower() or "specialized database" in ans1.lower():
        print("PASS: Answer contains relevant information from PDF")
    else:
        print("FAIL: Answer does not match PDF content")

    print("\n--- TEST 3: Technical Question (ANN Algorithms) ---")
    q2 = "What are the common ANN algorithms used in vector databases?"
    print(f"Question: {q2}")
    ans2 = query_rag(index, q2)
    print(f"Answer: {ans2}")
    if "hnsw" in ans2.lower() or "ivf" in ans2.lower() or "pq" in ans2.lower():
        print("PASS: Correct technical context retrieved")
    else:
        print("FAIL: Technical context not retrieved correctly")

    print("\n--- TEST 4: Unavailable Information ---")
    q3 = "What is the capital of France?"
    print(f"Question: {q3}")
    ans3 = query_rag(index, q3)
    print(f"Answer: {ans3}")
    if "not available" in ans3.lower() or "unavailable" in ans3.lower():
        print("PASS: Correctly indicates information unavailable")
    else:
        print("PARTIAL: Did not clearly indicate unavailable info")

    print("\n--- TEST 5: Specific Embedding Model Question ---")
    q4 = "What embedding model produces 384-dimensional vectors?"
    print(f"Question: {q4}")
    ans4 = query_rag(index, q4)
    print(f"Answer: {ans4}")
    if "bge-small" in ans4.lower() or "384" in ans4:
        print("PASS: Specific technical detail retrieved")
    else:
        print("FAIL: Specific detail not retrieved")

    print("\n--- TEST 6: Multiple Vector Databases ---")
    q5 = "Name some popular vector databases mentioned in the document."
    print(f"Question: {q5}")
    ans5 = query_rag(index, q5)
    print(f"Answer: {ans5}")
    dbs = ["pinecone", "weaviate", "chroma", "qdrant", "milvus"]
    found = [db for db in dbs if db in ans5.lower()]
    if len(found) >= 3:
        print(f"PASS: Found {len(found)} databases: {found}")
    else:
        print(f"PARTIAL: Only found {found}")

    print("\n--- TEST 7: Use Cases ---")
    q6 = "What are the use cases for vector databases?"
    print(f"Question: {q6}")
    ans6 = query_rag(index, q6)
    print(f"Answer: {ans6}")
    use_cases = ["semantic search", "recommendation", "rag", "anomaly", "image search"]
    found_uc = [uc for uc in use_cases if uc in ans6.lower()]
    if len(found_uc) >= 3:
        print(f"PASS: Found {len(found_uc)} use cases: {found_uc}")
    else:
        print(f"PARTIAL: Only found {found_uc}")

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()