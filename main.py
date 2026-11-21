"""
RAG Pipeline using LlamaIndex and Ollama
Loads PDF documents, creates vector index, and answers questions using local LLM.
"""

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def setup_rag():
    """Configure LLM, embeddings, and load documents to build index."""
    # Configure local Ollama LLM (smallest available: qwen2 0.5B)
    Settings.llm = Ollama(model="qwen2:0.5b", request_timeout=360.0)

    # Configure embeddings (per instructions: BAAI/bge-small-en-v1.5)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Load PDF documents from data directory
    documents = SimpleDirectoryReader("./data").load_data()

    if not documents:
        print("No PDF documents found in ./data directory.")
        print("Please add PDF files to the data/ folder and run again.")
        return None

    # Get unique file paths
    file_paths = set()
    for doc in documents:
        filepath = doc.metadata.get("file_path", "unknown")
        file_paths.add(filepath)

    print(f"Loaded {len(documents)} document chunk(s) from {len(file_paths)} file(s):")
    for fp in sorted(file_paths):
        print(f"  - {fp}")
    print("Building index...")

    # Create vector index from documents
    index = VectorStoreIndex.from_documents(documents)

    print("Index built successfully.")
    return index


def query_rag(index, question: str) -> str:
    """Query the RAG system with a question."""
    query_engine = index.as_query_engine(
        response_mode="compact"
    )
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


def main():
    """Main interactive Q&A loop."""
    print("=" * 50)
    print("RAG PDF Question-Answering System")
    print("Using LlamaIndex + Ollama (Llama 3) + BGE Embeddings")
    print("=" * 50)

    # Build the RAG index
    index = setup_rag()

    if index is None:
        return

    print("\nReady to answer questions. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if not question:
                continue

            print("\nSearching documents...")
            answer = query_rag(index, question)
            print(f"\nAnswer: {answer}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()