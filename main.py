"""CLI entrypoint and compatibility re-exports for the core pipeline."""

from src.core.answer_generation import generate_extractive_answer
from src.core.chunking import chunk_text
from src.core.document_loader import find_all_documents, read_document_content
from src.core.embeddings import create_embeddings
from src.core.retrieval import build_search_index, find_relevant_chunks


def main():
    """Run the assistant from the terminal for quick local testing."""
    print("=" * 70)
    print("Student Academic Knowledge Assistant")
    print("=" * 70)

    print("\nStep 1/6: Looking for PDF and TXT files in the data folder...")
    documents = find_all_documents()

    if not documents:
        print("No PDF or TXT files were found in the data folder.")
        return

    print(f"Found {len(documents)} document(s):")
    for doc in documents:
        print(f"- {doc.name}")

    print("\nStep 2/6: Reading document content...")
    all_documents = []
    for doc in documents:
        content = read_document_content(doc)
        all_documents.extend(content)

    print(f"Read {len(all_documents)} page/file entries.")

    print("\nStep 3/6: Splitting text into chunks...")
    chunks = chunk_text(all_documents)
    print(f"Created {len(chunks)} chunks.")

    print("\nStep 4/6: Creating embeddings...")
    embeddings, model = create_embeddings(chunks)
    print(f"Created {len(embeddings)} embeddings.")

    print("\nStep 5/6: Building search index...")
    index = build_search_index(embeddings)
    print("Search index is ready.")

    print("\nStep 6/6: You can now ask questions.")
    print("-" * 70)

    while True:
        question = input("\nAsk a question (type 'quit' to exit): ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("\nThanks for using the assistant.")
            break

        if not question:
            print("Please type a valid question.")
            continue

        relevant = find_relevant_chunks(question, index, chunks, model)
        answer, answer_sources = generate_extractive_answer(question, relevant)

        print("\n" + "=" * 70)
        print(f"Results for: {question}")
        print("=" * 70)

        print("\nBest answer (from your documents):")
        print(answer)

        if answer_sources:
            print("\nAnswer sources:")
            for src in sorted(set(answer_sources)):
                print(f"- {src}")

        for idx, chunk in enumerate(relevant, 1):
            print(f"\nResult {idx} | Source: {chunk['source']}")
            print(f"Content: {chunk['text'][:300]}...")

        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()