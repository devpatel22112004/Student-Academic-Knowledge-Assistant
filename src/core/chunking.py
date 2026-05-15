from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(all_documents):
    """Split document text into overlapping chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
    )

    chunks = []
    for source, text in all_documents:
        split_texts = text_splitter.split_text(text)
        for chunk in split_texts:
            chunks.append(
                {
                    "source": source,
                    "text": chunk,
                    "chunk_id": len(chunks),
                }
            )

    return chunks