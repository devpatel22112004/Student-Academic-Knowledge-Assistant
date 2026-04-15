
from pathlib import Path
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================================
# STEP 1: Find all PDF and TXT files in the data folder
# ============================================================================
def find_all_documents():
    """
    Finds all PDF and TXT files in the 'data' folder.
    Returns a list of file paths.
    """
    data_path = Path("data")
    documents = []
    
    # Search for PDF files
    documents.extend(data_path.rglob("*.pdf"))
    
    # Search for TXT files
    documents.extend(data_path.rglob("*.txt"))
    
    return sorted(documents)


# ============================================================================
# STEP 2: Read the content from PDF and TXT files
# ============================================================================
def read_document_content(file_path):
    """
    Reads content from a PDF or TXT file.
    Returns a list of (page_info, text_content) tuples.
    """
    content = []
    file_name = file_path.name
    
    if file_path.suffix.lower() == ".pdf":
        # Read PDF file
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            content.append((f"{file_name} - Page {page_num}", text))
    
    elif file_path.suffix.lower() == ".txt":
        # Read TXT file
        text = file_path.read_text(encoding="utf-8")
        content.append((file_name, text))
    
    return content


# ============================================================================
# STEP 3: Break down text into smaller chunks
# ============================================================================
def chunk_text(all_documents):
    """
    Takes document content and splits it into manageable chunks.
    Each chunk is 1000 characters with 200 character overlap.
    Returns a list of chunks with their source information.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    chunks = []
    for source, text in all_documents:
        split_texts = text_splitter.split_text(text)
        for i, chunk in enumerate(split_texts):
            chunks.append({
                "source": source,
                "text": chunk,
                "chunk_id": len(chunks)
            })
    
    return chunks


# ============================================================================
# STEP 4: Convert text chunks into vector embeddings
# ============================================================================
def create_embeddings(chunks):
    """
    Converts text chunks into vector embeddings using a pre-trained model.
    This allows us to compare questions with document chunks by similarity.
    """
    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Extract just the text from chunks
    texts = [chunk["text"] for chunk in chunks]
    
    # Convert texts to embeddings (vectors)
    embeddings = model.encode(texts)
    
    return embeddings, model


# ============================================================================
# STEP 5: Create FAISS index for fast similarity search
# ============================================================================
def build_search_index(embeddings):
    """
    Creates a FAISS index from embeddings.
    This allows us to quickly find similar chunks to a query.
    """
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    
    # Add embeddings to the index
    index.add(np.array(embeddings, dtype=np.float32))
    
    return index


# ============================================================================
# STEP 6: Find relevant chunks for a user question
# ============================================================================
def find_relevant_chunks(question, index, chunks, model, num_results=5):
    """
    Takes a user question, finds similar chunks from documents.
    Returns the most relevant chunks.
    """
    # Convert question to embedding
    question_embedding = model.encode([question])
    
    # Search for top-k similar chunks
    distances, indices = index.search(
        np.array(question_embedding, dtype=np.float32),
        num_results
    )
    
    # Get the actual chunks
    relevant_chunks = [chunks[i] for i in indices[0]]
    
    return relevant_chunks


# ============================================================================
# MAIN PROGRAM
# ============================================================================
def main():
    print("=" * 70)
    print("STUDENT ACADEMIC KNOWLEDGE ASSISTANT")
    print("=" * 70)
    
    # Step 1: Find documents
    print("\n[1/6] Finding PDF and TXT files in 'data' folder...")
    documents = find_all_documents()
    
    if not documents:
        print("❌ No PDF or TXT files found in 'data' folder!")
        return
    
    print(f"✓ Found {len(documents)} documents:")
    for doc in documents:
        print(f"   - {doc.name}")
    
    # Step 2: Read document content
    print("\n[2/6] Reading document content...")
    all_documents = []
    for doc in documents:
        content = read_document_content(doc)
        all_documents.extend(content)
    
    print(f"✓ Read {len(all_documents)} pages/files")
    
    # Step 3: Chunk text
    print("\n[3/6] Breaking down text into chunks...")
    chunks = chunk_text(all_documents)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 4: Create embeddings
    print("\n[4/6] Converting chunks to embeddings...")
    embeddings, model = create_embeddings(chunks)
    print(f"✓ Created {len(embeddings)} embeddings")
    
    # Step 5: Build search index
    print("\n[5/6] Building search index...")
    index = build_search_index(embeddings)
    print("✓ Index built successfully")
    
    # Step 6: Get user question and find answers
    print("\n[6/6] Ready for questions!")
    print("-" * 70)
    
    while True:
        question = input("\nYour question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Academic Knowledge Assistant!")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        # Find relevant chunks
        relevant = find_relevant_chunks(question, index, chunks, model)
        
        print("\n" + "=" * 70)
        print(f"RESULTS FOR: {question}")
        print("=" * 70)
        
        for idx, chunk in enumerate(relevant, 1):
            print(f"\n[Result {idx}] - Source: {chunk['source']}")
            print(f"Content: {chunk['text'][:300]}...")
        
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
