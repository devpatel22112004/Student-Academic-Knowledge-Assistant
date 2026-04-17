from pathlib import Path  # File path handling 
import re  # Regex for keyword extraction and sentence splitting
import faiss  # Fast vector similarity search
import numpy as np  # Array operations for embeddings
from pypdf import PdfReader  # Extract text from PDF pages
from sentence_transformers import SentenceTransformer  # Text to 384D embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Smart text chunking


# STEP 1: Find all PDF and TXT files in the data folder
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


# STEP 2: Read the content from PDF and TXT files
def read_document_content(file_path):
    """
    Reads content from a PDF or TXT file.
    Returns a list of (page_info, text_content) tuples.
    """
    content = [] #EMPTY LIST TO HOLD ALL CONTENT FROM THE FILE
    file_name = file_path.name 
    
    if file_path.suffix.lower() == ".pdf": 
        # Read PDF file
        reader = PdfReader(file_path)
        for page_num, page in enumerate(reader.pages, start=1): #LOOP THROUGH EACH PAGE IN THE PDF
            text = page.extract_text()
            content.append((f"{file_name} - Page {page_num}", text)) #APPEND THE PAGE INFO AND TEXT TO THE CONTENT LIST
                                                                    #("algorithms.pdf - Page 1", "Binary Search works by...")
    elif file_path.suffix.lower() == ".txt":
        # Read TXT file
        text = file_path.read_text(encoding="utf-8") #encoding="utf-8" isliye, taaki Hindi/English special characters sahi read ho saken.
        content.append((file_name, text))
    
    return content


# STEP 3: Break down text into smaller chunks
def chunk_text(all_documents):
    """
    Takes document content and splits it into manageable chunks.
    Each chunk is 700 characters with 120 character overlap.
    Returns a list of chunks with their source information.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )
    
    chunks = []
    for source, text in all_documents: #LOOP THROUGH EACH DOCUMENT CONTENT (SOURCE, TEXT) AND SPLIT THE TEXT INTO CHUNKS
        split_texts = text_splitter.split_text(text) 
        for i, chunk in enumerate(split_texts): #LOOP THROUGH EACH CHUNK AND APPEND IT TO THE CHUNKS LIST WITH SOURCE INFO AND CHUNK ID
            chunks.append({
                "source": source,
                "text": chunk,
                "chunk_id": len(chunks)
            })
    
    return chunks


# STEP 4: Convert text chunks into vector embeddings
def create_embeddings(chunks):
    """
    Converts text chunks into vector embeddings using a pre-trained model.
    This allows us to compare questions with document chunks by similarity.
    """
    # Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Extract just the text from chunks
    texts = [chunk["text"] for chunk in chunks] #LOOP THROUGH IN CHUNKS  ONLY EXTRACT TEXT ONLY
    
    # Convert texts to embeddings (vectors)
    embeddings = model.encode(texts) 
    
    return embeddings, model #RETURN BEACUSE QUERY ALSO WE ARE GOING TO USE THIS MODEL TO CONVERT QUERY INTO EMBEDDING


# STEP 5: Create FAISS index for fast similarity search
def build_search_index(embeddings): 
    """
    Creates a FAISS index from embeddings.
    This allows us to quickly find similar chunks to a query.
    """
    # Normalize embeddings and use cosine-like search with inner product.
    emb = np.array(embeddings, dtype=np.float32) #CONVERT EMBEDDINGS TO NUMPY ARRAY OF TYPE FLOAT32 BECAUSE FAISS WORKS WITH NUMPY ARRAYS
    faiss.normalize_L2(emb) 

    # Create FAISS index
    index = faiss.IndexFlatIP(emb.shape[1])
    
    # Add embeddings to the index
    index.add(emb)
    
    return index


def extract_keywords(text):
    """
    Extract simple keywords from query text for lexical matching.
    """
    stop_words = {
        "the", "is", "a", "an", "and", "or", "to", "of", "in", "on", "for",
        "from", "what", "who", "when", "where", "why", "how", "explain", "describe"
    }
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in stop_words]


def lexical_overlap_score(query, chunk_text):
    """
    Score chunk by keyword overlap with query.
    """
    q_keywords = set(extract_keywords(query))
    if not q_keywords:
        return 0.0
    c_words = set(re.findall(r"[a-zA-Z0-9]+", chunk_text.lower()))
    hits = len(q_keywords.intersection(c_words))
    return hits / len(q_keywords)


# STEP 6: Find relevant chunks for a user question
def find_relevant_chunks(question, index, chunks, model, num_results=5):
    """
    Takes a user question, finds similar chunks from documents.
    Returns the most relevant chunks.
    """
    # Convert question to embedding
    question_embedding = model.encode([question])
    question_embedding = np.array(question_embedding, dtype=np.float32)
    faiss.normalize_L2(question_embedding)
    
    # Search for top-k similar chunks
    # Retrieve larger candidate set, then rerank with lexical overlap.
    candidate_k = min(max(num_results * 3, 10), len(chunks))
    dense_scores, indices = index.search(question_embedding, candidate_k)

    candidates = []
    for rank, chunk_idx in enumerate(indices[0]):
        dense = float(dense_scores[0][rank])
        lexical = lexical_overlap_score(question, chunks[chunk_idx]["text"])
        # Weighted hybrid score improves exact-topic retrieval.
        hybrid = (0.7 * dense) + (0.3 * lexical)
        candidates.append((hybrid, chunk_idx))

    candidates.sort(key=lambda x: x[0], reverse=True)

    # Get top chunks after hybrid reranking.
    relevant_chunks = [chunks[idx] for _, idx in candidates[:num_results]]
    
    return relevant_chunks


def generate_extractive_answer(question, relevant_chunks, max_sentences=3):
    """
    Create a short grounded answer by selecting best-matching sentences
    from retrieved chunks.
    """
    keywords = set(extract_keywords(question))
    sentence_candidates = []

    for chunk in relevant_chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
        for sent in sentences:
            if len(sent.strip()) < 20:
                continue
            words = set(re.findall(r"[a-zA-Z0-9]+", sent.lower()))
            overlap = len(keywords.intersection(words)) if keywords else 0
            sentence_candidates.append((overlap, sent.strip(), chunk["source"]))

    sentence_candidates.sort(key=lambda x: x[0], reverse=True)
    selected = sentence_candidates[:max_sentences]

    if not selected:
        return "I could not find an exact sentence-level answer in the indexed documents.", []

    answer = " ".join([item[1] for item in selected])
    sources = [item[2] for item in selected]
    return answer, sources


# MAIN PROGRAM - Run everything step by step
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
        answer, answer_sources = generate_extractive_answer(question, relevant)
        
        print("\n" + "=" * 70)
        print(f"RESULTS FOR: {question}")
        print("=" * 70)

        print("\nBest Answer (from your documents):")
        print(answer)

        if answer_sources:
            print("\nAnswer Sources:")
            for src in sorted(set(answer_sources)):
                print(f"- {src}")
        
        for idx, chunk in enumerate(relevant, 1):
            print(f"\n[Result {idx}] - Source: {chunk['source']}")
            print(f"Content: {chunk['text'][:300]}...")
        
        print("\n" + "-" * 70)


if __name__ == "__main__":
    main()
