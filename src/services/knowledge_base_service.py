from src.core.chunking import chunk_text
from src.core.document_loader import read_uploaded_documents
from src.core.embeddings import create_embeddings
from src.services.vector_db_service import get_pinecone_service
from src.utils.user_files import (
    add_user_file, 
    get_all_file_hashes,
)
import time
import hashlib


def build_knowledge_base(uploaded_files, user_id="default", check_duplicates_only=False):
    """
    Convert uploaded files into chunks, embeddings, and store in Pinecone.
    
    Args:
        uploaded_files: List of uploaded file objects (can be None for loading existing KB)
        user_id: User ID for multi-user support
        check_duplicates_only: If True, only check for duplicates without processing
    
    Returns:
        Dictionary with knowledge base info including model for querying, or duplicate info
    """
    if uploaded_files is None or len(uploaded_files) == 0:
        _, model = create_embeddings([{"text": "dummy", "source": "dummy"}])
        vector_db = get_pinecone_service()
        return {
            "chunks": [],
            "model": model,
            "vector_db": vector_db,
            "user_id": user_id,
            "vectors_count": 0,
            "initialized": True
        }
    
    documents = read_uploaded_documents(uploaded_files)
    if not documents:
        return None

    vector_db = get_pinecone_service()

    unique_sources = {}
    duplicate_files = set()
    new_files = {}

    for doc in documents:
        source, text = doc
        filename = source.split(" - Page ")[0] if " - Page " in source else source
        if filename not in unique_sources:
            unique_sources[filename] = ""
        unique_sources[filename] += text

    user_file_hashes = get_all_file_hashes(user_id)
    
    for filename, combined_text in unique_sources.items():
        file_content = combined_text.encode('utf-8')
        file_hash = hashlib.md5(file_content).hexdigest()

        if file_hash in user_file_hashes:
            duplicate_files.add(filename)
            print(f"📋 Duplicate detected (local DB): {filename} [hash: {file_hash[:8]}...]")
        elif vector_db.file_exists(file_hash, user_id):
            duplicate_files.add(filename)
            print(f"📋 Duplicate detected (Pinecone): {filename} [hash: {file_hash[:8]}...]")
        else:
            new_files[filename] = (file_hash, combined_text)
            print(f"✨ New file: {filename} [hash: {file_hash[:8]}...]")

    if len(duplicate_files) > 0 and len(new_files) == 0:
        print(f"⚠️ ALL FILES ARE DUPLICATES: {duplicate_files}")
        _, model = create_embeddings([{"text": "dummy", "source": "dummy"}])
        return {
            "error": "duplicate",
            "message": f"⚠️ ALL {len(duplicate_files)} file(s) are already in your knowledge base!",
            "details": "These files were previously uploaded. You can ask questions about them, but no new documents were added.",
            "duplicate_files": [{"name": name, "status": "already_uploaded"} for name in sorted(duplicate_files)],
            "chunks": [],
            "model": model,
            "vector_db": vector_db,
            "user_id": user_id,
            "vectors_count": 0,
            "initialized": True
        }

    if len(duplicate_files) > 0 and len(new_files) > 0:
        print(f"⚠️ MIXED: {len(duplicate_files)} duplicates + {len(new_files)} new files")

    if len(new_files) == 0:
        return None

    documents_to_process = []
    for doc in documents:
        source, text = doc
        filename = source.split(" - Page ")[0] if " - Page " in source else source
        if filename in new_files:
            documents_to_process.append(doc)

    if documents_to_process:
        chunks = chunk_text(documents_to_process)
    else:
        chunks = []

    if chunks:
        embeddings, model = create_embeddings(chunks)
    else:
        _, model = create_embeddings([{"text": "dummy", "source": "dummy"}])
        embeddings = []

    file_hash_map = {}
    for filename, (file_hash, _) in new_files.items():
        file_hash_map[filename] = file_hash

    vectors_batch = []
    for idx, chunk in enumerate(chunks):
        embedding_list = embeddings[idx].tolist() if hasattr(embeddings[idx], 'tolist') else embeddings[idx]

        source = chunk.get("source", "Unknown")
        filename = source.split(" - Page ")[0] if " - Page " in source else source

        file_hash = file_hash_map.get(filename, "unknown")
        chunk_id = f"{user_id}_{filename}_{idx}_{int(time.time())}"

        vectors_batch.append((
            chunk_id,
            embedding_list,
            {
                "text": chunk.get("text", ""),
                "source": chunk.get("source", "Unknown"),
                "chunk_id": chunk.get("chunk_id", idx),
                "user_id": user_id,
                "file_hash": file_hash,
                "timestamp": int(time.time())
            }
        ))

    try:
        if vectors_batch:
            vector_db.upsert_batch(vectors_batch)
            print(f"✅ Stored {len(vectors_batch)} vectors in Pinecone")

        for filename, (file_hash, _) in new_files.items():
            add_user_file(user_id, filename, file_hash)

    except Exception as e:
        print(f"❌ Error storing vectors: {e}")
        raise

    result = {
        "chunks": chunks,
        "model": model,
        "vector_db": vector_db,
        "user_id": user_id,
        "vectors_count": len(vectors_batch),
        "initialized": True
    }

    if duplicate_files:
        result["error"] = "mixed"
        result["message"] = f"✅ Processed {len(new_files)} new file(s)  |  ⚠️ Skipped {len(duplicate_files)} existing file(s)"
        result["details"] = "New files were added to your knowledge base. Duplicate files (previously uploaded) were skipped."
        result["duplicate_files"] = [{"name": name, "status": "already_uploaded"} for name in sorted(duplicate_files)]
        result["new_files"] = list(new_files.keys())
        print(f"✅ MIXED RESULT: Added {len(new_files)} new, skipped {len(duplicate_files)} duplicates")
    else:
        result["message"] = f"✅ Successfully processed {len(new_files)} file(s)"
        result["new_files"] = list(new_files.keys())
        print(f"✅ ALL NEW: Added {len(new_files)} files")
    
    return result