from src.core.chunking import chunk_text
from src.core.document_loader import read_uploaded_documents
from src.core.embeddings import create_embeddings
from src.services.vector_db_service import get_pinecone_service
from src.utils.user_files import (
    add_user_file, 
    get_user_files, 
    file_hash_exists,
    get_all_file_hashes,
    get_file_names_by_hashes
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
    # If no files but KB requested (loading existing), just return initialized KB
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

    # Get Pinecone service
    vector_db = get_pinecone_service()
    
    # STEP 1: Deduplicate within current upload AND check with Pinecone
    unique_sources = {}
    duplicate_files = set()
    new_files = {}
    
    # Extract unique filenames (not pages) and compute file-level hash
    for doc in documents:
        # doc is a tuple: (source, text)
        # source looks like "rcbinfo.pdf - Page 1" or just "file.txt"
        source, text = doc
        
        # Extract just the filename (before " - Page" if exists)
        filename = source.split(" - Page ")[0] if " - Page " in source else source
        
        # Store combined text for this file (deduplicate within upload)
        if filename not in unique_sources:
            unique_sources[filename] = ""
        unique_sources[filename] += text
    
    # STEP 2: Check each unique file against BOTH local DB and Pinecone
    # This double-check ensures no duplicates slip through
    user_file_hashes = get_all_file_hashes(user_id)  # Check local DB first (faster)
    
    for filename, combined_text in unique_sources.items():
        file_content = combined_text.encode('utf-8')
        file_hash = hashlib.md5(file_content).hexdigest()
        
        # Check 1: Local database (fast, reliable)
        if file_hash in user_file_hashes:
            duplicate_files.add(filename)
            print(f"📋 Duplicate detected (local DB): {filename} [hash: {file_hash[:8]}...]")
        # Check 2: Pinecone (in case it wasn't in local DB yet)
        elif vector_db.file_exists(file_hash, user_id):
            duplicate_files.add(filename)
            print(f"📋 Duplicate detected (Pinecone): {filename} [hash: {file_hash[:8]}...]")
        else:
            new_files[filename] = (file_hash, combined_text)
            print(f"✨ New file: {filename} [hash: {file_hash[:8]}...]")
    
    # STEP 3: Handle different scenarios
    if len(duplicate_files) > 0 and len(new_files) == 0:
        # ❌ ALL files are duplicates - return initialized KB for querying
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
            "initialized": True  # KB is ready for querying!
        }
    
    if len(duplicate_files) > 0 and len(new_files) > 0:
        # ⚠️ MIXED: Some duplicates, some new - only process NEW files
        print(f"⚠️ MIXED: {len(duplicate_files)} duplicates + {len(new_files)} new files")
        # (will be handled below)
        pass
    
    # STEP 4: Process ONLY new files (skip duplicates - no re-embedding!)
    if len(new_files) == 0:
        # No new files to process
        return None
    
    # Filter documents to only include NEW files
    documents_to_process = []
    for doc in documents:
        source, text = doc
        filename = source.split(" - Page ")[0] if " - Page " in source else source
        if filename in new_files:
            documents_to_process.append(doc)
    
    # Only chunk and embed NEW files
    if documents_to_process:
        chunks = chunk_text(documents_to_process)
    else:
        chunks = []
    
    # Create embeddings (only for new chunks)
    if chunks:
        embeddings, model = create_embeddings(chunks)
    else:
        _, model = create_embeddings([{"text": "dummy", "source": "dummy"}])
        embeddings = []
    
    # Prepare vectors for batch upload
    # Create a map of filename to file_hash for all new files
    file_hash_map = {}
    for filename, (file_hash, _) in new_files.items():
        file_hash_map[filename] = file_hash
    
    vectors_batch = []
    for idx, chunk in enumerate(chunks):
        embedding_list = embeddings[idx].tolist() if hasattr(embeddings[idx], 'tolist') else embeddings[idx]
        
        # Get the source filename (remove " - Page X" suffix)
        source = chunk.get("source", "Unknown")
        filename = source.split(" - Page ")[0] if " - Page " in source else source
        
        # Use the file-level hash
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
    
    # Store embeddings in Pinecone
    try:
        if vectors_batch:
            vector_db.upsert_batch(vectors_batch)
            print(f"✅ Stored {len(vectors_batch)} vectors in Pinecone")
        
        # Add new files to user's file list (only new files, not duplicates)
        for filename, (file_hash, _) in new_files.items():
            add_user_file(user_id, filename, file_hash)
            
    except Exception as e:
        print(f"❌ Error storing vectors: {e}")
        raise
    
    # Return KB with new files result
    result = {
        "chunks": chunks,
        "model": model,
        "vector_db": vector_db,
        "user_id": user_id,
        "vectors_count": len(vectors_batch),
        "initialized": True
    }
    
    # If there were duplicates too, add that info
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
    
    # STEP 4: All files are NEW - proceed with processing
    # Chunk the documents
    chunks = chunk_text(documents)
    
    # Create embeddings
    embeddings, model = create_embeddings(chunks)