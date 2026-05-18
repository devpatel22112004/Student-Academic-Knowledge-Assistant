# 🔧 CELERY IMPLEMENTATION - STEP BY STEP

## Phase 1: Setup & Installation (Week 1)

### Day 1: Install Redis & Celery

#### Step 1: Install Redis

**On Mac:**
```bash
brew install redis
brew services start redis
redis-cli ping  # Should return: PONG
```

**On Linux:**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
redis-cli ping  # Should return: PONG
```

**On Windows:**
```bash
# Download from: https://redis.io/download
# Run installer
# Or use WSL (Windows Subsystem for Linux)
```

#### Step 2: Update requirements.txt

Add to your `requirements.txt`:
```
celery==5.3.1
redis==5.0.0
```

Then install:
```bash
pip install -r requirements.txt
```

#### Step 3: Verify Installation

```bash
# Test Redis
redis-cli ping
# Output: PONG ✅

# Test Celery
python -c "import celery; print(celery.__version__)"
# Output: 5.3.1 ✅
```

---

### Day 2-3: Create Celery Configuration

#### Step 1: Create `celery_app.py`

**File: celery_app.py**
```python
"""
Celery application configuration
Configure task queue and worker settings
"""

from celery import Celery
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Celery app
celery_app = Celery(
    'student_assistant',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

# Configuration
celery_app.conf.update(
    # Task serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    
    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task settings
    task_track_started=True,
    task_time_limit=30*60,  # 30 min hard limit
    task_soft_time_limit=25*60,  # 25 min soft limit
    
    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    
    # Worker settings
    worker_prefetch_multiplier=4,
    worker_max_tasks_per_child=100,
)

@celery_app.task(bind=True)
def debug_task(self):
    """Test task to verify Celery is working"""
    return f'Task {self.request.id} executed successfully!'
```

#### Step 2: Create `tasks.py`

**File: tasks.py**
```python
"""
Celery background tasks for file processing
"""

from celery_app import celery_app
from src.core.chunking import chunk_text
from src.core.document_loader import read_uploaded_documents
from src.core.embeddings import create_embeddings
from src.services.vector_db_service import get_pinecone_service
from src.utils.user_files import add_user_file
import hashlib
import time
import logging

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, max_retries=3)
def process_file_task(self, file_content, filename, user_id):
    """
    Process uploaded file in background
    
    Args:
        file_content: File content as string/bytes
        filename: Original filename
        user_id: User's email or ID
        
    Returns:
        Dictionary with success status and chunk count
    """
    try:
        logger.info(f"Starting to process file: {filename} for user: {user_id}")
        
        # Update status: Starting
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 0,
                'total': 100,
                'status': 'Reading document...'
            }
        )
        
        # 1. STEP 1: Read document
        logger.info("Step 1: Reading document...")
        documents = [(filename, file_content)]
        
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 25,
                'total': 100,
                'status': 'Chunking text...'
            }
        )
        
        # 2. STEP 2: Chunk text
        logger.info("Step 2: Chunking text...")
        chunks = chunk_text(documents)
        
        if not chunks:
            raise ValueError("No text found in file")
        
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 50,
                'total': 100,
                'status': f'Creating embeddings for {len(chunks)} chunks...'
            }
        )
        
        # 3. STEP 3: Create embeddings
        logger.info(f"Step 3: Creating embeddings for {len(chunks)} chunks...")
        embeddings, model = create_embeddings(chunks)
        
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 75,
                'total': 100,
                'status': 'Storing in database...'
            }
        )
        
        # 4. STEP 4: Store in Pinecone
        logger.info("Step 4: Storing vectors in Pinecone...")
        vector_db = get_pinecone_service()
        
        # Prepare vectors batch
        vectors_batch = []
        for idx, chunk in enumerate(chunks):
            embedding_list = embeddings[idx].tolist() if hasattr(embeddings[idx], 'tolist') else embeddings[idx]
            chunk_id = f"{user_id}_{filename}_{idx}_{int(time.time())}"
            
            vectors_batch.append((
                chunk_id,
                embedding_list,
                {
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", filename),
                    "user_id": user_id,
                    "filename": filename,
                    "chunk_id": idx,
                    "timestamp": int(time.time())
                }
            ))
        
        # Upload to Pinecone
        vector_db.upsert_batch(vectors_batch)
        logger.info(f"Stored {len(vectors_batch)} vectors in Pinecone")
        
        # 5. STEP 5: Update local tracking
        logger.info("Step 5: Updating file tracking...")
        file_hash = hashlib.md5(file_content.encode() if isinstance(file_content, str) else file_content).hexdigest()
        add_user_file(user_id, filename, file_hash)
        
        # Success!
        logger.info(f"Successfully processed file: {filename}")
        
        self.update_state(
            state='PROCESSING',
            meta={
                'current': 100,
                'total': 100,
                'status': 'Complete!'
            }
        )
        
        return {
            'status': 'success',
            'filename': filename,
            'chunks_processed': len(chunks),
            'vectors_stored': len(vectors_batch),
            'file_hash': file_hash
        }
        
    except Exception as exc:
        logger.error(f"Error processing file {filename}: {str(exc)}", exc_info=True)
        
        # Update status: Failed
        self.update_state(
            state='FAILURE',
            meta={
                'error': str(exc),
                'filename': filename
            }
        )
        
        # Retry up to 3 times with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@celery_app.task
def get_task_status(task_id):
    """
    Get the status of a task
    Used by frontend to check progress
    """
    from celery.result import AsyncResult
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        'state': result.state,
        'current': result.info.get('current', 0) if isinstance(result.info, dict) else 0,
        'total': result.info.get('total', 100) if isinstance(result.info, dict) else 100,
        'status': result.info.get('status', '') if isinstance(result.info, dict) else '',
    }


@celery_app.task
def cleanup_old_results():
    """
    Scheduled task to cleanup old results
    Run this once per day
    """
    logger.info("Cleaning up old results...")
    # Task results expire automatically after result_expires time
    logger.info("Cleanup complete")
    return 'Cleanup successful'
```

---

### Day 4: Test Celery Setup

**Test File: test_celery.py**
```python
"""Test if Celery is working correctly"""

import time
from celery_app import celery_app

def test_celery():
    # Send a simple test task
    print("Sending test task...")
    result = celery_app.send_task('celery_app.debug_task')
    
    # Wait for result
    print("Waiting for result (max 10 seconds)...")
    try:
        output = result.get(timeout=10)
        print(f"✅ SUCCESS: {output}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("Testing Celery setup...\n")
    
    # Verify Redis is running
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis is running\n")
    except:
        print("❌ Redis is NOT running!")
        print("Start Redis: redis-server")
        exit(1)
    
    # Test Celery
    if test_celery():
        print("\n✅ Celery is working correctly!")
        print("\nNext steps:")
        print("1. Modify sidebar.py to use Celery tasks")
        print("2. Run: celery -A tasks worker --loglevel=info")
        print("3. Run: streamlit run app.py")
    else:
        print("\n❌ Celery test failed!")
```

**Run test:**
```bash
# Terminal 1: Start Redis (if not running)
redis-server

# Terminal 2: Start Celery worker
celery -A tasks worker --loglevel=info

# Terminal 3: Run test
python test_celery.py
```

Expected output:
```
✅ Redis is running

Sending test task...
Waiting for result (max 10 seconds)...
✅ SUCCESS: Task xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx executed successfully!

✅ Celery is working correctly!
```

---

### Day 5: Integrate with Sidebar

**Modified: frontend/components/sidebar.py**
```python
import streamlit as st
import time
from tasks import process_file_task
from celery_app import celery_app
from celery.result import AsyncResult

def render_sidebar():
    """Render sidebar with file upload using Celery"""
    
    # ... existing code ...
    
    if st.button("Process Documents", use_container_width=True):
        if not files:
            st.warning("Please upload at least one PDF or TXT file.")
        else:
            # Get user_id
            user_id = st.session_state.current_user.get("email", "default") if st.session_state.current_user else "default"
            
            # Send all files to Celery tasks
            task_ids = []
            st.info(f"📤 Sending {len(files)} file(s) to processing queue...")
            
            for file in files:
                # Read file content once
                file_content = file.read()
                
                # Send to Celery (non-blocking!)
                task = process_file_task.delay(
                    file_content=file_content,
                    filename=file.name,
                    user_id=user_id
                )
                task_ids.append({
                    'id': task.id,
                    'name': file.name,
                    'started_at': time.time()
                })
            
            # Store in session
            st.session_state.processing_tasks = task_ids
            st.session_state.upload_started = time.time()
            
            st.success(f"✅ {len(files)} file(s) queued for processing!")
            st.info("📊 Processing status below:")
            
            # Show progress for each file
            progress_container = st.container()
            status_container = st.container()
            
            with progress_container:
                # Update progress for 30 seconds or until done
                for _ in range(30):
                    time.sleep(1)
                    
                    all_done = True
                    for task_info in task_ids:
                        task_result = AsyncResult(task_info['id'], app=celery_app)
                        
                        if task_result.state == 'PENDING':
                            st.write(f"⏳ {task_info['name']}: Waiting...")
                            all_done = False
                        elif task_result.state == 'PROCESSING':
                            progress = task_result.info.get('current', 0) / task_result.info.get('total', 100)
                            st.progress(progress, text=f"{task_info['name']}: {progress*100:.0f}%")
                            all_done = False
                        elif task_result.state == 'SUCCESS':
                            st.success(f"✅ {task_info['name']}: Complete!")
                        elif task_result.state == 'FAILURE':
                            st.error(f"❌ {task_info['name']}: Error")
                            all_done = False
                    
                    if all_done:
                        break
            
            # Set KB as initialized
            st.session_state.kb = {
                "initialized": True,
                "user_id": user_id
            }
            st.session_state.uploaded_names = [f.name for f in files]
```

---

## Phase 2: Progress Monitoring (Week 2)

### Add Status API

**File: status_api.py**
```python
"""API endpoints for task status monitoring"""

from celery.result import AsyncResult
from celery_app import celery_app

def get_task_status(task_id):
    """Get current status of a task"""
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        'id': task_id,
        'state': result.state,
        'current': result.info.get('current', 0) if isinstance(result.info, dict) else 0,
        'total': result.info.get('total', 100) if isinstance(result.info, dict) else 100,
        'status': result.info.get('status', '') if isinstance(result.info, dict) else '',
        'result': result.result if result.state == 'SUCCESS' else None,
        'error': str(result.info) if result.state == 'FAILURE' else None,
    }

def get_all_tasks_status(task_ids):
    """Get status of multiple tasks"""
    return [get_task_status(task_id) for task_id in task_ids]

def cancel_task(task_id):
    """Cancel a running task"""
    celery_app.control.revoke(task_id, terminate=True)
    return {'status': 'Task cancelled'}
```

---

## Phase 3: Production Setup (Week 3)

### Setup Celery Flower (Monitoring Dashboard)

```bash
# Install Flower
pip install flower

# Start Flower (separate terminal)
celery -A tasks --broker=redis://localhost:6379/0 flower --port=5555

# Access web UI
open http://localhost:5555
```

### Docker Setup (Optional)

**File: docker-compose.yml**
```yaml
version: '3'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  celery-worker:
    build: .
    command: celery -A tasks worker --loglevel=info
    depends_on:
      - redis
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
  
  flower:
    image: mher/flower
    command: celery --broker=redis://redis:6379/0 flower --port=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis
```

**Usage:**
```bash
docker-compose up
```

---

## 📊 Full Architecture After Implementation

```
┌──────────────────────────┐
│  Streamlit App           │
│  (User Interface)        │
│  ├─ Upload Files         │
│  ├─ Check Status         │
│  └─ See Results          │
└──────┬───────────────────┘
       │ (Send task)
       ↓
┌──────────────────────────┐
│  Redis Broker            │
│  (Message Queue)         │
│  ├─ Task 1: process file │
│  ├─ Task 2: process file │
│  └─ Task 3: process file │
└──────┬───────────────────┘
   ├────┬──────┬───────┐
   ↓    ↓      ↓       ↓
┌──────────────────────────┐
│  Celery Workers (1-4)    │
│  ├─ Worker 1: Processing │
│  ├─ Worker 2: Processing │
│  └─ Worker 3: Processing │
└──────────────────────────┘
   ├─ Pinecone (Vector DB)
   └─ Local JSON DB
```

---

## ✅ Verification Checklist

- [ ] Redis installed and running
- [ ] Celery and redis Python packages installed
- [ ] celery_app.py created
- [ ] tasks.py created
- [ ] Test Celery with test_celery.py (success)
- [ ] sidebar.py modified to use Celery
- [ ] Can process files in background
- [ ] Progress bar shows percentage
- [ ] Flower monitoring dashboard working
- [ ] Multiple workers can run
- [ ] Task retries working
- [ ] Production documentation complete

---

## 🚀 Running in Production

```bash
# Terminal 1: Start Redis
redis-server

# Terminal 2: Start Celery workers (4 workers)
celery -A tasks worker --loglevel=info -c 4

# Terminal 3: Start Flower (monitoring)
celery -A tasks flower

# Terminal 4: Start Streamlit app
streamlit run app.py

# Access:
# App: http://localhost:8501
# Flower: http://localhost:5555
```

---

## 💡 Tips & Troubleshooting

### Issue: Redis not found
```bash
# Check if Redis is running
redis-cli ping
# If error, start Redis:
redis-server
```

### Issue: "No module named celery"
```bash
pip install celery redis
```

### Issue: Tasks not being processed
```bash
# Check if worker is running
celery -A tasks worker --loglevel=debug

# Check Redis connection
redis-cli
ping  # Should return PONG
```

### Scale to 4 workers
```bash
celery -A tasks worker --loglevel=info -c 4
```

---

You now have a complete, production-ready Celery + Redis setup! 🎉
