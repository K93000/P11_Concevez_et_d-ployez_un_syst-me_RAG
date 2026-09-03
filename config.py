import os
from dotenv import load_dotenv

load_dotenv()

# Modèles Mistral AI
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
EMBEDDING_MODEL = "mistral-embed"
LLM_MODEL = "mistral-small-latest"

# Retriever & Vector DB
TOP_K = 5
FAISS_INDEX_PATH = "faiss_index_events"

# Chunking
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100